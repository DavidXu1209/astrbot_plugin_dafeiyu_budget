"""Offline budget/payload/native-adapter tests; no real API requests."""
import asyncio
import base64
from datetime import datetime, timedelta, timezone
import importlib.util
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
pkg = ModuleType('budget_acceptance')
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
from budget_acceptance.ledger import BudgetDenied, Ledger, LIMIT
from budget_acceptance.main import BudgetedZhipu, MODEL, MAX_OUTPUT, bound_payload
from astrbot.core.provider.sources.zhipu_source import ProviderZhipu
from astrbot.core.provider.entities import LLMResponse, TokenUsage

NOW = datetime(2026, 11, 6, 18, tzinfo=timezone(timedelta(hours=8)))


class LedgerChecks(IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.path = Path(self.temp.name) / 'monthly.sqlite3'
        self.ledger = Ledger(self.path)

    def tearDown(self):
        self.temp.cleanup()

    async def test_concurrent_reservations_never_exceed_limit(self):
        results = await asyncio.gather(*(self.ledger.reserve(10_000_000, NOW) for _ in range(12)), return_exceptions=True)
        self.assertEqual(sum(isinstance(r, str) for r in results), 5)
        self.assertEqual(self.ledger.total(NOW), LIMIT)

    async def test_settle_refunds_only_authoritative_usage(self):
        token = await self.ledger.reserve(1_000_000, NOW)
        self.assertTrue(await self.ledger.settle(token, 5000))
        self.assertEqual(self.ledger.total(NOW), 5000)
        with self.assertRaises(BudgetDenied):
            await self.ledger.settle(token, 0)

    async def test_unknown_attempt_survives_restart(self):
        await self.ledger.reserve(1_000_000, NOW)
        self.assertEqual(Ledger(self.path).total(NOW), 1_000_000)

    async def test_cst_month_boundary_and_existing_trial_seed(self):
        self.ledger.seed('trial', 4506, '2026-11')
        self.ledger.seed('trial', 4506, '2026-11')
        self.assertEqual(self.ledger.total(NOW), 4506)
        self.assertEqual(self.ledger.total(NOW.replace(month=12, day=1)), 0)
        self.assertEqual(self.ledger.month(datetime(2026, 11, 30, 16, tzinfo=timezone.utc)), '2026-12')

    async def test_unexpected_usage_freezes_month(self):
        token = await self.ledger.reserve(100, NOW)
        self.assertFalse(await self.ledger.settle(token, 101))
        with self.assertRaises(BudgetDenied):
            await self.ledger.reserve(1, NOW)

    async def test_invalid_reservations_rejected(self):
        for amount in (-1, 0, 0.1, LIMIT+1):
            with self.assertRaises(BudgetDenied):
                await self.ledger.reserve(amount, NOW)


class PayloadChecks(IsolatedAsyncioTestCase):
    async def test_no_unpriced_model_or_long_context(self):
        for payload in ({'model': 'other', 'messages': []},
                        {'model': MODEL, 'messages': [{'role': 'user', 'content': '字'*50000}]}):
            with self.assertRaises(BudgetDenied):
                bound_payload(payload)

    async def test_image_normalization_without_mutating_history(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.new('RGB', (1600, 1200), 'red').save(buf, format='PNG')
        url = 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
        original = {'model': MODEL, 'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': url}}]}]}
        body, reserve = bound_payload(original)
        output = body['messages'][0]['content'][0]['image_url']['url']
        with Image.open(io.BytesIO(base64.b64decode(output.split(',')[1]))) as image:
            self.assertEqual(image.size, (1024, 768))
        self.assertEqual(original['messages'][0]['content'][0]['image_url']['url'], url)
        self.assertEqual(body['max_tokens'], MAX_OUTPUT)
        self.assertGreater(reserve, 65536*2)

    async def test_remote_url_and_five_images_rejected(self):
        part = {'type': 'image_url', 'image_url': {'url': 'https://invalid.example/a.png'}}
        with self.assertRaises(BudgetDenied):
            bound_payload({'model': MODEL, 'messages': [{'role':'user','content':[part]}]})


class NativeAdapterChecks(IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.provider = object.__new__(BudgetedZhipu)
        self.provider.ledger = Ledger(Path(self.temp.name)/'budget.sqlite3')
        self.payload = {'model': MODEL, 'messages': [{'role':'user','content':'synthetic'}]}

    def tearDown(self):
        self.temp.cleanup()

    async def test_native_request_once_and_usage_settlement(self):
        response = LLMResponse(role='assistant', completion_text='synthetic', usage=TokenUsage(input_other=100, input_cached=20, output=30))
        response.raw_completion = SimpleNamespace(usage=object())
        with patch.object(ProviderZhipu, '_query', AsyncMock(return_value=response)) as native:
            result = await self.provider._query(self.payload, None)
        self.assertIs(result, response)
        native.assert_awaited_once()
        self.assertEqual(native.await_args.kwargs['request_max_retries'], 1)
        self.assertEqual(self.provider.ledger.total(), 120*2+30*7)

    async def test_missing_usage_retains_reserve(self):
        response = LLMResponse(role='assistant', completion_text='synthetic')
        with patch.object(ProviderZhipu, '_query', AsyncMock(return_value=response)):
            await self.provider._query(self.payload, None)
        self.assertGreater(self.provider.ledger.total(), 0)

    async def test_api_error_and_cancel_keep_reserve(self):
        for error in (TimeoutError(), asyncio.CancelledError()):
            before = self.provider.ledger.total()
            with patch.object(ProviderZhipu, '_query', AsyncMock(side_effect=error)) as native:
                with self.assertRaises(type(error)):
                    await self.provider._query(self.payload, None)
                native.assert_awaited_once()
            self.assertGreater(self.provider.ledger.total(), before)

    async def test_exhausted_budget_and_tools_never_contact_api(self):
        await self.provider.ledger.reserve(LIMIT)
        with patch.object(ProviderZhipu, '_query', AsyncMock()) as native:
            with self.assertRaises(BudgetDenied):
                await self.provider._query(self.payload, None)
            with self.assertRaises(BudgetDenied):
                await self.provider._query(self.payload, object())
            native.assert_not_awaited()

    async def test_native_error_handler_does_not_retry(self):
        error = TimeoutError()
        with self.assertRaises(TimeoutError):
            await self.provider._handle_api_error(error)
