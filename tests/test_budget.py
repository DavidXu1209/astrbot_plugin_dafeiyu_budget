"""Offline budget/payload/native-adapter tests; no real API requests."""
import asyncio
import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]


def _install_astrbot_stub():
    """Let the suite run without a full AstrBot install. Real packages win."""
    try:
        import astrbot.core.provider.entities  # noqa: F401
        import astrbot.core.provider.sources.zhipu_source  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    @dataclass
    class TokenUsage:
        input_other: int = 0
        input_cached: int = 0
        output: int = 0

        @property
        def input(self):
            return self.input_other + self.input_cached

    @dataclass
    class LLMResponse:
        role: str
        completion_text: str = ''
        usage: TokenUsage | None = None
        raw_completion: object = None

    class ProviderZhipu:
        async def _query(self, payloads, tools, *, request_max_retries=None, conversation_id=None):
            raise NotImplementedError

        async def _handle_api_error(self, error, *args, **kwargs):
            raise error

    class _Logger:
        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            pass

    def _package(name):
        module = ModuleType(name)
        module.__path__ = []
        module.__package__ = name
        sys.modules[name] = module
        return module

    def _module(name, **attrs):
        module = ModuleType(name)
        module.__package__ = name.rpartition('.')[0]
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    class ProviderType:
        CHAT_COMPLETION = 'chat_completion'

    class Star:
        def __init__(self, context):
            self.context = context

    class StarTools:
        @staticmethod
        def get_data_dir(name):
            return Path('.')

    def register_provider_adapter(*args, **kwargs):
        def decorate(cls):
            return cls
        return decorate

    _package('astrbot')
    api = _package('astrbot.api')
    api.logger = _Logger()
    _module('astrbot.api.provider', ProviderType=ProviderType)
    _module('astrbot.api.star', Star=Star, StarTools=StarTools)
    _package('astrbot.core')
    _package('astrbot.core.provider')
    _module(
        'astrbot.core.provider.register',
        provider_cls_map={},
        provider_registry=[],
        register_provider_adapter=register_provider_adapter,
    )
    _package('astrbot.core.provider.sources')
    _module('astrbot.core.provider.sources.zhipu_source', ProviderZhipu=ProviderZhipu)
    _module('astrbot.core.provider.entities', LLMResponse=LLMResponse, TokenUsage=TokenUsage)


_install_astrbot_stub()
pkg = ModuleType('budget_acceptance')
pkg.__path__ = [str(ROOT)]
sys.modules[pkg.__name__] = pkg
from budget_acceptance.ledger import BudgetDenied, Ledger, LIMIT
from budget_acceptance.main import (
    INPUT_MICRO_YUAN,
    MAX_OUTPUT,
    MODEL,
    OUTPUT_MICRO_YUAN,
    BudgetedZhipu,
    bound_payload,
    prepare_provider_config,
)
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
        for amount in (-1, 0, True, 0.1, LIMIT+1):
            with self.assertRaises(BudgetDenied):
                await self.ledger.reserve(amount, NOW)

    async def test_freeze_stops_at_the_month_boundary(self):
        token = await self.ledger.reserve(100, NOW)
        self.assertFalse(await self.ledger.settle(token, 101))
        december = NOW.replace(month=12, day=1)
        self.assertEqual(self.ledger.total(december), 0)
        self.assertIsInstance(await self.ledger.reserve(1, december), str)

    async def test_seed_rejects_freeze_marker_names(self):
        with self.assertRaises(ValueError):
            self.ledger.seed('freeze-2026-11', 1, '2026-11')


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

    def _png(self, size=(8, 8)):
        from PIL import Image
        buf = io.BytesIO()
        Image.new('RGB', size, 'red').save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

    async def test_remote_url_and_five_images_rejected(self):
        remote = {'type': 'image_url', 'image_url': {'url': 'https://invalid.example/a.png'}}
        with self.assertRaises(BudgetDenied):
            bound_payload({'model': MODEL, 'messages': [{'role':'user','content':[remote]}]})
        images = [{'type': 'image_url', 'image_url': {'url': self._png()}} for _ in range(5)]
        with self.assertRaises(BudgetDenied):
            bound_payload({'model': MODEL, 'messages': [{'role':'user','content': images}]})

    async def test_unpriced_media_malformed_image_and_smuggled_bytes(self):
        for part in (
            {'type': 'input_audio', 'input_audio': {'data': 'abc', 'format': 'mp3'}},
            {'type': 'video_url', 'video_url': {'url': 'https://invalid.example/a.mp4'}},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(b'not-an-image').decode()}},
            {'type': 'image_url', 'image_url': {'url': self._png()}, 'smuggle': '字' * 50000},
        ):
            with self.assertRaises(BudgetDenied):
                bound_payload({'model': MODEL, 'messages': [{'role': 'user', 'content': [part]}]})

    async def test_provider_config_rejects_unofficial_base_and_blank_key(self):
        with self.assertRaises(BudgetDenied):
            prepare_provider_config({'api_base': 'https://example.invalid/v4', 'key': ['sk-test']})
        with self.assertRaises(BudgetDenied):
            prepare_provider_config({'api_base': 'https://open.bigmodel.cn/api/paas/v4/', 'key': ['  ']})
        config = prepare_provider_config({
            'api_base': 'https://open.bigmodel.cn/api/paas/v4',
            'key': 'sk-test',
        })
        self.assertEqual(config['key'], ['sk-test'])
        self.assertEqual(config['custom_extra_body']['max_tokens'], MAX_OUTPUT)


class NativeAdapterChecks(IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.provider = object.__new__(BudgetedZhipu)
        self.provider.ledger = Ledger(Path(self.temp.name)/'budget.sqlite3')
        self.provider.api_keys = ['sk-test']
        self.payload = {'model': MODEL, 'messages': [{'role':'user','content':'synthetic'}]}

    def _priced(self, input_other=100, input_cached=20, output=30, raw=None):
        response = LLMResponse(
            role='assistant',
            completion_text='synthetic',
            usage=TokenUsage(input_other=input_other, input_cached=input_cached, output=output),
        )
        response.raw_completion = SimpleNamespace(usage=object() if raw is None else raw)
        return response

    def tearDown(self):
        self.temp.cleanup()

    async def test_native_request_once_and_usage_settlement(self):
        response = self._priced()
        with patch.object(ProviderZhipu, '_query', AsyncMock(return_value=response)) as native:
            result = await self.provider._query(self.payload, None)
        self.assertIs(result, response)
        native.assert_awaited_once()
        self.assertEqual(native.await_args.kwargs['request_max_retries'], 1)
        self.assertEqual(self.provider.ledger.total(), 120*INPUT_MICRO_YUAN+30*OUTPUT_MICRO_YUAN)

    async def test_raw_usage_and_reasoning_tokens_cannot_underbill(self):
        response = self._priced(0, 0, 0, raw={
            'prompt_tokens': 10.2,
            'completion_tokens': 4,
            'completion_tokens_details': {'reasoning_tokens': 9},
        })
        with patch.object(ProviderZhipu, '_query', AsyncMock(return_value=response)):
            await self.provider._query(self.payload, None)
        self.assertEqual(self.provider.ledger.total(), 11*INPUT_MICRO_YUAN+9*OUTPUT_MICRO_YUAN)
        # reasoning_tokens is a subset of completion_tokens here, so it must not be added twice.
        subset = self._priced(0, 0, 0, raw={
            'prompt_tokens': 8,
            'completion_tokens': 40,
            'completion_tokens_details': {'reasoning_tokens': 9},
        })
        with patch.object(ProviderZhipu, '_query', AsyncMock(return_value=subset)):
            await self.provider._query(self.payload, None)
        self.assertEqual(
            self.provider.ledger.total(),
            11*INPUT_MICRO_YUAN+9*OUTPUT_MICRO_YUAN + 8*INPUT_MICRO_YUAN+40*OUTPUT_MICRO_YUAN,
        )

    async def test_zero_usage_keeps_reserve_and_garbage_freezes_month(self):
        response = self._priced(0, 0, 0)
        with patch.object(ProviderZhipu, '_query', AsyncMock(return_value=response)):
            await self.provider._query(self.payload, None)
        kept = self.provider.ledger.total()
        self.assertGreater(kept, 0)
        bad = self._priced(0, 0, 0, raw={'prompt_tokens': -1, 'completion_tokens': 3})
        with patch.object(ProviderZhipu, '_query', AsyncMock(return_value=bad)):
            await self.provider._query(self.payload, None)
        with self.assertRaises(BudgetDenied):
            await self.provider.ledger.reserve(1)

    async def test_settle_failure_still_returns_reply(self):
        response = self._priced()
        with patch.object(ProviderZhipu, '_query', AsyncMock(return_value=response)):
            with patch.object(self.provider.ledger, 'settle', AsyncMock(side_effect=RuntimeError('db'))):
                result = await self.provider._query(self.payload, None)
        self.assertIs(result, response)
        self.assertGreater(self.provider.ledger.total(), 0)

    async def test_missing_key_never_contacts_api(self):
        self.provider.api_keys = ['']
        with patch.object(ProviderZhipu, '_query', AsyncMock()) as native:
            with self.assertRaises(BudgetDenied):
                await self.provider._query(self.payload, None)
            native.assert_not_awaited()
            self.assertEqual(self.provider.ledger.total(), 0)

    async def test_stream_is_one_metered_request(self):
        response = self._priced()
        with patch.object(ProviderZhipu, '_query', AsyncMock(return_value=response)) as native:
            chunks = [item async for item in self.provider._query_stream(self.payload, None)]
        self.assertEqual(chunks, [response])
        native.assert_awaited_once()
        self.assertEqual(self.provider.ledger.total(), 120*INPUT_MICRO_YUAN+30*OUTPUT_MICRO_YUAN)

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
