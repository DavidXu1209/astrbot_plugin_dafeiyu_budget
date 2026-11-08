"""Reuse AstrBot's native Zhipu adapter, meter every paid outbound query."""
import asyncio
import base64
import copy
import io
import json
import math
import threading

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.provider import ProviderType
from astrbot.api.star import Star, StarTools
from astrbot.core.provider.register import provider_cls_map, provider_registry, register_provider_adapter
from astrbot.core.provider.sources.zhipu_source import ProviderZhipu

from .ledger import BudgetDenied, Ledger

TYPE = 'dafeiyu_budgeted_zhipu'
MODEL = 'glm-5.3-flashx'
MAX_OUTPUT = 2048
OFFICIAL_API_BASE = 'https://open.bigmodel.cn/api/paas/v4'
# Micro-yuan per token. Cached input is billed at the full input price.
INPUT_MICRO_YUAN = 2
OUTPUT_MICRO_YUAN = 7
# Above this, micro-yuan cost no longer fits a signed 64-bit SQLite integer.
MAX_USAGE_TOKENS = (2**62) // 8
_IMAGE_LOCK = threading.Lock()
_MAX_PIXELS = 40_000_000
_UNPRICED_PARTS = frozenset({
    'audio', 'audio_url', 'input_audio',
    'video', 'video_url', 'input_video',
    'file', 'input_file', 'document',
    'image',
})


def has_api_key(keys):
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list):
        return False
    return any(isinstance(item, str) and item.strip() for item in keys)


def prepare_provider_config(provider_config):
    """Copy provider settings and pin every field the ledger's prices assume."""
    config = copy.deepcopy(provider_config)
    api_base = config.get('api_base', '')
    if not isinstance(api_base, str) or api_base.rstrip('/') != OFFICIAL_API_BASE:
        raise BudgetDenied('大肥鱼钱包只走智谱官方按量入口。')
    if not has_api_key(config.get('key')):
        raise BudgetDenied('大肥鱼钱包还没放进智谱按量 API Key。')
    if isinstance(config.get('key'), str):
        config['key'] = [config['key']]
    config['custom_extra_body'] = {
        'thinking': {'type': 'enabled'},
        'reasoning_effort': 'low',
        'max_tokens': MAX_OUTPUT,
    }
    return config


def _tokens(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('token count is not a number')
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('token count is not finite')
        value = math.ceil(value)
    if value < 0 or value > MAX_USAGE_TOKENS:
        raise ValueError('token count is out of range')
    return int(value)


def _optional_token(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        if name not in obj:
            return None
        value = obj[name]
    elif not hasattr(obj, name):
        return None
    else:
        value = getattr(obj, name)
    if value is None:
        return None
    return _tokens(value)


def _child(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _normalized_pair(usage):
    if usage is None:
        return None
    try:
        return (_tokens(usage.input), _tokens(usage.output))
    except (AttributeError, TypeError, ValueError):
        return 'bad'


def _raw_pair(usage):
    if usage is None:
        return None
    try:
        prompt = _optional_token(usage, 'prompt_tokens')
        completion = _optional_token(usage, 'completion_tokens')
        details = _child(usage, 'completion_tokens_details')
        reasoning = _optional_token(details, 'reasoning_tokens') if details is not None else None
    except (TypeError, ValueError):
        return 'bad'
    if prompt is None and completion is None and reasoning is None:
        return None
    output_values = [value for value in (completion, reasoning) if value is not None]
    return (prompt or 0, max(output_values) if output_values else 0)


def quote_usage(response):
    """Return micro-yuan cost, None when usage is missing, or 'bad' when unreadable.

    Adapter usage and the provider raw usage are both read. The charge is the
    higher input and the higher output, so a parsed TokenUsage of zeros cannot
    refund a reserve when the raw payload still reports tokens. Reasoning tokens
    count when they exceed completion tokens.
    """
    normalized = _normalized_pair(getattr(response, 'usage', None))
    raw_usage = getattr(getattr(response, 'raw_completion', None), 'usage', None)
    raw = _raw_pair(raw_usage)
    if normalized == 'bad' or raw == 'bad':
        return 'bad'
    pairs = [pair for pair in (normalized, raw) if isinstance(pair, tuple)]
    if not pairs:
        return None
    incoming = max(pair[0] for pair in pairs)
    outgoing = max(pair[1] for pair in pairs)
    if incoming == 0 and outgoing == 0:
        return None
    return incoming * INPUT_MICRO_YUAN + outgoing * OUTPUT_MICRO_YUAN


def _compress_image(url):
    from PIL import Image, ImageOps
    if (
        not isinstance(url, str)
        or not url.startswith('data:image/')
        or len(url) > 24_000_000
        or 'base64' not in url.split(',', 1)[0]
    ):
        raise BudgetDenied('图片必须是内联的 data:image，大肥鱼钱包才肯付。')
    try:
        encoded = url.split(',', 1)[1]
        raw = base64.b64decode(encoded, validate=True)
    except Exception:
        raise BudgetDenied('图片必须是内联的 data:image，大肥鱼钱包才肯付。') from None
    with _IMAGE_LOCK:
        previous_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = _MAX_PIXELS
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > _MAX_PIXELS:
                    raise BudgetDenied('图片太大，大肥鱼钱包没有预留。')
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((1024, 1024))
                stream = io.BytesIO()
                image.save(stream, format='JPEG', quality=85)
        except BudgetDenied:
            raise
        except Exception as exc:
            logger.warning('图片解析失败：%s', type(exc).__name__)
            raise BudgetDenied('图片打不开，大肥鱼钱包没有预留。') from None
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()


def bound_payload(payload):
    """Bound text bytes and static visual size before reserving any money."""
    if not isinstance(payload, dict):
        raise BudgetDenied('不支持的请求内容。')
    if payload.get('model') != MODEL:
        raise BudgetDenied('大肥鱼钱包只给 glm-5.3-flashx 付过价。')
    result = copy.deepcopy(payload)
    count = 0

    def walk(value):
        nonlocal count
        if isinstance(value, dict):
            part_type = value.get('type')
            if part_type in _UNPRICED_PARTS:
                raise BudgetDenied('大肥鱼钱包只支付文本和已核价的图片。')
            if part_type == 'image_url':
                count += 1
                if count > 4:
                    raise BudgetDenied('单次最多 4 张图片，多出来的大肥鱼钱包不付。')
                image_url = value.get('image_url')
                if not isinstance(image_url, dict) or 'url' not in image_url:
                    raise BudgetDenied('图片必须是内联的 data:image，大肥鱼钱包才肯付。')
                value['image_url']['url'] = _compress_image(image_url['url'])
                measured = {}
                for key, item in value.items():
                    if key != 'image_url':
                        measured[key] = walk(item)
                        continue
                    measured[key] = {
                        inner: ('[bounded 1024px image]' if inner == 'url' else walk(inner_value))
                        for inner, inner_value in item.items()
                    }
                return measured
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise BudgetDenied('不支持的请求内容。')

    measured = walk(result)
    text_bytes = len(json.dumps(measured, ensure_ascii=False).encode())
    if text_bytes > 131072:
        raise BudgetDenied('这次太长了，大肥鱼钱包不肯预留，请新开会话。')
    result['max_tokens'] = MAX_OUTPUT
    # Text byte fallback upper bound plus 65,536 input-token allowance per
    # 1024px image and fixed request overhead. Cached input uses full price.
    reserve = (text_bytes + count * 65536 + 4096) * INPUT_MICRO_YUAN + MAX_OUTPUT * OUTPUT_MICRO_YUAN
    return result, reserve


def format_yuan(micro):
    """Format an integer micro-yuan amount as yuan without binary floats."""
    if isinstance(micro, bool) or not isinstance(micro, int):
        raise ValueError('yuan amount must be an integer number of micro-yuan')
    sign = '-' if micro < 0 else ''
    whole, frac = divmod(abs(micro), 1_000_000)
    body = f'{whole}.{frac:06d}'.rstrip('0').rstrip('.')
    if '.' not in body:
        body += '.00'
    elif len(body.split('.', 1)[1]) == 1:
        body += '0'
    return f'{sign}{body} 元'


def render_wallet(summary):
    states = summary['states']
    spent = states.get('settled', 0) + states.get('overrun', 0) + states.get('prior_trial', 0)
    held = states.get('reserved', 0)
    lines = [
        f"大肥鱼钱包 · {summary['month']}",
        f"额度 {format_yuan(summary['limit'])}",
        f"已花 {format_yuan(spent)}",
        f"预留未结 {format_yuan(held)}",
        f"还剩 {format_yuan(summary['remaining'])}",
    ]
    if summary['frozen']:
        lines.append('本月已冻上，下月自动解开。')
    return '\n'.join(lines)


@register_provider_adapter(TYPE, '大肥鱼的共享月预算钱包',
    provider_type=ProviderType.CHAT_COMPLETION, provider_display_name='大肥鱼钱包（50元/月）',
    default_config_tmpl={'id': 'dafeiyu_wallet', 'provider': 'zhipu', 'type': TYPE,
                         'provider_type': 'chat_completion', 'enable': True, 'key': [],
                         'model': MODEL,
                         'api_base': 'https://open.bigmodel.cn/api/paas/v4/', 'timeout': 45})
class BudgetedZhipu(ProviderZhipu):
    def __init__(self, provider_config, provider_settings):
        config = prepare_provider_config(provider_config)
        super().__init__(config, provider_settings)
        self.ledger = Ledger(StarTools.get_data_dir('astrbot_plugin_dafeiyu_budget') / 'monthly.sqlite3')

    async def _settle_response(self, token, response):
        try:
            quoted = quote_usage(response)
            if quoted is None:
                logger.warning('未获得权威用量，本次预留额度保留。')
                return
            if quoted == 'bad':
                await self.ledger.freeze(token)
                logger.error('智谱用量无法核验，本月付费通道已冻结。')
                return
            if not await self.ledger.settle(token, quoted):
                logger.error('智谱返回用量超出预留估算，本月付费通道已冻结。本次核算 %s 微元。', quoted)
        except Exception as exc:
            logger.error('结算失败，本次预留额度保留：%s', type(exc).__name__)

    async def _query(self, payloads, tools, *, request_max_retries=None, conversation_id=None):
        if tools:
            raise BudgetDenied('大肥鱼钱包还没开放自动工具调用。')
        if not has_api_key(getattr(self, 'api_keys', None)):
            raise BudgetDenied('大肥鱼钱包还没放进智谱按量 API Key。')
        payloads, reserve = await asyncio.to_thread(bound_payload, payloads)
        token = await self.ledger.reserve(reserve)
        # Unknown failures/cancellation retain the reserve; there is no refund
        # without authoritative usage. request_max_retries counts the first
        # attempt, so 1 disables the native transport retry.
        response = await super()._query(payloads, tools, request_max_retries=1, conversation_id=conversation_id)
        await self._settle_response(token, response)
        return response

    async def _query_stream(self, payloads, tools, **kwargs):
        # All current QQ/background tasks use non-streaming. Keep one metered
        # request even if a caller asks for a stream; never create a second run.
        yield await self._query(payloads, tools, **kwargs)

    async def _handle_api_error(self, error, *args, **kwargs):
        raise error  # No native repair/retry loop for paid requests.


class BudgetPlugin(Star):
    def __init__(self, context, config=None):
        super().__init__(context)
        self.config = config

    def _ledger(self):
        return Ledger(StarTools.get_data_dir('astrbot_plugin_dafeiyu_budget') / 'monthly.sqlite3')

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command('钱包', alias={'余额'})
    async def wallet(self, event):
        """查看大肥鱼钱包本月余额。仅管理员。"""
        yield event.plain_result(render_wallet(await self._ledger().summary()))

    async def initialize(self):
        manager = self.context.provider_manager
        for config in manager.providers_config:
            merged = manager.get_merged_provider_config(config)
            provider_id = merged.get('id')
            if (
                merged.get('type') == TYPE
                and merged.get('enable')
                and provider_id
                and provider_id not in manager.inst_map
            ):
                await manager.load_provider(config)

    async def terminate(self):
        manager = self.context.provider_manager
        for provider_id, provider in list(manager.inst_map.items()):
            if isinstance(provider, BudgetedZhipu):
                await manager.terminate_provider(provider_id)
        metadata = provider_cls_map.get(TYPE)
        if metadata and metadata.cls_type is BudgetedZhipu:
            provider_cls_map.pop(TYPE, None)
            if metadata in provider_registry:
                provider_registry.remove(metadata)
