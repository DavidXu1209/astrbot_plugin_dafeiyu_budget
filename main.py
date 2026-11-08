"""Reuse AstrBot's native Zhipu adapter, meter every paid outbound query."""
import base64
import copy
import io
import json

from astrbot.api import logger
from astrbot.api.provider import ProviderType
from astrbot.api.star import Star, StarTools
from astrbot.core.provider.register import provider_cls_map, provider_registry, register_provider_adapter
from astrbot.core.provider.sources.zhipu_source import ProviderZhipu

from .ledger import BudgetDenied, Ledger

TYPE = 'dafeiyu_budgeted_zhipu'
MODEL = 'glm-5.3-flashx'
MAX_OUTPUT = 2048


def bound_payload(payload):
    """Bound text bytes and static visual size before reserving any money."""
    from PIL import Image, ImageOps
    if payload.get('model') != MODEL:
        raise BudgetDenied('预算通道只允许已核价的FlashX模型。')
    result = copy.deepcopy(payload)
    count = 0
    def walk(value):
        nonlocal count
        if isinstance(value, dict):
            if value.get('type') == 'image_url':
                count += 1
                if count > 4:
                    raise BudgetDenied('单次最多处理4张图片。')
                url = value['image_url']['url']
                if not isinstance(url, str) or not url.startswith('data:image/') or len(url) > 24_000_000:
                    raise BudgetDenied('图片必须先由原生适配器解析为有限的本地图片。')
                raw = base64.b64decode(url.split(',', 1)[1], validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.width * image.height > 40_000_000:
                        raise BudgetDenied('图片尺寸过大。')
                    image = ImageOps.exif_transpose(image).convert('RGB')
                    image.thumbnail((1024, 1024))
                    stream = io.BytesIO()
                    image.save(stream, format='JPEG', quality=85)
                value['image_url']['url'] = 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()
                return {'type': 'image_url', 'image_url': {'url': '[bounded 1024px image]'}}
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise BudgetDenied('不支持的请求内容。')
    measured = walk(result)
    text_bytes = len(json.dumps(measured, ensure_ascii=False).encode())
    if text_bytes > 131072:
        raise BudgetDenied('上下文超过付费通道的长度保护线，请新开会话。')
    result['max_tokens'] = MAX_OUTPUT
    # Text byte fallback upper bound plus 65,536 input-token allowance per
    # 1024px image and fixed request overhead. Cached input uses full price.
    reserve = (text_bytes + count * 65536 + 4096) * 2 + MAX_OUTPUT * 7
    return result, reserve


@register_provider_adapter(TYPE, '共享月预算的智谱原生视觉通道',
    provider_type=ProviderType.CHAT_COMPLETION, provider_display_name='GLM 月预算通道（50元/月）',
    default_config_tmpl={'id': 'glm_budget', 'provider': 'zhipu', 'type': TYPE,
                         'provider_type': 'chat_completion', 'enable': True, 'key': [],
                         'model': MODEL,
                         'api_base': 'https://open.bigmodel.cn/api/paas/v4/', 'timeout': 45})
class BudgetedZhipu(ProviderZhipu):
    def __init__(self, provider_config, provider_settings):
        config = copy.deepcopy(provider_config)
        if config.get('api_base', '').rstrip('/') != 'https://open.bigmodel.cn/api/paas/v4':
            raise BudgetDenied('仅允许智谱官方按量API入口。')
        config['custom_extra_body'] = {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low', 'max_tokens': MAX_OUTPUT}
        super().__init__(config, provider_settings)
        self.ledger = Ledger(StarTools.get_data_dir('astrbot_plugin_dafeiyu_budget') / 'monthly.sqlite3')

    async def _query(self, payloads, tools, *, request_max_retries=None, conversation_id=None):
        if tools:
            raise BudgetDenied('此预算通道尚未开放自动工具调用。')
        payloads, reserve = bound_payload(payloads)
        token = await self.ledger.reserve(reserve)
        # Unknown failures/cancellation retain the reserve; there is no refund
        # without authoritative usage. Native transport and helper get one try.
        response = await super()._query(payloads, tools, request_max_retries=1, conversation_id=conversation_id)
        usage = response.usage
        raw_usage = getattr(getattr(response, 'raw_completion', None), 'usage', None)
        if usage is not None and raw_usage is not None:
            cost = int(usage.input) * 2 + int(usage.output) * 7
            if not await self.ledger.settle(token, cost):
                logger.error('智谱返回用量超出预留估算，本月付费通道已冻结。')
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

    async def initialize(self):
        manager = self.context.provider_manager
        for config in manager.providers_config:
            merged = manager.get_merged_provider_config(config)
            if merged.get('type') == TYPE and merged.get('enable') and merged['id'] not in manager.inst_map:
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
