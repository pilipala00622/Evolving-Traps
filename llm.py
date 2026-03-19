'''
LLM统一接口模块 - 合并了配置、模型类和工具函数
其他脚本可以直接通过模型名调用：from llm import get_model_answer
from llm import get_model_answer
answer = get_model_answer('模型名', '提示词')
'''

import os
import json
import time
import copy
import uuid
import datetime
import hmac
import hashlib
import base64
import threading
try:
    import pandas as pd
except ImportError:
    pd = None
import requests
from requests.adapters import HTTPAdapter
from multiprocessing import Process, Lock
from tqdm import tqdm
import functools
import dataclasses
from typing import Any, Dict, List, Optional
import jinja2

# 延迟导入以避免循环导入
try:
    from werewolf_12_actors.utils import Deserializable, parse_json as utils_parse_json
except ImportError:
    Deserializable = object
    def utils_parse_json(text):
        try:
            return json.loads(text)
        except:
            return None
try:
    from werewolf_12_actors.config import RETRIES as CONFIG_RETRIES
except ImportError:
    CONFIG_RETRIES = 20

def _get_int_env(name: str, default: int) -> int:
    """读取 int 环境变量，读取失败时回退默认值。"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

# generate() 的解析重试次数（避免与底层 HTTP 重试叠加造成超长等待）
RETRIES = _get_int_env("WEREWOLF_GENERATE_RETRIES", CONFIG_RETRIES)

# ==================== API 配置 ====================
# 混元API配置
HUNYUAN_TOKEN = "Bearer pdTpdS1JaLGy7Eeivx0pkfTifZt9Wimq"
HUNYUAN_PRE_TOKEN = "Bearer m76j0DxNRCLSoOE4THbYB2mOswFe2mqA"
HUNYUAN_T1_TOKEN = "Bearer xUlzAe8ExJz5OpL6Hg86Z7YgdsVNml48"  # hunyuan-t1专用token
HUNYUAN_API_URL = "http://hunyuanapi.woa.com/openapi/chat/completions"

# 蒸馏系统API配置
# 注意：不同模型使用不同的认证方式
# - 通用模型（CommonLLM）：使用 APP_ID + APP_KEY 进行HMAC认证
# - 混元模型（HunYuanLLM）：使用 Bearer Token（HUNYUAN_TOKEN 或 HUNYUAN_PRE_TOKEN），不使用 APP_KEY
APP_ID = "data_eval_api_offline"
APP_KEY = "BH8A3rvw7BgNubI1"  # 通用模型的API Key

APP_hunyuan_token = "m76j0DxNRCLSoOE4THbYB2mOswFe2mqA"  # 混元模型备用Token（与HUNYUAN_PRE_TOKEN的值相同，去掉Bearer前缀）

# APP_ID = "nPDzwTuC_dylannlu"
# APP_KEY = "T2tfMfi3m04OqRPl"

# API请求扩展信息
EXTENSION = {
    # "task_category": {
    #     "level1": "文生文",
    #     "level2": "文生文-能力评测",
    #     "level3": "文生文-能力评测-评测执行"
    # },
    "task_creator": "yaxinxu",
    "task_id": "",
    "task_name": "游戏task",
    "task_source": "9",
    "caller_token": 'x'
}

# ==================== 模型配置 ====================
# 保证大版本评测需要的版本模型，1106进行更新
COMMON_MODEL_MARKERS = {
    "gpt-5.4":["api_openai_gpt-5.4-2026-03-05",{"stream": True, "reasoning": {"effort": "xhigh"}}],
    "gpt-5.2":["api_openai_gpt-5.2-response", {"stream": True, "reasoning": {"effort": "xhigh"}}],
    "GPT-5-pro":["api_openai_gpt-5-pro",{"reasoning": {"effort": "high"}}],
    "gpt5-weiruan":["api_azure_openai_gpt-5-response",{"stream": True, "reasoning": {"effort": "high"}}],
    "GPT-4.1-mini":"api_openai_gpt-4.1-mini",
    "gpt4.1":"api_azure_openai_gpt-4.1",
    "gpt4o":["api_openai_chatgpt-4o-latest",{"model": "chatgpt-4o-latest"}],
    "openai_o3":["api_azure_openai_o3",{"reasoning_effort": "high"}],
    "gpt-5.1": ["api_openai_gpt-5.1-response", {"reasoning": {"effort": "high"}, "stream":True}],
    
    # doubao
    "doubao-seed-1.8":["api_doubao_doubao-seed-1-8-251228", {"stream": True, "max_tokens": 32000, "reasoning_effort": "high"}],
    "doubao-seed-thinking-1015":["api_doubao_doubao-seed-1-6-251015", {"stream": True, "max_tokens": 32000, "reasoning_effort": "high"}],
    "doubao-seed-thinking-0715":"api_doubao_Doubao-Seed-1.6-thinking-250715",
    "doubao-seed-nonthinking-0615":["api_doubao_Doubao-Seed-1.6-250615",{"thinking": {"type": "disabled"}, "max_tokens": 16000}],
    "doubao-1.5-pro-32k":"api_doubao_Doubao-1.5-pro-32k-250115",
    
    # qwen系列
    # 0202更新
    "qwen3.5-plus":["api_ali_qwen3.5-plus",{"max_tokens": 65536, "enable_thinking": True}],
    "qwen3-235b": ["api_ali_qwen-plus-2025-07-28", {"max_tokens": 32768, "enable_thinking": True}],
    "qwen3-max-preview-thinking": ["api_ali_qwen3-max-preview", {"max_tokens": 32768, "enable_thinking": True}],
    # "qwen3-235b": "api_ali_qwen3-235b-a22b-thinking-2507",
    # "qwen3-235b":["api_ali_qwen3-max",{"max_tokens": 32768}],
    "qwen3-max": ["api_ali_qwen3-max",{"max_tokens": 32768}],
    "qwen3-nonthinking-2507": ["api_ali_qwen3-235b-a22b-instruct-2507",{"max_new_tokens": 64000}],
    
    # gemini系列
    "gemini-3.1-pro":"api_naci_default_gemini-3.1-pro-preview", # 0227做最新的修正，naci
    "gemini-2.5-pro":["api_google_gemini-2.5-pro",{"generationConfig": {"thinkingConfig": {"thinkingBudget": 32768}}}],
    "gemini-2.5-flash":["api_google_gemini-2.5-flash",{"generationConfig": {"thinkingConfig": {"thinkingBudget": 24576}}}],
    "gemini-3-pro-preview":["api_google_gemini-3-pro-preview",{"stream": True}],
    "gemini-3-flash":["api_google_gemini-3-flash-preview",{"stream": True}],
    "gemini-3-flash-preview-naci-default":["api_naci_default_gemini-3-flash-preview",{"stream": True}],
    "gemini-3-pro-image-preview-naci-default":["api_naci_default_gemini-3-pro-image-preview",{"stream": True}],
    "gemini-3-pro-preview-naci-default":["api_naci_default_gemini-3-pro-preview",{"stream": True}],
    
    # minimax
    "Minimax-M2":["api_minimax_MiniMax-M2",{"stream": True, "max_completion_tokens": 64000}],
    "minimax-m1":"api_minimax_MiniMax-M1",
    
    # claude
    "aws-claude-opus-4.6":["api_aws_third_anthropic.claude-opus-4-6-v1", {"thinking": {"type": "adaptive"}, "max_tokens": 128000, "output_config": {"effort": "max"}}],
    "aws-claude-sonnet-4.6":["api_aws_third_anthropic.claude-sonnet-4-6", {"thinking": {"type": "adaptive"}, "max_tokens": 65536, "output_config": {"effort": "high"}}],
    
    "claude-opus-4.6":["api_anthropic_claude-opus-4-6", {"thinking": {"type": "adaptive"}, "max_tokens": 128000, "output_config": {"effort": "max"}}],
    "claude-sonnet-4.6":["api_anthropic_claude-sonnet-4-6", {"thinking": {"type": "adaptive"}, "max_tokens": 65536, "output_config": {"effort": "high"}}],
    "claude-opus-4.5":["api_aws_third_anthropic.claude-opus-4-5-20251101-v1:0",{"max_tokens": 64000}],
    "claude-haiku-4.5-fast":["api_anthropic_claude-haiku-4-5-20251001",{"max_tokens": 32000}],
    "claude-haiku-4.5-slow":["api_anthropic_claude-haiku-4-5-20251001",{"thinking": {"type": "enabled", "budget_tokens": 10000}, "max_tokens": 20000}],
    "cladue-opus-4.1-think":["api_azure_claude_claude-opus-4-1-20250805",{"thinking": {"type": "enabled", "budget_tokens": 13000}, "max_tokens": 20000}],
    "cladue-opus-4.1-nonthink":"api_azure_claude_claude-opus-4-1-20250805",
    "claude-sonnet-4.5-fast":["api_anthropic_claude-sonnet-4-5-20250929",{"max_tokens": 32000}],
    # 更改了新的接口
    "claude-sonnet-4.5-slow":["api_aws_third_anthropic.claude-sonnet-4-5-20250929-v1:0", {"thinking": {"type": "enabled", "budget_tokens": 10000}, "max_tokens": 20000}],

    # glm
    "glm-4.7":["api_zhipu_glm-4.7", {"stream": True, "thinking": {"type": "enabled"}, "max_tokens": 128000}],
    "glm-4.5-thinking":["api_zhipu_glm-4.5",{"stream": True, "thinking": {"type": "enabled"}, "max_tokens": 32768}],    
    "glm-4.5-nonthinking":["api_zhipu_glm-4.5",{"stream": True, "thinking": {"type": "disabled"}, "max_tokens": 32768}],
    "glm-4.6-thinking":["api_zhipu_glm-4.6",{"stream": True, "thinking": {"type": "enabled"}, "max_tokens": 32000}],    
    "glm-4.6-nonthinking":["api_zhipu_glm-4.6",{"stream": True, "thinking": {"type": "disabled"}, "max_tokens": 32000}],
    
    # deepseek
    "deepSeek-v3.2-speciale":"api_deepseek_deepseek-v3.2-speciale",
    "deepseek-v3.2":["api_deepseek_deepseek-reasoner",{"max_tokens": 64000}],
    "deepseek-v3.2-nonthinking":["api_deepseek_deepseek-chat",{"max_tokens": 8192}],
    "deepseek-r1":"api_doubao_DeepSeek-R1-250120",
    "deepseek-v3":"api_doubao_deepseek-v3-250324",
    
    # kimi
    "kimi-k2.5":["api_moonshot_kimi-k2.5",{"stream": True, "thinking": {"type": "enabled"}, "max_tokens": 128000, "temperature": 1.0}],
    "kimi-k2-0905":["api_moonshot_kimi-k2-0905-preview", {"stream": True, "max_tokens": 250000}],
    "kimi-k2-0711":["api_moonshot_kimi-k2-0711-preview", {"stream": True, "max_tokens": 100000}],
    "kimi-k2-thinking":["api_moonshot_kimi-k2-thinking", {"stream": True, "max_tokens": 128000, "temperature": 1.0}],
    
    # 待确认
    "longcat-flash":"api_longcat_LongCat-Flash-Chat",
    
    # wenxin-1114更新
    "ernie-5.0-thinking": ["api_bd_ernie-5.0-thinking-preview", {"stream": True, "max_tokens": 65536}],

    # grok
    "grok-4":["api_xai_grok-4-latest", {"max_tokens": 64000, "stream":True}],
    
}

# 混元模型配置
HUNYUAN_MODEL_MARKERS = {
    "hunyuan-t1-online-latest": ["hunyuan-t1-latest", False],
    "hunyuan-turbos-latest": ["hunyuan-turbos-latest", False],
    "hunyuan-turbos-longtext": ["hunyuan-turbos-longtext-128k-20250325", False],
    "hunyuan-t1-longtext": ["hunyuan-t1-longtext-128k-20250324", False],
    "hunyuan-t1-dev-3": ["hunyuan-t1-dev-3", True], # hunyuan2.0
    "hunyuan-t1-dev-2": ["hunyuan-t1-dev-2", True],
    "hunyuan-T1-2.4-preveiw": ["hunyuan-t1-20250715", False],
    "hunyuan-T1-2.5": ["hunyuan-t1-20250822", False],
    "hunyuan-2.0-thinking-dev-20251012": ["hunyuan-2.0-thinking-dev-20251012", False],
    "hunyuan-2.0-thinking-20251109": ["hunyuan-2.0-thinking-20251109", False]
}

# ==================== 处理配置 ====================
# 多进程配置
DEFAULT_NUM_PROCESSES = 2
DEFAULT_BATCH_SIZE = 1

# 重试配置：与 config.RETRIES 统一，所有部署/API 均重试 20 次
DEFAULT_MAX_RETRIES = _get_int_env("WEREWOLF_MAX_RETRIES", CONFIG_RETRIES)
RETRY_SLEEP_SECONDS = _get_int_env("WEREWOLF_RETRY_SLEEP_SECONDS", 10)
COMMON_MAX_RETRIES = _get_int_env("WEREWOLF_COMMON_MAX_RETRIES", CONFIG_RETRIES)
COMMON_RETRY_SLEEP_SECONDS = _get_int_env("WEREWOLF_COMMON_RETRY_SLEEP_SECONDS", 10)
HUNYUAN_MAX_RETRIES = _get_int_env("WEREWOLF_HUNYUAN_MAX_RETRIES", CONFIG_RETRIES)
HUNYUAN_RETRY_SLEEP_SECONDS = _get_int_env("WEREWOLF_HUNYUAN_RETRY_SLEEP_SECONDS", 10)
# ==================== 重试回退 ====================
def _exp_backoff_sleep(base_seconds: int, attempt: int, max_seconds: int = 60) -> None:
    """指数回退：base * 2^attempt，上限 max_seconds"""
    sleep_seconds = min(max_seconds, base_seconds * (2 ** attempt))
    time.sleep(sleep_seconds)
DEFAULT_TIMEOUT = _get_int_env("WEREWOLF_DEFAULT_TIMEOUT", 300)  # 默认超时时间（秒）
SLOW_THINKING_TIMEOUT = _get_int_env("WEREWOLF_SLOW_THINKING_TIMEOUT", 1200)  # 慢思考模型超时时间（秒）

# 慢思考模型列表（需要更长超时时间的模型）
SLOW_THINKING_MODELS = [
    "Deepseek-V3.1-Thinking",
    "Doubao-seed-1.6-thinking-0715",
    "Doubao-1.5-thinking-pro-m-250428-fast",
    "gpt5",
    "o3-pro",
    "gemini-2.5-pro",
    "Grok-4"
]

_thread_local = threading.local()

def _get_requests_session() -> requests.Session:
    """为每个线程复用 requests 会话，减少握手开销。"""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _thread_local.session = session
    return session

# ==================== 工具函数 ====================
def get_simple_auth(source, SecretId, SecretKey):
    """生成API认证头"""
    dateTime = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    auth = (
        'hmac id="'
        + SecretId
        + '", algorithm="hmac-sha1", headers="date source", signature="'
    )
    signStr = "date: " + dateTime + "\n" + "source: " + source
    sign = hmac.new(SecretKey.encode(), signStr.encode(), hashlib.sha1).digest()
    sign = base64.b64encode(sign).decode()
    sign = auth + sign + '"'
    return sign, dateTime

def get_common_api_headers():
    """获取通用API请求头"""
    API_VERSION = "v2.03"
    source = "autoeval"
    sign, dateTime = get_simple_auth(source, APP_ID, APP_KEY)
    headers = {
        "Apiversion": API_VERSION,
        "Authorization": sign,
        "Date": dateTime,
        "Source": source,
    }
    return headers


def _extract_text_from_answer(answer):
    """从标准 answer 字段中提取文本。"""
    if isinstance(answer, list):
        text_parts = []
        for item in answer:
            if isinstance(item, dict) and item.get("type") == "text":
                value = item.get("value", "")
                if value:
                    text_parts.append(value)
        return "".join(text_parts)
    if isinstance(answer, str):
        return answer
    return str(answer) if answer else ""


def _extract_text_from_stream_response(response_data):
    """在 answer 为空时，回退解析 request_detail.response 中的 SSE 文本。"""
    request_detail = response_data.get("request_detail")
    if not isinstance(request_detail, dict):
        return ""

    raw_response = request_detail.get("response")
    if not raw_response or not isinstance(raw_response, str):
        return ""

    text_parts = []
    for raw_line in raw_response.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            if isinstance(delta, dict):
                content = delta.get("content", "")
                if content:
                    text_parts.append(content)

            message = choice.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
                if content:
                    text_parts.append(content)

    return "".join(text_parts)


def _extract_reasoning_payload(payload):
    """递归提取响应里和 reasoning / thinking 相关的字段。"""
    results = {}

    def walk(value, prefix="root"):
        if isinstance(value, dict):
            for key, child in value.items():
                lower_key = str(key).lower()
                child_prefix = f"{prefix}.{key}"
                if any(token in lower_key for token in ("reason", "think", "cot", "chain_of_thought")):
                    results[child_prefix] = child
                walk(child, child_prefix)
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                walk(child, f"{prefix}[{idx}]")

    walk(payload)
    return results

def get_model_timeout(model_name):
    """根据模型名称获取超时时间"""
    if model_name in SLOW_THINKING_MODELS:
        return SLOW_THINKING_TIMEOUT
    return DEFAULT_TIMEOUT


# ==================== LLM 类 ====================
class CommonLLM():
    """通用大语言模型接口"""
    
    def __init__(self, model_name):
        self.requrl = "http://trpc-gpt-eval.production.polaris:8080/api/v1/data_eval"
        self.model_name = model_name
        self.model_marker = COMMON_MODEL_MARKERS[model_name]
        self.params = None
        self.trace_id = None  # 存储请求的trace_id
        self.usage_info = {}  # 存储token使用信息
        self.latency = 0  # 存储请求耗时（客户端计算）
        self.timing_info = {}  # 存储API返回的时间戳信息
        self.raw_response_data = {}
        self.reasoning_payload = {}
        
        # 设置超时时间
        self.timeout = get_model_timeout(model_name)
        
        # 处理模型参数
        if isinstance(self.model_marker, list):
            self.model_marker, self.params = self.model_marker

    def get_model_answer(self, prompt, history=[]):
        """调用模型获取回答"""
        headerdata = get_common_api_headers()
        messages = history + [{'role': 'user', 'content': [{"type": "text", "value": prompt}]}]
        request_id = str(uuid.uuid4())
        system_prompt = ''
        
        if messages[0]['role'] == 'system':
            system_prompt = messages.pop(0)['content'][0]['value']
            
        post_data = {
            "request_id": request_id,
            "model_marker": self.model_marker,
            "session_id": "prod-32345",
            "messages": messages,
            "timeout": self.timeout,
        }
        post_data['extension'] = EXTENSION
        
        if system_prompt != '':
            post_data['system'] = system_prompt
        if self.params is not None:
            post_data['params'] = self.params
            
        times = 0
        response = None
        ans = ''
        
        # 重置元数据
        self.trace_id = request_id
        self.usage_info = {}
        self.latency = 0
        self.timing_info = {}
        self.raw_response_data = {}
        self.reasoning_payload = {}
        
        while times < COMMON_MAX_RETRIES:
            try:
                # 记录开始时间
                start_time = time.time()
                
                response = _get_requests_session().post(
                    self.requrl, json=post_data, headers=dict(headerdata), timeout=self.timeout
                )
                
                # 记录结束时间并计算耗时（即使失败也记录）
                end_time = time.time()
                self.latency = end_time - start_time
                
                response_data = json.loads(response.text)
                self.raw_response_data = response_data
                self.reasoning_payload = _extract_reasoning_payload(response_data)
                
                # 检查API是否返回错误
                if 'code' in response_data and response_data['code'] != 0:
                    error_msg = response_data.get('msg', '未知错误')
                    error_code = response_data['code']
                    
                    # 检查是否是速率限制错误（429 Resource exhausted）
                    # 即使 code 是 2000，如果错误消息包含速率限制信息，也应该允许重试
                    is_rate_limit_error = (
                        '429' in error_msg or 
                        'Resource exhausted' in error_msg or 
                        'rate limit' in error_msg.lower() or
                        'quota' in error_msg.lower()
                    )
                    
                    # 定义不可重试的错误码（配置错误、权限错误等）
                    # 这些错误重试也没有意义，应该快速失败
                    # 但速率限制错误（429）除外，应该允许重试
                    NON_RETRYABLE_ERROR_CODES = [
                        2000,  # 模型不存在或没有访问权限（但速率限制除外）
                        2001,  # 参数错误
                        2002,  # 权限错误
                    ]
                    
                    # 检查是否是模型不存在或权限错误（但排除速率限制错误）
                    if error_code in NON_RETRYABLE_ERROR_CODES and not is_rate_limit_error:
                        # 对于真正的配置错误，直接抛出异常，不进入重试循环
                        # 这样可以让调用者快速知道是配置问题，而不是临时错误
                        raise Exception(f"API返回错误 (code: {error_code}): {error_msg}。这是配置错误，不会重试。")
                    else:
                        # 对于其他错误（包括速率限制错误），抛出异常让重试机制处理
                        # 速率限制错误会通过重试机制等待后重试
                        raise Exception(f"API返回错误 (code: {error_code}): {error_msg}")
                
                # 检查answer字段是否存在且不为None
                if 'answer' not in response_data or response_data['answer'] is None:
                    error_msg = response_data.get('msg', 'answer字段为空')
                    raise Exception(f"API返回answer为空: {error_msg}")
                
                answer = response_data["answer"]
                ans = _extract_text_from_answer(answer)
                if not ans:
                    # 某些 stream=True 的通用模型不会回填 answer，
                    # 但会把完整 SSE 原文放到 request_detail.response 里。
                    ans = _extract_text_from_stream_response(response_data)
                
                # 提取token使用信息（优先使用cost_info，因为它包含cost字段）
                if 'cost_info' in response_data:
                    self.usage_info = response_data['cost_info'].copy()
                    # 确保字段名统一（cost_info中可能使用不同的字段名）
                    if 'prompt_tokens' not in self.usage_info and 'input_tokens' in self.usage_info:
                        self.usage_info['prompt_tokens'] = self.usage_info['input_tokens']
                    if 'completion_tokens' not in self.usage_info and 'output_tokens' in self.usage_info:
                        self.usage_info['completion_tokens'] = self.usage_info['output_tokens']
                elif 'usage' in response_data:
                    self.usage_info = response_data['usage'].copy()
                
                # 提取耗时相关的时间戳字段
                if 'created_unix' in response_data:
                    self.timing_info['created_unix'] = response_data['created_unix']
                if 'first_token_unix' in response_data:
                    self.timing_info['first_token_unix'] = response_data['first_token_unix']
                if 'finished_unix' in response_data:
                    self.timing_info['finished_unix'] = response_data['finished_unix']
                
                # 计算耗时（如果时间戳都存在）
                if 'created_unix' in self.timing_info and 'finished_unix' in self.timing_info:
                    # 时间戳是毫秒，转换为秒
                    server_latency = (self.timing_info['finished_unix'] - self.timing_info['created_unix']) / 1000.0
                    self.timing_info['server_latency'] = round(server_latency, 3)
                
                # 计算首token时间（如果存在）
                if 'created_unix' in self.timing_info and 'first_token_unix' in self.timing_info and self.timing_info['first_token_unix'] > 0:
                    time_to_first_token = (self.timing_info['first_token_unix'] - self.timing_info['created_unix']) / 1000.0
                    self.timing_info['time_to_first_token'] = round(time_to_first_token, 3)
                
                # 提取trace_id (如果接口返回了的话)
                if 'trace_id' in response_data:
                    self.trace_id = response_data['trace_id']
                elif 'request_id' in response_data:
                    self.trace_id = response_data['request_id']
                
                break
            except Exception as e:
                # 检查是否是配置错误（不可重试的错误）
                error_str = str(e)
                if '这是配置错误，不会重试' in error_str or 'prompt违规' in error_str:
                    # 对于配置错误，直接抛出，不重试
                    raise
                
                # 检查是否是速率限制错误，如果是，使用更长的等待时间
                is_rate_limit_error = (
                    '429' in error_str or 
                    'Resource exhausted' in error_str or 
                    'rate limit' in error_str.lower() or
                    'quota' in error_str.lower()
                )
                
                # 即使失败，也更新耗时（如果response存在的话）
                if 'start_time' in locals():
                    self.latency = time.time() - start_time
                
                print(f"[重试 {times + 1}/{COMMON_MAX_RETRIES}] {e}")
                if response is not None and 'The response was filtered due to the prompt' in response.text:
                    raise Exception(f"{self.model_name}文生文接口调用失败, prompt违规")
                if response is not None:
                    print(f"{self.model_name}文生文接口调用失败,{e}, {request_id}, {response.text[:200]}...")  # 只打印前200字符
                else:
                    print(f"{self.model_name}文生文接口调用失败, 接口超时，{request_id}")
                
                # 如果还有重试机会，等待后继续
                if times < COMMON_MAX_RETRIES - 1:
                    if is_rate_limit_error:
                        # 速率限制错误，使用更长的等待时间（指数回退，但基础时间更长）
                        wait_time = min(300, COMMON_RETRY_SLEEP_SECONDS * (2 ** times) * 3)  # 最多等待5分钟
                        print(f"⚠️ 检测到速率限制错误，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                    else:
                        _exp_backoff_sleep(COMMON_RETRY_SLEEP_SECONDS, times)
                ans = "none"
            times += 1
            
        return ans


class HunYuanLLM():
    """混元大语言模型接口"""
    
    def __init__(self, model_name):
        self.model_name, pre = HUNYUAN_MODEL_MARKERS[model_name]
        self.trace_id = None  # 存储最后一次请求的trace_id
        self.usage_info = {}  # 存储token使用信息
        self.latency = 0  # 存储请求耗时
        self.raw_response_data = {}
        self.reasoning_payload = {}
        
        # 检查是否是需要使用新API的模型（hunyuan-T1-2.5 和 hunyuan-2.0-thinking-20251109）
        if model_name == "hunyuan-T1-2.5" or model_name == "hunyuan-2.0-thinking-20251109":
            # 这些模型使用新的API地址（没有v1路径）和专用token
            self.hy_url = "http://hunyuanapi.woa.com/openapi/chat/completions"
            self.token = HUNYUAN_T1_TOKEN
        elif pre:
            # 预发布环境
            self.hy_url = "http://hunyuanapipre-release.woa.com/openapi/v1/chat/completions"
            self.token = HUNYUAN_PRE_TOKEN
        else:
            # 正式环境（其他模型仍使用v1路径）
            self.hy_url = "http://hunyuanapi.woa.com/openapi/v1/chat/completions"
            self.token = HUNYUAN_TOKEN

        self.hy_headers = {
            "Content-Type": "application/json",
            "Authorization": self.token,
            "Sensitive-Business": "true" # 敏感业务，调用hunyuan需要加白名单！
        }
    
    def get_model_answer(self, prompt, history=[], stream=None):
        """
        调用混元模型获取回答
        
        参数:
            prompt: 提示词
            history: 对话历史
            stream: 是否使用流式响应（None时自动判断：使用新API的模型默认True，其他False）
        """
        if len(history) > 0 and isinstance(history[0]['content'], list):
            history = copy.deepcopy(history)
            for message in history:
                message['content'] = message['content'][0]['value']
                
        message = history + [{'role': 'user', 'content': prompt}]
        
        # 自动判断是否使用stream（hunyuan-T1-2.5默认使用stream）
        if stream is None:
            # 通过检查URL判断是否是hunyuan-T1-2.5（使用新API的模型）
            stream = "openapi/chat/completions" in self.hy_url and "v1" not in self.hy_url
        
        post_data = {
            "model": self.model_name, 
            "messages": message
        }
        
        # 只有使用v1路径的模型才添加enable_enhancement参数
        if "v1" in self.hy_url:
            post_data["enable_enhancement"] = False
        
        # 如果使用stream模式，添加stream参数
        if stream:
            post_data["stream"] = True
        
        hy_answer = "none"
        tries = HUNYUAN_MAX_RETRIES
        self.trace_id = None
        self.usage_info = {}
        self.latency = 0
        self.raw_response_data = {}
        self.reasoning_payload = {}
        
        for i in range(tries):
            try:
                # 记录开始时间
                start_time = time.time()
                
                if stream:
                    # 流式响应处理
                    answer_parts = []
                    response = _get_requests_session().post(
                        self.hy_url, 
                        json=post_data, 
                        headers=self.hy_headers,
                        stream=True,
                        timeout=300
                    )
                    
                    if response.status_code != 200:
                        error_text = response.text
                        raise Exception(f"HTTP错误 {response.status_code}: {error_text}")
                    
                    # 逐行读取SSE格式的流式响应
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # 去掉 "data: " 前缀
                                if data_str.strip() == '[DONE]':
                                    break
                                
                                try:
                                    chunk_data = json.loads(data_str)
                                    if chunk_data:
                                        self.raw_response_data = chunk_data
                                        extracted = _extract_reasoning_payload(chunk_data)
                                        if extracted:
                                            self.reasoning_payload.update(extracted)
                                    
                                    # 提取trace_id（通常在第一条消息中）
                                    if self.trace_id is None and 'id' in chunk_data:
                                        self.trace_id = chunk_data['id']
                                    
                                    # 提取内容增量
                                    if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                        delta = chunk_data['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            answer_parts.append(content)
                                    
                                    # 提取usage信息（通常在最后一条消息中）
                                    if 'usage' in chunk_data:
                                        self.usage_info = chunk_data['usage']
                                        
                                except json.JSONDecodeError:
                                    continue
                    
                    # 合并完整回答
                    hy_answer = ''.join(answer_parts)
                    
                else:
                    # 非流式响应处理
                    response = _get_requests_session().post(
                        self.hy_url, 
                        json=post_data, 
                        headers=self.hy_headers,
                        timeout=300
                    )
                    response_data = json.loads(response.text)
                    self.raw_response_data = response_data
                    self.reasoning_payload = _extract_reasoning_payload(response_data)
                    
                    # 提取答案
                    hy_answer = response_data["choices"][0]["message"]["content"]
                    
                    # 提取trace_id
                    self.trace_id = response_data.get('id')
                    
                    # 提取token使用信息
                    if 'usage' in response_data:
                        self.usage_info = response_data['usage']
                
                # 记录结束时间并计算耗时
                end_time = time.time()
                self.latency = end_time - start_time
                
                break
            except Exception as e:
                print(e)
                if 'response' in locals() and hasattr(response, 'text'):
                    print(f"{self.model_name} try again!", response.text, self.trace_id)
                else:
                    print(f"{self.model_name} try again!", self.trace_id)
                _exp_backoff_sleep(HUNYUAN_RETRY_SLEEP_SECONDS, i)
                
        return hy_answer.strip('\n').strip()

class LLM():
    """统一的LLM接口类"""

    def __init__(self, model_name):
        self.model_name = model_name
        if 'hunyuan' in model_name:
            self.llm = HunYuanLLM(model_name=model_name)
        else:
            # 全部非混元模型统一走 CommonLLM
            self.llm = CommonLLM(model_name=model_name)
    
    def get_model_answer(self, prompt, history=[]):
        """获取模型回答"""
        answer = self.llm.get_model_answer(prompt, history)
        if answer == 'none':
            raise Exception("模型调用失败")
        
        return answer.strip('\n').strip()
    
    @property
    def trace_id(self):
        """获取trace_id"""
        return getattr(self.llm, 'trace_id', None)
    
    @property
    def usage_info(self):
        """获取usage信息"""
        return getattr(self.llm, 'usage_info', {})
    
    @property
    def latency(self):
        """获取请求耗时"""
        return getattr(self.llm, 'latency', 0)
    
    @property
    def timing_info(self):
        """获取时间戳信息"""
        return getattr(self.llm, 'timing_info', {})

    @property
    def raw_response_data(self):
        """获取最近一次请求的原始响应。"""
        return getattr(self.llm, 'raw_response_data', {})

    @property
    def reasoning_payload(self):
        """获取最近一次请求中 reasoning / thinking 相关字段。"""
        return getattr(self.llm, 'reasoning_payload', {})


# ==================== 简化接口函数 ====================
# 全局LLM实例缓存，避免重复创建
_llm_cache = {}
_token_usage_by_round = {}


def _extract_round_num(worldstate: Dict[str, Any]) -> Optional[int]:
    """从worldstate中提取轮次编号。"""
    for key in ("round", "round_num", "round_number"):
        if key in worldstate:
            value = worldstate.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return None


def _normalize_usage_info(usage: Dict[str, Any]) -> Dict[str, int]:
    """统一token字段名，返回标准化token信息。"""
    if not usage:
        return {}
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _record_token_usage(round_num: int, model_name: str, usage: Dict[str, Any]) -> None:
    """记录每轮每个模型的token消耗。"""
    if round_num is None:
        return
    normalized = _normalize_usage_info(usage)
    if not normalized:
        return
    round_bucket = _token_usage_by_round.setdefault(round_num, {})
    model_bucket = round_bucket.setdefault(
        model_name,
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
    )
    model_bucket["prompt_tokens"] += normalized["prompt_tokens"]
    model_bucket["completion_tokens"] += normalized["completion_tokens"]
    model_bucket["total_tokens"] += normalized["total_tokens"]
    model_bucket["calls"] += 1


def get_round_token_usage() -> Dict[int, Dict[str, Dict[str, int]]]:
    """获取按轮次聚合的token消耗信息。"""
    return copy.deepcopy(_token_usage_by_round)


def reset_round_token_usage(round_num: Optional[int] = None) -> None:
    """清空token消耗统计（可选清空指定轮次）。"""
    if round_num is None:
        _token_usage_by_round.clear()
        return
    _token_usage_by_round.pop(round_num, None)


def _get_or_create_llm(model_name: str) -> "LLM":
    """内部使用：获取缓存LLM实例。"""
    if model_name not in _llm_cache:
        _llm_cache[model_name] = LLM(model_name=model_name)
    return _llm_cache[model_name]

def get_model_answer(model_name, prompt, history=[], use_cache=True):
    """
    简化接口：直接通过模型名调用模型获取回答
    
    参数:
        model_name: 模型名称（如 'gpt4o', 'hunyuan-t1-online-latest' 等）
        prompt: 提示词
        history: 对话历史（可选）
        use_cache: 是否使用缓存（默认True，同一模型名会复用LLM实例）
    
    返回:
        str: 模型回答
    
    示例:
        from llm import get_model_answer
        
        # 简单调用
        answer = get_model_answer('gpt4o', '你好')
        
        # 带历史对话
        history = [{'role': 'user', 'content': [{'type': 'text', 'value': '你好'}]}]
        answer = get_model_answer('gpt4o', '请继续', history=history)
    """
    # 检查模型名是否有效
    if 'hunyuan' in model_name:
        if model_name not in HUNYUAN_MODEL_MARKERS:
            raise ValueError(f"未知的混元模型: {model_name}")
    else:
        if model_name not in COMMON_MODEL_MARKERS:
            raise ValueError(f"未知的模型: {model_name}")
    
    # 使用缓存或创建新实例
    if use_cache:
        if model_name not in _llm_cache:
            _llm_cache[model_name] = LLM(model_name=model_name)
        llm = _llm_cache[model_name]
    else:
        llm = LLM(model_name=model_name)
    
    return llm.get_model_answer(prompt, history)


def get_llm_instance(model_name):
    """
    获取LLM实例（用于需要访问trace_id、usage_info等元数据的场景）
    
    参数:
        model_name: 模型名称
    
    返回:
        LLM: LLM实例
    
    示例:
        from llm import get_llm_instance
        
        llm = get_llm_instance('gpt4o')
        answer = llm.get_model_answer('你好')
        print(f"Trace ID: {llm.trace_id}")
        print(f"Tokens: {llm.usage_info.get('total_tokens', 0)}")
        print(f"Latency: {llm.latency:.3f}s")
    """
    if 'hunyuan' in model_name:
        if model_name not in HUNYUAN_MODEL_MARKERS:
            raise ValueError(f"未知的混元模型: {model_name}")
    else:
        if model_name not in COMMON_MODEL_MARKERS:
            raise ValueError(f"未知的模型: {model_name}")

    return LLM(model_name=model_name)


# ==================== LmLog 和 generate 函数 ====================
@dataclasses.dataclass
class LmLog(Deserializable):
    """语言模型调用日志"""
    prompt: str
    raw_resp: str
    result: Any

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "prompt": self.prompt,
            "raw_resp": self.raw_resp,
            "result": self.result
        }

    @classmethod
    def from_json(cls, data: Dict[Any, Any]):
        return cls(**data)


def format_prompt(prompt_template, worldstate) -> str:
    """格式化提示词模板"""
    return jinja2.Template(prompt_template).render(worldstate)


def generate(
    prompt_template: str,
    response_schema: Dict[str, Any],
    worldstate: Dict[str, Any],
    model: str,
    temperature: float = 1.0,
    allowed_values: Optional[List[Any]] = None,
    result_key: Optional[str] = None,
) -> tuple[Any, LmLog]:
    """从语言模型生成文本并解析结果。

    参数:
        prompt_template: 提示词的 Jinja 模板。
        response_schema: 预期响应的模式。
        worldstate: 要渲染到提示词中的世界状态。
        model: 要使用的语言模型。
        temperature: 语言模型的采样温度。
        allowed_values: 结果的可选允许值列表。如果提供，
          生成将重试直到获得允许值范围内的结果。
        result_key: 从解析结果中提取特定值的可选键。
          如果未提供，则返回整个解析结果。

    返回:
        包含结果（如果失败则为 None）和 LmLog 的元组。
    """
    prompt = format_prompt(prompt_template, worldstate)
    
    # 如果提供了 response_schema，在 prompt 中添加 JSON 格式要求
    if response_schema:
        schema_str = json.dumps(response_schema, ensure_ascii=False, indent=2)
        json_instruction = f"\n\n请使用符合以下模式的有效 JSON 对象响应:\n```json\n{schema_str}\n```\n\n确保您的响应是可以直接解析的有效 JSON。"
        if model == "glm-4.7":
            json_instruction += "\n\n请直接以 JSON 对象开头（即以 { 开头），或将唯一一个 JSON 对象放在 ```json 代码块中，不要先输出其他分析内容。必须返回单个 JSON 对象，不要返回 JSON 数组（不要以 [ 开头）。"
        prompt = prompt + json_instruction
    
    raw_responses = []
    # 注意: temperature 参数不被 get_model_answer 直接支持，
    # 但模型可能在模型设置中配置了 temperature
    
    for attempt in range(RETRIES):
        raw_resp = None
        try:
            # 使用缓存LLM实例以获取usage信息
            llm = _get_or_create_llm(model_name=model)
            raw_resp = llm.get_model_answer(prompt=prompt, history=[])
            round_num = _extract_round_num(worldstate)
            _record_token_usage(round_num, model, llm.usage_info)
            
            # 解析 JSON 响应
            result = utils_parse_json(raw_resp)
            # 兼容模型返回数组的情况（如 [{"say": "..."}]），统一当作 dict 使用，避免 'list' object has no attribute 'get'
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
                result = result[0]
            elif result is not None and not isinstance(result, dict):
                result = None
            log = LmLog(prompt=prompt, raw_resp=raw_resp, result=result)

            if result and result_key:
                result = result.get(result_key)

            if allowed_values is None or result in allowed_values:
                return result, log

        except Exception as e:
            print(f"由于异常重试: {e}")
        raw_responses.append(raw_resp)

    # 在连接之前过滤掉 None 值
    valid_responses = [str(r) if r is not None else "None" for r in raw_responses]
    return None, LmLog(
        prompt=prompt, raw_resp="-------".join(valid_responses), result=None
    )


# ==================== 工具函数 ====================
def retry_get_model_answer(llm, prompt, history=None, max_retries=DEFAULT_MAX_RETRIES):
    """
    带重试的模型调用函数
    
    参数:
        llm: LLM实例
        prompt: 提示词
        history: 对话历史
        max_retries: 最大重试次数
    """
    if history is None:
        history = []
        
    for i in range(max_retries):
        try:
            response = llm.get_model_answer(prompt, history)
            return response
                
        except Exception as e:
            print(f"Error: {e}")
            print(f"Retrying... ({i + 1}/{max_retries})")
            if i < max_retries - 1:
                _exp_backoff_sleep(RETRY_SLEEP_SECONDS, i)
            
    raise Exception(f"Failed to get model answer after {max_retries} retries")


def retry_on_none(max_retries):
    """装饰器：如果函数返回None则重试"""
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            for attempt in range(max_retries + 1):
                result = func(*args, **kwargs)
                if result is not None:
                    return result
                print(f"[RetryDecorator] Attempt {attempt + 1} failed, retrying...")
            return None
        return wrapper_retry
    return decorator_retry


# ==================== 数据处理函数 ====================
def check_question(record, llm):
    """
    处理单条问题记录
    
    参数:
        record: 数据记录
        llm: LLM实例
    """
    # 支持中英文字段名
    if '问题' in record:
        question = record['问题']
    elif 'question' in record:
        question = record['question']
    else:
        raise KeyError("记录中未找到'问题'或'question'字段")
    
    try:
        result = retry_get_model_answer(llm, question)
        print(f"处理结果: {result}")
        
        # 将结果添加到记录中
        model_name = llm.model_name
        record[f'{model_name}_answer'] = str(result)
        
        # 保存所有模型的元数据信息（使用属性访问，更简洁）
        # Trace ID
        if llm.trace_id:
            record[f'{model_name}_trace_id'] = llm.trace_id
        
        # 请求耗时
        if llm.latency > 0:
            record[f'{model_name}_latency'] = round(llm.latency, 3)  # 保留3位小数
        
        # Token使用信息
        if llm.usage_info:
            usage = llm.usage_info
            record[f'{model_name}_prompt_tokens'] = usage.get('prompt_tokens', 0)
            record[f'{model_name}_completion_tokens'] = usage.get('completion_tokens', 0)
            record[f'{model_name}_total_tokens'] = usage.get('total_tokens', 0)
            
            # 如果有成本信息也保存
            if 'cost' in usage:
                record[f'{model_name}_cost'] = usage.get('cost', 0)
        
        # 打印元数据摘要
        print(f"元数据 - Trace ID: {llm.trace_id}, "
              f"耗时: {llm.latency:.3f}s, "
              f"Tokens: {llm.usage_info.get('total_tokens', 'N/A')}")
        print("处理完成")
        
    except Exception as e:
        print(f"处理失败: {e}")
        # 即使失败，也尝试保存可能获取到的元数据
        model_name = llm.model_name
        record['error'] = str(e)
        record[f'{model_name}_answer'] = ""
        
        # 保存失败时的元数据（如果有的话）
        if llm.trace_id:
            record[f'{model_name}_trace_id'] = llm.trace_id
        if llm.latency > 0:
            record[f'{model_name}_latency'] = round(llm.latency, 3)
    
    return record


# ==================== 多进程处理 ====================
def single_process(data, save_path, lock, get_request_response_func, batch_size=DEFAULT_BATCH_SIZE, **kwargs):
    """
    单进程处理函数，逐条调用接口，成功返回结果批量写入文件
    """
    buffer = []
    for i in tqdm(range(len(data)), desc=f"Process {os.getpid()}"):
        record = data.iloc[i].to_dict()
        try:
            record = get_request_response_func(record, **kwargs)
        except Exception as e:
            print(f"Error processing record {record.get('id', 'unknown')}: {e}")
            continue
            
        if (record is None) or (record == {}):
            continue
            
        buffer.append(record)

        # 达到批量大小，写入文件
        if len(buffer) >= batch_size:
            with lock:
                with open(save_path, 'a', encoding='utf-8') as f:
                    for rec in buffer:
                        json.dump(rec, f, ensure_ascii=False)
                        f.write('\n')
            buffer.clear()

    # 写入剩余未写入的数据
    if buffer:
        with lock:
            with open(save_path, 'a', encoding='utf-8') as f:
                for rec in buffer:
                    json.dump(rec, f, ensure_ascii=False)
                    f.write('\n')

    # 返回所有成功结果的DataFrame
    return pd.DataFrame(buffer)


def split_list(lst, n):
    """将列表均匀分成n份"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def multi_process(data, save_path, num_process=DEFAULT_NUM_PROCESSES, 
                 get_request_response_func=check_question, remove_path=False, 
                 id_name='id', **kwargs):
    """
    多进程处理主函数
    
    参数:
        data: 数据DataFrame
        save_path: 保存地址
        num_process: 进程数
        get_request_response_func: 处理单条数据的函数
        remove_path: 是否删除已存在的输出文件
        id_name: ID字段名
        **kwargs: 传给处理函数的额外参数
    """
    if remove_path and os.path.exists(save_path):
        os.remove(save_path)
    elif (not remove_path) and os.path.exists(save_path):
        # 读取已有结果，跳过已处理的数据
        with open(save_path, 'r', encoding='utf-8') as f:
            result = [json.loads(line) for line in f if line.strip()]
        result = pd.DataFrame(result)
        if len(result) > 0:
            question_ids = list(result[id_name].unique())
            data = data[~data[id_name].isin(question_ids)]
        print(f'已有{len(result)}条数据，剩余{len(data)}条数据')
    
    data_list = split_list(data, num_process)
    processes = []
    lock = Lock()
    
    for i in range(num_process):
        p = Process(target=single_process, 
                   args=(data_list[i], save_path, lock, get_request_response_func), 
                   kwargs=kwargs)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 读取最终结果
    if os.path.exists(save_path):
        data = pd.read_json(save_path, orient='records', lines=True)
        return data
    else:
        return pd.DataFrame()


# ==================== 主运行函数 ====================
def load_data(input_file):
    """加载输入数据"""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))

    # 用DataFrame包装，便于后续处理
    data = pd.DataFrame(data)

    # 检查是否有id字段，如果没有则自动生成
    if 'id' not in data.columns:
        data['id'] = range(len(data))
        
    return data


def main(model_name, input_file=None, output_file=None, 
         num_processes=DEFAULT_NUM_PROCESSES, remove_path=False):
    """
    主运行函数
    
    参数:
        model_name: 模型名称
        input_file: 输入文件路径（必须提供）
        output_file: 输出文件路径（必须提供）
        num_processes: 进程数
        remove_path: 是否删除已存在的输出文件
    """
    # 检查必需参数
    if input_file is None:
        raise ValueError("必须提供 input_file 参数")
    if output_file is None:
        raise ValueError("必须提供 output_file 参数")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    print(f"加载数据: {input_file}")
    data = load_data(input_file)
    print(f"数据加载完成，共{len(data)}条记录")
    print(data.head())
    
    print(f"初始化模型: {model_name}")
    llm = LLM(model_name=model_name)

    print(f"开始处理，进程数: {num_processes}")
    result = multi_process(
        data, 
        save_path=output_file,
        num_process=num_processes, 
        get_request_response_func=check_question, 
        remove_path=remove_path, 
        id_name='id', 
        llm=llm
    )

    if result.empty:
        print("结果文件不存在或为空。")
    else:
        print(f"处理完成，结果保存到: {output_file}")
        print(f"成功处理 {len(result)} 条记录")

    return result


if __name__ == "__main__":
    # 示例用法
    # 方式1: 使用简化接口
    # from llm import get_model_answer
    # answer = get_model_answer('gpt4o', '你好')
    # print(answer)
    
    # 方式2: 使用LLM类
    # llm = LLM('gpt4o')
    # answer = llm.get_model_answer('你好')
    # print(f"Answer: {answer}")
    # print(f"Trace ID: {llm.trace_id}")
    
    # 方式3: 批量处理（原main函数）
    main(
        model_name='deepseek-v3',  # 快思考，然后o3-pro去做评测
        input_file='//question_production/multiprocess_api_utils/merged_data.jsonl',
        output_file='kimi_k2_0729.jsonl',
        num_processes=20,
        remove_path=False
    )
