import argparse
import asyncio
import base64
import dataclasses
import json
import mimetypes
import os
import time
import hashlib
import hmac
import datetime
from datetime import datetime, timezone
import urllib
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientTimeout
import dataclasses_json

TokenGenerator = AsyncGenerator[str, None]
ApiResult = Tuple[aiohttp.ClientResponse, TokenGenerator]

AZURE_OPENAI_API_VERSION = "2024-02-15-preview"
MAX_TTFT = 9.99
MAX_TOTAL_TIME = 99.99

from dotenv import load_dotenv
load_dotenv()  # 默认会加载根目录下的.env文件

with open("llm.json", 'r', encoding='utf-8') as f:
    llm_info = json.load(f)

@dataclasses.dataclass
class InputFile:
    @classmethod
    def from_file(cls, path: str):
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            raise ValueError(f"Unknown file type: {path}")
        with open(path, "rb") as f:
            data = f.read()
        return cls(mime_type, data)

    mime_type: str
    data: bytes

    @property
    def base64_data(self):
        return base64.b64encode(self.data).decode("utf-8")


@dataclasses.dataclass
class ApiMetrics(dataclasses_json.DataClassJsonMixin):
    model: str
    context: Optional[str] = None
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    ttr: Optional[float] = None
    ttft: Optional[float] = None
    tps: Optional[float] = None
    input_tokens: Optional[int] = None
    num_tokens: Optional[int] = None
    total_time: Optional[float] = None
    output: Optional[str] = None
    error: Optional[str] = None


@dataclasses.dataclass
class ApiContext:
    session: aiohttp.ClientSession
    index: int
    name: str
    func: Callable
    model: str
    prompt: str
    files: List[InputFile]
    temperature: float
    max_tokens: int
    detail: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __init__(self, session, index, name, func, args, prompt, files):
        self.session = session
        self.index = index
        self.name = name
        self.func = func
        self.model = args.model
        self.prompt = prompt
        self.files = files
        self.detail = args.detail
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.api_key = args.api_key
        self.base_url = args.base_url
        self.metrics = ApiMetrics(model=self.name)

    async def run(self, on_token: Optional[Callable[["ApiContext", str], None]] = None):
        response = None
        try:
            start_time = time.time()
            first_token_time = None
            response, chunk_gen = await self.func(self)
            self.metrics.ttr = time.time() - start_time
            if response.ok:
                if chunk_gen:
                    self.metrics.num_tokens = 0
                    self.metrics.output = ""
                    async for chunk in chunk_gen:
                        self.metrics.output += chunk
                        self.metrics.num_tokens += 1
                        if not first_token_time:
                            first_token_time = time.time()
                            self.metrics.ttft = first_token_time - start_time
                        if on_token:
                            on_token(self, chunk)
            else:
                self.metrics.error = f"{response.status} {response.reason}"
        except TimeoutError:
            self.metrics.error = "Timeout"
        except aiohttp.ClientError as e:
            self.metrics.error = str(e)
        end_time = time.time()
        if self.metrics.num_tokens:
            token_time = end_time - first_token_time
            self.metrics.total_time = end_time - start_time
            self.metrics.tps = min((self.metrics.num_tokens - 1) / token_time, 999)
        elif self.metrics.error:
            self.metrics.ttft = MAX_TTFT
            self.metrics.tps = 0.0
            self.metrics.total_time = MAX_TOTAL_TIME
        provider, model = self.name.split("/")
        self.metrics.context = llm_info[provider][model]["context"]
        self.metrics.input_price = llm_info[provider][model]["input_price"]
        self.metrics.output_price = llm_info[provider][model]["output_price"]
        if response:
            await response.release()


async def post(
    ctx: ApiContext,
    url: str,
    headers: dict,
    data: dict,
    make_chunk_gen: Optional[Callable[[aiohttp.ClientResponse], TokenGenerator]] = None,
    max_retries: int = 3,
    base_timeout: int = 30,
    input_tokens=None,
): 
    retries = 0
    print("="*10, url, "="*5, data)
    while retries < max_retries:
        try:
            timeout = ClientTimeout(total=base_timeout * (2 ** retries))
            response = await ctx.session.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
            chunk_gen = make_chunk_gen(response, input_tokens) if make_chunk_gen else None
            return response, chunk_gen
        except asyncio.TimeoutError:
            retries += 1
            if retries < max_retries:
                wait_time = 2 ** retries
                print(f"请求超时,{wait_time}秒后进行第{retries}次重试...")
                await asyncio.sleep(wait_time)
            else:
                print("达到最大重试次数,放弃请求")
                raise
        except Exception as e:
            print(f"=={url}===发生错误: {e}")
            raise

def get_api_key(ctx: ApiContext, env_var: str) -> str:
    if ctx.api_key:
        return ctx.api_key
    if env_var in os.environ:
        return os.environ[env_var]
    raise ValueError(f"Missing API key: {env_var}")


def make_headers(
    auth_token: Optional[str] = None,
    api_key: Optional[str] = None,
    x_api_key: Optional[str] = None,
):
    headers = {
        "content-type": "application/json",
    }
    if auth_token:
        headers["authorization"] = f"Bearer {auth_token}"
    if api_key:
        headers["api-key"] = api_key
    if x_api_key:
        headers["x-api-key"] = x_api_key
    return headers


def make_openai_url_and_headers(ctx: ApiContext, path: str):
    url = ctx.base_url or "https://api.openai.com/v1"
    hostname = urllib.parse.urlparse(url).hostname
    use_azure_openai = hostname and hostname.endswith("openai.azure.com")
    if use_azure_openai:
        api_key = get_api_key(ctx, "AZURE_OPENAI_API_KEY")
        headers = make_headers(api_key=api_key)
        url += f"/openai/deployments/{ctx.model.replace('.', '')}{path}?api-version={AZURE_OPENAI_API_VERSION}"
    else:
        api_key = get_api_key(ctx, "OPENAI_API_KEY")
        headers = make_headers(auth_token=api_key)
        url += path
    return url, headers


def make_openai_messages(ctx: ApiContext):
    if not ctx.files:
        return [{"role": "user", "content": ctx.prompt}]

    content: List[Dict[str, Any]] = [{"type": "text", "text": ctx.prompt}]
    for file in ctx.files:
        if not file.mime_type.startswith("image/"):
            raise ValueError(f"Unsupported file type: {file.mime_type}")
        url = f"data:{file.mime_type};base64,{file.base64_data}"
        image_url = {"url": url}
        if ctx.detail:
            image_url["detail"] = ctx.detail
        content.append({"type": "image_url", "image_url": image_url})
    return [{"role": "user", "content": content}]


def make_openai_chat_body(ctx: ApiContext, **kwargs):
    # Models differ in how they want to receive the prompt, so
    # we let the caller specify the key and format.
    body = {
        "model": ctx.model,
        "max_tokens": ctx.max_tokens,
        "temperature": ctx.temperature,
        "stream": True,
    }
    for key, value in kwargs.items():
        body[key] = value
    return body


async def make_sse_chunk_gen(response: aiohttp.ClientResponse) -> AsyncGenerator[Dict[str, Any], None]:
    try:
        async for line in response.content:
            line = line.decode("utf-8").strip()
            if line.startswith("data:"):
                content = line[5:].strip()
                if content == "[DONE]":
                    break
                yield json.loads(content)
    except asyncio.TimeoutError:
        print("读取流式响应内容时超时")
        raise
    except Exception as e:
        print(f"处理流式响应内容时发生错误: {e}")
        raise

async def openai_chunk_gen(response: aiohttp.ClientResponse, input_tokens=None) -> TokenGenerator:
    tokens = 0
    try:
        async for chunk in make_sse_chunk_gen(response):
            if chunk.get("choices") and chunk["choices"][0]["delta"]:
                delta_content = chunk["choices"][0]["delta"].get("content")
                if delta_content:
                    tokens += 1
                    yield delta_content
            usage = chunk.get("usage")
            if usage:
                num_output_tokens = usage.get("completion_tokens")
                while tokens < num_output_tokens:
                    tokens += 1
                    yield ""
    except asyncio.TimeoutError:
        print("计算 token 时超时")
        raise
    except Exception as e:
        print(f"计算 token 时发生错误: {e}")
        raise

async def baidu_chunk_gen(response: aiohttp.ClientResponse, input_tokens=None) -> TokenGenerator:
    tokens = 0
    try:
        async for chunk in make_sse_chunk_gen(response):
            if chunk.get("result"):
                delta_content = chunk.get("result")
                if delta_content:
                    yield delta_content
            usage = chunk.get("usage")
            if usage:
                print("!"*10, usage)
                num_output_tokens = usage.get("completion_tokens")
                while tokens < num_output_tokens:
                    tokens += 1
                    yield ""
    except asyncio.TimeoutError:
        print("计算 token 时超时")
        raise
    except Exception as e:
        print(f"计算 token 时发生错误: {e}")
        raise


async def tencent_chunk_gen(response: aiohttp.ClientResponse, input_tokens=None) -> TokenGenerator:
    tokens = 0
    try:
        async for chunk in make_sse_chunk_gen(response):
            if chunk.get("Choices") and chunk["Choices"][0]["Delta"]:
                delta_content = chunk["Choices"][0]["Delta"].get("Content")
                if delta_content:
                    tokens += 1
                    yield delta_content
            usage = chunk.get("Usage")
            if usage:
                num_output_tokens = usage.get("CompletionTokens")
                while tokens < num_output_tokens:
                    tokens += 1
                    yield ""
    except asyncio.TimeoutError:
        print("计算 token 时超时")
        raise
    except Exception as e:
        print(f"计算 token 时发生错误: {e}")
        raise
    
async def minimax_chunk_gen(response, input_tokens=None) -> TokenGenerator:
    tokens = 0
    inputtokens = input_tokens if input_tokens else 0
    async for chunk in make_sse_chunk_gen(response):
        if chunk.get("choices") and chunk.get("choices")[0].get("delta"):
            delta_content = chunk["choices"][0]["delta"].get("content")
            if delta_content:
                tokens += 1
                yield delta_content
        usage = chunk.get("usage")
        if usage:
            # num_output_tokens = total_tokens - input_tokens
            num_output_tokens = usage.get("total_tokens")-inputtokens
            while tokens < num_output_tokens:
                tokens += 1
                yield ""

async def qwen_chunk_gen(response, input_tokens=None) -> TokenGenerator:
    tokens = 0
    async for chunk in make_sse_chunk_gen(response):
        if chunk.get("choices") and chunk.get("choices")[0].get("delta"):
            delta_content = chunk["choices"][0]["delta"].get("content")
            if delta_content:
                tokens += 1
                yield delta_content
        while tokens < 13:
            tokens += 1
            yield ""
 
                   
async def openai_chat(ctx: ApiContext, path: str = "/chat/completions") -> ApiResult:
    url, headers = make_openai_url_and_headers(ctx, path)
    data = make_openai_chat_body(ctx, messages=make_openai_messages(ctx))
    return await post(ctx, url, headers, data, openai_chunk_gen)


async def openai_embed(ctx: ApiContext) -> ApiResult:
    url, headers = make_openai_url_and_headers(ctx, "/embeddings")
    data = {"model": ctx.model, "input": ctx.prompt}
    return await post(ctx, url, headers, data)


def make_anthropic_messages(prompt: str, files: Optional[List[InputFile]] = None):
    """Formats the prompt as a text chunk and any images as image chunks.
    Note that Anthropic's image protocol is somewhat different from OpenAI's."""
    if not files:
        return [{"role": "user", "content": prompt}]

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for file in files:
        if not file.mime_type.startswith("image/"):
            raise ValueError(f"Unsupported file type: {file.mime_type}")
        source = {
            "type": "base64",
            "media_type": file.mime_type,
            "data": file.base64_data,
        }
        content.append({"type": "image", "source": source})
    return [{"role": "user", "content": content}]


async def anthropic_chat(ctx: ApiContext) -> ApiResult:
    """Make an Anthropic chat completion request. The request protocol is similar to OpenAI's,
    but the response protocol is completely different."""

    async def chunk_gen(response) -> TokenGenerator:
        tokens = 0
        async for chunk in make_sse_chunk_gen(response):
            delta = chunk.get("delta")
            if delta and delta.get("type") == "text_delta":
                tokens += 1
                yield delta["text"]
            usage = chunk.get("usage")
            if usage:
                num_tokens = usage.get("output_tokens")
                while tokens < num_tokens:
                    tokens += 1
                    yield ""

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": get_api_key(ctx, "ANTHROPIC_API_KEY"),
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "messages-2023-12-15",
    }
    data = make_openai_chat_body(
        ctx, messages=make_anthropic_messages(ctx.prompt, ctx.files)
    )
    return await post(ctx, url, headers, data, chunk_gen)


async def cohere_chat(ctx: ApiContext) -> ApiResult:
    """Make a Cohere chat completion request."""

    async def chunk_gen(response) -> TokenGenerator:
        tokens = 0
        async for line in response.content:
            chunk = json.loads(line)
            if chunk.get("event_type") == "text-generation" and "text" in chunk:
                tokens += 1
                yield chunk["text"]

    url = "https://api.cohere.ai/v1/chat"
    headers = make_headers(auth_token=get_api_key(ctx, "COHERE_API_KEY"))
    data = make_openai_chat_body(ctx, message=ctx.prompt)
    return await post(ctx, url, headers, data, chunk_gen)


async def cloudflare_chat(ctx: ApiContext) -> ApiResult:
    """Make a Cloudflare chat completion request. The protocol is similar to OpenAI's,
    but the URL doesn't follow the same scheme and the response structure is different.
    """

    async def chunk_gen(response) -> TokenGenerator:
        async for chunk in make_sse_chunk_gen(response):
            yield chunk["response"]

    account_id = os.environ["CF_ACCOUNT_ID"]
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{ctx.model}"
    )
    headers = make_headers(auth_token=get_api_key(ctx, "CF_API_KEY"))
    data = make_openai_chat_body(ctx, messages=make_openai_messages(ctx))
    return await post(ctx, url, headers, data, chunk_gen)


async def make_json_chunk_gen(response) -> AsyncGenerator[Dict[str, Any], None]:
    """Hacky parser for the JSON streaming format used by Google Vertex AI."""
    buf = ""
    async for line in response.content:
        # Eat the first array bracket, we'll do the same for the last one below.
        line = line.decode("utf-8").strip()
        if not buf and line.startswith("["):
            line = line[1:]
        # Split on comma-only lines, otherwise concatenate.
        if line == ",":
            yield json.loads(buf)
            buf = ""
        else:
            buf += line
    yield json.loads(buf[:-1])


def get_google_access_token():
    from google.auth.transport import requests
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(
        "service_account.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    if not creds.token:
        creds.refresh(requests.Request())
    return creds.token


def make_google_url_and_headers(ctx: ApiContext, method: str):
    region = "us-west1"
    project_id = os.environ["GCP_PROJECT"]
    url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{ctx.model}:{method}"
    api_key = ctx.api_key
    if not api_key:
        api_key = get_google_access_token()
    headers = make_headers(auth_token=api_key)
    return url, headers


def make_gemini_messages(prompt: str, files: List[InputFile]):
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for file in files:
        parts.append(
            {"inline_data": {"mime_type": file.mime_type, "data": file.base64_data}}
        )

    return [{"role": "user", "parts": parts}]


async def gemini_chat(ctx: ApiContext) -> ApiResult:
    async def chunk_gen(response) -> TokenGenerator:
        tokens = 0
        async for chunk in make_json_chunk_gen(response):
            content = chunk["candidates"][0].get("content")
            if content and "parts" in content:
                part = content["parts"][0]
                if "text" in part:
                    tokens += 1
                    yield part["text"]
            usage = chunk.get("usageMetadata")
            if usage:
                num_tokens = usage.get("candidatesTokenCount")
                while tokens < num_tokens:
                    tokens += 1
                    yield ""

    # The Google AI Gemini API (URL below) doesn't return the number of generated tokens.
    # Instead we use the Google Cloud Vertex AI Gemini API, which does return the number of tokens, but requires an Oauth credential.
    # Also, setting safetySettings to BLOCK_NONE is not supported in the Vertex AI Gemini API, at least for now.
    if True:
        url, headers = make_google_url_and_headers(ctx, "streamGenerateContent")
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{ctx.model}:streamGenerateContent?key={get_api_key(ctx, 'GOOGLE_GEMINI_API_KEY')}"
        headers = make_headers()
    harm_categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
    data = {
        "contents": make_gemini_messages(ctx.prompt, ctx.files),
        "generationConfig": {
            "temperature": ctx.temperature,
            "maxOutputTokens": ctx.max_tokens,
        },
        "safetySettings": [
            {"category": category, "threshold": "BLOCK_NONE"}
            for category in harm_categories
        ],
    }
    return await post(ctx, url, headers, data, chunk_gen)


async def cohere_embed(ctx: ApiContext) -> ApiResult:
    url = "https://api.cohere.ai/v1/embed"
    headers = make_headers(auth_token=get_api_key(ctx, "COHERE_API_KEY"))
    data = {
        "model": ctx.model,
        "texts": [ctx.prompt],
        "input_type": "search_query",
    }
    return await post(ctx, url, headers, data)


async def make_fixie_chunk_gen(response) -> TokenGenerator:
    text = ""
    async for line in response.content:
        line = line.decode("utf-8").strip()
        obj = json.loads(line)
        curr_turn = obj["turns"][-1]
        if (
            curr_turn["role"] == "assistant"
            and curr_turn["messages"]
            and "content" in curr_turn["messages"][-1]
        ):
            if curr_turn["state"] == "done":
                break
            new_text = curr_turn["messages"][-1]["content"]
            # Sometimes we get a spurious " " message
            if new_text == " ":
                continue
            if new_text.startswith(text):
                delta = new_text[len(text) :]
                text = new_text
                yield delta
            else:
                print(f"Warning: got unexpected text: '{new_text}' vs '{text}'")


async def fixie_chat(ctx: ApiContext) -> ApiResult:
    url = f"https://api.fixie.ai/api/v1/agents/{ctx.model}/conversations"
    headers = make_headers(auth_token=get_api_key(ctx, "FIXIE_API_KEY"))
    data = {"message": ctx.prompt, "runtimeParameters": {}}
    return await post(ctx, url, headers, data, make_fixie_chunk_gen)


async def fake_chat(ctx: ApiContext) -> ApiResult:
    class FakeResponse(aiohttp.ClientResponse):
        def __init__(self, status, reason):
            self.status = status
            self.reason = reason

        # async def release(self):
        # pass

    async def make_fake_chunk_gen(output: str):
        for word in output.split():
            yield word + " "
            await asyncio.sleep(0.05)

    output = "This is a fake response."
    if ctx.index % 2 == 0:
        response = FakeResponse(200, "OK")
    else:
        response = FakeResponse(500, "Internal Server Error")
    sleep = 0.5 * (ctx.index + 1)
    max_sleep = ctx.session.timeout.total
    if max_sleep:
        await asyncio.sleep(min(sleep, max_sleep))
    if sleep > max_sleep:
        raise TimeoutError
    return (response, make_fake_chunk_gen(output))


def make_display_name(provider: str, model: str) -> str:
    model_segments = model.split("/")
    if provider:
        # We already have a provider, so just need to add the model name.
        # If we've got a model name, add the end of the split to the provider.
        # Otherwise, we have model.domain.com, so we need to swap to domain.com/model.
        if model:
            name = provider + "/" + model_segments[-1]
        else:
            domain_segments = provider.split(".")
            name = ".".join(domain_segments[1:]) + "/" + domain_segments[0]
    elif len(model_segments) > 1:
        # We've got a provider/model string, from which we need to get the provider and model.
        provider = model_segments[0]
        name = provider + "/" + model_segments[-1]
    return name

async def doubao_chat(ctx: ApiContext) -> ApiResult:
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('ByteDance_API_KEY')}"
    }
    # 定义请求数据
    data = {
        "model": os.getenv(f"{ctx.model}"),
        "messages": [
            {
                "role": "user",
                "content": ctx.prompt
            }
        ],
        "stream": True
    }
    return await post(ctx, url, headers, data, openai_chunk_gen)

async def zhipu_chat(ctx: ApiContext) -> ApiResult:
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('Zhipu_API_KEY')}"
    }
    # 定义请求数据
    data = {
        "model": ctx.model,
        "messages": [
            {
                "role": "user",
                "content": ctx.prompt
            }
        ],
        "stream": True
    }
    return await post(ctx, url, headers, data, openai_chunk_gen)

async def kimi_chat(ctx: ApiContext) -> ApiResult:
    url = "https://api.moonshot.cn/v1/chat/completions"
    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('MoonShot_API_KEY')}"
    }
    # 定义请求数据
    data = {
        "model": ctx.model,
        "messages": [
            {
                "role": "user",
                "content": ctx.prompt
            }
        ],
        "stream": True
    }
    return await post(ctx, url, headers, data, openai_chunk_gen)

async def qwen_chat(ctx: ApiContext) -> ApiResult:
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('AliCLoud_API_KEY')}"
    }
    # 定义请求数据
    data = {
        "model": ctx.model,
        "messages": [
            {
                "role": "user",
                "content": ctx.prompt
            }
        ],
        "stream": True
    }
    return await post(ctx, url, headers, data, qwen_chunk_gen)


async def yi_chat(ctx: ApiContext) -> ApiResult:
    url = "https://api.lingyiwanwu.com/v1/chat/completions"
    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('LingYi_API_KEY')}"
    }
    # 定义请求数据
    data = {
        "model": ctx.model,
        "messages": [
            {
                "role": "user",
                "content": ctx.prompt
            }
        ],
        "stream": True
    }
    return await post(ctx, url, headers, data, openai_chunk_gen)

async def minimax_chat(ctx: ApiContext) -> ApiResult:
    url = "https://api.minimax.chat/v1/text/chatcompletion_v2"
    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('MiniMax_API_KEY')}"
    }
    # 定义请求数据
    minimax_input_tokens = {
        "abab6.5-chat": 81,
        "abab6.5s-chat": 81,
        "abab6.5t-chat": 104,
        "abab6.5g-chat": 104,
        "abab5.5-chat": 172,
        "abab5.5s-chat": 104
    }
    data = {
        "model": ctx.model,
        "messages": [
            {
                "role": "user",
                "content": ctx.prompt
            }
        ],
        "stream": True
    }
    return await post(ctx, url, headers, data, minimax_chunk_gen, input_tokens=minimax_input_tokens[ctx.model])

async def baichuan_chat(ctx: ApiContext) -> ApiResult:
    url = "https://api.baichuan-ai.com/v1/chat/completions"
    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('BaiChuan_API_KEY')}"
    }
    # 定义请求数据
    data = {
        "model": ctx.model,
        "messages": [
            {
                "role": "user",
                "content": ctx.prompt
            }
        ],
        "stream": True
    }
    return await post(ctx, url, headers, data, openai_chunk_gen)
    
async def tencent_chat(ctx: ApiContext) -> ApiResult:
    # 密钥参数
    secret_id = os.environ.get("Tencent_Secret_ID")
    secret_key = os.environ.get("Tencent_Secret_KEY")
    service = "hunyuan"
    host = "hunyuan.tencentcloudapi.com"
    endpoint = "https://" + host
    region = "ap-beijing"
    action = "ChatCompletions" 
    version = "2023-09-01"
    algorithm = "TC3-HMAC-SHA256"
    timestamp = int(time.time())
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    # 更新参数
    params = {
        "Model": ctx.model,
        "Messages": [
            {
                "Role": "user",
                "Content": "使用以下模板创建一个自我介绍,name填入'莫尔索',age填入'18',hobby填入'打羽毛球': '我的名字是{name},今年{age}岁,我最大的爱好是{hobby}"
            }
        ],
        "Stream": True
    }

    # 步骤 1：拼接规范请求串
    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    ct = "application/json; charset=utf-8"
    payload = json.dumps(params)
    canonical_headers = "content-type:%s\nhost:%s\nx-tc-action:%s\n" % (ct, host, action.lower())
    signed_headers = "content-type;host;x-tc-action"
    hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = (http_request_method + "\n" +
                         canonical_uri + "\n" +
                         canonical_querystring + "\n" +
                         canonical_headers + "\n" +
                         signed_headers + "\n" +
                         hashed_request_payload)

    # 步骤 2：拼接待签名字符串
    credential_scope = date + "/" + service + "/" + "tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = (algorithm + "\n" +
                      str(timestamp) + "\n" +
                      credential_scope + "\n" +
                      hashed_canonical_request)

    # 步骤 3：计算签名
    def sign(key, msg):
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = sign(secret_date, service)
    secret_signing = sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    # 步骤 4：拼接 Authorization
    authorization = (algorithm + " " +
                     "Credential=" + secret_id + "/" + credential_scope + ", " +
                     "SignedHeaders=" + signed_headers + ", " +
                     "Signature=" + signature)

    # 发送 POST 请求
    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json; charset=utf-8",
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": version,
        "X-TC-Region": region
    }
    return await post(ctx, endpoint, headers, params, tencent_chunk_gen)


async def silicon_chat(ctx: ApiContext) -> ApiResult:
    Silicon_Model = {
        "Qwen2-72B-Instruct": "Qwen/Qwen2-72B-Instruct",
        "Qwen2-57B-A14B-Instruct": "Qwen/Qwen2-57B-A14B-Instruct",
        "Qwen2-7B-Instruct": "Qwen/Qwen2-7B-Instruct",
        "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
        "Qwen1.5-110B-Chat": "Qwen/Qwen1.5-110B-Chat",
        "Qwen1.5-32B-Chat": "Qwen/Qwen1.5-32B-Chat",
        "Qwen1.5-14B-Chat": "Qwen/Qwen1.5-14B-Chat",
        "Qwen1.5-7B-Chat": "Qwen/Qwen1.5-7B-Chat",
        "glm-4-9b-chat": "THUDM/glm-4-9b-chat",
        "chatglm3-6b": "THUDM/chatglm3-6b",
        "DeepSeek-Coder-V2-Instruct":"deepseek-ai/DeepSeek-Coder-V2-Instruct",
        "deepseek-llm-67b-chat": "deepseek-ai/deepseek-llm-67b-chat",
        "DeepSeek-V2-Chat": "deepseek-ai/DeepSeek-V2-Chat",
        "Yi-1.5-34B-Chat-16K": "01-ai/Yi-1.5-34B-Chat-16K",
        "Yi-1.5-9B-Chat-16K": "01-ai/Yi-1.5-9B-Chat-16K",
        "Yi-1.5-6B-Chat": "01-ai/Yi-1.5-6B-Chat",     
    }
    
    url = "https://api.siliconflow.cn/v1/chat/completions"
    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('SiliconCloud_API_KEY')}",
    }
    # 定义请求数据
    data = {
        "model": Silicon_Model[ctx.model],
        "messages": [
            {
                "role": "user",
                "content": ctx.prompt
            }
        ],
        "stream": True
    }
    return await post(ctx, url, headers, data, openai_chunk_gen)

async def deepseek_chat(ctx: ApiContext) -> ApiResult:
    url = "https://api.deepseek.com/chat/completions"
    # 定义请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('DeepSeek_API_KEY')}",
    }
    # 定义请求数据
    data = {
        "model": ctx.model,
        "messages": [
            {
                "role": "user",
                "content": ctx.prompt
            }
        ],
        "stream": True
    }
    return await post(ctx, url, headers, data, openai_chunk_gen)

def get_access_token():
    import requests
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
        
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={os.getenv('Baidu_API_KEY')}&client_secret={os.getenv('Baidu_Secret_KEY')}"
    
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

async def baidu_chat(ctx: ApiContext) -> ApiResult:
    url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{ctx.model}?access_token={get_access_token()}"
    # 定义请求头
    headers = {
        "Content-Type": "application/json"
    }
    # 定义请求数据
    data = {
        "messages": [
            {
                "role": "user",
                "content": ctx.prompt
            }
        ],
        "stream": True
    }
    return await post(ctx, url, headers, data, baidu_chunk_gen)

def make_context(
    session: aiohttp.ClientSession,
    index: int,
    args: argparse.Namespace,
    prompt: Optional[str] = None,
    files: Optional[List[InputFile]] = None,
) -> ApiContext:
    model = args.model
    provider = args.provider
    func = None  # 初始化函数变量
    if provider == "字节跳动":
        func = doubao_chat
    elif provider == "通义千问":
        func = qwen_chat
    elif provider == "腾讯混元":
        func = tencent_chat
    elif provider == "百度文心":
        func = baidu_chat
    elif provider == "智谱AI":
        func = zhipu_chat
    elif provider == "月之暗面":
        func = kimi_chat
    elif provider == "零一万物":
        func = yi_chat
    elif provider == "MiniMax":
        func = minimax_chat
    elif provider == "百川智能":
        func = baichuan_chat
    elif provider == "SiliconCloud":
        func = silicon_chat
    elif provider == "深度求索":
        func = deepseek_chat
    else:
        raise ValueError(f"Unknown model: {model}")
    # 假设 make_display_name 是一个有效的函数
    name = args.display_name or make_display_name(provider, model)
    return ApiContext(session, index, name, func, args, prompt or "", files or [])