import argparse
import asyncio
import dataclasses
import datetime
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import dataclasses_json
# import gcloud.aio.storage as gcs

import llm_benchmark
import llm_request

from dotenv import load_dotenv
load_dotenv()  # 默认会加载根目录下的.env文件

DEFAULT_DISPLAY_LENGTH = 64
DEFAULT_FORMAT = "json"
# DEFAULT_GCS_BUCKET = "thefastest-data"

GPT_4O = "gpt-4o"
GPT_4_TURBO = "gpt-4-turbo"
GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
GPT_35_TURBO = "gpt-3.5-turbo"
GPT_35_TURBO_0125 = "gpt-3.5-turbo-0125"
GPT_35_TURBO_1106 = "gpt-3.5-turbo-1106"
LLAMA_3_70B_CHAT = "llama-3-70b-chat"
LLAMA_3_8B_CHAT = "llama-3-8b-chat"
MIXTRAL_8X22B_INSTRUCT = "mixtral-8x22b-instruct"
MIXTRAL_8X7B_INSTRUCT = "mixtral-8x7b-instruct"
PHI_2 = "phi-2"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--format",
    "-F",
    choices=["text", "json"],
    default="text",
    help="Output results in the specified format",
)
parser.add_argument(
    "--mode",
    "-m",
    choices=["text", "image", "audio", "video"],
    default="text",
    help="Mode to run benchmarks for",
)
parser.add_argument(
    "--filter",
    "-r",
    help="Filter models by name",
)
parser.add_argument(
    "--spread",
    "-s",
    type=float,
    default=0.0,
    help="Spread the requests out over the specified time in seconds",
)
parser.add_argument(
    "--display-length",
    "-l",
    type=int,
    default=DEFAULT_DISPLAY_LENGTH,
    help="Amount of the generation response to display",
)
parser.add_argument(
    "--store",
    action="store_true",
    help="Store the results in the configured GCP bucket",
)


def _dict_to_argv(d: Dict[str, Any]) -> List[str]:
    return [
        f"--{k.replace('_', '-')}" + (f"={v}" if v or v == 0 else "")
        for k, v in d.items()
    ]


class _Llm:
    """
    We maintain a dict of params for the llm, as well as any
    command-line flags that we didn't already handle. We'll
    turn this into a single command line for llm_benchmark.run
    to consume, which allows us to reuse the parsing logic
    from that script, rather than having to duplicate it here.
    """

    def __init__(self, model: str, display_name: Optional[str] = None, **kwargs):
        self.args = {
            "model": model,
            "format": "none",
            **kwargs,
        }
        if display_name:
            self.args["display_name"] = display_name

    async def run(self, pass_argv: List[str], spread: float) -> asyncio.Task:
        if spread:
            await asyncio.sleep(spread)
        full_argv = _dict_to_argv(self.args) + pass_argv
        return await llm_benchmark.run(full_argv)


class _AnyscaleLlm(_Llm):
    """See https://docs.endpoints.anyscale.com/text-generation/query-a-model"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "anyscale.com/" + (display_model or model),
            api_key=os.getenv("ANYSCALE_API_KEY"),
            base_url="https://api.endpoints.anyscale.com/v1",
        )


class _CloudflareLlm(_Llm):
    """See https://developers.cloudflare.com/workers-ai/models/"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "cloudflare.com/" + (display_model or model),
        )


class _DatabricksLlm(_Llm):
    """See https://docs.databricks.com/en/machine-learning/foundation-models/supported-models.html"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "databricks.com/" + (display_model or model),
            api_key=os.getenv("DATABRICKS_TOKEN"),
            base_url="https://adb-1558081827343359.19.azuredatabricks.net/serving-endpoints",
        )


class _FireworksLlm(_Llm):
    """See https://fireworks.ai/models"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "fireworks.ai/" + (display_model or model),
            api_key=os.getenv("FIREWORKS_API_KEY"),
            base_url="https://api.fireworks.ai/inference/v1",
        )


class _GroqLlm(_Llm):
    """See https://console.groq.com/docs/models"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "groq.com/" + (display_model or model),
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )


class _OctoLlm(_Llm):
    """See https://octo.ai/docs/getting-started/inference-models#serverless-endpoints"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "octo.ai/" + (display_model or model),
            api_key=os.getenv("OCTOML_API_KEY"),
            base_url="https://text.octoai.run/v1",
        )


class _PerplexityLlm(_Llm):
    """See https://docs.perplexity.ai/docs/model-cards"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "perplexity.ai/" + (display_model or model),
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai",
        )


class _TogetherLlm(_Llm):
    """See https://docs.together.ai/docs/inference-models"""

    def __init__(self, model: str, display_model: Optional[str] = None):
        super().__init__(
            model,
            "together.ai/" + (display_model or model),
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1",
        )


def _text_models():
    AZURE_EASTUS2_OPENAI_API_KEY = os.getenv("AZURE_EASTUS2_OPENAI_API_KEY")
    return [
        # GPT-4o
        _Llm(GPT_4O),
        _Llm(
            GPT_4O,
            api_key=AZURE_EASTUS2_OPENAI_API_KEY,
            base_url="https://fixie-openai-sub-with-gpt4.openai.azure.com",
        ),
        _Llm(GPT_4O, base_url="https://fixie-westus.openai.azure.com"),
        _Llm(
            GPT_4O,
            api_key=os.getenv("AZURE_NCENTRALUS_OPENAI_API_KEY"),
            base_url="https://fixie-centralus.openai.azure.com",
        ),
        # GPT-4 Turbo
        _Llm(GPT_4_TURBO),
        # GPT-4 Turbo Previews
        _Llm(GPT_4_0125_PREVIEW),
        _Llm(
            GPT_4_0125_PREVIEW,
            api_key=os.getenv("AZURE_SCENTRALUS_OPENAI_API_KEY"),
            base_url="https://fixie-scentralus.openai.azure.com",
        ),
        _Llm(GPT_4_1106_PREVIEW),
        _Llm(GPT_4_1106_PREVIEW, base_url="https://fixie-westus.openai.azure.com"),
        _Llm(
            GPT_4_1106_PREVIEW,
            api_key=AZURE_EASTUS2_OPENAI_API_KEY,
            base_url="https://fixie-openai-sub-with-gpt4.openai.azure.com",
        ),
        _Llm(
            GPT_4_1106_PREVIEW,
            api_key=os.getenv("AZURE_FRCENTRAL_OPENAI_API_KEY"),
            base_url="https://fixie-frcentral.openai.azure.com",
        ),
        _Llm(
            GPT_4_1106_PREVIEW,
            api_key=os.getenv("AZURE_SECENTRAL_OPENAI_API_KEY"),
            base_url="https://fixie-secentral.openai.azure.com",
        ),
        _Llm(
            GPT_4_1106_PREVIEW,
            api_key=os.getenv("AZURE_UKSOUTH_OPENAI_API_KEY"),
            base_url="https://fixie-uksouth.openai.azure.com",
        ),
        # GPT-3.5
        _Llm(GPT_35_TURBO_0125),
        _Llm(GPT_35_TURBO_1106),
        _Llm(GPT_35_TURBO_1106, base_url="https://fixie-westus.openai.azure.com"),
        _Llm(
            GPT_35_TURBO,
            api_key=AZURE_EASTUS2_OPENAI_API_KEY,
            base_url="https://fixie-openai-sub-with-gpt4.openai.azure.com",
        ),
        # Claude
        _Llm("claude-3-opus-20240229"),
        _Llm("claude-3-sonnet-20240229"),
        _Llm("claude-3-haiku-20240307"),
        # Cohere
        _Llm("command-r-plus"),
        _Llm("command-r"),
        _Llm("command-light"),
        # Gemini
        _Llm("gemini-pro"),
        _Llm("gemini-1.5-pro-preview-0514"),
        _Llm("gemini-1.5-flash-preview-0514"),
        # Mistral 8x22b
        # _Llm(
        #    "mistral-large",  # is this the same?
        #    api_key=os.getenv("AZURE_EASTUS2_MISTRAL_API_KEY"),
        #    base_url="https://fixie-mistral-serverless.eastus2.inference.ai.azure.com/v1",
        # ),
        _AnyscaleLlm("mistralai/Mixtral-8x22B-Instruct-v0.1", MIXTRAL_8X22B_INSTRUCT),
        _FireworksLlm(
            "accounts/fireworks/models/mixtral-8x22b-instruct", MIXTRAL_8X22B_INSTRUCT
        ),
        _OctoLlm("mixtral-8x22b-instruct", MIXTRAL_8X22B_INSTRUCT),
        _TogetherLlm("mistralai/Mixtral-8x22B-Instruct-v0.1", MIXTRAL_8X22B_INSTRUCT),
        # Mistral 8x7b
        _AnyscaleLlm("mistralai/Mixtral-8x7B-Instruct-v0.1", MIXTRAL_8X7B_INSTRUCT),
        _DatabricksLlm("databricks-mixtral-8x7b-instruct", MIXTRAL_8X7B_INSTRUCT),
        _FireworksLlm(
            "accounts/fireworks/models/mixtral-8x7b-instruct", MIXTRAL_8X7B_INSTRUCT
        ),
        _GroqLlm("mixtral-8x7b-32768", MIXTRAL_8X7B_INSTRUCT),
        _OctoLlm("mixtral-8x7b-instruct", MIXTRAL_8X7B_INSTRUCT),
        _TogetherLlm("mistralai/Mixtral-8x7B-Instruct-v0.1", MIXTRAL_8X7B_INSTRUCT),
        # Function calling Mistral 8x7b
        _FireworksLlm("accounts/fireworks/models/firefunction-v1", "firefunction-v1"),
        # Llama 3 70b
        _AnyscaleLlm("meta-llama/Llama-3-70b-chat-hf", LLAMA_3_70B_CHAT),
        _DatabricksLlm("databricks-meta-llama-3-70b-instruct", LLAMA_3_70B_CHAT),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3-70b-instruct", LLAMA_3_70B_CHAT
        ),
        _GroqLlm("llama3-70b-8192", LLAMA_3_70B_CHAT),
        _OctoLlm("meta-llama-3-70b-instruct", LLAMA_3_70B_CHAT),
        _PerplexityLlm("llama-3-70b-instruct", LLAMA_3_70B_CHAT),
        _TogetherLlm("meta-llama/Llama-3-70b-chat-hf", LLAMA_3_70B_CHAT),
        # Function calling with Llama 3 70b
        _FireworksLlm(
            "accounts/fireworks/models/firefunction-v2-rc", "firefunction-v2"
        ),
        # Llama 3 8b
        _AnyscaleLlm("meta-llama/Llama-3-8b-chat-hf", LLAMA_3_8B_CHAT),
        _CloudflareLlm("@cf/meta/llama-3-8b-instruct", LLAMA_3_8B_CHAT),
        _FireworksLlm(
            "accounts/fireworks/models/llama-v3-8b-instruct", LLAMA_3_8B_CHAT
        ),
        _GroqLlm("llama3-8b-8192", LLAMA_3_8B_CHAT),
        _OctoLlm("meta-llama-3-8b-instruct", LLAMA_3_8B_CHAT),
        _PerplexityLlm("llama-3-8b-instruct", LLAMA_3_8B_CHAT),
        _TogetherLlm("meta-llama/Llama-3-8b-chat-hf", LLAMA_3_8B_CHAT),
        # Phi-2
        _CloudflareLlm("@cf/microsoft/phi-2", PHI_2),
        _TogetherLlm("microsoft/phi-2", PHI_2),
    ]


def _image_models():
    return [
        _Llm(GPT_4O),
        _Llm(GPT_4_TURBO),
        _Llm("gpt-4-vision-preview", base_url="https://fixie-westus.openai.azure.com"),
        _Llm("claude-3-opus-20240229"),
        _Llm("claude-3-sonnet-20240229"),
        _Llm("gemini-pro-vision"),
        _Llm("gemini-1.5-pro-preview-0514"),
        _Llm("gemini-1.5-flash-preview-0514"),
        _FireworksLlm("accounts/fireworks/models/firellava-13b", "firellava-13b"),
    ]


def _av_models():
    return [
        # _Llm(GPT_4O),
        _Llm("gemini-1.5-pro-preview-0514"),
        _Llm("gemini-1.5-flash-preview-0514"),
    ]


# def _get_models(mode: str, filter: Optional[str] = None):
#     mode_map = {
#         "text": _text_models,
#         "image": _image_models,
#         "audio": _av_models,
#         "video": _av_models,
#     }
#     if mode not in mode_map:
#         raise ValueError(f"Unknown mode {mode}")
#     models = mode_map[mode]()
#     return [m for m in models if not filter or filter in m.args["model"].lower()]


def _get_prompt(mode: str) -> List[str]:
    if mode == "text":
        return ["讲个笑话"]
    elif mode == "image":
        return [
            "Based on the image, explain what will happen next.",
            "--file",
            "media/image/inception.jpeg",
        ]
    elif mode == "audio":
        return [
            "Summarize the information in the audio clip.",
            "--file",
            "media/audio/news.wav",
        ]
    elif mode == "video":
        return [
            "What color is the logo on the screen and how does it relate to what the actor is saying?",
            "--file",
            "media/video/psa.webm",
        ]
    raise ValueError(f"Unknown mode {mode}")


@dataclasses.dataclass
class _Response(dataclasses_json.DataClassJsonMixin):
    time: str
    duration: str
    region: str
    cmd: str
    results: List[llm_request.ApiMetrics]


def _format_response(
    response: _Response, format: str, dlen: int = 0
) -> Tuple[str, str]:
    if format == "json":
        return response.to_json(indent=2, ensure_ascii=False), "application/json"
    else:
        s = (
            "| Provider/Model                             | TTR  | TTFT | TPS | Tok | Total |"
            f" {'Response':{dlen}.{dlen}} |\n"
            "| :----------------------------------------- | ---: | ---: | --: | --: | ----: |"
            f" {':--':-<{dlen}.{dlen}} |\n"
        )

        for r in response.results:
            ttr = r.ttr or 0.0
            ttft = r.ttft or 0.0
            tps = r.tps or 0.0
            num_tokens = r.num_tokens or 0
            total_time = r.total_time or 0.0
            output = r.error or r.output.replace("\n", "\\n").strip()
            s += (
                f"| {r.model:42} | {ttr:4.2f} | {ttft:4.2f} | "
                f"{tps:3.0f} | {num_tokens:3} | {total_time:5.2f} | "
                f"{output:{dlen}.{dlen}} |\n"
            )

        s += f"\ntime: {response.time}, duration: {response.duration} region: {response.region}, cmd: {response.cmd}\n"
        return s, "text/markdown"


def get_models():
    with open("llm.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    models = []
    for provider, model_info in data.items():
        for model in model_info:
            models.append((provider, model))
    return models        
    
async def _run(models) -> Tuple[str, str]:
    """
    This function is invoked either from the webapp (via run) or the main function below.
    The args we know about are stored in args, and any unknown args are stored in pass_argv,
    which we'll pass to the _Llm.run function, who will turn them back into a
    single list of flags for consumption by the llm_benchmark.run function.
    """
    time_start = datetime.datetime.now()
    time_str = time_start.isoformat()
    region = os.getenv("REGION", "local")
    tasks = []
    # for m in models:
    m_runtime = _Llm(model=models[1])
    tasks.append(asyncio.create_task(m_runtime.run(["-P",models[0],"-m", models[1]], 3)))
    await asyncio.gather(*tasks)
    results = [t.result() for t in tasks if t.result() is not None]
    elapsed = datetime.datetime.now() - time_start
    elapsed_str = f"{elapsed.total_seconds():.2f}s"
    response = _Response(time_str, elapsed_str, region, "", results)
    return response.to_dict()
    # return _format_response(response, DEFAULT_FORMAT, DEFAULT_DISPLAY_LENGTH)

async def main():
    time_start = datetime.datetime.now()
    time_str = time_start.isoformat()
    region = os.getenv("REGION", "local")
    
    with open("llm.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    llm_metrices = []
    # 国内大部分模型API的 RPS 较低，故不再并发访问测试    
    for provider, model_info in data.items():
        for model in model_info:
            time.sleep(5)
            llm_metrice = await _run((provider, model))
            print(llm_metrice["results"])
            llm_metrices +=llm_metrice["results"]
    elapsed = datetime.datetime.now() - time_start
    elapsed_str = f"{elapsed.total_seconds():.2f}s"
    result = {
        "time": time_str,
        "duration": elapsed_str,
        "region": region,
        "prompt": "使用以下模板创建一个自我介绍,name填入'莫尔索',age填入'18',hobby填入'打羽毛球':'我的名字是{name},今年{age}岁,我最大的爱好是{hobby}",
        "results": llm_metrices
    }
    with open("./2024-07-29.json", 'w') as f:
	    f.write(json.dumps(result, indent=4, ensure_ascii=False))
      
if __name__ == "__main__":
    asyncio.run(main())
