"""LLM 接入层：自定义模型 + API（OpenAI 兼容协议，零额外依赖）。

设计目标：不写死任何服务商。只要对方提供 OpenAI 兼容的 /chat/completions，
改配置就能换模型 —— 智谱 GLM / DeepSeek / Moonshot / 通义(兼容模式) /
SiliconFlow / OpenAI / Ollama 本地 / vLLM 自部署 全部适用。

配置优先级：命令行参数 > 配置文件 > 环境变量（LLM_BASE_URL / LLM_API_KEY / LLM_MODEL）。
配置值支持 ${环境变量} 引用（如 api_key: ${ZHIPUAI_API_KEY}），避免密钥明文进仓库。

新协议适配（比如要接原生 Anthropic API）：实现同样的 complete(prompt, system) 即可，
抽取管线只依赖这个方法。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

#: 项目根目录（src/kg/llm.py 向上三级）
ROOT = Path(__file__).resolve().parents[2]

#: 默认配置文件名（放项目根目录；llm_config.yaml 被 .gitignore 忽略，可放密钥）
DEFAULT_CONFIG_NAME = "llm_config.yaml"

#: 环境变量兜底
ENV_BASE_URL = "LLM_BASE_URL"
ENV_API_KEY = "LLM_API_KEY"
ENV_MODEL = "LLM_MODEL"
ENV_EMBEDDING_MODEL = "LLM_EMBEDDING_MODEL"
ENV_EMBEDDING_BASE_URL = "LLM_EMBEDDING_BASE_URL"

_INTERP_RE = re.compile(r"\$\{(\w+)\}")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class LLMConfigError(RuntimeError):
    """配置缺失/不合法。报错信息要能直接告诉用户怎么修。"""


class LLMError(RuntimeError):
    """请求失败或响应不符合预期。"""


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """一次 LLM 调用所需的全部配置。embedding_* 可缺省 -> 用本地哈希向量降级。"""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    embedding_model: str = ""      # 留空 = 不用 API 嵌入，走本地 HashEmbedder
    embedding_base_url: str = ""   # 留空 = 沿用 base_url
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 120
    chunk_chars: int = 4000
    max_retries: int = 1
    gleans: int = 0  # GraphRAG 式补抽轮数：首抽后追问遗漏，提升召回（费 token）

    def validate(self) -> None:
        """缺什么就报什么，并给出三种配置途径的指引。"""
        missing = [k for k, v in (("base_url", self.base_url), ("model", self.model)) if not v]
        if not self.api_key and not self._is_local(self.base_url):
            missing.append("api_key")
        if missing:
            raise LLMConfigError(
                f"LLM 配置缺少 {', '.join(missing)}。三种配置途径（优先级从高到低）：\n"
                f"  1. 命令行参数： --base-url ... --api-key ... --model ...\n"
                f"  2. 配置文件：  复制 llm_config.example.yaml 为 llm_config.yaml 后填写\n"
                f"  3. 环境变量：  {ENV_BASE_URL} / {ENV_API_KEY} / {ENV_MODEL}"
            )

    @staticmethod
    def _is_local(base_url: str) -> bool:
        """本地推理服务（Ollama 等）通常不校验密钥，api_key 可为空。"""
        try:
            host = re.sub(r"^\w+://", "", base_url).split("/")[0].split(":")[0]
            return host.lower() in _LOCAL_HOSTS
        except Exception:  # noqa: BLE001
            return False


def _interp(value: object) -> object:
    """把字符串里的 ${VAR} 替换为环境变量值；未定义的变量替换为空串。"""
    if not isinstance(value, str):
        return value
    return _INTERP_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


def _pick(*candidates):
    """取第一个非空候选（0 / False 也是合法值，不能简单 or）。"""
    for c in candidates:
        if c is not None and c != "":
            return c
    return None


def load_config(config_path: Optional[str | Path] = None, **overrides) -> LLMConfig:
    """加载 LLM 配置：文件 + 环境变量 + 覆盖参数三路合并。

    overrides 里只传调用方真正提供的键（CLI 没给的参数不要传进来，
    否则会顶掉文件/环境变量的值）。
    """
    path = Path(config_path) if config_path else ROOT / DEFAULT_CONFIG_NAME
    file_data: dict = {}
    if path.exists():
        file_data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    env_of = {"base_url": ENV_BASE_URL, "api_key": ENV_API_KEY, "model": ENV_MODEL,
              "embedding_model": ENV_EMBEDDING_MODEL,
              "embedding_base_url": ENV_EMBEDDING_BASE_URL}

    def merged(key: str):
        env_val = os.environ.get(env_of[key]) if key in env_of else None
        return _pick(overrides.get(key), _interp(file_data.get(key)), env_val)

    cfg = LLMConfig(
        base_url=str(merged("base_url") or ""),
        api_key=str(merged("api_key") or ""),
        model=str(merged("model") or ""),
        embedding_model=str(merged("embedding_model") or ""),
        embedding_base_url=str(_pick(merged("embedding_base_url"),
                                     merged("base_url")) or ""),
        temperature=float(_pick(merged("temperature"), LLMConfig.temperature)),
        max_tokens=int(_pick(merged("max_tokens"), LLMConfig.max_tokens)),
        timeout=int(_pick(merged("timeout"), LLMConfig.timeout)),
        chunk_chars=int(_pick(merged("chunk_chars"), LLMConfig.chunk_chars)),
        max_retries=int(_pick(merged("max_retries"), LLMConfig.max_retries)),
        gleans=int(_pick(merged("gleans"), LLMConfig.gleans)),
    )
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------
# 客户端（OpenAI 兼容 /chat/completions）
# --------------------------------------------------------------------------


def api_endpoint(base_url: str, suffix: str) -> str:
    """base_url -> 具体端点地址（suffix 如 chat/completions、embeddings）。"""
    b = base_url.strip().rstrip("/")
    if not b:
        raise LLMConfigError("base_url 为空")
    if b.endswith(suffix):
        return b
    return f"{b}/{suffix}"


def chat_url(base_url: str) -> str:
    """base_url -> 完整 chat/completions 地址。填到域名或 /v1 级别都行。"""
    return api_endpoint(base_url, "chat/completions")


class LLMClient:
    """最小可用的 OpenAI 兼容客户端。

    抽取管线只依赖 complete(prompt, system)，换协议时实现同名方法即可替换本类。
    """

    def __init__(self, cfg: LLMConfig) -> None:
        cfg.validate()
        self.cfg = cfg

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        """调 OpenAI 兼容 /embeddings，返回与 texts 等长的向量列表。"""
        base = self.cfg.embedding_base_url or self.cfg.base_url
        url = api_endpoint(base, "embeddings")
        body = {"model": self.cfg.embedding_model, "input": texts}
        data = self._post(url, body)
        try:
            rows = sorted(data["data"], key=lambda d: d.get("index", 0))
            return [list(map(float, d["embedding"])) for d in rows]
        except (KeyError, TypeError, ValueError):
            raise LLMError(
                f"响应里没有 data[*].embedding：{json.dumps(data, ensure_ascii=False)[:300]}")

    def complete(self, prompt: str, system: str = "") -> str:
        url = api_endpoint(self.cfg.base_url, "chat/completions")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                data = self._post(url, body)
                return self._content(data)
            except LLMError as exc:
                last_err = exc
                # 限流/网关抖动值得重试；参数错误重试也没用
                if not getattr(exc, "retryable", False):
                    raise
                if attempt < self.cfg.max_retries:
                    time.sleep(2 * (attempt + 1))
        raise last_err  # type: ignore[misc]

    def _post(self, url: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:  # 本地服务可不带
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        req = urllib.request.Request(
            url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            err = LLMError(f"HTTP {exc.code} 来自 {self.cfg.model} @ {url}\n  {detail}")
            err.retryable = exc.code in (429, 500, 502, 503, 504)
            raise err from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"连不上 {url}（{exc.reason}）。检查 base_url 与网络") from exc
        except TimeoutError as exc:
            err = LLMError(f"请求超时（>{self.cfg.timeout}s）：{url}")
            err.retryable = True
            raise err from exc

    @staticmethod
    def _content(data: dict) -> str:
        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"响应里没有 choices[0].message.content：{json.dumps(data, ensure_ascii=False)[:300]}")
