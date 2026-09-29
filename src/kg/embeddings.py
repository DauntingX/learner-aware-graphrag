"""嵌入与向量索引层（GraphRAG 的向量召回底座）。

两条路子，按配置自动选择：
  1. APIEmbedder —— OpenAI 兼容 /embeddings 接口（embedding_model 配置了就用它），
     语义检索质量最好；与 LLM 接入层共用 base_url/api_key，同样支持自定义模型与 API。
  2. HashEmbedder —— 本地字符 n-gram 特征哈希向量（无 API / 离线测试用）。
     词面级相似度，确定性、零依赖；语义弱于真嵌入，但足以跑通全链路与对比评测。

向量索引持久化为 JSON（这个量级 < 1MB，可读性优先；升级 Qdrant 时只换这一层）。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .llm import LLMClient, LLMConfig


# --------------------------------------------------------------------------
# 嵌入器
# --------------------------------------------------------------------------


class HashEmbedder:
    """字符 n-gram 特征哈希嵌入（降级/测试用）。

    中文按字 bigram（单字 fallback）、ASCII 按词 unigram，
    用带符号哈希散进固定维度后 L2 归一化。同词面的文本相似，语义无关的接近正交。
    """

    DIM = 512

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def _tokens(self, text: str) -> List[str]:
        text = str(text).strip().lower()
        tokens: List[str] = []
        for seg in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text):
            if re.match(r"^[\u4e00-\u9fff]+$", seg):
                tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
                if len(seg) == 1:
                    tokens.append(seg)
            else:
                tokens.append(seg)
        return tokens

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        counts: Dict[str, int] = {}
        for tok in self._tokens(text):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, c in counts.items():
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(c))  # sublinear tf
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.embed_one(t) for t in texts])


class APIEmbedder:
    """OpenAI 兼容 /embeddings 客户端封装（批量、L2 归一化）。"""

    BATCH = 16

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        outs: List[List[float]] = []
        for i in range(0, len(texts), self.BATCH):
            outs.extend(self.client.embeddings(list(texts[i:i + self.BATCH])))
        arr = np.asarray(outs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def embedder_identity(embedder) -> tuple[str, str]:
    """Return a stable, non-secret identity for index compatibility checks."""
    if isinstance(embedder, HashEmbedder):
        return "hash", f"char-ngram-md5-v1:dim={embedder.dim}"
    if isinstance(embedder, APIEmbedder):
        cfg = embedder.client.cfg
        endpoint = (cfg.embedding_base_url or cfg.base_url).rstrip("/")
        # Keep endpoint credentials and private hostnames out of index metadata.
        endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
        return "api", f"{cfg.embedding_model}:endpoint-sha256={endpoint_hash}"
    # An extension embedder can expose a stable cache identity of its own.
    return (f"{type(embedder).__module__}.{type(embedder).__qualname__}",
            str(getattr(embedder, "cache_identity", "unknown")))


def get_embedder(cfg: Optional[LLMConfig] = None, client: Optional[LLMClient] = None):
    """按配置选择嵌入器：配置了 embedding_model 用 API，否则本地 Hash。"""
    if cfg is not None and cfg.embedding_model:
        return APIEmbedder(client or LLMClient(cfg))
    return HashEmbedder()


# --------------------------------------------------------------------------
# 向量索引（节点 + 社区两类条目）
# --------------------------------------------------------------------------


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> List[tuple[int, float]]:
    """返回 [(行号, 余弦相似度)] 按相似度降序，最多 k 个正相似条目。"""
    if matrix.shape[0] == 0:
        return []
    sims = matrix @ query_vec
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order if sims[i] > 0]


class VectorIndex:
    """极简向量索引：条目 id -> 向量 + 元数据，JSON 持久化。"""

    def __init__(self, kind: str, model_tag: str) -> None:
        self.kind = kind  # "hash" 或 API 模型名，用于检测索引与当前嵌入器是否匹配
        self.model_tag = model_tag
        self.ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.meta: Dict[str, dict] = {}
        self.cache_signature: str = ""

    @classmethod
    def build(cls, kind: str, model_tag: str, items: List[tuple[str, str, dict]],
              embedder) -> "VectorIndex":
        """items: (id, 文本, 元数据)。文本做嵌入，元数据原样存。"""
        idx = cls(kind, model_tag)
        idx.ids = [i for i, _t, _m in items]
        idx.vectors = embedder.embed_texts([t for _i, t, _m in items])
        idx.meta = {i: m for i, _t, m in items}
        return idx

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": self.kind,
            "model_tag": self.model_tag,
            "dim": int(self.vectors.shape[1]),
            "ids": self.ids,
            "vectors": [[round(float(x), 6) for x in row] for row in self.vectors],
            "meta": self.meta,
            "cache_signature": self.cache_signature,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        idx = cls(raw["kind"], raw["model_tag"])
        idx.ids = raw["ids"]
        idx.vectors = np.asarray(raw["vectors"], dtype=np.float32)
        idx.meta = raw.get("meta", {})
        idx.cache_signature = raw.get("cache_signature", "")
        return idx

    def search(self, query_text: str, embedder, k: int) -> List[tuple[str, float]]:
        """返回 [(id, 相似度)] 降序。"""
        if not self.ids or self.vectors is None or self.vectors.shape[0] == 0:
            return []
        qv = embedder.embed_texts([query_text])[0]
        return [(self.ids[i], s) for i, s in cosine_topk(qv, self.vectors, k)]
