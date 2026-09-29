"""GraphRAG 检索引擎：向量召回 + 图多跳扩展 + 掌握度加权。

与朴素 RAG 的三个本质区别（这是本项目的核心差异化）：
  1. **图扩展**：向量只负责找到种子节点，随后沿图边扩展前置知识、易混淆概念 ——
     朴素 RAG 永远召不回「文本不相似但教学上必需」的前置知识；
  2. **掌握度加权**：排序分 = 相似度 ×(1 - 0.85×掌握度)。同一个查询，薄弱/未接触的
     子图排在前面，已掌握的自动降权 —— 上下文里的每个节点都带掌握度标注，
     LLM 一眼就知道「哪些要细讲、哪些一句带过」；
  3. **双粒度检索**：local（节点级）+ global（社区摘要级，communities.py），
     后者回答「这个领域整体有什么」的鸟瞰型问题。

检索结果组装成带角色分节的教学上下文，直接可塞进 LLM 提示词，也通过
MCP 工具 search_knowledge 暴露给 Agent。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .communities import BAND_LABELS, all_communities
from .embeddings import VectorIndex, cosine_topk, embedder_identity
from .extract import ExtractError, parse_llm_json
from .learner import LearnerModel
from .schema import band_of
from .store import KnowledgeGraph

#: 掌握度压制系数：扩展候选用 score = sim ×(1 - MASTERY_SUPPRESSION×mastery) 排序。
#: 种子按纯相似度选（意图优先，加权不与用户问的东西竞争）；
#: 加权的战场是扩展填充 —— 图邻居多于槽位时薄弱者先进上下文。
MASTERY_SUPPRESSION = 0.85

DEFAULT_TOP_K = 3
DEFAULT_MAX_CONTEXT = 12
DEFAULT_PREREQ_DEPTH = 1

ROLE_ORDER = {"target": 0, "prereq": 1, "confusable": 2, "related": 3}
ROLE_LABELS = {
    "target": "目标概念（按相关度）",
    "prereq": "建议先懂（前置知识，薄弱优先）",
    "confusable": "易混淆对照",
    "related": "相关概念",
}


# --------------------------------------------------------------------------
# 索引构建
# --------------------------------------------------------------------------


INDEX_PIPELINE_VERSION = 2


def index_signature(kg: KnowledgeGraph, learner: LearnerModel, embedder) -> str:
    """Fingerprint everything that affects node vectors or community summaries.

    Sort graph records so a semantically identical graph produces the same key
    regardless of YAML insertion order. Mastery matters because community
    summaries include it and are themselves embedded.
    """
    nodes = sorted((str(nid), data) for nid, data in kg.g.nodes(data=True))
    edges = sorted(
        ((str(src), str(dst), str(key), data)
         for src, dst, key, data in kg.g.edges(keys=True, data=True)),
        key=lambda row: (row[0], row[1], row[2]),
    )
    payload = {
        "pipeline_version": INDEX_PIPELINE_VERSION,
        "graph_nodes": nodes,
        "graph_edges": edges,
        "mastery": sorted((str(nid), learner.mastery_or_zero(nid)) for nid in kg.g.nodes),
        "embedder": embedder_identity(embedder),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def node_index_text(node) -> str:
    """节点 -> 参与嵌入的文本。

    名称重复 3 次做字段加权：查询命中的往往就是概念名本身（如「RLHF 的训练流程」），
    不加权会被长描述稀释（实测 0.14 -> 0.21，决定该概念能否进种子）。
    """
    parts = [node.name] * 3 + list(node.aliases or []) + [node.domain, node.desc]
    return " ".join(parts)


def build_index(kg: KnowledgeGraph, learner: LearnerModel, embedder,
                out_path: Optional[str | Path] = None) -> dict:
    """构建节点索引与社区索引。返回 {"nodes": VectorIndex, "communities": [dict]}。"""
    items = []
    for nid in kg.g.nodes:
        node = kg.node(nid)
        m = learner.mastery_or_zero(nid)
        items.append((nid, node_index_text(node),
                      {"name": node.name, "domain": node.domain, "mastery": round(m, 3)}))
    kind, model_tag = embedder_identity(embedder)
    node_idx = VectorIndex.build(kind, model_tag, items, embedder)

    comms = all_communities(kg, learner)
    comm_idx = VectorIndex.build(
        kind, model_tag,
        [(c["id"], c["summary"], c) for c in comms],
        embedder,
    )
    signature = index_signature(kg, learner, embedder)
    node_idx.cache_signature = signature
    comm_idx.cache_signature = signature
    index = {"nodes": node_idx, "communities": comm_idx, "community_meta": comms}
    if out_path is not None:
        save_index(index, out_path)
    return index


def save_index(index: dict, path: str | Path) -> None:
    p = Path(path)
    index["nodes"].save(p)
    index["communities"].save(str(p) + ".communities.json")


def load_index(path: str | Path) -> dict:
    p = Path(path)
    nodes = VectorIndex.load(p)
    communities = VectorIndex.load(str(p) + ".communities.json")
    if (not nodes.cache_signature or
            nodes.cache_signature != communities.cache_signature):
        raise ValueError("索引版本过旧或节点/社区索引不匹配，需重建")
    community_meta = []
    for cid in communities.ids:
        meta = communities.meta.get(cid)
        if not isinstance(meta, dict) or not {"id", "summary", "members", "hubs"} <= meta.keys():
            raise ValueError("社区索引缺少全局检索摘要，需重建")
        community_meta.append(meta)
    return {"nodes": nodes, "communities": communities,
            "community_meta": community_meta}


def index_is_current(index: dict, kg: KnowledgeGraph, learner: LearnerModel,
                     embedder) -> bool:
    """Reject stale, partial, or incompatible on-disk/in-memory indexes."""
    nodes: VectorIndex = index["nodes"]
    communities: VectorIndex = index["communities"]
    signature = index_signature(kg, learner, embedder)
    kind, model_tag = embedder_identity(embedder)
    if (nodes.cache_signature != signature or communities.cache_signature != signature or
            nodes.kind != kind or communities.kind != kind or
            nodes.model_tag != model_tag or communities.model_tag != model_tag):
        return False
    if set(nodes.ids) != set(kg.g.nodes) or len(nodes.ids) != len(kg.g.nodes):
        return False
    if not all(nid in nodes.meta for nid in nodes.ids):
        return False
    if set(communities.ids) != {c["id"] for c in index.get("community_meta", [])}:
        return False
    for vi in (nodes, communities):
        if vi.vectors is None or vi.vectors.ndim != 2 or len(vi.ids) != vi.vectors.shape[0]:
            return False
    return True


def load_or_build_index(kg: KnowledgeGraph, learner: LearnerModel, embedder,
                        path: str | Path, rebuild: bool = False) -> tuple[dict, bool]:
    """Load a matching index or rebuild it when inputs or files have changed."""
    if not rebuild:
        try:
            index = load_index(path)
            if index_is_current(index, kg, learner, embedder):
                return index, False
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return build_index(kg, learner, embedder, out_path=path), True


# --------------------------------------------------------------------------
# 检索结果
# --------------------------------------------------------------------------


@dataclass
class RAGResult:
    query: str
    mode: str  # local / global
    seeds: List[dict] = field(default_factory=list)  # {id, name, sim, score, mastery}
    context_nodes: List[dict] = field(default_factory=list)  # {id, name, role, ...}
    context_text: str = ""
    stats: dict = field(default_factory=dict)


def _entry(nid: str, role: str, kg: KnowledgeGraph, learner: LearnerModel,
           sim: float = 0.0) -> dict:
    node = kg.node(nid)
    m = learner.mastery_or_zero(nid)
    band = band_of(m).value
    return {
        "id": nid,
        "name": node.name if node else nid,
        "role": role,
        "domain": node.domain if node else "",
        "desc": node.desc if node else "",
        "evidence_refs": [ref.model_dump(mode="json") for ref in getattr(node, "evidence_refs", [])]
        if node else [],
        "difficulty": node.difficulty if node else 3,
        "mastery": round(m, 3),
        "band": BAND_LABELS[band],
        "sim": round(sim, 4),
    }


def _name_desc(kg: KnowledgeGraph, nid: str) -> str:
    node = kg.node(nid)
    if not node:
        return nid
    return f"{node.name}（{node.domain}）{node.desc}"


def _prereq_order(nids: Sequence[str], learner: LearnerModel,
                  kg: KnowledgeGraph) -> List[str]:
    """前置知识的教学序：薄弱的先讲（掌握度升序），同水平低难度优先。"""
    return sorted(nids, key=lambda n: (learner.mastery_or_zero(n),
                                       kg.node(n).difficulty if kg.node(n) else 3))


def _render_context(nodes: List[dict]) -> str:
    """按角色分节渲染教学上下文，直接可塞进提示词。"""
    by_role: Dict[str, List[dict]] = {}
    for n in nodes:
        by_role.setdefault(n["role"], []).append(n)
    lines = []
    for role in sorted(by_role, key=lambda r: ROLE_ORDER.get(r, 9)):
        lines.append(f"【{ROLE_LABELS.get(role, role)}】")
        for n in by_role[role]:
            desc = f"：{n['desc']}" if n.get("desc") else ""
            lines.append(f"- {n['name']}（{n['band']}，掌握度 {n['mastery']:.2f}）{desc}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 查询扩展（LightRAG：关键词多路召回 + 伪相关反馈）
# --------------------------------------------------------------------------


KEYWORD_PROMPT = """从用户问题里抽出用于知识图谱检索的关键词，两类都要有：
1. 具体概念词（问题里或隐含的实体/术语名，如「KV 缓存」「反向传播」）
2. 主题词（更宽泛的领域词，如「推理优化」）
最多 {max_kw} 个。只输出 JSON：{{"keywords": ["...", "..."]}}

用户问题：{query}"""


def extract_query_keywords(query: str, client, max_kw: int = 6) -> List[str]:
    """LLM 抽取查询关键词（LightRAG 的查询侧处理）。任何失败返回空列表，
    检索退化为纯原查询 —— 关键词是增益项不是依赖项。"""
    try:
        raw = client.complete(KEYWORD_PROMPT.format(query=query, max_kw=max_kw),
                              system="只输出一个 JSON 对象。")
        obj = parse_llm_json(raw)
        kws = [str(k).strip() for k in obj.get("keywords", []) if str(k).strip()]
        return kws[:max_kw]
    except Exception:  # noqa: BLE001
        return []


#: 伪相关反馈的 Rocchio β 权重：final = 原查询相似度 + β×质心相似度。
#: 用加法混合而不是 max-pool —— max 会让「与首轮 cluster 相似的泛化节点」
#: 压过原始查询的目标（实测把 graph_rag 自己挤出了种子）。
PSEUDO_FEEDBACK_BETA = 0.4


def _multi_query_hits(query: str, extra_queries: Optional[List[str]],
                      pseudo_feedback: bool, kg: KnowledgeGraph,
                      index: dict, embedder, k: int) -> List[Tuple[str, float]]:
    """多查询召回：关键词 max-pooling + 伪相关反馈 Rocchio 加法混合。

    - 原查询 + LLM 关键词（如有）之间取 max —— 它们是同一意图的不同表述；
    - 反馈质心按 Rocchio 加到原始查询相似度上 —— 反馈只做微调，不与意图竞争
      （「模型训练时梯度消失怎么办」靠质心把 LSTM/循环神经网络带进种子）。
    """
    node_idx: VectorIndex = index["nodes"]
    if node_idx.vectors is None or node_idx.vectors.shape[0] == 0:
        return []
    texts = [query] + [q for q in (extra_queries or []) if q and q != query]
    qv = embedder.embed_texts(texts)                       # (m, d)
    sims = node_idx.vectors @ qv.T                         # (n, m)
    base = sims.max(axis=1)

    def top(best_vec):
        order = np.argsort(-best_vec)[:k]
        return [(node_idx.ids[int(i)], float(best_vec[i])) for i in order if best_vec[i] > 0]

    hits = top(base)
    if pseudo_feedback and hits:
        names = [kg.node(nid).name for nid, _ in hits[:3] if kg.node(nid)]
        if names:
            nv = embedder.embed_texts(names).mean(axis=0)
            norm = float(np.linalg.norm(nv))
            if norm > 0:
                nv = nv / norm
                best = base + PSEUDO_FEEDBACK_BETA * (node_idx.vectors @ nv)
                boosted = top(best)
                # 意图保底：原查询的 top-1 永远是种子第 1，反馈只竞争剩余名额。
                # 反馈质心会放大首轮噪声（首轮集群互相重叠时尤其严重），
                # 不设保底会把用户真正问的东西挤出种子（实测踩过）。
                top1 = hits[0]
                rest = [h for h in boosted if h[0] != top1[0]][: max(k - 1, 0)]
                hits = [top1] + rest
    return hits


# --------------------------------------------------------------------------
# local search
# --------------------------------------------------------------------------


def local_search(query: str, kg: KnowledgeGraph, learner: LearnerModel,
                 index: dict, embedder, top_k: int = DEFAULT_TOP_K,
                 max_context: int = DEFAULT_MAX_CONTEXT,
                 prereq_depth: int = DEFAULT_PREREQ_DEPTH,
                 extra_queries: Optional[List[str]] = None,
                 pseudo_feedback: bool = True,
                 rerank_client=None) -> RAGResult:
    """GraphRAG local search：向量找种子 -> 图扩展前置/易混淆 -> 掌握度加权填充。

    extra_queries：LLM 抽取的查询关键词（见 extract_query_keywords），多路 max-pooling。
    rerank_client：可选 LLM 重排 —— 扩展填充顺序交给 LLM 按「教学相关性」排序
    （LightRAG 的可选 rerank），失败自动退回掌握度排序。
    """
    node_idx: VectorIndex = index["nodes"]
    graph_ids = set(kg.g.nodes)
    raw_hits = [(nid, sim) for nid, sim in _multi_query_hits(
        query, extra_queries, pseudo_feedback, kg, index, embedder, k=top_k * 4)
        if nid in graph_ids]

    # scored 保持 _multi_query_hits 给定的顺序 —— 那里面编码了「意图优先」策略
    # （原查询 top-1 保底，反馈只竞争剩余名额）。这里绝不能按相似度重排：
    # 质心加分会把用户真正问的东西又挤出去（实测踩过两次）。
    scored = []
    for nid, sim in raw_hits:
        m = learner.mastery_or_zero(nid)
        scored.append({"id": nid, "sim": sim, "mastery": round(m, 3),
                       "score": sim * (1 - MASTERY_SUPPRESSION * m)})
    seeds = scored[:top_k]

    if not seeds:
        return RAGResult(query=query, mode="local", stats={"reason": "无正相似度召回"})

    # 图扩展：前置（教学必需，薄弱优先）-> 易混淆/相关（跨种子全局薄弱优先填充，
    # 掌握度加权只作用于"谁进上下文"的填充决策，不与查询意图竞争）
    seen: Dict[str, str] = {}
    context: List[dict] = []
    for s in seeds:
        seen[s["id"]] = "target"
        context.append(_entry(s["id"], "target", kg, learner, sim=s["sim"]))
    for s in seeds:
        for pid in _prereq_order(kg.prerequisites(s["id"], depth=prereq_depth), learner, kg):
            if pid not in seen and len(context) < max_context:
                seen[pid] = "prereq"
                context.append(_entry(pid, "prereq", kg, learner))

    fill: List[tuple[float, str, str]] = []  # (掌握度, 种子序, 节点id)
    for si, s in enumerate(seeds):
        conf = set(kg.confusable(s["id"]))
        nbrs = set(conf)
        for e in kg.edges_of(s["id"]):
            other = e["target"] if e["source"] == s["id"] else e["source"]
            if other not in seen:
                nbrs.add(other)
        for nid in nbrs - set(seen):
            fill.append((learner.mastery_or_zero(nid), f"{si}", nid))
    fill.sort()  # 默认全局按掌握度升序：最薄弱的邻居先进上下文

    if rerank_client is not None and fill:
        # LightRAG 式可选 rerank：填充顺序交给 LLM 按「对回答该问题的教学相关性」重排，
        # 解析失败自动退回掌握度排序（rerank 是增益项不是依赖项）
        cand_ids = [nid for _m, _si, nid in fill]
        cand_lines = "\n".join(
            f'- "{nid}": {_name_desc(kg, nid)}' for nid in cand_ids)
        rerank_prompt = (
            f"用户问题：{query}\n候选知识概念：\n{cand_lines}\n"
            f'请按「对回答该问题最有教学帮助」从高到低排序，只输出 id 的 JSON 数组，如 ["a","b"]。')
        try:
            raw = rerank_client.complete(rerank_prompt, system="只输出一个 JSON 数组。")
            import json as _json

            ordered = _json.loads(raw.strip().strip("`").replace("'", '"'))
            order_map = {str(nid): i for i, nid in enumerate(ordered)}
            fill.sort(key=lambda t: order_map.get(t[2], len(order_map)))
        except Exception:  # noqa: BLE001
            pass  # 退回掌握度排序

    for _m, _si, nid in fill:
        if len(context) >= max_context:
            break
        if nid in seen:
            continue
        role = "confusable" if nid in set(kg.confusable(nid)) or any(
            nid in set(kg.confusable(s["id"])) for s in seeds) else "related"
        seen[nid] = role
        context.append(_entry(nid, role, kg, learner))

    return RAGResult(
        query=query,
        mode="local",
        seeds=[{**s, "name": kg.node(s["id"]).name if kg.node(s["id"]) else s["id"]}
               for s in seeds],
        context_nodes=context,
        context_text=_render_context(context),
        stats={
            "candidates": len(raw_hits),
            "context_size": len(context),
            "prereq_added": sum(1 for n in context if n["role"] == "prereq"),
        },
    )


# --------------------------------------------------------------------------
# global search
# --------------------------------------------------------------------------


def global_search(query: str, kg: KnowledgeGraph, learner: LearnerModel,
                  index: dict, embedder, top_k: int = 2) -> RAGResult:
    """GraphRAG global search：查询匹配社区摘要，回答领域鸟瞰型问题。"""
    comm_idx: VectorIndex = index["communities"]
    hits = comm_idx.search(query, embedder, k=top_k)
    meta = {c["id"]: c for c in index.get("community_meta", [])}

    context: List[dict] = []
    sections = []
    for cid, sim in hits:
        c = meta.get(cid)
        if not c:
            continue
        members = []
        for nid in c.get("members", []):
            if kg.node(nid):
                members.append(_entry(nid, f"community:{cid}", kg, learner, sim=sim))
        context.extend(members)
        hub_line = "、".join(f"{m['name']}（{m['band']}）"
                             for m in members if m["id"] in set(c.get("hubs", [])))
        sections.append(f"【{c['summary']}】\n核心节点：{hub_line}")

    return RAGResult(query=query, mode="global", context_nodes=context,
                     context_text="\n\n".join(sections),
                     stats={"communities": [c for c, _ in hits]})


# --------------------------------------------------------------------------
# 对比评测：朴素 RAG vs GraphRAG
# --------------------------------------------------------------------------


def compare_rag(kg: KnowledgeGraph, learner: LearnerModel, index: dict, embedder,
                eval_queries: List[dict], k_naive: int = 6,
                **local_kw) -> dict:
    """在同一批查询上对比朴素 RAG 与 GraphRAG。

    指标（均为查询均值）：
      golden_hit        期望节点进入召回前列的比例
      prereq_coverage   期望节点 depth<=2 前置链被上下文覆盖的比例 ——
                        朴素 RAG 的结构性盲区（前置知识文本上不相似）
      mastered_ratio    上下文中已掌握节点的占比 —— 越低说明「讲废话」越少
      context_size      上下文规模（透明化对比口径）
    """
    from kg.paths import prerequisite_graph  # noqa: E402

    import networkx as nx

    # 默认给两种检索相同的节点预算；显式传入 max_context 时尊重调用方。
    local_kw.setdefault("max_context", k_naive)
    dg = prerequisite_graph(kg)
    rows = []
    for item in eval_queries:
        query, golden = item["query"], item["golden"]
        golden = [g for g in golden if g in kg.g]

        # 朴素：纯向量 top-k，无扩展无加权
        naive_ids = [nid for nid, _s in index["nodes"].search(query, embedder, k=k_naive)
                     if nid in kg.g]
        # GraphRAG：种子+扩展+掌握度加权
        graph = local_search(query, kg, learner, index, embedder, **local_kw)
        graph_ids = [n["id"] for n in graph.context_nodes]

        # 期望节点两跳内的全部前置（朴素 RAG 的结构性盲区）
        prereqs: set = set()
        for g in golden:
            if g not in dg:
                continue
            for nid in nx.ancestors(dg, g):
                if nx.shortest_path_length(dg, nid, g) <= 2:
                    prereqs.add(nid)

        def cov(ids: List[str]) -> float:
            if not prereqs:
                return 1.0  # 没有前置链的查询不计入短板
            return len(prereqs & set(ids)) / len(prereqs)

        def mastered_ratio(ids: List[str]) -> float:
            if not ids:
                return 0.0
            return sum(1 for i in ids if learner.mastery_or_zero(i) >= 0.85) / len(ids)

        rows.append({
            "query": query,
            "golden_hit_naive": bool(set(golden) & set(naive_ids)),
            "golden_hit_graph": bool(set(golden) & set(graph_ids)),
            "prereq_cov_naive": cov(naive_ids),
            "prereq_cov_graph": cov(graph_ids),
            "mastered_naive": mastered_ratio(naive_ids),
            "mastered_graph": mastered_ratio(graph_ids),
            "size_naive": len(naive_ids),
            "size_graph": len(graph_ids),
        })

    n = len(rows) or 1
    agg = {
        "queries": len(rows),
        "golden_hit_naive": round(sum(r["golden_hit_naive"] for r in rows) / n, 3),
        "golden_hit_graph": round(sum(r["golden_hit_graph"] for r in rows) / n, 3),
        "prereq_coverage_naive": round(sum(r["prereq_cov_naive"] for r in rows) / n, 3),
        "prereq_coverage_graph": round(sum(r["prereq_cov_graph"] for r in rows) / n, 3),
        "mastered_ratio_naive": round(sum(r["mastered_naive"] for r in rows) / n, 3),
        "mastered_ratio_graph": round(sum(r["mastered_graph"] for r in rows) / n, 3),
        "context_size_naive": round(sum(r["size_naive"] for r in rows) / n, 1),
        "context_size_graph": round(sum(r["size_graph"] for r in rows) / n, 1),
    }
    return {"summary": agg, "per_query": rows}


# --------------------------------------------------------------------------
# 答案侧评测（RAGAS 思想：faithfulness / relevancy，LLM-as-judge）
# --------------------------------------------------------------------------

ANSWER_SYSTEM = "你是 AI 学习教练。基于给定的图谱上下文回答问题：已掌握的概念一句带过，薄弱/未接触的重点讲，先补前置知识再讲目标概念；只引用上下文里的概念。"

JUDGE_PROMPT = """你是严格的评测员。根据【上下文】逐条核对【回答】：
1. faithfulness（忠实度 0~1）：回答中的论断是否都能被上下文支持，编造上下文之外的概念要扣分
2. relevancy（切题度 0~1）：回答是否切中问题要害
只输出 JSON：{{"faithfulness": 0.x, "relevancy": 0.x, "comment": "一句话理由"}}

【上下文】
{context}

【问题】{query}

【回答】{answer}"""


def evaluate_answers(kg: KnowledgeGraph, learner: LearnerModel, index: dict, embedder,
                     client, eval_queries: List[dict], k_naive: int = 6,
                     **local_kw) -> dict:
    """答案侧对比评测（借鉴 RAGAS 的 LLM-as-judge 思路）。

    同一批查询，分别用朴素 RAG 上下文与 GraphRAG 上下文生成回答，再由 LLM 判分：
      - faithfulness / relevancy：judge 打分（0~1）
      - golden 覆盖：期望概念名是否出现在回答文本里（确定性字符串核对，不依赖 judge）
    需要 LLM 客户端；与 --compare（检索侧指标）互补成完整评测故事。
    """
    rows = []
    for item in eval_queries:
        query = item["query"]
        golden_names = [kg.node(g).name for g in item.get("golden", []) if kg.node(g)]

        naive_ids = [nid for nid, _ in index["nodes"].search(query, embedder, k=k_naive)
                     if nid in kg.g]
        naive_ctx = _render_context([_entry(n, "naive", kg, learner) for n in naive_ids])
        graph = local_search(query, kg, learner, index, embedder, **local_kw)

        per_method = {}
        for tag, ctx in (("naive", naive_ctx), ("graph", graph.context_text)):
            if not ctx:
                per_method[tag] = {"answer": "", "faithfulness": 0.0,
                                   "relevancy": 0.0, "golden_covered": 0.0}
                continue
            answer = client.complete(
                f"图谱上下文：\n{ctx}\n\n问题：{query}", system=ANSWER_SYSTEM)
            try:
                judge = parse_llm_json(client.complete(
                    JUDGE_PROMPT.format(context=ctx[:2000], query=query, answer=answer),
                    system="只输出一个 JSON 对象。"))
                faith = max(0.0, min(1.0, float(judge.get("faithfulness", 0))))
                rel = max(0.0, min(1.0, float(judge.get("relevancy", 0))))
            except (ExtractError, TypeError, ValueError):
                faith, rel = 0.0, 0.0
            hits = sum(1 for name in golden_names if name and name in answer)
            per_method[tag] = {
                "answer": answer,
                "faithfulness": round(faith, 3),
                "relevancy": round(rel, 3),
                "golden_covered": round(hits / len(golden_names), 3) if golden_names else 1.0,
            }
        rows.append({"query": query, **{f"{t}_{k}": v for t, r in per_method.items()
                                        for k, v in r.items()}})

    n = len(rows) or 1

    def avg(tag, key):
        return round(sum(r[f"{tag}_{key}"] for r in rows) / n, 3)

    return {
        "summary": {
            "queries": len(rows),
            "faithfulness_naive": avg("naive", "faithfulness"),
            "faithfulness_graph": avg("graph", "faithfulness"),
            "relevancy_naive": avg("naive", "relevancy"),
            "relevancy_graph": avg("graph", "relevancy"),
            "golden_covered_naive": avg("naive", "golden_covered"),
            "golden_covered_graph": avg("graph", "golden_covered"),
        },
        "per_query": rows,
    }
