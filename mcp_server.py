#!/usr/bin/env python
"""MCP Server：把知识图谱 + 学习者画像外挂给 Agent（P5，本项目最核心的能力）。

定位：让 Agent 在给用户讲解 / 布置任务**之前**，先问这张图 ——
「这个人现在会什么、卡在哪、下一步该学什么」。

接入 ZCode / Claude Desktop 等 MCP 客户端（stdio 传输）::

    {
      "mcpServers": {
        "ai-kg": {
          "command": "python",
          "args": ["E:/桌面/tupu/mcp_server.py"]
        }
      }
    }

工具一览：
    get_learner_profile   知识画像：各掌握度分档有哪些概念、平均水平
    check_readiness       就绪检查：做某事/学某点之前，前置是否满足、缺什么
    get_learning_path     学习路径：到目标的教学序步骤 + 缺口标注
    recommend_next        下一步推荐：前置已满足、解锁最多的知识点
    query_node            节点查询：概念详情 + 前置/后继/易混淆邻居
    list_domains          领域概览：各领域规模与平均掌握度
    update_mastery        证据回写：把对话/答题中暴露的掌握证据记回画像

本模块只在 stdout 上跑 MCP 协议，任何日志都必须走 stderr。
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from kg import (  # noqa: E402
    Evidence,
    EvidenceType,
    LearnerModel,
    load_ontology_merged,
    learning_path,
    recommend_next as kg_recommend_next,
)
from kg.paths import PREREQ_SATISFIED
from kg.schema import BAND_COLORS, band_of, compute_mastery
from kg.store import KnowledgeGraph
from kg.embeddings import get_embedder  # noqa: E402
from kg.rag import (global_search, index_is_current, load_or_build_index,
                    local_search)  # noqa: E402

# ------------------------------------------------------------------
# 装载：本体每次启动时由 seed + 机器补丁层合成（SQLite 可能滞后，不作为数据源）
# ------------------------------------------------------------------


def _load_kg() -> KnowledgeGraph:
    onto, _files = load_ontology_merged(
        ROOT / "data" / "seed_ontology.yaml",
        ROOT / "data" / "ontology_generated.yaml",
    )
    return KnowledgeGraph.from_ontology(onto)


KG = _load_kg()
LEARNER_PATH = ROOT / "data" / "learner_state.json"

BAND_LABELS = {
    "untouched": "未接触",
    "weak": "薄弱",
    "learning": "学习中",
    "solid": "较扎实",
    "mastered": "已掌握",
}

mcp = MCPServer(
    "ai-knowledge-graph",
    instructions=(
        "AI 领域知识图谱 + 学习者画像 + GraphRAG 检索服务。"
        "在给用户讲解任何 AI 概念、布置学习任务之前，先调 check_readiness 或 get_learner_profile "
        "感知用户水平：已掌握的别重复讲，薄弱的先补。"
        "回答 AI 学习类问题前先调 search_knowledge 拉取带掌握度标注的检索上下文"
        "（GraphRAG：向量召回 + 图扩展前置知识 + 掌握度加权，上下文里薄弱概念需重点讲）。"
        "update_mastery 用于把对话中暴露的掌握证据（用户讲清了一个概念/答对了题）记回画像。"
    ),
)


def _learner() -> LearnerModel:
    """每次读取本地私人画像；缺失时从空画像开始，避免把演示数据当作用户状态。"""
    return LearnerModel.load(LEARNER_PATH)


def _name(nid: str) -> str:
    node = KG.node(nid)
    return node.name if node else nid


def _resolve(query: str):
    """把用户传入的关键词解析为唯一节点；命中多个且无精确匹配时返回候选。"""
    hits = KG.find(query)
    if not hits:
        return None, []
    q = query.strip().lower()
    for h in hits:
        alias_pool = [a.lower() for a in (h.aliases or [])]
        if h.id.lower() == q or h.name.lower() == q or q in alias_pool:
            return h, hits
    return hits[0], hits


def _short(s: str, limit: int = 80) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ------------------------------------------------------------------
# 工具 1：知识画像
# ------------------------------------------------------------------


@mcp.tool()
def get_learner_profile(domain: str = "") -> dict:
    """获取用户的知识画像：各掌握度分档（已掌握/较扎实/学习中/薄弱/未接触）分别有哪些概念。

    domain 可选，传领域名（如 "大模型工程"）只看该领域；留空看全图。
    Agent 讲解前先调这个，避免把用户已掌握的内容从头讲一遍。
    """
    learner = _learner()
    ids = sorted(KG.g.nodes)
    if domain:
        ids = [n for n in ids if (KG.node(n).domain if KG.node(n) else "") == domain]
    buckets: dict[str, list] = defaultdict(list)
    touched = 0
    total_mastery = 0.0
    for nid in ids:
        m = learner.mastery_or_zero(nid)
        band = band_of(m).value
        buckets[band].append({"id": nid, "name": _name(nid), "mastery": round(m, 3)})
        if m > 0:
            touched += 1
            total_mastery += m
    return {
        "user_id": learner.user_id,
        "scope": domain or "全部",
        "total_concepts": len(ids),
        "touched": touched,
        "average_mastery_of_touched": round(total_mastery / touched, 3) if touched else 0.0,
        "bands": {
            band: {"label": BAND_LABELS[band], "color": BAND_COLORS[band], "concepts": items}
            for band, items in buckets.items()
        },
    }


# ------------------------------------------------------------------
# 工具 2：就绪检查 —— 「学习者感知」的核心
# ------------------------------------------------------------------


@mcp.tool()
def check_readiness(target: str) -> dict:
    """就绪检查：判断用户当前是否具备学习/使用 target 的前置知识。

    返回 verdict（ready / needs_review / not_ready）、缺口清单（含建议补的顺序）与可快速略过的项。
    Agent 在讲解或布置涉及 target 的任务之前，应先调本工具决定从哪里讲起。
    """
    node, hits = _resolve(target)
    if node is None:
        return {"error": f"图中没有匹配 “{target}” 的知识点", "suggestions": []}

    from kg.paths import prerequisite_graph

    import networkx as nx

    dg = prerequisite_graph(KG)
    prereqs = sorted(nx.ancestors(dg, node.id))
    learner = _learner()

    gaps, review, ok = [], [], []
    for pid in prereqs:
        m = learner.mastery_or_zero(pid)
        pnode = KG.node(pid)
        entry = {
            "id": pid,
            "name": _name(pid),
            "mastery": round(m, 3),
            "difficulty": pnode.difficulty if pnode else 3,
            "domain": pnode.domain if pnode else "",
        }
        if m < 0.35:
            entry["action"] = "需要先学"
            gaps.append(entry)
        elif m < PREREQ_SATISFIED:
            entry["action"] = "建议复习"
            review.append(entry)
        else:
            ok.append(entry)

    if gaps:
        verdict = "not_ready"
    elif review:
        verdict = "needs_review"
    else:
        verdict = "ready"

    return {
        "target": node.id,
        "target_name": node.name,
        "verdict": verdict,
        "verdict_meaning": {
            "ready": "前置全部满足，可直接进入目标",
            "needs_review": "前置基本满足，但有薄弱项建议先复习",
            "not_ready": "存在硬缺口，应先补齐 gaps 里的概念",
        }[verdict],
        "prerequisites_total": len(prereqs),
        "gaps": sorted(gaps, key=lambda x: -x["difficulty"]),
        "review": review,
        "satisfied": [e["name"] for e in ok],
        "other_matches": [h.name for h in hits[1:4]] if len(hits) > 1 else [],
    }


# ------------------------------------------------------------------
# 工具 3：学习路径
# ------------------------------------------------------------------


@mcp.tool()
def get_learning_path(target: str, include_known: bool = True) -> dict:
    """计算从用户当前水平到 target 的学习路径（教学序拓扑排序，缺口标注）。

    返回 steps 数组，按建议学习顺序排列，每步含掌握度与状态（gap/learning/known）。
    include_known=False 时跳过已掌握的节点，路径更短。
    """
    node, hits = _resolve(target)
    if node is None:
        return {"error": f"图中没有匹配 “{target}” 的知识点"}
    try:
        result = learning_path(KG, node.id, _learner(), include_known=include_known)
    except ValueError as exc:
        return {"error": str(exc)}
    for step in result["steps"]:
        step["desc"] = _short(step.get("desc", ""))
    result["steps"] = [
        {k: v for k, v in s.items() if k in
         ("order", "id", "name", "mastery", "status", "reason", "difficulty", "domain", "desc")}
        for s in result["steps"]
    ]
    return result


# ------------------------------------------------------------------
# 工具 4：下一步推荐
# ------------------------------------------------------------------


@mcp.tool()
def recommend_next(limit: int = 5) -> dict:
    """推荐用户「现在最该学」的知识点：前置已满足、且解锁后续最多的优先。"""
    picks = kg_recommend_next(KG, _learner(), limit=max(1, min(limit, 20)))
    return {"recommendations": picks}


# ------------------------------------------------------------------
# 工具 5：节点查询
# ------------------------------------------------------------------


@mcp.tool()
def query_node(keyword: str) -> dict:
    """查询知识点的详情与邻居：前置、后继（学会后能解锁什么）、易混淆概念。

    关键词可写 id、名称或别名。命中多个时返回 candidates 供再次精确查询。
    """
    node, hits = _resolve(keyword)
    if node is None:
        return {"error": f"图中没有匹配 “{keyword}” 的知识点"}
    learner = _learner()
    prereqs = KG.prerequisites(node.id, depth=1)
    return {
        "id": node.id,
        "name": node.name,
        "type": node.type.value,
        "domain": node.domain,
        "difficulty": node.difficulty,
        "desc": node.desc,
        "evidence_refs": [ref.model_dump(mode="json") for ref in getattr(node, "evidence_refs", [])],
        "aliases": node.aliases,
        "mastery": round(learner.mastery_or_zero(node.id), 3),
        "band": BAND_LABELS[band_of(learner.mastery_or_zero(node.id)).value],
        "prerequisites": [{"id": p, "name": _name(p), "mastery": round(learner.mastery_or_zero(p), 3)}
                          for p in prereqs],
        "unlocks": [_name(d) for d in KG.dependents(node.id)],
        "confusable_with": [{"id": c, "name": _name(c)} for c in KG.confusable(node.id)],
        "candidates": [h.name for h in hits[:5]] if len(hits) > 1 else [],
    }


# ------------------------------------------------------------------
# 工具 6：领域概览
# ------------------------------------------------------------------


@mcp.tool()
def list_domains() -> dict:
    """列出图谱所有知识领域及各领域的规模、平均掌握度 —— 用于判断该从哪个领域切入。"""
    learner = _learner()
    by_domain: dict[str, list[str]] = defaultdict(list)
    for nid in KG.g.nodes:
        node = KG.node(nid)
        by_domain[node.domain if node else "未分类"].append(nid)
    rows = []
    for dom, ids in sorted(by_domain.items(), key=lambda x: -len(x[1])):
        ms = [learner.mastery_or_zero(n) for n in ids]
        touched = [m for m in ms if m > 0]
        rows.append({
            "domain": dom,
            "concepts": len(ids),
            "average_mastery": round(sum(touched) / len(touched), 3) if touched else 0.0,
            "untouched": sum(1 for m in ms if m == 0),
        })
    return {"domains": rows}


# ------------------------------------------------------------------
# 工具 7：证据回写 —— 闭环的关键
# ------------------------------------------------------------------


@mcp.tool()
def update_mastery(concept_id: str, evidence_type: str = "conversation",
                   desc: str = "", weight: float = 0.3,
                   revoke_last: bool = False) -> dict:
    """给某知识点记录一条掌握证据，并重算掌握度（多证据加权 + 时间衰减）。

    evidence_type 取值: self_report / quiz / project / conversation / code。
    典型场景：用户在对话里讲清楚了一个概念、答对了一道题 —— Agent 据实回写，
    下次 check_readiness 就能看到变化。weight 为该证据的权重 0~1。
    revoke_last=True 时撤销该概念最近一条有效证据（发现用户其实没掌握时用），
    证据保留在轨迹里但不再参与掌握度计算。
    """
    node, hits = _resolve(concept_id)
    if node is None:
        return {"error": f"图中没有匹配 “{concept_id}” 的知识点"}
    try:
        etype = EvidenceType(evidence_type)
    except ValueError:
        return {"error": f"evidence_type 必须是 {[e.value for e in EvidenceType]} 之一"}

    weight = max(0.0, min(1.0, weight))
    learner = _learner()
    st = learner.states.get(node.id)
    if st is not None and not st.evidence and st.mastery > 0:
        # 历史状态只有总分没有明细：把它折算成一条等效证据，避免重算时丢失
        st.evidence.append(Evidence(type=EvidenceType.SELF_REPORT,
                                    desc="历史掌握度折算", weight=st.mastery,
                                    ts=st.last_touched))

    today = date.today().isoformat()
    if revoke_last:
        if st is None or not any(not e.revoked for e in st.evidence):
            return {"error": f"{node.name} 没有可撤销的有效证据"}
        for ev in reversed(st.evidence):
            if not ev.revoked:
                ev.revoked = True
                revoked_desc = ev.desc
                break
        st.mastery = compute_mastery(st.evidence, st.last_touched)
        learner.states[node.id] = st
        learner.save(LEARNER_PATH)
        return {"concept": node.name, "revoked": revoked_desc,
                "new_mastery": round(st.mastery, 4),
                "band": BAND_LABELS[band_of(st.mastery).value],
                "evidence_count": len(st.evidence),
                "note": "画像已持久化；被撤销证据保留在轨迹里不再生效"}

    ev = Evidence(type=etype, desc=desc or f"由 Agent 记录（{evidence_type}）",
                  weight=weight, ts=today)
    st = learner.mark(node.id, evidence=ev, source=evidence_type, ts=today)
    learner.save(LEARNER_PATH)

    return {
        "concept": node.name,
        "new_mastery": round(st.mastery, 4),
        "band": BAND_LABELS[band_of(st.mastery).value],
        "evidence_count": len(st.evidence),
        "note": "画像已持久化，可视化页面重新构建后可见",
    }


# ------------------------------------------------------------------
# 工具 8：GraphRAG 检索 —— 学习者感知的检索增强
# ------------------------------------------------------------------


def _load_rag_index(learner: LearnerModel):
    """加载（或首次构建）向量索引；嵌入器按配置选 API 语义向量或哈希降级。"""
    global _RAG_CACHE
    try:
        from kg.llm import load_config

        cfg = load_config()
    except Exception:  # noqa: BLE001  # 无 LLM 配置也能检索（哈希降级）
        cfg = None
    embedder = get_embedder(cfg)
    if _RAG_CACHE is not None and index_is_current(_RAG_CACHE[0], KG, learner, embedder):
        return _RAG_CACHE
    idx_path = ROOT / "data" / "vector_index.json"
    index, _built = load_or_build_index(KG, learner, embedder, idx_path)
    _RAG_CACHE = (index, embedder)
    return _RAG_CACHE


_RAG_CACHE = None


@mcp.tool()
def search_knowledge(query: str, mode: str = "local", max_context: int = 12) -> dict:
    """GraphRAG 学习者感知检索：拉取回答 AI 学习问题所需的图谱上下文。

    与朴素向量检索的区别：向量只找种子节点，随后沿图边扩展前置知识与易混淆概念
    （朴素 RAG 永远召不回文本不相似但教学必需的前置），扩展填充按掌握度加权
    （薄弱子图优先）。返回的 context_text 已按角色分节、逐节点标注掌握度，
    可直接作为讲解依据：已掌握(0.85+)一句带过，薄弱/未接触重点讲。
    mode=global 返回社区摘要粒度（适合「某领域整体有什么」的鸟瞰问题）。
    """
    learner = _learner()
    index, embedder = _load_rag_index(learner)
    max_context = max(3, min(max_context, 25))
    if mode == "global":
        r = global_search(query, KG, learner, index, embedder)
    else:
        r = local_search(query, KG, learner, index, embedder, max_context=max_context)
    return {
        "query": query,
        "mode": r.mode,
        "seeds": r.seeds,
        "context": r.context_nodes,
        "context_text": r.context_text,
        "stats": r.stats,
    }


def main() -> None:
    mcp.run()  # stdio 传输


if __name__ == "__main__":
    main()
