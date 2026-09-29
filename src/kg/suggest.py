"""前置关系自动补全（P2 的落地）：找图谱缺口 -> LLM 建议前置边 -> 人工审核 -> 走 ingest。

确定性缺口检测（不依赖 LLM）：
  - 候选 = 前置 DAG 中**没有前置**的节点，按「解锁后续数量」降序 ——
    一个被很多节点依赖的概念却没有前置，要么是真起点，要么是漏了边；
  - 再用「描述共现」信号辅助：候选的 desc 里提到了图中其他节点的名称，
    提示 LLM 优先检查这些配对。

LLM 只做建议，入库仍走 ingest 的全套防线（schema 校验、成环拒绝、补丁层），
**产出到 data/inbox/ 由人工过目后手动入库** —— 自动编排不等于自动生效。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from .extract import parse_llm_json
from .ingest import Ontology
from .llm import LLMClient
from .paths import prerequisite_graph
from .schema import RelationType

SUGGEST_SYSTEM = "你是知识图谱编排专家。只输出一个 JSON 对象，不要任何解释。"

SUGGEST_PROMPT = """下面是一张 AI 知识图谱的候选节点，它们在「前置知识」关系（prerequisite_of）上
是空的（没有任何前置）。判断其中哪些**确实缺失了前置关系**，给出补全建议。

## 全部已有节点（source/target 只能用这些 id）
{nodes}

## 候选节点（无前置，按被依赖程度排序）
{candidates}

## 规则
1. 只有「不懂它就很难懂候选节点」的关系才算 prerequisite_of，方向：基础 -> 进阶
2. 数学/基础概念（如线性代数）没有前置是正常的，不要硬造
3. 最多 {max_edges} 条，每条给出置信度和一句话理由
4. 只输出 JSON：{{"edges": [{{"source": "...", "target": "...", "confidence": 0到1, "reason": "..."}}]}}
   没有建议就输出 {{"edges": []}}"""


def find_candidates(onto: Ontology, limit: int = 12) -> List[dict]:
    """无前置的节点按解锁数排序；desc 里提到的其他节点作为共现提示。"""
    onto_tmp = onto
    from .ingest import KnowledgeGraph  # 局部导入避免环

    kg = KnowledgeGraph.from_ontology(onto_tmp)
    dg = prerequisite_graph(kg)
    name_to_id = {n.name: n.id for n in onto.nodes}
    out = []
    for nid in sorted(dg.nodes):
        if list(dg.predecessors(nid)):
            continue  # 有前置，不是缺口
        node = kg.node(nid)
        mentioned = [name_to_id[name] for name in name_to_id
                     if name != node.name and name in (node.desc or "")]
        out.append({
            "id": nid,
            "name": node.name,
            "domain": node.domain,
            "desc": node.desc,
            "unlock_count": len(kg.dependents(nid)),
            "mentioned_ids": mentioned[:5],
        })
    out.sort(key=lambda x: -x["unlock_count"])
    return out[:limit]


def suggest_prerequisites(onto: Ontology, client: LLMClient, limit: int = 12,
                          max_edges: int = 10) -> Tuple[dict, dict]:
    """返回 (可直接交给 ingest 的 payload, 统计)。payload 含 nodes:[] + 建议边。"""
    candidates = find_candidates(onto, limit=limit)
    node_lines = "\n".join(f"- {n.id} | {n.name} | {n.domain}" for n in onto.nodes)
    cand_lines = "\n".join(
        f"- {c['id']}（{c['name']}，解锁 {c['unlock_count']} 个后续）"
        + (f" 描述里提到: {', '.join(c['mentioned_ids'])}" if c["mentioned_ids"] else "")
        for c in candidates)
    prompt = SUGGEST_PROMPT.format(nodes=node_lines, candidates=cand_lines, max_edges=max_edges)
    raw = client.complete(prompt, system=SUGGEST_SYSTEM)
    obj = parse_llm_json(raw)
    edges = [e for e in obj.get("edges", []) if isinstance(e, dict)]
    for e in edges:
        e.setdefault("type", "prerequisite_of")
        e.setdefault("confidence", 0.6)
        if e.get("reason"):
            e["note"] = f"LLM 建议: {e['reason']}"
    payload = {"source": "llm_prerequisite_suggestion", "nodes": [], "relations": edges}
    stats = {"candidates": len(candidates), "suggested": len(edges)}
    return payload, stats
