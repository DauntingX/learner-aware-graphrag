"""学习路径规划。

这是"自动编排"的落地：基于 prerequisite_of 构成的 DAG，
用拓扑排序把"从当前水平到目标"的学习顺序算出来，并标注缺口。

也是前端"学习路径动画"的数据来源。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx

from .learner import GAP_THRESHOLD, MASTERED_THRESHOLD, LearnerModel
from .schema import BAND_COLORS, RelationType, band_of
from .store import KnowledgeGraph

#: 前置知识"算掌握了"的判定阈值（比自己的 MASTERED 松一些，
#: 因为前置只需够用，不必精通）
PREREQ_SATISFIED = 0.5


def prerequisite_graph(kg: KnowledgeGraph) -> nx.DiGraph:
    """抽出只含 prerequisite_of 边的有向图。"""
    dg = nx.DiGraph()
    dg.add_nodes_from(kg.g.nodes)
    for s, t, d in kg.g.edges(data=True):
        if d.get("type") == RelationType.PREREQUISITE_OF.value:
            dg.add_edge(s, t)
    return dg


def _status_of(learner: LearnerModel, node_id: str) -> str:
    m = learner.mastery_or_zero(node_id)
    if m >= MASTERED_THRESHOLD:
        return "known"
    if m >= GAP_THRESHOLD:
        return "learning"
    return "gap"


_STATUS_REASON = {
    "gap": "前置缺口，建议优先补上",
    "learning": "已有基础，复习巩固即可",
    "known": "已掌握，可快速跳过",
}


def _pedagogical_order(sub: nx.DiGraph, depth: Dict[str, int], kg: KnowledgeGraph) -> List[str]:
    """带教学优先级的拓扑排序。

    纯拓扑排序只保证"前置在前"，但同一层的顺序是随意的 ——
    会出现"向量数据库"排在"线性代数"前面这种合法但不合理的结果。

    这里用 Kahn 算法 + 优先队列，同层内按两条规则出队：
      1. depth 大的优先（离目标越远越基础，先学）
      2. depth 相同时难度低的优先
    """
    import heapq

    def difficulty(nid: str) -> int:
        node = kg.node(nid)
        return node.difficulty if node else 3

    indeg = {n: sub.in_degree(n) for n in sub.nodes}
    heap = [(-depth.get(n, 0), difficulty(n), n) for n in sub.nodes if indeg[n] == 0]
    heapq.heapify(heap)

    out: List[str] = []
    while heap:
        _neg_depth, _diff, nid = heapq.heappop(heap)
        out.append(nid)
        for succ in sub.successors(nid):
            indeg[succ] -= 1
            if indeg[succ] == 0:
                heapq.heappush(heap, (-depth.get(succ, 0), difficulty(succ), succ))

    if len(out) != len(sub.nodes):  # pragma: no cover - DAG 已提前校验
        raise ValueError("拓扑排序未能覆盖全部节点，可能存在环")
    return out


def learning_path(
    kg: KnowledgeGraph,
    target: str,
    learner: Optional[LearnerModel] = None,
    include_known: bool = True,
) -> dict:
    """计算到达 target 的学习路径。

    返回结构直接给前端做动画消费：steps 是按建议学习顺序排列的序列。
    """
    if target not in kg.g:
        raise KeyError(f"图中不存在节点: {target}")
    learner = learner or LearnerModel()
    dg = prerequisite_graph(kg)

    if not nx.is_directed_acyclic_graph(dg):
        cycles = list(nx.simple_cycles(dg))
        raise ValueError(f"prerequisite_of 存在环路，无法拓扑排序: {cycles[:3]}")

    scope = nx.ancestors(dg, target) | {target}
    sub = dg.subgraph(scope)

    # depth：该知识点距离目标还有几跳（0 = 目标本身）。越大说明越基础。
    depth: Dict[str, int] = {}
    for n in sub.nodes:
        try:
            depth[n] = nx.shortest_path_length(sub, n, target)
        except nx.NetworkXNoPath:  # pragma: no cover - sub 内理论上必可达
            depth[n] = -1

    ordered = _pedagogical_order(sub, depth, kg)

    steps: List[dict] = []
    for nid in ordered:
        status = _status_of(learner, nid)
        if not include_known and status == "known":
            continue
        node = kg.node(nid)
        mastery = learner.mastery_or_zero(nid)
        band = band_of(mastery).value
        steps.append(
            {
                "order": len(steps),  # 从 0 开始，前端动画按此索引播放
                "id": nid,
                "name": node.name if node else nid,
                "type": node.type.value if node else "",
                "domain": node.domain if node else "",
                "difficulty": node.difficulty if node else 3,
                "desc": node.desc if node else "",
                "depth": depth.get(nid, -1),
                "mastery": round(mastery, 4),
                "band": band,
                "color": BAND_COLORS[band],
                "status": status,
                "reason": _STATUS_REASON[status],
            }
        )

    gaps = [s["name"] for s in steps if s["status"] == "gap"]
    target_node = kg.node(target)
    return {
        "target": target,
        "target_name": target_node.name if target_node else target,
        "total": len(steps),
        "gap_count": len(gaps),
        "known_count": sum(1 for s in steps if s["status"] == "known"),
        "gaps": gaps,
        "steps": steps,
    }


def recommend_next(
    kg: KnowledgeGraph,
    learner: Optional[LearnerModel] = None,
    limit: int = 5,
) -> List[dict]:
    """推荐"现在最该学"的知识点。

    规则：前置已满足（都 >= PREREQ_SATISFIED）但自己还没掌握，按难度升序。
    """
    learner = learner or LearnerModel()
    dg = prerequisite_graph(kg)
    picks: List[dict] = []
    for nid in kg.g.nodes:
        if learner.is_mastered(nid):
            continue
        preds = list(dg.predecessors(nid))
        if all(learner.mastery_or_zero(p) >= PREREQ_SATISFIED for p in preds):
            node = kg.node(nid)
            picks.append(
                {
                    "id": nid,
                    "name": node.name if node else nid,
                    "domain": node.domain if node else "",
                    "difficulty": node.difficulty if node else 3,
                    "mastery": round(learner.mastery_or_zero(nid), 4),
                    "unlocks": len(kg.dependents(nid)),
                }
            )
    picks.sort(key=lambda x: (-x["unlocks"], x["difficulty"]))
    return picks[:limit]
