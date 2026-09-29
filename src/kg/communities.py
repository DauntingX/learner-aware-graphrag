"""社区检测与社区摘要（GraphRAG「全局检索」的基础）。

GraphRAG 的标志性设计是双粒度检索：
  - local search：查询 -> 具体节点及邻居（rag.py）
  - global search：查询 -> 社区摘要（本模块），回答「这个领域整体有什么」类问题

社区用 Louvain（对无向投影图，全关系类型参与），摘要 V1 用确定性抽取式生成：
领域构成 + 按度中心性排序的核心节点 + 学习者画像概览。
不依赖 LLM，可离线测试；后续 P2 可换成 LLM 生成式摘要。
"""

from __future__ import annotations

from typing import Dict, List

import networkx as nx

from .learner import LearnerModel
from .schema import MasteryBand, band_of
from .store import KnowledgeGraph

BAND_LABELS = {
    MasteryBand.UNTOUCHED.value: "未接触",
    MasteryBand.WEAK.value: "薄弱",
    MasteryBand.LEARNING.value: "学习中",
    MasteryBand.SOLID.value: "较扎实",
    MasteryBand.MASTERED.value: "已掌握",
}


def undirected_projection(kg: KnowledgeGraph) -> nx.Graph:
    """全关系类型的无向加权投影：同对节点多重关系权重叠加。"""
    G = nx.Graph()
    G.add_nodes_from(kg.g.nodes)
    for s, t, d in kg.g.edges(data=True):
        w = float(d.get("confidence", 1.0))
        if G.has_edge(s, t):
            G[s][t]["weight"] += w
        else:
            G.add_edge(s, t, weight=w)
    return G


def detect_communities(kg: KnowledgeGraph, seed: int = 42) -> List[List[str]]:
    """Louvain 社区划分，返回按节点数降序的社区列表。孤立节点单独成社区。"""
    G = undirected_projection(kg)
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    return sorted([sorted(c) for c in comms], key=len, reverse=True)


def summarize_community(kg: KnowledgeGraph, learner: LearnerModel, members: List[str],
                        community_id: int) -> dict:
    """抽取式社区摘要：领域构成 + 核心节点（度中心性 Top5）+ 画像概览。"""
    sub = kg.g.subgraph(members)
    und = undirected_projection(kg).subgraph(members)
    centrality = nx.degree_centrality(und)
    hubs = sorted(members, key=lambda n: -centrality.get(n, 0))[:5]

    domain_counts: Dict[str, int] = {}
    for n in members:
        node = kg.node(n)
        dom = node.domain if node else "未分类"
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
    domains = sorted(domain_counts.items(), key=lambda x: -x[1])

    masteries = [learner.mastery_or_zero(n) for n in members]
    touched = [m for m in masteries if m > 0]
    bands: Dict[str, int] = {}
    for n in members:
        b = band_of(learner.mastery_or_zero(n)).value
        bands[b] = bands.get(b, 0) + 1

    summary = (
        f"社区 #{community_id}：{len(members)} 个知识点，"
        f"主要领域 {'、'.join(f'{d}({c})' for d, c in domains[:3])}。"
        f"核心节点：{'、'.join(kg.node(h).name if kg.node(h) else h for h in hubs)}。"
        f"学习者画像：{BAND_LABELS[band_of(sum(masteries) / len(masteries)).value]}"
        f"（平均掌握度 {sum(touched) / len(touched):.2f}，未接触 {bands.get('untouched', 0)} 个）"
        if touched else
        f"社区 #{community_id}：{len(members)} 个知识点，主要领域 "
        f"{'、'.join(f'{d}({c})' for d, c in domains[:3])}。核心节点："
        f"{'、'.join(kg.node(h).name if kg.node(h) else h for h in hubs)}。学习者画像：全部未接触。"
    )
    return {
        "id": f"community_{community_id}",
        "size": len(members),
        "members": members,
        "domains": [d for d, _c in domains],
        "hubs": hubs,
        "summary": summary,
        "avg_mastery": round(sum(touched) / len(touched), 3) if touched else 0.0,
    }


def all_communities(kg: KnowledgeGraph, learner: LearnerModel) -> List[dict]:
    return [summarize_community(kg, learner, members, i)
            for i, members in enumerate(detect_communities(kg))]
