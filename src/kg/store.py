"""图存储层。

轻量档实现：内存里用 NetworkX 存图，磁盘上用 SQLite 持久化。
后续迁移 Neo4j 时，只需要换掉这个类的实现，上层接口保持不变
（这就是为什么要有这一层抽象）。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import networkx as nx
import yaml

from .schema import (
    BAND_COLORS,
    RELATION_STYLES,
    MasteryBand,
    Node,
    NodeType,
    Ontology,
    Relation,
    RelationType,
    band_of,
)


class KnowledgeGraph:
    """知识图谱的读写门面。

    内部约定：NetworkX 有向图，节点属性 = Node 模型字段，边属性 = Relation 模型字段。
    """

    def __init__(self) -> None:
        # 用 MultiDiGraph 而不是 DiGraph：同一对节点之间可能同时存在多种关系。
        # 例如 rnn -> lstm 既是 prerequisite_of（先学 RNN）又是 confusable_with（易混）。
        # 普通 DiGraph 一对节点只能有一条边，后加入的会**静默覆盖**前面的，
        # 造成关系丢失 —— 这是建图时非常隐蔽的陷阱，踩过一次就该记下来。
        self.g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    @classmethod
    def from_ontology(cls, onto: Ontology) -> "KnowledgeGraph":
        kg = cls()
        for n in onto.nodes:
            kg.g.add_node(n.id, **n.model_dump(mode="json"))
        for r in onto.relations:
            src, dst = r.source, r.target
            if src not in kg.g:
                raise ValueError(f"关系引用了不存在的节点: {src}")
            if dst not in kg.g:
                raise ValueError(f"关系引用了不存在的节点: {dst}")
            # key 用关系类型：同一对节点可以挂多条不同类型的边且互不覆盖
            kg.g.add_edge(src, dst, key=r.type.value, **r.model_dump(mode="json"))
        return kg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KnowledgeGraph":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_ontology(Ontology(**raw))

    # ------------------------------------------------------------------
    # 基本查询
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.g.number_of_nodes()

    def node(self, node_id: str) -> Optional[Node]:
        if node_id not in self.g:
            return None
        data = dict(self.g.nodes[node_id])
        data.pop("id", None)
        return Node(id=node_id, **data)

    def find(self, keyword: str) -> List[Node]:
        """按 id / 名称 / 别名模糊查找节点。"""
        kw = keyword.strip().lower()
        hits: List[Node] = []
        for nid, data in self.g.nodes(data=True):
            pool = [nid, data.get("name", "")] + list(data.get("aliases", []) or [])
            if any(kw in str(x).lower() for x in pool):
                hits.append(self.node(nid))  # type: ignore[arg-type]
        return hits

    def edges_of(self, node_id: str, rel_type: Optional[RelationType] = None) -> List[dict]:
        out = []
        for s, t, d in self.g.out_edges(node_id, data=True):
            if rel_type is None or d.get("type") == rel_type.value:
                out.append({"source": s, "target": t, **d})
        for s, t, d in self.g.in_edges(node_id, data=True):
            if rel_type is None or d.get("type") == rel_type.value:
                out.append({"source": s, "target": t, **d})
        return out

    def prerequisites(self, node_id: str, depth: int = 2) -> List[str]:
        """返回 node_id 的祖先节点（沿 prerequisite_of 反向遍历）。

        语义：要学 node_id，需要先掌握哪些知识。
        prerequisite_of 的方向是 A -> B（A 是 B 的前置），
        所以"B 的前置" = 沿入边反向走。
        """
        if node_id not in self.g:
            return []
        seen: set[str] = set()
        frontier = [node_id]
        for _ in range(depth):
            nxt: List[str] = []
            for cur in frontier:
                for s, _t, d in self.g.in_edges(cur, data=True):
                    if d.get("type") == RelationType.PREREQUISITE_OF.value and s not in seen:
                        seen.add(s)
                        nxt.append(s)
            frontier = nxt
            if not frontier:
                break
        return sorted(seen)

    def dependents(self, node_id: str) -> List[str]:
        """返回 node_id 的直接后继（学会它之后能学什么）。"""
        if node_id not in self.g:
            return []
        return sorted(
            t
            for _s, t, d in self.g.out_edges(node_id, data=True)
            if d.get("type") == RelationType.PREREQUISITE_OF.value
        )

    def confusable(self, node_id: str) -> List[str]:
        """返回与 node_id 易混淆的概念。"""
        out = set()
        for s, t, d in self.g.edges(node_id, data=True):
            if d.get("type") == RelationType.CONFUSABLE_WITH.value:
                out.add(t if s == node_id else s)
        for s, t, d in self.g.in_edges(node_id, data=True):
            if d.get("type") == RelationType.CONFUSABLE_WITH.value:
                out.add(s if t == node_id else t)
        return sorted(out)

    # ------------------------------------------------------------------
    # 统计与自检
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        from collections import Counter

        type_counter = Counter(d.get("type") for _n, d in self.g.nodes(data=True))
        edge_counter = Counter(d.get("type") for _s, _t, d in self.g.edges(data=True))
        domains = Counter(d.get("domain") for _n, d in self.g.nodes(data=True))
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "node_types": dict(type_counter),
            "relation_types": dict(edge_counter),
            "domains": dict(domains),
            "roots": self.roots(),
            "isolated": [n for n in self.g.nodes if self.g.degree(n) == 0],
        }

    def roots(self) -> List[str]:
        """没有任何前置知识的节点 —— 学习的天然起点。"""
        return sorted(
            n
            for n in self.g.nodes
            if not any(
                d.get("type") == RelationType.PREREQUISITE_OF.value
                for _s, _t, d in self.g.in_edges(n, data=True)
            )
        )

    def check_dag(self) -> List[List[str]]:
        """检查 prerequisite_of 子图是否有环。返回所有环路列表（空列表 = 健康）。

        这是"自动编排"阶段必须做的自检 —— 前置关系必须构成 DAG，
        否则学习路径无法拓扑排序。
        """
        dg = nx.DiGraph()
        dg.add_nodes_from(self.g.nodes)
        for s, t, d in self.g.edges(data=True):
            if d.get("type") == RelationType.PREREQUISITE_OF.value:
                dg.add_edge(s, t)
        return list(nx.simple_cycles(dg))

    def multi_relation_pairs(self) -> Dict[str, List[str]]:
        """找出"一对节点之间存在多种关系"的组合。

        这些正是 DiGraph 会丢失、MultiDiGraph 才保得住的关系。
        自检时打印出来，方便确认建图正确。
        """
        from collections import defaultdict

        pairs: Dict[str, List[str]] = defaultdict(list)
        for s, t, d in self.g.edges(data=True):
            pairs[f"{s} -> {t}"].append(d.get("type", ""))
        return {k: sorted(v) for k, v in pairs.items() if len(v) > 1}

    def leaves(self) -> List[str]:
        """出度为 0 的知识点（学到这里算阶段性终点）。"""
        return sorted(
            n
            for n in self.g.nodes
            if not any(
                d.get("type") == RelationType.PREREQUISITE_OF.value
                for _s, _t, d in self.g.out_edges(n, data=True)
            )
        )

    # ------------------------------------------------------------------
    # 持久化（SQLite）
    # ------------------------------------------------------------------

    def save_sqlite(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        try:
            con.executescript(
                """
                DROP TABLE IF EXISTS nodes;
                DROP TABLE IF EXISTS edges;
                CREATE TABLE nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT, type TEXT, domain TEXT,
                    descr TEXT, difficulty INTEGER, aliases TEXT,
                    evidence_refs TEXT
                );
                CREATE TABLE edges (
                    source TEXT, target TEXT, type TEXT,
                    confidence REAL, note TEXT, evidence_refs TEXT,
                    PRIMARY KEY (source, target, type)
                );
                """
            )
            con.executemany(
                "INSERT INTO nodes VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        n,
                        d.get("name"),
                        d.get("type"),
                        d.get("domain"),
                        d.get("desc"),
                        d.get("difficulty"),
                        json.dumps(d.get("aliases") or [], ensure_ascii=False),
                        json.dumps(d.get("evidence_refs") or [], ensure_ascii=False),
                    )
                    for n, d in self.g.nodes(data=True)
                ],
            )
            con.executemany(
                "INSERT OR REPLACE INTO edges VALUES (?,?,?,?,?,?)",
                [
                    (s, t, d.get("type"), d.get("confidence", 1.0), d.get("note", ""),
                     json.dumps(d.get("evidence_refs") or [], ensure_ascii=False))
                    for s, t, d in self.g.edges(data=True)
                ],
            )
            con.commit()
        finally:
            con.close()

    @classmethod
    def from_sqlite(cls, path: str | Path) -> "KnowledgeGraph":
        con = sqlite3.connect(path)
        try:
            kg = cls()
            node_columns = {row[1] for row in con.execute("PRAGMA table_info(nodes)")}
            edge_columns = {row[1] for row in con.execute("PRAGMA table_info(edges)")}
            for row in con.execute("SELECT * FROM nodes"):
                if "evidence_refs" in node_columns:
                    nid, name, ntype, domain, descr, diff, aliases, refs = row
                else:  # 兼容旧的、可重建的 SQLite 导出文件
                    nid, name, ntype, domain, descr, diff, aliases = row
                    refs = "[]"
                kg.g.add_node(
                    nid,
                    name=name,
                    type=ntype,
                    domain=domain,
                    desc=descr,
                    difficulty=diff,
                    aliases=json.loads(aliases or "[]"),
                    evidence_refs=json.loads(refs or "[]"),
                )
            for row in con.execute("SELECT * FROM edges"):
                if "evidence_refs" in edge_columns:
                    s, t, rtype, conf, note, refs = row
                else:
                    s, t, rtype, conf, note = row
                    refs = "[]"
                kg.g.add_edge(s, t, key=rtype, type=rtype, confidence=conf,
                              note=note, evidence_refs=json.loads(refs or "[]"))
            return kg
        finally:
            con.close()

    # ------------------------------------------------------------------
    # 导出给前端
    # ------------------------------------------------------------------

    def to_payload(self, learner_getter=None) -> dict:
        """把图导出为 Web 前端友好的 JSON。

        learner_getter: (node_id) -> mastery 或 None。
        传入后每个节点会带上 mastery / band / color 三个可视化字段。
        """
        nodes = []
        for nid, d in self.g.nodes(data=True):
            mastery = 0.0
            if learner_getter is not None:
                m = learner_getter(nid)
                mastery = float(m) if m is not None else 0.0
            band = band_of(mastery).value
            nodes.append(
                {
                    "id": nid,
                    "name": d.get("name", nid),
                    "type": d.get("type"),
                    "domain": d.get("domain"),
                    "desc": d.get("desc", ""),
                    "difficulty": d.get("difficulty", 3),
                    "aliases": d.get("aliases") or [],
                    "mastery": round(mastery, 4),
                    "band": band,
                    "color": BAND_COLORS[band],
                }
            )
        links = []
        for s, t, d in self.g.edges(data=True):
            rtype = d.get("type")
            style = RELATION_STYLES.get(rtype, {"color": "#B4B2A9", "dash": "", "width": 1.0})
            links.append(
                {
                    "id": f"{s}__{t}__{rtype}",
                    "source": s,
                    "target": t,
                    "type": rtype,
                    "note": d.get("note", ""),
                    "color": style["color"],
                    "dash": style["dash"],
                    "width": style["width"],
                }
            )
        return {"nodes": nodes, "links": links}
