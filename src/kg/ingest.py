"""增量入库层（P1 的落库半边，LLM 无关）。

抽取管线（LLM 结构化抽取）的产物按下面的契约交给本模块，
由它完成 消歧 -> 校验 -> 合并 -> 自检 -> 回写。

输入契约（extractor 的输出格式）::

    {
      "source": "notes/transformer.md 可选溯源",
      "nodes": [
        {"id": "rope", "name": "旋转位置编码", "type": "Technique",
         "domain": "大模型工程", "difficulty": 4, "desc": "...", "aliases": ["RoPE"]}
      ],
      "relations": [
        {"source": "rope", "target": "transformer", "type": "applied_in", "confidence": 0.9}
      ]
    }

source / target 既可以写已有节点的 id，也可以写名称或别名 —— 消歧顺序：
  id 精确 > 名称精确 > 别名精确 > difflib 模糊（阈值 0.88，唯一最佳才命中）> 新建实体

三条硬规则：
  1. 新的 prerequisite_of 边不允许制造环路（会破坏拓扑排序），违规边单独拒绝而非整批失败；
  2. 自环、悬空端点、schema 校验失败一律进 rejected 报告，绝不静默丢弃；
  3. 人工维护的 seed 本体只读。增量与字段级补丁写入机器层 ontology_generated.yaml，
     装载时 seed 在下、机器层在上按字段叠加 —— 用户后续改 seed 的字段不受机器层旧值影响。

掌握这层的接口：
    merge_payload(onto, payload)      纯函数：合并 + 报告，不落盘
    load_ontology_merged(*paths)      两层叠加装载（build_graph.py 用）
    ingest_file(payload, data_dir)    文件级入口：读 -> 合并 -> 自检 -> 原子回写
"""

from __future__ import annotations

import difflib
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import yaml
from pydantic import ValidationError

from .schema import Node, Ontology, Relation, RelationType, SourceRef
from .store import KnowledgeGraph

#: 模糊匹配阈值。太低会把"大模型"和"大语言模型"错并成同一实体。
FUZZY_CUTOFF = 0.88

#: 嵌入对齐阈值（opt-in）：名称向量余弦超过它且唯一最佳才判为同一实体。
#: 借鉴 neo4j llm-graph-builder 的 embedding 实体对齐，语义上比编辑距离更准；
#: 保守取高阈值 + 要求唯一，宁可不并也不错并。
EMB_ALIGN_CUTOFF = 0.92
EMB_ALIGN_MARGIN = 0.02

#: 机器增量层的默认文件名
GENERATED_NAME = "ontology_generated.yaml"

#: seed 层的默认文件名
SEED_NAME = "seed_ontology.yaml"


# --------------------------------------------------------------------------
# 消歧索引
# --------------------------------------------------------------------------


def _norm(s: Any) -> str:
    """匹配用归一化：小写 + 去多余空白。"""
    return re.sub(r"\s+", " ", str(s).strip().lower())


def slugify(name: str) -> str:
    """从名称生成可作节点 id 的 slug；纯中文等无法转写时退回散列。"""
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not s:
        s = "x" + hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    return s


class EntityIndex:
    """现有本体 + 本批次已接收节点的统一查找索引。

    embedder 可选：提供时模糊匹配升级为名称向量余弦（语义对齐），
    未提供则退回 difflib 编辑距离。
    """

    def __init__(self, onto: Ontology, embedder=None) -> None:
        self.by_id: Dict[str, Node] = {}
        self.by_name: Dict[str, str] = {}
        self.by_alias: Dict[str, str] = {}
        self._fuzzy_pool: List[Tuple[str, str]] = []  # (归一化文本, node_id)
        self._embedder = embedder
        self._emb_texts: List[str] = []  # 与 _fuzzy_pool 对齐的原始文本（参与嵌入）
        for n in onto.nodes:
            self._register(n)

    def _register(self, n: Node) -> None:
        self.by_id[n.id] = n
        self.by_name[_norm(n.name)] = n.id
        for a in n.aliases or []:
            self.by_alias[_norm(a)] = n.id
        self._fuzzy_pool.append((_norm(n.name), n.id))
        self._emb_texts.append(n.name)
        for a in n.aliases or []:
            self._fuzzy_pool.append((_norm(a), n.id))
            self._emb_texts.append(a)
        self._emb_matrix = None  # 惰性构建：有查询时才算一次

    def add(self, n: Node) -> None:
        """把（可能新建的）节点纳入索引，供同批次后续关系引用。"""
        self._register(n)

    def _embedding_matrix(self):
        if self._embedder is None:
            return None
        if self._emb_matrix is None or self._emb_matrix.shape[0] != len(self._emb_texts):
            import numpy as np

            self._emb_matrix = self._embedder.embed_texts(self._emb_texts)
        return self._emb_matrix

    def _embedding_align(self, mention: str) -> Optional[str]:
        """名称向量余弦对齐：唯一且最佳显著领先才命中。"""
        import numpy as np

        mat = self._embedding_matrix()
        if mat is None or mat.shape[0] == 0:
            return None
        qv = self._embedder.embed_texts([mention])[0]
        sims = mat @ qv
        order = np.argsort(-sims)[:2]
        if len(order) == 0 or sims[order[0]] < EMB_ALIGN_CUTOFF:
            return None
        if len(order) > 1 and sims[order[0]] - sims[order[1]] < EMB_ALIGN_MARGIN:
            return None  # 并列歧义，宁可新建
        return self._fuzzy_pool[int(order[0])][1]

    def resolve(self, mention: str, proposed_id: str = "") -> Tuple[Optional[str], str]:
        """把一个实体指称解析为节点 id。返回 (node_id 或 None, 命中方式)。

        命中方式取值：id / name / alias / fuzzy / none。
        fuzzy 优先用嵌入对齐（提供 embedder 时），否则 difflib。
        """
        mention = str(mention).strip()
        if not mention:
            return None, "none"
        if proposed_id and proposed_id in self.by_id:
            return proposed_id, "id"
        if mention in self.by_id:
            return mention, "id"
        nid = self.by_name.get(_norm(mention))
        if nid:
            return nid, "name"
        nid = self.by_alias.get(_norm(mention))
        if nid:
            return nid, "alias"
        if self._embedder is not None:
            nid = self._embedding_align(mention)
            if nid:
                return nid, "fuzzy"
        # difflib 兜底：要求唯一最佳命中，多个并列视为歧义，宁可新建也不猜
        matches = difflib.get_close_matches(
            _norm(mention), [t for t, _ in self._fuzzy_pool], n=2, cutoff=FUZZY_CUTOFF
        )
        if len(matches) == 1:
            for text, nid in self._fuzzy_pool:
                if text == matches[0]:
                    return nid, "fuzzy"
        return None, "none"


# --------------------------------------------------------------------------
# 合并
# --------------------------------------------------------------------------


class IngestReport:
    """一次入库的结构化报告 —— 抽取管线的质量反馈就靠它。"""

    def __init__(self) -> None:
        self.added_nodes: List[str] = []
        self.updated_nodes: List[str] = []  # 合并了别名/描述的已有节点
        self.added_relations: List[str] = []
        self.updated_relations: List[str] = []  # 同 key 关系被更高置信度覆盖
        self.fuzzy_resolved: List[Dict[str, str]] = []  # {mention, node_id}
        self.rejected: List[Dict[str, str]] = []  # {item, reason}

    def summary(self) -> str:
        lines = [
            f"新增节点 {len(self.added_nodes)}，更新节点 {len(self.updated_nodes)}，"
            f"新增关系 {len(self.added_relations)}，覆盖关系 {len(self.updated_relations)}，"
            f"拒绝 {len(self.rejected)}"
        ]
        for m in self.fuzzy_resolved:
            lines.append(f"  [模糊] “{m['mention']}” -> {m['node_id']}")
        for r in self.rejected:
            lines.append(f"  [拒绝] {r['item']}  原因：{r['reason']}")
        return "\n".join(lines)


def _reject(report: IngestReport, item: str, reason: str) -> None:
    report.rejected.append({"item": item, "reason": reason})


def _rel_key(r: Relation) -> Tuple[str, str, str]:
    t = r.type.value if hasattr(r.type, "value") else str(r.type)
    return (r.source, r.target, t)


def _merge_refs(old: List[SourceRef], new: List[SourceRef]) -> List[SourceRef]:
    """保留同一事实来自多个分片的证据，同时让重复入库保持幂等。"""
    result = list(old)
    seen = {(r.document, r.chunk_id, r.quote) for r in result}
    for ref in new:
        key = (ref.document, ref.chunk_id, ref.quote)
        if key not in seen:
            result.append(ref)
            seen.add(key)
    return result


def merge_payload(onto: Ontology, payload: dict, embedder=None) -> Tuple[Ontology, IngestReport]:
    """把抽取产物合并进本体，返回新本体 + 报告。不修改传入对象。

    embedder 可选：提供时实体消歧升级为名称向量对齐（见 EntityIndex）。

    合并语义：
      - 节点：同 id 视为同一实体，aliases 取并集，仅给空字段补值，不覆盖已有内容；
      - 关系：以 (source, target, type) 为 key 去重，保留置信度更高的一条；
      - prerequisite_of 增量边逐一做环路检查，会成环的单独拒绝。
    """
    report = IngestReport()
    index = EntityIndex(onto, embedder=embedder)
    nodes: Dict[str, Node] = {n.id: n for n in onto.nodes}
    relations: Dict[Tuple[str, str, str], Relation] = {_rel_key(r): r for r in onto.relations}

    # ---- 1. 节点 ----
    for raw in payload.get("nodes", []):
        raw = dict(raw)
        if not raw.get("evidence_refs") and payload.get("source"):
            raw["evidence_refs"] = [{"document": str(payload["source"]),
                                      "chunk_id": "unknown"}]
        label = f"node:{raw.get('id') or raw.get('name') or '?'}"
        try:
            # 抽取器可以不提供 id：先放行 schema，稍后由名称生成
            node = Node(**{**raw, "id": raw.get("id") or ""})
        except ValidationError as exc:
            _reject(report, label, f"schema 校验失败：{exc.errors()[0]['msg']}")
            continue

        nid, how = index.resolve(node.name, proposed_id=node.id)
        if how == "id" and node.id and _norm(index.by_id[node.id].name) != _norm(node.name):
            # 提议的 id 已存在但名称不同：是撞车而非同一实体，走新建分支
            nid, how = None, "none"
        if how == "fuzzy" and nid:
            report.fuzzy_resolved.append({"mention": node.name, "node_id": nid})
        if nid:
            old = nodes[nid]
            merged_alias = sorted(set(old.aliases) | set(node.aliases))
            merged_refs = _merge_refs(old.evidence_refs, node.evidence_refs)
            changed = (
                merged_alias != old.aliases
                or (not old.desc and node.desc)
                or (not old.domain and node.domain)
                or merged_refs != old.evidence_refs
            )
            nodes[nid] = old.model_copy(
                update={
                    "aliases": merged_alias,
                    "desc": old.desc or node.desc,
                    "domain": old.domain or node.domain,
                    "evidence_refs": merged_refs,
                }
            )
            index.add(nodes[nid])
            if changed:
                report.updated_nodes.append(nid)
            continue

        # 新建实体：id 撞车（同名不同实体）时追加后缀
        new_id = node.id or slugify(node.name)
        if new_id in index.by_id:
            new_id = f"{new_id}_2"
        node = node.model_copy(update={"id": new_id})
        nodes[new_id] = node
        index.add(node)
        report.added_nodes.append(new_id)

    # ---- 2. 关系 ----
    prereq_dg = nx.DiGraph()
    prereq_dg.add_nodes_from(nodes)
    for r in relations.values():
        if r.type == RelationType.PREREQUISITE_OF:
            prereq_dg.add_edge(r.source, r.target)

    for raw in payload.get("relations", []):
        raw = dict(raw)
        if not raw.get("evidence_refs") and payload.get("source"):
            raw["evidence_refs"] = [{"document": str(payload["source"]),
                                      "chunk_id": "unknown"}]
        s_raw, t_raw, rtype = raw.get("source"), raw.get("target"), raw.get("type")
        label = f"rel:{s_raw}-[{rtype}]->{t_raw}"
        try:
            rel = Relation(**raw)
        except ValidationError as exc:
            _reject(report, label, f"schema 校验失败：{exc.errors()[0]['msg']}")
            continue

        sid = _endpoint_id(index, s_raw, payload, report)
        tid = _endpoint_id(index, t_raw, payload, report)
        if not sid or not tid:
            missing = s_raw if not sid else t_raw
            _reject(report, label, f"端点“{missing}”无法消歧且不在本批次节点中")
            continue
        if sid == tid:
            _reject(report, label, "自环关系无意义")
            continue

        key = (sid, tid, rel.type.value)
        existing = relations.get(key)
        if existing is not None:
            refs = _merge_refs(existing.evidence_refs, rel.evidence_refs)
            if rel.confidence > existing.confidence or refs != existing.evidence_refs:
                selected = rel if rel.confidence > existing.confidence else existing
                relations[key] = selected.model_copy(update={
                    "source": sid, "target": tid, "evidence_refs": refs,
                })
                report.updated_relations.append(f"{sid}-[{rel.type.value}]->{tid}")
            continue

        if rel.type == RelationType.PREREQUISITE_OF and nx.has_path(prereq_dg, tid, sid):
            # 新边 s->t 会成环 当且仅当 t 已经能到达 s
            _reject(report, label, f"会制造前置环路（{tid} 已可达 {sid}），破坏学习路径拓扑排序")
            continue

        relations[key] = rel.model_copy(update={"source": sid, "target": tid})
        if rel.type == RelationType.PREREQUISITE_OF:
            prereq_dg.add_edge(sid, tid)
        report.added_relations.append(f"{sid}-[{rel.type.value}]->{tid}")

    return Ontology(nodes=list(nodes.values()), relations=list(relations.values())), report


def _endpoint_id(index: EntityIndex, mention: Any, payload: dict,
                 report: IngestReport) -> Optional[str]:
    """关系端点消歧：先在本批次 nodes 里匹配（抽取器常用名称指称），再走全量索引。"""
    if mention is None:
        return None
    mention = str(mention).strip()
    if not mention:
        return None
    for raw in payload.get("nodes", []):
        name = str(raw.get("name", ""))
        if raw.get("id") == mention or (_norm(name) == _norm(mention) and name):
            nid, how = index.resolve(name or mention, proposed_id=str(raw.get("id") or ""))
            if nid:
                if how == "fuzzy":
                    report.fuzzy_resolved.append({"mention": mention, "node_id": nid})
                return nid
    nid, how = index.resolve(mention)
    if nid and how == "fuzzy":
        report.fuzzy_resolved.append({"mention": mention, "node_id": nid})
    return nid


# --------------------------------------------------------------------------
# 两层装载与回写（seed 只读，generated 是补丁层）
# --------------------------------------------------------------------------


def load_ontology_merged(*paths: str | Path) -> Tuple[Ontology, List[Path]]:
    """按顺序叠加装载多份本体文件。

    语义：前面的文件是基底，后面的文件是**补丁层** ——
      - 节点：同 id 只用补丁里给出的字段覆盖（字段级合并），其余字段保留基底值；
      - 关系：同 (source, target, type) 整条覆盖。
    """
    nodes: Dict[str, Node] = {}
    relations: Dict[Tuple[str, str, str], Relation] = {}
    loaded: List[Path] = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for n in raw.get("nodes", []):
            nid = n.get("id")
            if not nid:
                raise ValueError(f"{p.name} 里存在缺少 id 的节点: {n}")
            if nid in nodes:  # 补丁：字段级覆盖
                base = nodes[nid].model_dump()
                base.update({k: v for k, v in n.items() if k != "id"})
                nodes[nid] = Node(**base)
            else:
                nodes[nid] = Node(**n)
        for r in raw.get("relations", []):
            rel = Relation(**r)
            relations[_rel_key(rel)] = rel
        loaded.append(p)
    if not loaded:
        raise FileNotFoundError(f"一份本体文件都没找到: {paths}")
    return Ontology(nodes=list(nodes.values()), relations=list(relations.values())), loaded


def _node_patch(old: Node, new: Node) -> dict:
    """old -> new 的字段级差异。id 必须带上：装载时靠它定位要打补丁的节点。"""
    dump = new.model_dump(mode="json", exclude_none=True)
    patch = {k: v for k, v in dump.items() if k != "id" and getattr(old, k) != v}
    return {"id": new.id, **patch}


def ingest_file(payload_path: str | Path, data_dir: str | Path,
                dry_run: bool = False, embedder=None) -> IngestReport:
    """把一份抽取产物入库到 data_dir 下的机器补丁层（ontology_generated.yaml）。

    流程：装载 seed+generated 现状 -> 合并 payload -> 全图自检 ->
    计算与现状的差异 -> （非 dry-run）只把差异原子写入补丁层。
    embedder 可选：开启嵌入实体对齐。
    """
    data_dir = Path(data_dir)
    seed = data_dir / SEED_NAME
    generated = data_dir / GENERATED_NAME
    onto, _files = load_ontology_merged(seed, generated)
    payload = _read_payload(payload_path)

    new_onto, report = merge_payload(onto, payload, embedder=embedder)

    # 合并后再做一次全图自检：merge 已逐一挡住成环边，这里是最后防线
    cycles = KnowledgeGraph.from_ontology(new_onto).check_dag()
    if cycles:
        raise ValueError(f"合并后 prerequisite_of 出现环路，拒绝入库：{cycles[:3]}")

    # 差异 = 本轮真正新增/被更新的内容；补丁层只存这些
    old_nodes = {n.id: n for n in onto.nodes}
    old_rels = {_rel_key(r): r for r in onto.relations}
    node_entries: Dict[str, dict] = {}
    for n in new_onto.nodes:
        if n.id not in old_nodes:
            node_entries[n.id] = n.model_dump(mode="json", exclude_none=True)
        elif n.id in report.updated_nodes:
            node_entries[n.id] = _node_patch(old_nodes[n.id], n)
    rel_entries: Dict[Tuple[str, str, str], dict] = {}
    for r in new_onto.relations:
        k = _rel_key(r)
        if k not in old_rels or r != old_rels[k]:
            rel_entries[k] = r.model_dump(mode="json", exclude_none=True)

    if not dry_run and (node_entries or rel_entries):
        _write_patch_layer(generated, node_entries, rel_entries)
    return report


def _read_payload(payload_path: str | Path) -> dict:
    p = Path(payload_path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".json":
        import json

        return json.loads(p.read_text(encoding="utf-8"))
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _write_patch_layer(out: Path, node_entries: Dict[str, dict],
                       rel_entries: Dict[Tuple[str, str, str], dict]) -> None:
    """把本轮差异并入补丁层并原子落盘（写临时文件 -> 替换，旧内容留 .bak）。"""
    # 读旧补丁层，同 id/同 key 的新条目覆盖旧条目
    old_nodes: Dict[str, dict] = {}
    old_rels: Dict[Tuple[str, str, str], dict] = {}
    if out.exists():
        raw = yaml.safe_load(out.read_text(encoding="utf-8")) or {}
        for n in raw.get("nodes", []):
            old_nodes[n.get("id", "")] = n
        for r in raw.get("relations", []):
            try:
                rel = Relation(**r)
                old_rels[_rel_key(rel)] = r
            except ValidationError:
                continue  # 旧补丁层里的坏条目不阻塞新入库

    for nid, entry in node_entries.items():
        old_nodes[nid] = {**old_nodes.get(nid, {}), **entry}
    old_rels.update(rel_entries)
    payload = {"nodes": list(old_nodes.values()), "relations": list(old_rels.values())}

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".yaml.tmp")
    tmp.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    if out.exists():
        out.with_suffix(".yaml.bak").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.replace(out)
