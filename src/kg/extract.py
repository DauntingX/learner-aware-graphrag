"""LLM 结构化抽取管线（P1 的抽取半边）。

流程：语料分片 -> 逐片调 LLM 抽三元组（带已有节点表辅助消歧）->
容错解析 JSON -> 失败重试 -> （可选）gleaning 补抽 -> 片级产物合并 -> 交给 ingest 入库。

分片有两种来源：
  - chunk_text：纯文本按空行段落分片（兜底）
  - notes.read_note -> sections_to_chunks：md/docx 按标题小节分片，
    每片带「笔记位置」面包屑（借鉴 LangChain MarkdownHeaderTextSplitter）
gleaning 补抽借鉴 Microsoft GraphRAG：首抽后再问一轮「还有什么遗漏」。

只依赖 LLMClient.complete(prompt, system) 这一个方法 —— 换模型/换协议
只需换客户端实现，本模块不动。

输出契约与 ingest 层一致：{"nodes": [...], "relations": [...]}。
"""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ingest import Ontology, _norm
from .llm import LLMConfig, LLMClient
from .notes import NoteSection, read_note

SYSTEM_PROMPT = "你是严谨的知识图谱构建专家。只输出一个 JSON 对象，不要输出任何解释、前后缀或代码块标记。"

_PROMPT_TEMPLATE = """从下面的学习笔记片段中抽取 AI 领域的知识三元组。

## 已有图谱节点（source/target 优先复用这些 id，不要为它们新建节点）
{existing}

## 节点 type 必须严格取其一
Concept（概念）/ Technique（技术）/ Model（模型）/ Task（任务）/ Tool（工具）/ Paper（论文）/ Skill（技能）

## 关系 type 必须严格取其一
prerequisite_of: A 是 B 的前置知识（方向：基础 -> 进阶）
part_of: A 是 B 的组成部分
related_to: 弱关联
evolved_from: A 由 B 演进而来（方向：新技术 -> 旧技术）
applied_in: A 应用于 B（任务/模型）
implemented_by: 概念 A 由工具 B 实现
described_in: 概念 A 出自论文 B
confusable_with: A 与 B 易混淆（无方向）

## 规则
1. 只输出一个 JSON 对象，结构如下（没有就给空数组）：
    {{"nodes": [{{"id": "...", "name": "...", "type": "...", "domain": "...", "difficulty": 1到5, "desc": "不超过40字", "aliases": ["..."], "evidence_quote": "原文短句"}}],
     "relations": [{{"source": "...", "target": "...", "type": "...", "confidence": 0到1, "evidence_quote": "原文短句"}}]}}
2. 新实体自己起英文 snake_case id，同一概念在不同片段里必须用同一 id；别名放进 aliases
3. 片段里没有依据的关系不要编造；difficulty：1 入门 ~ 5 硬核
4. desc 用一句话说清它是什么/为什么重要
5. 每个节点和关系的 evidence_quote 必须从待抽取片段逐字复制一段不超过120字的依据；找不到则填空字符串

## 待抽取片段
{chunk}"""


class ExtractError(RuntimeError):
    """LLM 输出无法解析为约定 JSON。"""


# --------------------------------------------------------------------------
# JSON 容错解析
# --------------------------------------------------------------------------


def parse_llm_json(text: str) -> dict:
    """从 LLM 输出里抠出第一个完整 JSON 对象。

    实际输出五花八门：```json 围栏、前后带说明文字、字符串里含花括号。
    用带字符串感知的括号深度扫描，而不是朴素 find("{")。
    """
    t = text.strip()
    if not t:
        raise ExtractError("LLM 返回了空内容")

    candidates = [t]
    if "```" in t:  # 去掉围栏行再试
        fenced = re.sub(r"```[a-zA-Z]*\n?", "", t).strip()
        candidates.insert(0, fenced)

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 深度扫描第一个平衡的 {...}（跳过字符串字面量）
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break  # 这一段不是合法 JSON，从下一个 { 再找
                    break
        start = t.find("{", start + 1)

    raise ExtractError(f"输出无法解析为 JSON 对象，原文开头：{t[:200]!r}")


# --------------------------------------------------------------------------
# 语料分片
# --------------------------------------------------------------------------


def chunk_text(text: str, chunk_chars: int = 4000) -> List[str]:
    """按空行分段落，尽量凑到 chunk_chars 上限；超长段落硬切。

    优先在段落边界断开，避免把一个概念拦腰截断。
    """
    chunk_chars = max(chunk_chars, 200)
    chunks: List[str] = []
    cur = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        while len(para) > chunk_chars:  # 单段超长：硬切，剩余部分继续走正常拼装
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(para[:chunk_chars])
            para = para[chunk_chars:]
        if not para:
            continue
        if not cur:
            cur = para
        elif len(cur) + len(para) + 2 <= chunk_chars:
            cur += "\n\n" + para
        else:
            chunks.append(cur)
            cur = para
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------


def existing_node_lines(onto: Ontology) -> str:
    """已有节点清单 -> 提示词用的紧凑列表。这是跨片段消歧的关键输入。"""
    lines = []
    for n in sorted(onto.nodes, key=lambda x: x.id):
        line = f"- {n.id} | {n.name}"
        if n.aliases:
            line += f" | 别名: {'/'.join(n.aliases[:3])}"
        lines.append(line)
    return "\n".join(lines)


def build_prompt(chunk: str, existing: str) -> str:
    return _PROMPT_TEMPLATE.format(existing=existing, chunk=chunk)


# --------------------------------------------------------------------------
# 抽取主流程
# --------------------------------------------------------------------------

GLEAN_PROMPT = """上面片段已经抽取过一轮。请再检查一遍，只补充**遗漏**的实体和关系：
没有遗漏就输出 {{"nodes": [], "relations": []}}，有就只输出新增部分的 JSON（结构与首轮相同）。"""


def _extract_one(client: LLMClient, prompt: str, max_retries: int,
                 gleans: int = 0) -> dict:
    """单片段抽取：解析失败重试 + GraphRAG 式 gleaning 补抽遗漏。"""
    last_err: Optional[ExtractError] = None
    payload: Optional[dict] = None
    p = prompt
    for attempt in range(max_retries + 1):
        raw = client.complete(p, system=SYSTEM_PROMPT)
        try:
            payload = parse_llm_json(raw)
            break
        except ExtractError as exc:
            last_err = exc
            p += ("\n\n注意：你上一次的输出无法解析为 JSON。"
                  "请严格只输出一个 JSON 对象，以 { 开头、} 结尾，不要任何其他文字。")
    if payload is None:
        raise last_err  # type: ignore[misc]

    # gleaning：首抽成功后追问遗漏（GraphRAG 的多轮抽取思想，召回率换 token）
    for _i in range(max(0, gleans)):
        raw = client.complete(prompt + "\n\n" + GLEAN_PROMPT, system=SYSTEM_PROMPT)
        try:
            extra = parse_llm_json(raw)
        except ExtractError:
            continue  # 补抽失败不影响首轮成果
        seen_n = {n.get("id") for n in payload.get("nodes", []) if isinstance(n, dict)}
        seen_r = {(r.get("source"), r.get("target"), r.get("type"))
                  for r in payload.get("relations", []) if isinstance(r, dict)}
        if isinstance(extra.get("nodes"), list):
            new_nodes = [n for n in extra["nodes"]
                         if isinstance(n, dict) and n.get("id") not in seen_n]
            payload["nodes"] = list(payload.get("nodes", [])) + new_nodes
            seen_n.update(n["id"] for n in new_nodes)
        if isinstance(extra.get("relations"), list):
            new_rels = [r for r in extra["relations"]
                        if isinstance(r, dict)
                        and (r.get("source"), r.get("target"), r.get("type")) not in seen_r]
            payload["relations"] = list(payload.get("relations", [])) + new_rels
            seen_r.update((r.get("source"), r.get("target"), r.get("type")) for r in new_rels)
    return payload


def _merge_chunk_payloads(payloads: List[dict]) -> dict:
    """同批片段产物去重合并：节点按 id 首见优先（别名并集），关系按 key 保最高置信度。"""
    nodes: Dict[str, dict] = {}
    for p in payloads:
        for n in p.get("nodes", []):
            nid = str(n.get("id") or "").strip()
            if not nid:
                continue
            if nid in nodes:
                old_alias = nodes[nid].get("aliases") or []
                extra = [a for a in (n.get("aliases") or []) if a not in old_alias]
                nodes[nid]["aliases"] = old_alias + extra
                for f in ("desc", "domain"):
                    if not nodes[nid].get(f) and n.get(f):
                        nodes[nid][f] = n[f]
                nodes[nid]["evidence_refs"] = _union_refs(
                    nodes[nid].get("evidence_refs", []), n.get("evidence_refs", []))
            else:
                nodes[nid] = dict(n)
    rels: Dict[Tuple[str, str, str], dict] = {}
    for p in payloads:
        for r in p.get("relations", []):
            key = (_norm(r.get("source", "")), _norm(r.get("target", "")),
                   str(r.get("type", "")))
            if key not in rels:
                rels[key] = dict(r)
                continue
            refs = _union_refs(rels[key].get("evidence_refs", []),
                               r.get("evidence_refs", []))
            if float(r.get("confidence", 0)) > float(rels[key].get("confidence", 0)):
                rels[key] = dict(r)
            rels[key]["evidence_refs"] = refs
    return {"nodes": list(nodes.values()), "relations": list(rels.values())}


def _union_refs(old: List[dict], new: List[dict]) -> List[dict]:
    result = list(old)
    seen = {(r.get("document"), r.get("chunk_id"), r.get("quote")) for r in old}
    for ref in new:
        key = (ref.get("document"), ref.get("chunk_id"), ref.get("quote"))
        if key not in seen:
            result.append(ref)
            seen.add(key)
    return result


def _attach_source_refs(payload: dict, chunk: str, document: str, chunk_index: int) -> dict:
    """由程序记录分片位置；模型引文仅在逐字命中原文时保留。"""
    match = re.match(r"【笔记位置：([^】]+)】", chunk)
    common = {
        "document": Path(document).name,
        "chunk_id": f"chunk-{chunk_index:04d}",
        "section": match.group(1) if match else "",
        "chunk_sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
    }
    for kind in ("nodes", "relations"):
        for item in payload.get(kind, []):
            if not isinstance(item, dict):
                continue
            raw_quote = item.pop("evidence_quote", "")
            quote = raw_quote.strip() if isinstance(raw_quote, str) else ""
            verified = bool(quote and quote in chunk)
            ref = {**common, "quote": quote[:240] if verified else "",
                   "quote_verified": verified}
            item["evidence_refs"] = _union_refs(item.get("evidence_refs", []), [ref])
    return payload


def sections_to_chunks(sections: List[NoteSection], chunk_chars: int = 4000) -> List[str]:
    """结构化小节 -> 分片文本：一个小节一片（天然语义边界），超长小节再细切。

    每片都带「笔记位置」面包屑前缀，LLM 能看到这段话在文档结构里的位置。
    """
    chunks: List[str] = []
    for sec in sections:
        full = sec.chunk_text()
        if len(full) <= chunk_chars:
            chunks.append(full)
            continue
        prefix = f"【笔记位置：{sec.breadcrumb}】"
        budget = max(chunk_chars - len(prefix) - 10, 200)
        cur = ""
        for para in re.split(r"\n\s*\n", sec.text):
            para = para.strip()
            if not para:
                continue
            while len(para) > budget:  # 单段超长硬切
                if cur:
                    chunks.append(prefix + "\n" + cur)
                    cur = ""
                chunks.append(prefix + "\n" + para[:budget])
                para = para[budget:]
            if not para:
                continue
            if not cur:
                cur = para
            elif len(cur) + len(para) + 2 <= budget:
                cur += "\n\n" + para
            else:
                chunks.append(prefix + "\n" + cur)
                cur = para
        if cur:
            chunks.append(prefix + "\n" + cur)
    return chunks


def _extract_chunks(chunks: List[str], client: LLMClient, cfg: LLMConfig, onto: Ontology,
                    max_chunks: Optional[int] = None,
                    document: str = "inline_text") -> Tuple[dict, dict]:
    """分片抽取主循环：单片失败不中断整批。"""
    existing = existing_node_lines(onto)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    payloads: List[dict] = []
    failed: List[str] = []
    for i, chunk in enumerate(chunks):
        prompt = build_prompt(chunk, existing)
        try:
            extracted = _extract_one(client, prompt, cfg.max_retries, gleans=cfg.gleans)
            payloads.append(_attach_source_refs(extracted, chunk, document, i))
        except Exception as exc:  # noqa: BLE001  # 网络错误/解析错误都不打断整批
            failed.append(f"chunk{i}({len(chunk)}字): {exc}")
    payload = _merge_chunk_payloads(payloads)
    stats = {
        "chunks": len(chunks),
        "ok_chunks": len(payloads),
        "failed_chunks": failed,
        "nodes": len(payload["nodes"]),
        "relations": len(payload["relations"]),
    }
    return payload, stats


def extract_text(text: str, client: LLMClient, cfg: LLMConfig,
                 onto: Ontology, max_chunks: Optional[int] = None,
                 document: str = "inline_text") -> Tuple[dict, dict]:
    """抽取纯文本（按空行段落分片）。结构化笔记请用 extract_note。"""
    return _extract_chunks(chunk_text(text, cfg.chunk_chars),
                           client, cfg, onto, max_chunks, document)


def extract_note(path: str | Path, client: LLMClient, cfg: LLMConfig,
                 onto: Ontology, max_chunks: Optional[int] = None) -> Tuple[dict, dict]:
    """抽取 md/docx 笔记：按标题小节分片（一节一片，带面包屑）。"""
    sections = read_note(Path(path))
    return _extract_chunks(sections_to_chunks(sections, cfg.chunk_chars),
                           client, cfg, onto, max_chunks, Path(path).name)
