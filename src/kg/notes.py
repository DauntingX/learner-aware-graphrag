"""笔记文件解析：md / docx -> 结构化小节。

借鉴来源：
  - LangChain MarkdownHeaderTextSplitter 的「按标题切分 + 层级面包屑」思路：
    每个小节携带完整的标题路径（如 "多模态笔记 > Vision Transformer"），
    抽取提示词里有上下文，跨节消歧更准；
  - mammoth 的语义转换思想：docx 不按视觉格式而按 Heading 样式取结构
    （本模块用 python-docx 直接遍历段落实现，零额外安装）。

md 与 docx 统一产出 NoteSection 列表，供抽取层「一节一片」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

#: ATX 标题行（# ~ ######），ATX 闭合形式（## 标题 ##）也兼容
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*(#:*)?\s*$")

_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class NoteSection:
    """笔记里的一个标题小节。无任何标题的文档整体算一节（level=0）。"""

    title: str  # 本节自身标题；无标题文档取首行或文件名
    breadcrumb: str  # 完整标题路径，"根 > 父节 > 本节"
    level: int  # 标题层级 1-6；文档级小节为 0
    text: str  # 小节正文（不含自身标题行）

    def chunk_text(self) -> str:
        """给 LLM 的文本：面包屑前缀 + 正文，让每个分片自带上下文。"""
        return f"【笔记位置：{self.breadcrumb}】\n{self.text}".strip()


def parse_markdown(text: str) -> List[NoteSection]:
    """按 ATX 标题切分 markdown；```/~~~ 围栏内的 # 不算标题。"""
    sections: List[NoteSection] = []
    stack: List[tuple[int, str]] = []  # (level, title) 标题栈
    buf: List[str] = []
    cur_title = ""
    cur_level = 0
    in_fence = False

    def flush():
        body = "\n".join(buf).strip()
        if not body and not cur_title:
            return
        if not body:
            return  # 空小节不产出，标题只作为后续小节的路径
        bc = " > ".join([t for _l, t in stack] or [cur_title or "正文"])
        sections.append(NoteSection(title=cur_title or "正文",
                                    breadcrumb=bc, level=cur_level, text=body))

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        m = None if in_fence else _MD_HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            cur_title = m.group(2).strip()
            cur_level = level
            # 弹栈到父级：新标题不能是同级或上级的子节
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, cur_title))
            buf = []
        else:
            buf.append(line)
    flush()
    return sections


def parse_docx(path: str | Path) -> List[NoteSection]:
    """python-docx 遍历段落，按 Heading 样式重建层级，产出与 md 相同的小节结构。

    内置标题样式（python-docx 读出的样式名为 'Heading 1'~'Heading 6'，中英文档一致）
    之外的样式一律当正文。
    """
    import docx  # python-docx

    doc = docx.Document(str(path))
    sections: List[NoteSection] = []
    stack: List[tuple[int, str]] = []
    cur_title = ""
    cur_level = 0
    buf: List[str] = []

    def _heading_level(style_name: str) -> Optional[int]:
        m = re.match(r"^(?:Heading|标题)\s*([1-6])$", style_name.strip(), re.IGNORECASE)
        return int(m.group(1)) if m else None

    def flush():
        body = "\n".join(buf).strip()
        if body:
            bc = " > ".join([t for _l, t in stack] or [cur_title or "正文"])
            sections.append(NoteSection(title=cur_title or "正文",
                                        breadcrumb=bc, level=cur_level, text=body))
        buf.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        level = _heading_level(para.style.name if para.style else "")
        if level:
            flush()
            cur_title, cur_level = text or "（无题小节）", level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, cur_title))
        elif text:
            buf.append(text)
    flush()
    return sections


def read_note(path: str | Path) -> List[NoteSection]:
    """按扩展名分发：.md/.markdown/.txt 走文本解析，.docx 走 python-docx。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".docx":
        return parse_docx(p)
    if suffix in (".md", ".markdown", ".txt"):
        return parse_markdown(p.read_text(encoding="utf-8", errors="replace"))
    raise ValueError(f"不支持的笔记格式: {p.name}（支持 .md/.markdown/.txt/.docx）")
