"""知识图谱数据模型。

本模块定义三层模型中的 L1（知识层）与 L2（学习者层）的数据结构：
- L1 知识层：Node / Relation（描述 AI 领域本身）
- L2 学习者层：LearnerState / Evidence（描述"我"在这张图上的状态）

两层共用同一批节点 id，这是本项目的核心设计。
"""

from __future__ import annotations

import datetime as _dt
import math
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# L1 知识层
# --------------------------------------------------------------------------


class NodeType(str, Enum):
    """节点类型。Concept 与 Skill 的区分很重要：前者是"知道"，后者是"会做"。"""

    CONCEPT = "Concept"
    TECHNIQUE = "Technique"
    MODEL = "Model"
    TASK = "Task"
    TOOL = "Tool"
    PAPER = "Paper"
    SKILL = "Skill"


class RelationType(str, Enum):
    """关系类型。边的语义才是知识图谱的灵魂。"""

    PREREQUISITE_OF = "prerequisite_of"  # A 是 B 的前置知识（驱动学习路径）
    PART_OF = "part_of"  # A 是 B 的组成部分（层级包含）
    RELATED_TO = "related_to"  # 弱关联、同领域
    EVOLVED_FROM = "evolved_from"  # 技术演进关系
    APPLIED_IN = "applied_in"  # 方法应用于某任务/模型
    IMPLEMENTED_BY = "implemented_by"  # 概念由某工具实现
    DESCRIBED_IN = "described_in"  # 概念出自某论文
    CONFUSABLE_WITH = "confusable_with"  # 易混淆对（驱动对比式复习）


class SourceRef(BaseModel):
    """抽取事实对应的原始分片；引文只有逐字匹配原文时才标记已验证。"""

    document: str
    chunk_id: str
    section: str = ""
    chunk_sha256: str = ""
    quote: str = ""
    quote_verified: bool = False


class Node(BaseModel):
    """知识图谱中的一个知识点。"""

    id: str
    name: str
    type: NodeType
    domain: str = ""
    desc: str = ""
    difficulty: int = Field(default=3, ge=1, le=5, description="1=入门, 5=硬核")
    aliases: List[str] = Field(default_factory=list, description="别名，用于实体消歧")
    evidence_refs: List[SourceRef] = Field(default_factory=list)


class Relation(BaseModel):
    """两个知识点之间的关系。"""

    source: str
    target: str
    type: RelationType
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    note: str = ""
    evidence_refs: List[SourceRef] = Field(default_factory=list)


class Ontology(BaseModel):
    """一份完整的本体（节点 + 关系）。从 seed_ontology.yaml 加载。"""

    nodes: List[Node]
    relations: List[Relation]

    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}


# --------------------------------------------------------------------------
# L2 学习者层
# --------------------------------------------------------------------------


class EvidenceType(str, Enum):
    SELF_REPORT = "self_report"  # 自评
    QUIZ = "quiz"  # 答题
    PROJECT = "project"  # 做过项目
    CONVERSATION = "conversation"  # 对话中暴露
    CODE = "code"  # 写过代码


class Evidence(BaseModel):
    """一条"我掌握某概念"的证据。

    revoked 借鉴 graphiti 的时序思想：证据可能被推翻（后来发现当时并没有真的掌握），
    标记失效而不是删除 —— 保留轨迹，重算掌握度时自动跳过。
    """

    type: EvidenceType
    desc: str
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    ts: Optional[str] = None
    revoked: bool = False


class LearnerState(BaseModel):
    """学习者对某个节点的状态。这就是"让 Agent 知道你水平"的数据底座。"""

    user_id: str = "me"
    concept_id: str
    mastery: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.3, ge=0.0, le=1.0, description="模型对这个判断有多确信")
    evidence: List[Evidence] = Field(default_factory=list)
    last_touched: Optional[str] = None
    source: str = "self_report"


# --------------------------------------------------------------------------
# 掌握度计算：证据加权 + 时间衰减
# --------------------------------------------------------------------------

#: 时间衰减半衰期（天）。60 天后未接触的记忆权重减半。
DECAY_HALF_LIFE_DAYS = 60.0


def decay_factor(last_touched: Optional[str], now: Optional[_dt.date] = None) -> float:
    """计算时间衰减系数，返回 (0, 1]。

    采用指数衰减：f = 0.5 ** (Δt / 半衰期)。
    没有记录时间时返回 1.0（不做衰减）。
    """
    if not last_touched:
        return 1.0
    now = now or _dt.date.today()
    try:
        last = _dt.date.fromisoformat(last_touched)
    except ValueError:
        return 1.0
    days = max((now - last).days, 0)
    return math.pow(0.5, days / DECAY_HALF_LIFE_DAYS)


def compute_mastery(evidence: List[Evidence], last_touched: Optional[str] = None,
                    now: Optional[_dt.date] = None) -> float:
    """由证据列表计算掌握度。

    规则：
      - 被标记 revoked 的证据不参与计算（保留在轨迹里但不生效）
      - 每条证据按自己的时间戳衰减，再按 1 - Π(1 - 有效权重_i) 聚合
      - 旧数据没有证据时间戳时，回退到状态的 last_touched
    """
    active = [ev for ev in evidence if not ev.revoked]
    if not active:
        return 0.0
    remain = 1.0
    for ev in active:
        weight = max(0.0, min(1.0, ev.weight))
        remain *= 1.0 - weight * decay_factor(ev.ts or last_touched, now)
    return round(1.0 - remain, 4)


# --------------------------------------------------------------------------
# 掌握度分档（用于可视化着色，遵循中文习惯：红=薄弱，绿=已掌握）
# --------------------------------------------------------------------------


class MasteryBand(str, Enum):
    UNTOUCHED = "untouched"  # 未接触 -> 灰
    WEAK = "weak"  # 薄弱 -> 红
    LEARNING = "learning"  # 学习中 -> 橙
    SOLID = "solid"  # 较扎实 -> 黄绿
    MASTERED = "mastered"  # 已掌握 -> 绿


#: 分档阈值：mastery 下界
BAND_THRESHOLDS = [
    (0.0001, MasteryBand.WEAK),
    (0.35, MasteryBand.LEARNING),
    (0.65, MasteryBand.SOLID),
    (0.85, MasteryBand.MASTERED),
]


def band_of(mastery: float) -> MasteryBand:
    """把 0~1 的掌握度映射到分档。"""
    if mastery <= 0.0:
        return MasteryBand.UNTOUCHED
    band = MasteryBand.WEAK
    for lower, b in BAND_THRESHOLDS:
        if mastery >= lower:
            band = b
    return band


#: 分档 -> 颜色（中文习惯：红涨绿跌在这里借用于"红=薄弱"）
BAND_COLORS = {
    MasteryBand.UNTOUCHED.value: "#B4B2A9",  # 灰
    MasteryBand.WEAK.value: "#E24B4A",  # 红 - 薄弱
    MasteryBand.LEARNING.value: "#EF9F27",  # 橙 - 学习中
    MasteryBand.SOLID.value: "#97C459",  # 黄绿 - 较扎实
    MasteryBand.MASTERED.value: "#1D9E75",  # 绿 - 已掌握
}

#: 关系类型 -> 可视化中的边样式
RELATION_STYLES = {
    RelationType.PREREQUISITE_OF.value: {"color": "#185FA5", "dash": "", "width": 1.8},
    RelationType.PART_OF.value: {"color": "#0F6E56", "dash": "", "width": 1.4},
    RelationType.RELATED_TO.value: {"color": "#B4B2A9", "dash": "4,3", "width": 1.0},
    RelationType.EVOLVED_FROM.value: {"color": "#534AB7", "dash": "6,3", "width": 1.4},
    RelationType.APPLIED_IN.value: {"color": "#D85A30", "dash": "", "width": 1.2},
    RelationType.IMPLEMENTED_BY.value: {"color": "#888780", "dash": "2,3", "width": 1.0},
    RelationType.DESCRIBED_IN.value: {"color": "#AFA9EC", "dash": "2,3", "width": 1.0},
    RelationType.CONFUSABLE_WITH.value: {"color": "#D4537E", "dash": "3,3", "width": 1.2},
}
