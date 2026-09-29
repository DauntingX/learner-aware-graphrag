"""AI 知识图谱 · 学习者模型 · Agent 外挂服务。

分层：
    schema   —— 数据模型（L1 知识层 + L2 学习者层）
    store    —— 图存储（NetworkX + SQLite，可迁移 Neo4j）
    learner  —— 学习者模型（掌握度画像）
    paths    —— 学习路径规划（拓扑排序 + 缺口标注）
    ingest   —— 增量入库（实体消歧 + 补丁层合并，P1 落库半边）
"""

from .ingest import (
    EntityIndex,
    IngestReport,
    GENERATED_NAME,
    ingest_file,
    load_ontology_merged,
    merge_payload,
    slugify,
)
from .learner import LearnerModel
from .paths import learning_path, prerequisite_graph, recommend_next
from .schema import (
    BAND_COLORS,
    Evidence,
    EvidenceType,
    LearnerState,
    MasteryBand,
    Node,
    NodeType,
    Ontology,
    Relation,
    RelationType,
    SourceRef,
    band_of,
    compute_mastery,
)
from .store import KnowledgeGraph

__all__ = [
    "KnowledgeGraph",
    "LearnerModel",
    "Ontology",
    "Node",
    "Relation",
    "NodeType",
    "RelationType",
    "SourceRef",
    "Evidence",
    "EvidenceType",
    "LearnerState",
    "MasteryBand",
    "band_of",
    "compute_mastery",
    "BAND_COLORS",
    "learning_path",
    "recommend_next",
    "prerequisite_graph",
    "EntityIndex",
    "IngestReport",
    "GENERATED_NAME",
    "ingest_file",
    "load_ontology_merged",
    "merge_payload",
    "slugify",
]

__version__ = "0.2.0"
