"""学习者模型（L2 层）。

职责：记录"我"在每个知识点上的状态，并对外提供掌握度查询。
这是让 Agent "知道用户水平"的数据底座。

持久化格式：一个简单的 JSON 文件，形如
{
  "user_id": "me",
  "states": {
     "attention": {"mastery": 0.72, "confidence": 0.6, "last_touched": "2026-09-10",
                   "source": "self_report", "evidence": [...]}
  }
}
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .schema import (
    BAND_COLORS,
    Evidence,
    EvidenceType,
    LearnerState,
    MasteryBand,
    band_of,
    compute_mastery,
    decay_factor,
)

#: 判定为"已掌握"的阈值
MASTERED_THRESHOLD = 0.85

#: 判定为"薄弱/缺口"的阈值
GAP_THRESHOLD = 0.35


class LearnerModel:
    """学习者掌握度模型。"""

    def __init__(self, user_id: str = "me", states: Optional[Dict[str, LearnerState]] = None):
        self.user_id = user_id
        self.states: Dict[str, LearnerState] = states or {}

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "LearnerModel":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        states = {
            cid: LearnerState(**{"concept_id": cid, **data})
            for cid, data in raw.get("states", {}).items()
        }
        return cls(user_id=raw.get("user_id", "me"), states=states)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "user_id": self.user_id,
            "states": {cid: st.model_dump(mode="json") for cid, st in self.states.items()},
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_mastery(self, concept_id: str, now: Optional[date] = None) -> Optional[float]:
        st = self.states.get(concept_id)
        if st is None:
            return None
        if st.evidence:
            return compute_mastery(st.evidence, st.last_touched, now)
        # 兼容只有自评分数、没有证据明细的旧画像。
        return round(st.mastery * decay_factor(st.last_touched, now), 4)

    def mastery_or_zero(self, concept_id: str) -> float:
        return self.get_mastery(concept_id) or 0.0

    def mark(
        self,
        concept_id: str,
        mastery: Optional[float] = None,
        evidence: Optional[Evidence] = None,
        source: str = "self_report",
        ts: Optional[str] = None,
    ) -> LearnerState:
        """记录一次学习状态。

        两种用法：
          - 直接给 mastery（自评 / 答题得分）
          - 给一条 evidence，由 compute_mastery 自动折算
        """
        st = self.states.get(concept_id) or LearnerState(concept_id=concept_id, user_id=self.user_id)
        if evidence is not None:
            # 旧证据缺时间戳时，先固定到它原来的更新时间，避免新记录刷新旧证据。
            for previous in st.evidence:
                if previous.ts is None:
                    previous.ts = st.last_touched
            if not st.evidence and st.mastery > 0:
                st.evidence.append(Evidence(
                    type=EvidenceType.SELF_REPORT, desc="历史掌握度折算",
                    weight=st.mastery, ts=st.last_touched,
                ))
            evidence.ts = evidence.ts or ts or date.today().isoformat()
            st.evidence.append(evidence)
        if mastery is not None:
            st.mastery = max(0.0, min(1.0, mastery))
            # 显式评分覆盖先前判断，保留撤销轨迹供审计。
            for previous in st.evidence:
                previous.revoked = True
            event_ts = ts or date.today().isoformat()
            st.evidence.append(Evidence(
                type=EvidenceType.SELF_REPORT, desc="显式掌握度评分",
                weight=st.mastery, ts=event_ts,
            ))
            st.last_touched = event_ts
        else:
            if ts:
                st.last_touched = ts
            elif evidence is not None:
                st.last_touched = evidence.ts
            st.mastery = compute_mastery(st.evidence, st.last_touched)
        st.source = source
        self.states[concept_id] = st
        return st

    # ------------------------------------------------------------------
    # 画像输出
    # ------------------------------------------------------------------

    def band(self, concept_id: str) -> str:
        return band_of(self.mastery_or_zero(concept_id)).value

    def color(self, concept_id: str) -> str:
        return BAND_COLORS[self.band(concept_id)]

    def is_gap(self, concept_id: str) -> bool:
        """是否是需要补的缺口。未接触也算缺口。"""
        return self.mastery_or_zero(concept_id) < GAP_THRESHOLD

    def is_mastered(self, concept_id: str) -> bool:
        return self.mastery_or_zero(concept_id) >= MASTERED_THRESHOLD

    def profile(self, node_ids: Iterable[str]) -> dict:
        """给定一组节点，输出掌握度画像（Agent 调用 get_learner_profile 时的返回体）。"""
        node_ids = list(node_ids)
        buckets: Dict[str, List[str]] = {b.value: [] for b in MasteryBand}
        for nid in node_ids:
            buckets[self.band(nid)].append(nid)
        touched = [n for n in node_ids if self.mastery_or_zero(n) > 0]
        avg = (
            round(sum(self.mastery_or_zero(n) for n in touched) / len(touched), 3)
            if touched
            else 0.0
        )
        return {
            "user_id": self.user_id,
            "total": len(node_ids),
            "touched": len(touched),
            "average_mastery": avg,
            "bands": buckets,
        }

    def summary(self, kg_node_name) -> str:
        """生成一段人类可读的知识画像文字，供 Agent 直接放进上下文。"""
        from collections import defaultdict

        by_band: Dict[str, List[str]] = defaultdict(list)
        for cid in self.states:
            mastery = self.mastery_or_zero(cid)
            by_band[mastery and band_of(mastery).value or MasteryBand.UNTOUCHED.value].append(
                kg_node_name(cid)
            )
        lines = [f"用户 {self.user_id} 的知识画像："]
        labels = {
            MasteryBand.MASTERED.value: "已掌握",
            MasteryBand.SOLID.value: "较扎实",
            MasteryBand.LEARNING.value: "学习中",
            MasteryBand.WEAK.value: "薄弱",
            MasteryBand.UNTOUCHED.value: "未接触",
        }
        for band, label in labels.items():
            items = by_band.get(band, [])
            if items:
                lines.append(f"  · {label}（{len(items)}）：{'、'.join(sorted(items)[:12])}")
        return "\n".join(lines)
