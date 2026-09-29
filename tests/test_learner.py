"""学习者画像的时间衰减与证据回写回归测试。"""

from __future__ import annotations

import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kg.learner import LearnerModel  # noqa: E402
from kg.schema import Evidence, EvidenceType, LearnerState, compute_mastery  # noqa: E402


def test_direct_score_decays_when_read():
    today = date(2026, 9, 28)
    touched = (today - timedelta(days=60)).isoformat()
    learner = LearnerModel(states={
        "a": LearnerState(concept_id="a", mastery=0.8, last_touched=touched)
    })
    assert learner.get_mastery("a", now=today) == 0.4
    assert learner.states["a"].mastery == 0.8, "读取不应改写原始状态"


def test_evidence_decays_independently():
    today = date(2026, 9, 28)
    old = (today - timedelta(days=60)).isoformat()
    evidence = [
        Evidence(type=EvidenceType.QUIZ, desc="旧测验", weight=0.6, ts=old),
        Evidence(type=EvidenceType.PROJECT, desc="新项目", weight=0.6, ts=today.isoformat()),
    ]
    # 旧证据有效权重 0.3；新证据 0.6；1 - (1-.3)*(1-.6) = .72。
    assert compute_mastery(evidence, now=today) == 0.72
    evidence[1].revoked = True
    assert compute_mastery(evidence, now=today) == 0.3


def test_new_evidence_does_not_refresh_old_evidence():
    today = date.today()
    old = (today - timedelta(days=60)).isoformat()
    learner = LearnerModel(states={
        "a": LearnerState(
            concept_id="a", mastery=0.6, last_touched=old,
            evidence=[Evidence(type=EvidenceType.QUIZ, desc="旧测验", weight=0.6)],
        )
    })
    learner.mark("a", evidence=Evidence(type=EvidenceType.PROJECT, desc="新项目", weight=0.6),
                 ts=today.isoformat())
    assert learner.states["a"].evidence[0].ts == old
    assert learner.get_mastery("a", now=today) == 0.72
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "learner.json"
        learner.save(path)
        reloaded = LearnerModel.load(path)
    assert reloaded.get_mastery("a", now=today) == 0.72


def test_profile_accepts_generator_once():
    learner = LearnerModel()
    profile = learner.profile(node for node in ("a", "b"))
    assert profile["total"] == 2
    assert len(profile["bands"]["untouched"]) == 2


def main() -> int:
    import traceback

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
