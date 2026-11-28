import pytest
from benchmarks.roboinf.compare_clear import compare

from roboscope_kit.runtime.benchmarks.roboinf.factory import Adapter, Factory


def test_random_receipts_cannot_be_replaced_by_paired_clear_results():
    from types import SimpleNamespace

    Factory.validate_suite("test_random")
    with pytest.raises(ValueError):
        Factory.validate_suite("val")
    adapter = Adapter(None, [{"case_id": "paired", "task": "task"}], "test_random")
    metadata = {
        "case_id": "paired",
        "task": "task",
        "task_success": True,
        "instruction_success": False,
        "domain": "test",
    }
    results = [
        SimpleNamespace(records=[SimpleNamespace(record_id="paired", success=True, metadata=metadata)])
    ]
    with pytest.raises(ValueError, match="Invalid"):
        adapter.aggregate(results)
    metadata["domain"] = "test_random"
    assert adapter.metadata()["suite"] == "test_random"
    assert adapter.aggregate(results)["tasks"]["task"] == {"cases": 1, "TR": 1, "JR": 0}


def judgments(n):
    return [
        {"case_id": str(i), "domain": "test", "task_success": False, "instruction_success": True}
        for i in range(n)
    ]


def test_macro_boundary_and_joint_semantics():
    cases = {str(i): "a" for i in range(100)}
    old = judgments(100)
    new = judgments(100)
    for row in new[:2]:
        row.update(task_success=True, instruction_success=False)
    result = compare(cases, old, new)
    assert result["passed"]
    assert result["macro"] == {"TR": {"delta_pp": 2}, "JR": {"delta_pp": 0}}
    new[2]["task_success"] = True
    assert not compare(cases, old, new)["passed"]


def test_per_task_gate_even_when_macro_matches():
    old, new = judgments(40), judgments(40)
    cases = {str(i): str(i // 20) for i in range(40)}
    for i in (0, 1):
        old[i]["task_success"] = new[20 + i]["task_success"] = True
    assert not compare(cases, old, new)["passed"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "unknown", "wrong_domain", "non_boolean"])
def test_invalid_records(bad):
    old, new = judgments(2), judgments(2)
    if bad == "missing":
        new.pop()
    elif bad == "duplicate":
        new.append(new[0])
    elif bad == "unknown":
        new[0]["case_id"] = "other"
    elif bad == "wrong_domain":
        new[0]["domain"] = "val"
    else:
        new[0]["instruction_success"] = 0
    with pytest.raises(ValueError):
        compare({"0": "a", "1": "b"}, old, new)
