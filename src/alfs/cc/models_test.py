"""Tests for CC task and output models."""

from pydantic import TypeAdapter

from alfs.cc.models import (
    CCInductionOutput,
    CCInductionTask,
    CCMorphRedirectOutput,
    CCMorphRedirectTask,
    CCOutput,
    CCRewriteOutput,
    CCRewriteTask,
    CCTask,
    CCTrimSenseOutput,
    CCTrimSenseTask,
    FormInfo,
    InductionSense,
    MorphRelation,
    RewrittenSense,
    SenseInfo,
)

_task_adapter: TypeAdapter[CCTask] = TypeAdapter(CCTask)
_output_adapter: TypeAdapter[CCOutput] = TypeAdapter(CCOutput)


def test_induction_task_roundtrip():
    task = CCInductionTask(
        id="abc",
        form="dog",
        contexts=["The dog barked.", "I walked the dog."],
        existing_defs=["a domestic animal"],
    )
    data = task.model_dump_json()
    parsed = _task_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionTask)
    assert parsed.form == "dog"
    assert len(parsed.contexts) == 2


def test_rewrite_task_roundtrip():
    task = CCRewriteTask(
        id="def",
        form="run",
        senses=[SenseInfo(id="s1", definition="to move quickly", pos="verb")],
    )
    data = task.model_dump_json()
    parsed = _task_adapter.validate_json(data)
    assert isinstance(parsed, CCRewriteTask)
    assert parsed.senses[0].definition == "to move quickly"


def test_trim_sense_task_roundtrip():
    task = CCTrimSenseTask(
        id="ghi",
        form="bank",
        senses=[
            SenseInfo(id="s1", definition="financial institution", pos="noun"),
            SenseInfo(id="s2", definition="side of a river", pos="noun"),
        ],
        examples=[["I went to the bank"], ["The river bank was muddy"]],
    )
    data = task.model_dump_json()
    parsed = _task_adapter.validate_json(data)
    assert isinstance(parsed, CCTrimSenseTask)
    assert len(parsed.senses) == 2
    assert len(parsed.examples) == 2


def test_morph_redirect_task_roundtrip():
    task = CCMorphRedirectTask(
        id="jkl",
        forms=[
            FormInfo(
                form="dogs",
                senses=[SenseInfo(id="s1", definition="plural of dog", pos="noun")],
            )
        ],
        inventory_forms=["dog", "dogs", "cat"],
    )
    data = task.model_dump_json()
    parsed = _task_adapter.validate_json(data)
    assert isinstance(parsed, CCMorphRedirectTask)
    assert parsed.forms[0].form == "dogs"
    assert "dog" in parsed.inventory_forms


def test_induction_output_roundtrip():
    output = CCInductionOutput(
        id="abc",
        form="dog",
        senses=[InductionSense(definition="a domestic animal", pos="noun")],
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionOutput)
    assert parsed.senses[0].pos == "noun"


def test_rewrite_output_roundtrip():
    output = CCRewriteOutput(
        id="def",
        form="run",
        senses=[RewrittenSense(definition="to move swiftly on foot", subsenses=None)],
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCRewriteOutput)


def test_trim_sense_output_roundtrip():
    output = CCTrimSenseOutput(
        id="ghi",
        form="bank",
        sense_num=None,
        reason="all senses distinct",
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCTrimSenseOutput)
    assert parsed.sense_num is None


def test_trim_sense_output_with_deletion():
    output = CCTrimSenseOutput(
        id="ghi",
        form="bank",
        sense_num=2,
        reason="redundant with sense 1",
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCTrimSenseOutput)
    assert parsed.sense_num == 2


def test_morph_redirect_output_roundtrip():
    output = CCMorphRedirectOutput(
        id="jkl",
        relations=[
            MorphRelation(
                derived_form="dogs",
                derived_sense_idx=0,
                base_form="dog",
                base_sense_idx=0,
                relation="plural",
                proposed_definition="plural of dog (n.)",
            )
        ],
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCMorphRedirectOutput)
    assert len(parsed.relations) == 1
    assert parsed.relations[0].relation == "plural"
