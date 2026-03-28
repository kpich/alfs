"""Tests for CC task and output models."""

from pydantic import TypeAdapter

from alfs.cc.models import (
    CCInductionOutput,
    CCInductionTask,
    CCOutput,
    CCTask,
    ContextLabel,
    InductionSense,
)
from alfs.data_models.occurrence import Occurrence

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
    assert parsed.occurrence_refs == []


def test_induction_task_with_occurrence_refs():
    task = CCInductionTask(
        id="abc",
        form="dog",
        contexts=["The dog barked.", "I walked the dog."],
        existing_defs=[],
        occurrence_refs=[
            Occurrence(doc_id="doc1", byte_offset=100),
            Occurrence(doc_id="doc2", byte_offset=200),
        ],
    )
    data = task.model_dump_json()
    parsed = _task_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionTask)
    assert len(parsed.occurrence_refs) == 2
    assert parsed.occurrence_refs[0].doc_id == "doc1"
    assert parsed.occurrence_refs[1].byte_offset == 200


def test_induction_output_roundtrip():
    output = CCInductionOutput(
        id="abc",
        form="dog",
        new_senses=[InductionSense(definition="a domestic animal", pos="noun")],
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionOutput)
    assert parsed.new_senses[0].pos == "noun"


def test_induction_output_new_fields_roundtrip():
    output = CCInductionOutput(
        id="abc",
        form="dog",
        new_senses=[InductionSense(definition="a domestic animal", pos="noun")],
        context_labels=[
            ContextLabel(context_idx=0, sense_idx=1),
            ContextLabel(context_idx=1, sense_idx=None),
        ],
        add_to_blocklist=False,
        blocklist_reason=None,
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionOutput)
    assert len(parsed.context_labels) == 2
    assert parsed.context_labels[0].sense_idx == 1
    assert parsed.context_labels[1].sense_idx is None
    assert parsed.add_to_blocklist is False


def test_induction_output_blocklist_case():
    output = CCInductionOutput(
        id="xyz",
        form="thrumbly",
        new_senses=[],
        add_to_blocklist=True,
        blocklist_reason="tokenization artifact",
    )
    data = output.model_dump_json()
    parsed = _output_adapter.validate_json(data)
    assert isinstance(parsed, CCInductionOutput)
    assert parsed.add_to_blocklist is True
    assert parsed.blocklist_reason == "tokenization artifact"
    assert parsed.new_senses == []


def test_induction_output_defaults():
    output = CCInductionOutput(id="abc", form="dog")
    assert output.new_senses == []
    assert output.context_labels == []
    assert output.add_to_blocklist is False
    assert output.blocklist_reason is None
