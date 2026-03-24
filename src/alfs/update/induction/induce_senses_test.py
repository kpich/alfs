import polars as pl

from alfs.data_models.alf import Alf, Sense
from alfs.data_models.sense_store import SenseStore
from alfs.data_models.update_target import UpdateTarget
from alfs.update.induction import induce_senses


def test_non_ascii_form_uses_other_prefix(tmp_path, monkeypatch):
    """Non-ASCII form 'ſendiþ' must look up other/occurrences.parquet, not ſ/."""
    form = "ſendiþ"
    doc_id = "doc001"
    byte_offset = 10

    # Build other/occurrences.parquet (where aggregate_occurrences puts it)
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    occ_df = pl.DataFrame(
        {"form": [form], "doc_id": [doc_id], "byte_offset": [byte_offset]}
    )
    occ_df.write_parquet(other_dir / "occurrences.parquet")

    # Build docs.parquet with matching text
    text = "some prefix " + form + " some suffix"
    docs_df = pl.DataFrame({"doc_id": [doc_id], "text": [text]})
    docs_path = tmp_path / "docs.parquet"
    docs_df.write_parquet(docs_path)

    # Write target.json
    target_path = tmp_path / "target.json"
    target = UpdateTarget(form=form)
    target_path.write_text(target.model_dump_json())

    output_path = tmp_path / "output.json"

    # Monkeypatch LLM to return a canned sense
    canned_response = {
        "all_covered": False,
        "senses": [{"definition": "to send", "examples": [1], "pos": "verb"}],
    }
    canned_verdict = {"is_valid": True, "reason": ""}

    call_count = {"n": 0}

    def fake_chat_json(model: str, prompt: str, format: object = None) -> object:
        n = call_count["n"]
        call_count["n"] += 1
        if n == 0:
            return canned_response
        return canned_verdict

    monkeypatch.setattr(
        "alfs.update.induction.induce_senses.llm.chat_json", fake_chat_json
    )

    induce_senses.run(
        target_file=target_path,
        seg_data_dir=tmp_path,
        docs=docs_path,
        output=output_path,
    )

    alf = Alf.model_validate_json(output_path.read_text())
    assert len(alf.senses) == 1
    assert alf.senses[0].definition == "to send"


def _setup_occurrences(tmp_path, form: str) -> tuple:
    """Write minimal occurrences + docs for a form; return (docs_path, target_path,
    output_path)."""
    prefix_dir = tmp_path / form[0]
    prefix_dir.mkdir(parents=True, exist_ok=True)
    occ_df = pl.DataFrame({"form": [form], "doc_id": ["d1"], "byte_offset": [5]})
    occ_df.write_parquet(prefix_dir / "occurrences.parquet")
    text = "     " + form + " suffix"
    docs_df = pl.DataFrame({"doc_id": ["d1"], "text": [text]})
    docs_path = tmp_path / "docs.parquet"
    docs_df.write_parquet(docs_path)
    target_path = tmp_path / "target.json"
    target_path.write_text(UpdateTarget(form=form).model_dump_json())
    output_path = tmp_path / "output.json"
    return docs_path, target_path, output_path


def test_redirect_form_sees_canonical_senses_as_existing(tmp_path, monkeypatch):
    """When inducting a redirect form, existing_defs should include the canonical form's
    senses."""
    senses_db = tmp_path / "senses.db"
    store = SenseStore(senses_db)
    dog = Alf(form="dog", senses=[Sense(definition="a domesticated animal")])
    dogs = Alf(form="dogs", senses=[], redirect="dog")
    store.write(dog)
    store.write(dogs)

    docs_path, target_path, output_path = _setup_occurrences(tmp_path, "dogs")

    prompts_seen: list[str] = []

    def fake_chat_json(model: str, prompt: str, format: object = None) -> object:
        prompts_seen.append(prompt)
        return {"all_covered": True, "senses": []}

    monkeypatch.setattr(
        "alfs.update.induction.induce_senses.llm.chat_json", fake_chat_json
    )

    induce_senses.run(
        target_file=target_path,
        seg_data_dir=tmp_path,
        docs=docs_path,
        output=output_path,
        senses_db=senses_db,
    )

    assert prompts_seen, "LLM was not called"
    assert "a domesticated animal" in prompts_seen[0]
