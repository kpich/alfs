import polars as pl

from alfs.data_models.alf import Alf
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

    def fake_chat_json(model, prompt, format=None):
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
