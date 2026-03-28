"""Induce senses for a word form using Ollama, or write a CC task file.

Usage (single target):
    python -m alfs.update.induction.induce_senses \\
        --target target.json --seg-data-dir by_prefix/ --docs docs.parquet \\
        --output senses.json --model qwen2.5:32b --context-chars 150 --max-samples 20 \\
        [--senses-db senses.db] [--labeled-db labeled.db] [--cc-tasks-dir DIR]

Usage (queue mode):
    python -m alfs.update.induction.induce_senses \\
        --queue-file ../alfs_data/induction_queue.yaml \\
        --blocklist-file ../alfs_data/blocklist.yaml \\
        --seg-data-dir by_prefix/ --docs docs.parquet \\
        --senses-db senses.db --labeled-db labeled.db \\
        --queue-dir ../clerk_queue \\
        [--model qwen2.5:32b] [--cc-tasks-dir DIR]
"""

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path
import random
import uuid

import polars as pl

from alfs.clerk.queue import enqueue
from alfs.clerk.request import AddSensesRequest
from alfs.data_models.alf import Alf, Sense, morph_base_form
from alfs.data_models.blocklist import Blocklist
from alfs.data_models.induction_queue import InductionQueue, InductionQueueEntry
from alfs.data_models.occurrence import Occurrence
from alfs.data_models.occurrence_store import OccurrenceStore
from alfs.data_models.pos import PartOfSpeech
from alfs.data_models.sense_store import SenseStore
from alfs.data_models.update_target import UpdateTarget
from alfs.encoding import context_window as _context_window
from alfs.seg.aggregate_occurrences import prefix as form_prefix
from alfs.update import llm
from alfs.update.induction import prompts

_SKIP_SENSE_KEY = "_skip"

_SENSE_SCHEMA = {
    "type": "object",
    "properties": {
        "all_covered": {"type": "boolean"},
        "add_to_blocklist": {"type": "boolean"},
        "blocklist_reason": {"type": "string"},
        "senses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "definition": {"type": "string"},
                    "examples": {"type": "array", "items": {"type": "integer"}},
                    "pos": {"type": "string", "enum": [p.value for p in PartOfSpeech]},
                },
                "required": ["definition", "examples", "pos"],
            },
        },
    },
    "required": ["senses"],
}

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["is_valid", "reason"],
}


def extract_context(text: str, byte_offset: int, form: str, context_chars: int) -> str:
    snippet, _ = _context_window(text, byte_offset, form, context_chars)
    return snippet


def _load_existing_defs(form: str, senses_db: Path) -> list[str]:
    """Load existing definitions for a form from senses.db."""
    store = SenseStore(senses_db)
    entry = store.read(form)
    if not entry:
        return []
    resolved = entry
    if entry.redirect:
        canonical = store.read(entry.redirect)
        if canonical:
            resolved = canonical
    existing_defs = [s.definition for s in resolved.senses]
    base_name = morph_base_form(resolved)
    if base_name is not None:
        base_entry = store.read(base_name)
        if base_entry is not None:
            existing_defs.extend(s.definition for s in base_entry.senses)
    return existing_defs


def _load_contexts(
    form: str,
    seg_data_dir: Path,
    docs: Path,
    pinned_occurrences: list[Occurrence],
    labeled_db: Path | None,
    max_samples: int,
    context_chars: int,
) -> tuple[list[str], list[Occurrence]]:
    """Extract context strings for a form. Returns (contexts, occurrence_refs).

    If pinned_occurrences is non-empty, use those. Otherwise sample from corpus,
    excluding well-labeled occurrences.
    """
    occ_path = seg_data_dir / form_prefix(form) / "occurrences.parquet"
    if not occ_path.exists():
        return [], []

    df = pl.read_parquet(str(occ_path)).filter(pl.col("form") == form)
    all_occurrences = list(df.select(["doc_id", "byte_offset"]).iter_rows(named=True))

    if pinned_occurrences:
        # Use the pinned occurrences from the queue entry
        samples_raw = [
            {"doc_id": o.doc_id, "byte_offset": o.byte_offset}
            for o in pinned_occurrences
        ]
        # Only keep pinned occurrences that actually exist in the parquet
        existing_set = {(r["doc_id"], r["byte_offset"]) for r in all_occurrences}
        samples_raw = [
            r for r in samples_raw if (r["doc_id"], r["byte_offset"]) in existing_set
        ]
        if not samples_raw:
            # Fall back to free sampling if none of the pinned ones exist
            samples_raw = random.sample(
                all_occurrences, min(max_samples, len(all_occurrences))
            )
    else:
        well_labeled: set[tuple[str, int]] = set()
        if labeled_db and labeled_db.exists():
            occ_store = OccurrenceStore(labeled_db)
            ldf = occ_store.query_form(form).filter(pl.col("rating").is_in([2]))
            for row in ldf.select(["doc_id", "byte_offset"]).iter_rows():
                well_labeled.add((row[0], row[1]))
        candidates = [
            o
            for o in all_occurrences
            if (o["doc_id"], o["byte_offset"]) not in well_labeled
        ]
        random.seed(42)
        samples_raw = random.sample(candidates, min(max_samples, len(candidates)))

    needed_doc_ids = list({o["doc_id"] for o in samples_raw})
    docs_df = (
        pl.scan_parquet(str(docs))
        .filter(pl.col("doc_id").is_in(needed_doc_ids))
        .collect(engine="streaming")
    )
    docs_map = dict(
        zip(docs_df["doc_id"].to_list(), docs_df["text"].to_list(), strict=False)
    )

    contexts = []
    occurrence_refs = []
    for occ in samples_raw:
        doc_id: str = occ["doc_id"]  # type: ignore[assignment]
        byte_offset: int = occ["byte_offset"]  # type: ignore[assignment]
        text = docs_map.get(doc_id, "")
        if not text:
            continue
        ctx = extract_context(text, byte_offset, form, context_chars)
        contexts.append(ctx)
        occurrence_refs.append(Occurrence(doc_id=doc_id, byte_offset=byte_offset))

    return contexts, occurrence_refs


def _apply_context_labels(
    form: str,
    occurrence_refs: list[Occurrence],
    proposed_senses: list[dict],  # raw LLM output with 'examples' field
    accepted_indices: list[int],  # which proposed sense indices (0-based) were accepted
    labeled_db: Path,
    model: str,
) -> None:
    """Write occurrence labels to labeled.db based on LLM examples field.

    Contexts claimed by an accepted sense → sense_key=str(accepted_position+1),
    rating=2.
    Contexts not claimed by any accepted sense → no label written.
    """
    if not occurrence_refs:
        return

    # Build map: context_idx (0-based) → accepted_position (0-based) or None
    accepted_set = set(accepted_indices)
    # Map from proposed_idx (0-based) → accepted_position (0-based)
    proposed_to_accepted: dict[int, int] = {}
    accepted_pos = 0
    for i in range(len(proposed_senses)):
        if i in accepted_set:
            proposed_to_accepted[i] = accepted_pos
            accepted_pos += 1

    # Map context_idx (1-based in examples) → final sense_key
    context_to_sense: dict[int, str] = {}  # 0-based context_idx → sense_key
    for proposed_idx, item in enumerate(proposed_senses):
        if proposed_idx not in proposed_to_accepted:
            continue
        final_pos = proposed_to_accepted[proposed_idx]
        sense_key = str(final_pos + 1)
        for ex_1based in item.get("examples", []):
            ctx_idx = ex_1based - 1  # convert to 0-based
            if 0 <= ctx_idx < len(occurrence_refs):
                context_to_sense[ctx_idx] = sense_key

    if not context_to_sense:
        return

    occ_store = OccurrenceStore(labeled_db)
    rows = []
    for ctx_idx, sense_key in context_to_sense.items():
        occ = occurrence_refs[ctx_idx]
        # (form, doc_id, byte_offset, sense_key, rating, synonyms)
        rows.append((form, occ.doc_id, occ.byte_offset, sense_key, 2, None))
    occ_store.upsert_many(rows, model)
    print(f"  Labeled {len(rows)} occurrence(s) for '{form}'")


def _induce_one(
    entry: InductionQueueEntry,
    seg_data_dir: Path,
    docs: Path,
    senses_db: Path | None,
    labeled_db: Path | None,
    clerk_queue_dir: Path,
    blocklist: Blocklist,
    model: str,
    context_chars: int,
    max_samples: int,
    cc_tasks_dir: Path | None,
) -> None:
    """Run induction for a single queued entry."""
    form = entry.form

    existing_defs: list[str] = []
    if senses_db and senses_db.exists():
        existing_defs = _load_existing_defs(form, senses_db)

    contexts, occurrence_refs = _load_contexts(
        form,
        seg_data_dir,
        docs,
        entry.occurrences,
        labeled_db,
        max_samples,
        context_chars,
    )

    if not contexts:
        print(f"No contexts found for '{form}'; skipping.")
        return

    if cc_tasks_dir:
        from alfs.cc.models import CCInductionTask

        task = CCInductionTask(
            id=str(uuid.uuid4()),
            form=form,
            contexts=contexts,
            existing_defs=existing_defs,
            occurrence_refs=occurrence_refs,
        )
        pending_dir = cc_tasks_dir / "pending" / "induction"
        pending_dir.mkdir(parents=True, exist_ok=True)
        task_path = pending_dir / f"{task.id}.json"
        task_path.write_text(task.model_dump_json())
        print(f"Wrote CC task for '{form}' to {task_path}")
        return

    prompt = prompts.induction_prompt(form, contexts, existing_defs)
    data = llm.chat_json(model, prompt, format=_SENSE_SCHEMA)

    # Handle blocklist decision
    if data.get("add_to_blocklist", False):
        reason = data.get("blocklist_reason") or None
        blocklist.add(form, reason)
        print(f"  Blocklisted '{form}': {reason}")
        if labeled_db and labeled_db.exists():
            OccurrenceStore(labeled_db).delete_by_form(form)
            print(f"  Deleted labeled occurrences for '{form}'")
        return

    if data.get("all_covered", False) and existing_defs:
        print(f"Existing senses cover all contexts for '{form}'; no new sense added.")
        return

    proposed = data.get("senses", [])
    accepted: list[Sense] = []
    accepted_indices: list[int] = []
    for i, item in enumerate(proposed):
        pos_str = item.get("pos")
        try:
            pos = PartOfSpeech(pos_str) if pos_str else None
        except ValueError:
            pos = None
        sense = Sense(
            definition=item["definition"],
            pos=pos,
            updated_by_model=model,
        )
        verdict = llm.chat_json(
            model,
            prompts.induction_critic_prompt(
                form,
                sense.definition,
                existing_defs + [s.definition for s in accepted],
            ),
            format=_CRITIC_SCHEMA,
        )
        if verdict.get("is_valid", True):
            accepted.append(sense)
            accepted_indices.append(i)
        else:
            print(f"  critic rejected '{form}': {verdict.get('reason', '')}")

    if accepted:
        request = AddSensesRequest(
            id=str(uuid.uuid4()),
            created_at=datetime.now(UTC),
            form=form,
            new_senses=accepted,
        )
        enqueue(request, clerk_queue_dir)
        print(f"  Queued {len(accepted)} sense(s) for '{form}'")

        # Label occurrences based on the LLM's examples assignments
        if labeled_db and labeled_db.exists() and occurrence_refs:
            _apply_context_labels(
                form,
                occurrence_refs,
                proposed,
                accepted_indices,
                labeled_db,
                model,
            )
    else:
        print(f"No accepted senses for '{form}'")


def run_from_queue(
    queue_file: str | Path,
    blocklist_file: str | Path,
    seg_data_dir: str | Path,
    docs: str | Path,
    senses_db: str | Path | None,
    labeled_db: str | Path | None,
    clerk_queue_dir: str | Path,
    model: str = "qwen2.5:32b",
    context_chars: int = 150,
    max_samples: int = 20,
    cc_tasks_dir: str | Path | None = None,
) -> None:
    """Dequeue all entries and run induction for each."""
    queue = InductionQueue(Path(queue_file))
    entries = queue.dequeue_all()
    if not entries:
        print("Induction queue is empty.")
        return

    print(f"Processing {len(entries)} queued form(s)...")
    blocklist = Blocklist(Path(blocklist_file))

    for entry in entries:
        print(f"\n--- {entry.form} ---")
        _induce_one(
            entry=entry,
            seg_data_dir=Path(seg_data_dir),
            docs=Path(docs),
            senses_db=Path(senses_db) if senses_db else None,
            labeled_db=Path(labeled_db) if labeled_db else None,
            clerk_queue_dir=Path(clerk_queue_dir),
            blocklist=blocklist,
            model=model,
            context_chars=context_chars,
            max_samples=max_samples,
            cc_tasks_dir=Path(cc_tasks_dir) if cc_tasks_dir else None,
        )


def run(
    target_file: str | Path,
    seg_data_dir: str | Path,
    docs: str | Path,
    output: str | Path,
    model: str = "qwen2.5:32b",
    context_chars: int = 150,
    max_samples: int = 20,
    senses_db: str | Path | None = None,
    labeled_db: str | Path | None = None,
    cc_tasks_dir: str | Path | None = None,
) -> None:
    """Original single-target induction mode (kept for backward compatibility)."""
    target = UpdateTarget.model_validate_json(Path(target_file).read_text())
    form = target.form

    existing_defs: list[str] = []
    if senses_db and Path(senses_db).exists():
        existing_defs = _load_existing_defs(form, Path(senses_db))

    occ_path = Path(seg_data_dir) / form_prefix(form) / "occurrences.parquet"
    if not occ_path.exists():
        alf = Alf(form=form, senses=[])
        Path(output).write_text(alf.model_dump_json())
        print(f"No occurrences parquet for '{form}' ({occ_path}); skipping.")
        return
    df = pl.read_parquet(str(occ_path)).filter(pl.col("form") == form)
    all_occurrences = list(df.select(["doc_id", "byte_offset"]).iter_rows(named=True))

    well_labeled: set[tuple[str, int]] = set()
    if labeled_db and Path(labeled_db).exists():
        occ_store = OccurrenceStore(Path(labeled_db))
        ldf = occ_store.query_form(form).filter(pl.col("rating").is_in([2]))
        for row in ldf.select(["doc_id", "byte_offset"]).iter_rows():
            well_labeled.add((row[0], row[1]))
    all_occurrences = [
        o
        for o in all_occurrences
        if (o["doc_id"], o["byte_offset"]) not in well_labeled
    ]

    random.seed(42)
    samples = random.sample(all_occurrences, min(max_samples, len(all_occurrences)))

    needed_doc_ids = list({occ["doc_id"] for occ in samples})
    docs_df = (
        pl.scan_parquet(str(docs))
        .filter(pl.col("doc_id").is_in(needed_doc_ids))
        .collect(engine="streaming")
    )
    docs_map = dict(
        zip(docs_df["doc_id"].to_list(), docs_df["text"].to_list(), strict=False)
    )

    contexts = []
    for occ in samples:
        text = docs_map.get(occ["doc_id"], "")
        if not text:
            continue
        ctx = extract_context(text, occ["byte_offset"], form, context_chars)
        contexts.append(ctx)

    if not contexts:
        alf = Alf(form=form, senses=[])
        Path(output).write_text(alf.model_dump_json())
        # TODO: investigate why doc_ids from occurrences parquet are absent from
        # docs.parquet for this form — this likely indicates a data pipeline bug
        print(f"No contexts found for '{form}'; skipping.")
        return

    if cc_tasks_dir:
        from alfs.cc.models import CCInductionTask

        occurrence_refs = [
            Occurrence(doc_id=s["doc_id"], byte_offset=s["byte_offset"])
            for s in samples
            if s["doc_id"] in docs_map
        ]
        task = CCInductionTask(
            id=str(uuid.uuid4()),
            form=form,
            contexts=contexts,
            existing_defs=existing_defs,
            occurrence_refs=occurrence_refs,
        )
        pending_dir = Path(cc_tasks_dir) / "pending" / "induction"
        pending_dir.mkdir(parents=True, exist_ok=True)
        task_path = pending_dir / f"{task.id}.json"
        task_path.write_text(task.model_dump_json())
        # Write empty output so downstream pipeline doesn't fail
        alf = Alf(form=form, senses=[])
        Path(output).write_text(alf.model_dump_json())
        print(f"Wrote CC task for '{form}' to {task_path}")
        return

    prompt = prompts.induction_prompt(form, contexts, existing_defs)
    data = llm.chat_json(model, prompt, format=_SENSE_SCHEMA)

    if data.get("all_covered", False) and existing_defs:
        alf = Alf(form=form, senses=[])
        Path(output).write_text(alf.model_dump_json())
        print(f"Existing senses cover all contexts for '{form}'; no new sense added.")
        return

    proposed = data.get("senses", [])
    accepted: list[Sense] = []
    for item in proposed:
        pos_str = item.get("pos")
        try:
            pos = PartOfSpeech(pos_str) if pos_str else None
        except ValueError:
            pos = None
        sense = Sense(
            definition=item["definition"],
            pos=pos,
            updated_by_model=model,
        )
        verdict = llm.chat_json(
            model,
            prompts.induction_critic_prompt(
                form,
                sense.definition,
                existing_defs + [s.definition for s in accepted],
            ),
            format=_CRITIC_SCHEMA,
        )
        if verdict.get("is_valid", True):
            accepted.append(sense)
        else:
            print(f"  critic rejected '{form}': {verdict.get('reason', '')}")

    alf = Alf(form=form, senses=accepted)
    Path(output).write_text(alf.model_dump_json())
    print(f"Wrote {len(accepted)} sense(s) for '{form}' to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Induce senses for a word form")

    # Queue mode args
    parser.add_argument(
        "--queue-file", default=None, help="Path to induction_queue.yaml (queue mode)"
    )
    parser.add_argument("--blocklist-file", default=None, help="Path to blocklist.yaml")
    parser.add_argument(
        "--queue-dir", default=None, help="Path to clerk queue directory (queue mode)"
    )

    # Single-target mode args
    parser.add_argument(
        "--target", default=None, help="Path to target.json (single-target mode)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for senses.json (single-target mode)",
    )

    # Shared args
    parser.add_argument(
        "--seg-data-dir", required=True, help="Path to by_prefix/ directory"
    )
    parser.add_argument("--docs", required=True, help="Path to docs.parquet")
    parser.add_argument("--model", default="qwen2.5:32b")
    parser.add_argument("--context-chars", type=int, default=150)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument(
        "--senses-db", default=None, help="Path to senses.db (optional)"
    )
    parser.add_argument(
        "--labeled-db", default=None, help="Path to labeled.db (optional)"
    )
    parser.add_argument(
        "--cc-tasks-dir",
        default=None,
        help="Path to CC tasks directory (writes task file instead of calling LLM)",
    )
    args = parser.parse_args()

    cc_tasks_dir = args.cc_tasks_dir or os.environ.get("CC_TASKS_DIR")

    if args.queue_file:
        if not args.queue_dir:
            parser.error("--queue-dir is required in queue mode")
        if not args.blocklist_file:
            parser.error("--blocklist-file is required in queue mode")
        run_from_queue(
            args.queue_file,
            args.blocklist_file,
            args.seg_data_dir,
            args.docs,
            args.senses_db,
            args.labeled_db,
            args.queue_dir,
            args.model,
            args.context_chars,
            args.max_samples,
            cc_tasks_dir,
        )
    else:
        if not args.target:
            parser.error("--target is required in single-target mode")
        if not args.output:
            parser.error("--output is required in single-target mode")
        run(
            args.target,
            args.seg_data_dir,
            args.docs,
            args.output,
            args.model,
            args.context_chars,
            args.max_samples,
            args.senses_db,
            args.labeled_db,
            cc_tasks_dir,
        )


if __name__ == "__main__":
    main()
