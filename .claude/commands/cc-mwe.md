You are a lexicographer reviewing multi-word expression (MWE) candidates. Process all pending MWE task files.

## Steps

1. Find all task files: use Glob with path `/Users/kpich/dev/alfs/cc_tasks/pending/mwe` and pattern `*.json`, then read each with the Read tool.

   **IMPORTANT: Never use Bash for file operations. Use Read to read files, Write to write files, and Bash only to delete files (`rm`).**

2. For each MWE task file, you will see:
   - `form`: the candidate MWE (e.g., "a priori", "take care", "well-known")
   - `components`: the individual spaCy tokens that make up the MWE (e.g., `["a", "priori"]`, `["well", "-", "known"]`)
   - `pmi`: pointwise mutual information score (higher = stronger association)
   - `corpus_count`: how many times this token sequence appears in the corpus
   - `contexts`: sample sentences containing the MWE
   - `occurrence_refs`: list of `{doc_id, byte_offset}` objects, parallel to `contexts`

3. For each candidate, decide one of three outcomes. **When you decide to approve**, also draft sense(s) for the MWE (see step 4). The contexts you are already reading are sufficient — no extra lookups needed.

   **A. Approve** — this is a genuine multi-word expression that deserves its own dictionary entry. Criteria:
   - The combination has a meaning that is **not compositionally derivable** from its parts (e.g., "kick the bucket" ≠ kick + bucket).
   - OR it is a **fixed phrase** with a specific conventional meaning (e.g., "a priori", "ad hoc", "vis-à-vis").
   - OR it is a **light verb construction** where the verb is semantically bleached and the meaning resides in the combination (e.g., "take care", "make do", "give way").
   - OR it is a **contraction** that was split by the tokenizer (e.g., "won't" split into "wo" + "n't").
   - OR it is a **hyphenated compound** with a meaning distinct from its unhyphenated parts (e.g., "well-known", "self-aware").
   - OR it is a **well-known proper name** — a notable person (e.g., "Martin Luther", "Barack Obama"), place (e.g., "New York", "Nibong Tebal"), organization, or publication title that a general English dictionary might include.
   - OR it is a **non-English phrase used in English** — a foreign-language expression that appears in English-language texts as a borrowing or citation (e.g., "Popol Vuh", "Corpus Juris Canonici", "Rodong Sinmun").

   **B. Skip** — the combination is NOT a genuine MWE, and it is unlikely to reappear (e.g., a one-off compositional phrase, a common noun-verb collocation like "the president"). High PMI alone is not sufficient — the combination must have a non-compositional, conventionalized, or encyclopedic meaning. **Do not use Skip for things that will reappear** — use Blocklist instead.

   **C. Blocklist** — not an MWE, and it will keep showing up as a false positive. Use this for:
   - Corpus-specific artifacts or boilerplate phrases (e.g., database column headers, MIME type strings, usernames)
   - **Individual journalists' bylines and photo credits** (e.g., Reuters/AP photographer or reporter names) — these recur systematically from news corpus structure and will always be false positives
   - Obscure proper names (individual bylines, minor fictional characters) that would not warrant a dictionary entry and will likely recur due to corpus repetition
   - Any proper name that appears repeatedly only because a single source document repeats it

4. **For approved MWEs, draft senses** based on the contexts you just read:
   - Write 1–3 senses covering the distinct meanings shown in the contexts.
   - Each sense needs a `definition` (concise, lexicographic style) and a `pos` (use: noun, verb, adjective, adverb, phrase, prefix, suffix, abbreviation, interjection, conjunction, preposition, determiner, pronoun, numeral, proper noun).
   - For each context, assign it to one of the new senses via `context_labels` using 0-indexed `context_idx` and 1-indexed `sense_idx` (matching the position in `new_senses`). If a context doesn't clearly illustrate any sense, omit it from `context_labels`.
   - Proper nouns (people, places, organizations) typically get a single sense: a brief identifying description (e.g., "Capital city of France.").

5. Write the output JSON to `/Users/kpich/dev/alfs/cc_tasks/done/mwe/<same_filename>` with this schema:

   For **approve**:
   ```json
   {
     "type": "mwe",
     "id": "<same id from task>",
     "form": "<canonical form — use proper casing as appropriate>",
     "action": "approve",
     "blocklist_reason": null,
     "occurrence_refs": [<copy the occurrence_refs array verbatim from the task file>],
     "new_senses": [
       {"definition": "<sense definition>", "pos": "<pos tag>"}
     ],
     "context_labels": [
       {"context_idx": 0, "sense_idx": 1},
       {"context_idx": 1, "sense_idx": 1}
     ]
   }
   ```

   For **skip** — write a done file, then delete the pending file (no senses needed):
   ```json
   {
     "type": "mwe",
     "id": "<same id from task>",
     "form": "<form from task>",
     "action": "skip",
     "blocklist_reason": null
   }
   ```
   Then delete the pending file.

   For **blocklist**:
   ```json
   {
     "type": "mwe",
     "id": "<same id from task>",
     "form": "<form from task>",
     "action": "blocklist",
     "blocklist_reason": "corpus artifact"
   }
   ```

   Then delete the pending file.

6. **Casing rules for approved MWEs:**
   - Fixed foreign phrases: use conventional casing (e.g., "a priori", "ad hoc" — lowercase)
   - Proper noun phrases: capitalize as conventionally used (e.g., "New York", "World War")
   - Contractions: use the conventional written form (e.g., "won't", "I'll", "can't")
   - Hyphenated compounds: use conventional casing (e.g., "well-known", "self-aware")
   - Light verb constructions: lowercase (e.g., "take care", "give way")

Process ALL matching pending files, not just the first one.
