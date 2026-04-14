You are a lexicographer reviewing multi-word expression (MWE) candidates. Process all pending MWE task files.

## Steps

1. Find all task files: use Glob with path `../cc_tasks/pending/mwe` and pattern `*.json`, then read each with the Read tool.

   **IMPORTANT: Never use Bash for file operations. Use Read to read files, Write to write files, and Bash only to delete files (`rm`).**

2. For each MWE task file, you will see:
   - `form`: the candidate MWE (e.g., "a priori", "take care", "well-known")
   - `components`: the individual spaCy tokens that make up the MWE (e.g., `["a", "priori"]`, `["well", "-", "known"]`)
   - `pmi`: pointwise mutual information score (higher = stronger association)
   - `corpus_count`: how many times this token sequence appears in the corpus
   - `contexts`: sample sentences containing the MWE
   - `occurrence_refs`: list of `{doc_id, byte_offset}` objects, parallel to `contexts`

3. For each candidate, decide one of three outcomes:

   **A. Approve** — this is a genuine multi-word expression that deserves its own dictionary entry. Criteria:
   - The combination has a meaning that is **not compositionally derivable** from its parts (e.g., "kick the bucket" ≠ kick + bucket).
   - OR it is a **fixed phrase** with a specific conventional meaning (e.g., "a priori", "ad hoc", "vis-à-vis").
   - OR it is a **light verb construction** where the verb is semantically bleached and the meaning resides in the combination (e.g., "take care", "make do", "give way").
   - OR it is a **contraction** that was split by the tokenizer (e.g., "won't" split into "wo" + "n't").
   - OR it is a **hyphenated compound** with a meaning distinct from its unhyphenated parts (e.g., "well-known", "self-aware").

   **B. Skip** — the combination is NOT a genuine MWE. It just happens to co-occur frequently due to topic or syntactic patterns (e.g., "the president", "United States", common noun-verb collocations). High PMI alone is not sufficient — the combination must have a non-compositional or conventionalized meaning.

   **C. Blocklist** — not an MWE, and it keeps showing up as a false positive. Use this for combinations that will never be MWEs but persistently have high PMI (e.g., corpus-specific artifacts, boilerplate phrases).

4. Write the output JSON to `../cc_tasks/done/mwe/<same_filename>` with this schema:

   For **approve**:
   ```json
   {
     "type": "mwe",
     "id": "<same id from task>",
     "form": "<canonical form — use proper casing as appropriate>",
     "action": "approve",
     "blocklist_reason": null
   }
   ```

   For **skip** — just delete the pending file and move to the next task. Do NOT write an output file.

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

5. **Casing rules for approved MWEs:**
   - Fixed foreign phrases: use conventional casing (e.g., "a priori", "ad hoc" — lowercase)
   - Proper noun phrases: capitalize as conventionally used (e.g., "New York", "World War")
   - Contractions: use the conventional written form (e.g., "won't", "I'll", "can't")
   - Hyphenated compounds: use conventional casing (e.g., "well-known", "self-aware")
   - Light verb constructions: lowercase (e.g., "take care", "give way")

Process ALL matching pending files, not just the first one.
