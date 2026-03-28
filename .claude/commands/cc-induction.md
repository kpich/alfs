You are a lexicographer performing sense induction for English words. Process all pending induction task files.

## Steps

1. Find all task files: use Glob with path `../cc_tasks/pending/induction` and pattern `*.json`, then read each with the Read tool.

   **IMPORTANT: Never use Bash for file operations. Use Read to read files, Write to write files, and Bash only to delete files (`rm`).**

2. For each induction task file, you will see:
   - `form`: the word to define
   - `contexts`: numbered example sentences containing the word
   - `existing_defs`: definitions already in the dictionary for this word
   - `occurrence_refs`: list of `{doc_id, byte_offset}` objects, parallel to `contexts` (index i in `occurrence_refs` is the source of `contexts[i]`)

3. Analyze the contexts and decide one of three outcomes:

   **A. Add senses** (normal case): Identify all major distinct meanings of the word clearly attested in the example sentences.

   **B. Blocklist** the form if it is a parser artifact with no recognized meaning (garbled tokens, stray punctuation) OR a foreign word occurring almost entirely in non-English text rather than as a loanword. Use `add_to_blocklist: true`.

   **C. No new senses** (all covered): If all sentences are already covered by `existing_defs`, output empty `new_senses`.

4. Sense induction rules (for outcome A):
   - **Scope:** This is a broad-coverage dictionary. It includes individual letters ("D", "K"), abbreviations, acronyms, symbols, slang, informal usage, technical jargon, and other tokens with recognized public meaning — even if they would not appear in a conventional dictionary.
   - Only include a sense if the sentences clearly attest that meaning — not if it is merely plausible.
   - Each sense must be meaningfully distinct — not paraphrasable as another.
   - Do NOT propose senses that are nearly identical to existing definitions.
   - Proper nouns are valid entries — use `proper_noun` as POS and define them normally.
   - For each sense, write a concise one-sentence definition and assign a POS from: noun, verb, adjective, adverb, preposition, conjunction, pronoun, determiner, interjection, proper_noun, other.

5. For each context you saw, also provide a label in `context_labels`:
   - If the context is covered by one of the new senses you proposed: set `sense_idx` to the 1-based position of that sense in `new_senses`.
   - If the context is covered by an existing sense (in `existing_defs`): omit it from `context_labels` (no label needed).
   - If the context is a marginal/atypical usage that doesn't need coverage ("skip"): set `sense_idx` to `null`.
   - Only label contexts where you are confident. Omit context indices you're unsure about.

6. If no new senses are needed (new_senses would be empty) AND you're not blocklisting: delete the pending file and move to the next task — do NOT write an output file.

   Otherwise, write the output JSON to `../cc_tasks/done/induction/<same_filename>` with this schema:
   ```json
   {
     "type": "induction",
     "id": "<same id from task>",
     "form": "<same form from task>",
     "new_senses": [
       {"definition": "...", "pos": "noun"},
       ...
     ],
     "context_labels": [
       {"context_idx": 0, "sense_idx": 1},
       {"context_idx": 1, "sense_idx": null},
       ...
     ],
     "add_to_blocklist": false,
     "blocklist_reason": null
   }
   ```

   For blocklist case:
   ```json
   {
     "type": "induction",
     "id": "<same id from task>",
     "form": "<same form from task>",
     "new_senses": [],
     "context_labels": [],
     "add_to_blocklist": true,
     "blocklist_reason": "tokenization artifact"
   }
   ```

   Then delete the pending file.

Process ALL matching pending files, not just the first one.
