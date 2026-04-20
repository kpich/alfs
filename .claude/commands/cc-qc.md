You are a lexicographer performing quality control on dictionary entries. For each entry, make all applicable improvements in a single pass. Process all pending qc task files.

## Important: Tool Use

**NEVER use Bash to read or write files.** Use the `Read` tool to read, `Write` tool to write output files, and `Bash` only to delete files (`rm`). Do not use `cat`, shell pipelines, or any other Bash file operations — these require repeated approval prompts.

## Steps

1. Find all task files: use `Glob` with `path="/Users/kpich/dev/alfs/cc_tasks/pending/qc"` and `pattern="*.json"`, then read each with the `Read` tool.

2. For each task file you will see:
   - `form`: the wordform under review
   - `senses`: list of senses, each with `id`, `definition`, `pos` (may be null)

3. For each form, apply **at most one entry-level action** OR **any combination of sense-level actions**. These two tiers cannot be mixed.

   ---

   ### Entry-level actions (pick at most one; precludes all sense-level changes)

   #### A. delete_entry — remove entirely and blocklist
   The form should be removed and added to the blocklist. Applies to:
   - Artifact forms: stray punctuation, markup fragments, tokenization errors (e.g. `5*!`, `--`, `</p>`)
   - Foreign words with no established use as English loanwords
   - Truly one-off proper names with no shared semantics (throwaway usernames, serial numbers, internal code names)

   **This dictionary is intentionally expansive.** Keep slang, technical jargon, neologisms, regional terms, archaic forms. Only delete forms that are genuinely not English by any standard.

   Set `delete_entry: true`, `delete_entry_reason: "brief reason"`.

   #### B. normalize_case — redirect case variant to canonical form
   A form whose only distinction from a more canonical form is capitalization. Decide the canonical casing (proper names → capitalized; common words → lowercase). Example: `"aaron"` → `"Aaron"`, `"Dogs"` → `"dogs"`.

   Do NOT normalize if the case variant carries a meaning absent from the canonical form (e.g. `His` as a divine pronoun, `POTS` as an acronym). When in doubt, use no action.

   Set `normalize_case: "<canonical_form>"`.

   #### C. spelling_variant_of — mark as spelling variant
   The form is the same word spelled differently (e.g. British vs. American English): `colour` → `color`, `analyse` → `analyze`. Only use this when certain the two spellings represent the same word with the same meaning — do not use it when the forms have distinct meanings (e.g. `metre` vs. `meter`).

   Set `spelling_variant_of: "<preferred_form>"`.

   ---

   ### Sense-level actions (combine freely; omit fields for senses needing no change)

   #### D. morph_rels — inflectional morphology
   The form is a regular inflection of another word. Qualifying categories:
   - Plural forms: `dogs` → `dog`, `boxes` → `box`
   - Verbal inflections: `walked` → `walk`, `running` → `run`, `goes` → `go`
   - Comparative/superlative: `faster` → `fast`, `best` → `good`

   Do NOT include derivational morphology (`happiness` ← `happy`), pronoun case changes (`him` → `he`), or semantic relations only.

   Only list senses that qualify; unlisted senses stay untouched. For each qualifying sense provide:
   - `sense_idx`: 0-based index into the senses list
   - `morph_base`: the base word (e.g. `"dog"`)
   - `morph_relation`: short label (e.g. `"plural"`, `"past tense"`, `"comparative"`)
   - `proposed_definition`: concise reference string (e.g. `"plural of dog (n.)"`)
   - `promote_to_parent`: `true` if the current definition has content not already on the base form; `false` if it would duplicate an existing base sense

   #### E. deleted_senses — remove redundant or misplaced senses
   Remove a sense if it is **fully semantically subsumed** by another sense on the same entry with no meaningful additional nuance — it adds nothing a reader would need. Be conservative: when in doubt, keep.

   Also remove a sense if it **only makes sense as part of a multi-word expression** — i.e., the definition crucially depends on another word to be meaningful. For example, a sense under "take" that really describes "take care" or "take down" belongs in the compound entry, not in the simplex. Delete with reason like `"belongs to MWE 'take care'"`. (The MWE pipeline handles creating proper compound entries.)

   Do NOT delete a sense merely because it overlaps with another, is specific, or has no examples. Only delete true duplicates, strict featureless subsets of another sense, or senses misplaced on a simplex entry that belong to an MWE.

   For each sense to delete: `{"sense_idx": <0-based>, "reason": "brief reason"}`.

   #### F. sense_rewrites — improve definition phrasing
   Rewrite a definition if it is:
   - **Muddled or unclear**: hard to parse, internally contradictory, or vague to the point of uselessness
   - **Self-referential**: uses the word itself as the core of its own explanation
   - **Excessively terse**: one word or a short phrase that conveys no information
   - **Padded**: bloated with unnecessary hedging or repetitive language
   - **Overly narrow**: scopes the word to a particular domain, register, or context of use when it has broader general application — e.g., defining a common word as if it only occurs in legal texts, a specific genre, or one narrow topic. The examples that surfaced during sense induction are a sample, not the full universe of usage; do not let them artificially restrict the definition.
   - **Conflates two distinct senses**: the definition clearly covers two different meanings that a dictionary would list separately. Rewrite to cover only the more central meaning; the other will be inducted as its own sense from the corpus.

   Preserve the meaning; improve only the phrasing. Do not change the scope or intent of the sense.

   For each rewrite: `{"sense_idx": <0-based>, "definition": "<improved text>"}`.

   #### G. pos_corrections — fix incorrect POS tags
   Correct a clearly wrong part-of-speech tag (e.g. a sense tagged `"verb"` that is obviously a noun). Only correct when clearly wrong — do not second-guess borderline cases.

   Valid POS values: `"noun"`, `"verb"`, `"adjective"`, `"adverb"`, `"preposition"`, `"conjunction"`, `"pronoun"`, `"determiner"`, `"interjection"`, `"proper_noun"`, `"other"`.

   **Note:** If the correct POS is `proper_noun` but the form is lowercased, prefer `normalize_case` (entry-level) over a POS correction.

   For each correction: `{"sense_idx": <0-based>, "pos": "<correct_pos>"}`.

   ---

   ### No action
   If none of the above applies, just delete the pending file — **do not write an output file**.

4. **Writing output**: Write output JSON to `/Users/kpich/dev/alfs/cc_tasks/done/qc/{same_filename}` using the `Write` tool, then delete the pending file with `Bash` (`rm /Users/kpich/dev/alfs/cc_tasks/pending/qc/{filename}`).

   Output schema (omit or leave as default any fields that don't apply):
   ```json
   {
     "type": "qc",
     "id": "<same id from task>",
     "form": "<the form>",

     "morph_rels": [
       {
         "sense_idx": 0,
         "morph_base": "dog",
         "morph_relation": "plural",
         "proposed_definition": "plural of dog (n.)",
         "promote_to_parent": true
       }
     ],

     "deleted_senses": [
       {"sense_idx": 2, "reason": "redundant with sense 0"}
     ],

     "sense_rewrites": [
       {"sense_idx": 1, "definition": "improved definition text"}
     ],

     "pos_corrections": [
       {"sense_idx": 0, "pos": "noun"}
     ],

     "delete_entry": false,
     "delete_entry_reason": null,

     "normalize_case": null,

     "spelling_variant_of": null
   }
   ```

   **Constraints:**
   - A sense_idx cannot appear in more than one of `deleted_senses`, `sense_rewrites`, `pos_corrections`, or `morph_rels`.
   - If any entry-level field is set (`delete_entry: true`, `normalize_case` non-null, or `spelling_variant_of` non-null), all sense-level lists must be empty.
   - Only one entry-level action may be used at a time.

5. **No action**: Just delete the pending file with `Bash` (`rm /Users/kpich/dev/alfs/cc_tasks/pending/qc/{filename}`). Do NOT write an output file.

Process ALL matching pending files, not just the first one.
