You are a lexicographer reviewing wordforms in an expansive English dictionary. For each form, decide whether it needs a morphological-relation tag, a deletion+blocklist, or no change. Process all pending morphrel_block task files.

## Important: Tool Use

**NEVER use Bash to read or write files.** Use the `Read` tool to read, `Write` tool to write output files, and `Bash` only to delete files (`rm`). Do not use `cat`, shell pipelines, or any other Bash file operations — these require repeated approval prompts.

## Steps

1. Find all task files: use `Glob` with `path="../cc_tasks/pending/morphrel_block"` and `pattern="*.json"`, then read each with the `Read` tool.

2. For each task file you will see:
   - `form`: the wordform under review
   - `senses`: list of senses, each with `id`, `definition`, `pos`

3. For each form, determine which of the following applies:

   ### A. morph_rel — inflectional morphology
   The form is a regular inflection of another word. Qualifying categories:
   - Plural forms: `dogs` → `dog`, `boxes` → `box`
   - Verbal inflections: `walked` → `walk`, `running` → `run`, `goes` → `go`
   - Comparative/superlative: `faster` → `fast`, `best` → `good`

   Do NOT include:
   - Derivational morphology (`happiness` ← `happy`, `driver` ← `drive`)
   - Pronoun case changes (`him` → `he`, `our` → `we`)
   - Semantic/etymological relations only

   A form may have only some senses that are morph links — only list those senses; unlisted senses stay untouched. Multiple morph_rels entries are fine (e.g. `dogs` could link to `dog (n.)` and `dog (v.)` separately).

   For each qualifying sense, provide:
   - `sense_idx`: 0-based index into the form's senses list
   - `morph_base`: the base word (e.g. `"dog"`)
   - `morph_relation`: short label (e.g. `"plural"`, `"past tense"`, `"comparative"`)
   - `proposed_definition`: concise reference string (e.g. `"plural of dog (n.)"`)
   - `promote_to_parent`: `true` if the current definition contains content not obviously already on the base form; `false` if it would duplicate an existing base sense

   ### B. delete — garbage / blocklist
   The form should be removed entirely and added to the blocklist. Applies to:
   - Artifact forms: stray punctuation, markup fragments, tokenization errors (e.g. `5*!`, `--`, `</p>`)
   - Forms that are foreign words with no established use as English loan words
   - Truly one-off proper names or abbreviations with no shared semantics (e.g. a unique username, a serial number). Note: proper names in general — including minor fictional characters, historical figures, place names, and surnames — should be **kept**, not deleted. Only delete a proper name if it is so contextually unique that no reader would look it up (e.g. a throwaway username, an internal code name, a serial number).

   **This dictionary is intentionally expansive** — it includes appreciably more than a typical desk dictionary. Keep slang, technical jargon, neologisms, regional terms, archaic forms, and words a broad community of speakers would recognize. Only delete forms that are genuinely not English words or phrases by any standard.

   Provide `blocklist_reason`: a brief explanation (e.g. `"tokenization artifact"`, `"foreign word (Spanish)"`, `"one-off proper name"`).

   ### C. normalize_case — case normalization
   A form whose only distinction from a more canonical form is capitalization. Decide the canonical casing (proper names → capitalized; common words → lowercase). Examples: `"aaron"` → canonical `"Aaron"`; `"Dogs"` → canonical `"dogs"`.

   Do NOT normalize if the case variant carries a meaning absent from the canonical form (e.g. `His` as a divine pronoun, `POTS` as an acronym). The test: would a reader specifically look up this form to find a meaning unavailable under the canonical form? If so, keep it (no action). When in doubt, use no action.

   The apply step handles the merge automatically:
   - If `canonical_form` already exists in the store → this variant's senses are discarded and the form is blocklisted
   - If `canonical_form` does not exist → senses are migrated to the canonical form before deleting this one

   Provide `canonical_form`: the correctly-cased target form (e.g. `"Aaron"`, `"dogs"`).

   ### D. no action — keep as-is
   The form is a valid dictionary entry with no morph/redirect/delete action needed. In this case: **do not write any output file** — just delete the pending file and move on.

4. **Writing output**: If an action (A, B, C, or D) applies, write output JSON to `../cc_tasks/done/morphrel_block/{same_filename}` using the `Write` tool, then delete the pending file with `Bash` (`rm ../cc_tasks/pending/morphrel_block/{filename}`).

   Output schema:
   ```json
   {
     "type": "morphrel_block",
     "id": "<same id from task>",
     "form": "<the form>",
     "action": "morph_rel" | "delete" | "normalize_case",

     // For morph_rel only:
     "morph_rels": [
       {
         "sense_idx": 0,
         "morph_base": "dog",
         "morph_relation": "plural",
         "proposed_definition": "plural of dog (n.)",
         "promote_to_parent": true
       }
     ],

     // For delete only:
     "blocklist_reason": "tokenization artifact",

     // For normalize_case only:
     "canonical_form": "Aaron"
   }
   ```

   For action types that don't use a field, omit it (or leave as default).

5. **No action**: Just delete the pending file with `Bash` (`rm ../cc_tasks/pending/morphrel_block/{filename}`). Do NOT write an output file.

Process ALL matching pending files, not just the first one.
