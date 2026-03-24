You are a lexicographer identifying morphological derivations between dictionary entries. Process all pending morph_redirect task files.

## Steps

1. Find all task files: use Glob with path `../cc_tasks/pending/morph_redirect` and pattern `*.json`, then read each.

2. For each morph_redirect task file, you will see:
   - `forms`: list of form objects, each with `form` (the word) and `senses` (list of sense objects with `id`, `definition`, `pos`)
   - `inventory_forms`: list of all form strings in the dictionary (for validating base forms)

3. For each form in the `forms` list, determine if any of its senses are regular inflections of another word in the inventory. Qualifying categories:
   - Plural forms (dogs -> dog, boxes -> box)
   - Verbal inflections (walked -> walk, running -> run, goes -> go)
   - Comparative or superlative forms (faster -> fast, best -> good)

   Do NOT include:
   - Pronoun case changes (our -> we, him -> he)
   - Derivational morphology (happiness -> happy, driver -> drive)
   - Semantic or etymological relations

   The base form must exist in `inventory_forms`. Do not propose a base that is not in the inventory.

   **Retained senses**: A form may have some senses that are morph links and other senses that are not. Only list the senses that qualify as morph links — unlisted senses are left untouched. Do not redirect a sense just because the form also has other morph-linked senses.

4. For each valid relation found, provide:
   - `derived_form`: the inflected form
   - `derived_sense_idx`: 0-based index into that form's senses
   - `base_form`: the base word
   - `base_sense_idx`: 0-based index into the base form's senses (you may need to look at other forms in the batch to find the base's senses, or use 0 if the base is not in the batch)
   - `relation`: short description (e.g. "plural", "past tense", "comparative")
   - `proposed_definition`: a concise reference like "plural of dog (n.)". Note: when this link is applied, the derived sense's current definition will be promoted to the base form's entry (added as a new sense there), and this proposed_definition will replace it on the derived form. Only list senses that are genuine inflections; unlisted senses remain in the derived form as independent meanings.
   - `promote_to_parent`: `true` if the derived sense's current definition represents content not already present in the base form's senses; `false` if the base already has an equivalent meaning (to avoid duplicating it).

5. If no morphological links are found (relations list would be empty): delete the pending file and move to the next task — do NOT write an output file.

   Otherwise, write the output JSON to `../cc_tasks/done/morph_redirect/{same_filename}` with this schema:
   ```json
   {
     "type": "morph_redirect",
     "id": "<same id from task>",
     "relations": [
       {
         "derived_form": "dogs",
         "derived_sense_idx": 0,
         "base_form": "dog",
         "base_sense_idx": 0,
         "relation": "plural",
         "proposed_definition": "plural of dog (n.)",
         "promote_to_parent": true
       }
     ]
   }
   ```
   Then delete the pending file.

Process ALL matching pending files, not just the first one.
