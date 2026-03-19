You are a lexicographer confirming British/American spelling variant pairs. Process all pending spelling_variant task files.

## Steps

1. Find all task files: use Glob with path `../cc_tasks/pending/spelling_variant` and pattern `*.json`, then read each.

2. For each spelling_variant task file, you will see:
   - `candidates`: list of pairs, each with `variant_form` (British spelling) and `preferred_form` (American spelling)

3. For each candidate pair, confirm that it is a genuine spelling variant — the same word spelled differently in British vs. American English. Reject pairs where:
   - The two forms are distinct words with different meanings (e.g. "metre" (unit) vs. "meter" (measuring device) are related but not mere spelling variants)
   - The transformation happened to produce a real word that is not actually the American spelling of the variant
   - One form is not a recognizable British spelling of the other

4. Write the output JSON to `../cc_tasks/done/spelling_variant/{same_filename}` with this schema:
   ```json
   {
     "type": "spelling_variant",
     "id": "<same id from task>",
     "confirmed": [
       {
         "variant_form": "colour",
         "preferred_form": "color"
       }
     ]
   }
   ```
   The `confirmed` list contains only the pairs you have confirmed as genuine spelling variants. It may be empty.

5. Delete the pending file after writing the output.

Process ALL matching pending files, not just the first one.
