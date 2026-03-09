You are a lexicographer improving dictionary entries. Process all pending rewrite task files.

## Steps

1. Find all task files: glob `../cc_tasks/pending/*.json`, read each, and filter for files where `type` is `"rewrite"`.

2. For each rewrite task file, you will see:
   - `form`: the word being defined
   - `senses`: list of current sense objects, each with `id`, `definition`, `subsenses`, and `pos`

3. Rewrite the definitions to be clearer and more precise. Rules:
   - Keep exactly the same number of senses. Do not add or remove senses.
   - Preserve the meaning of each sense; improve only the phrasing.
   - Definitions must not be self-referential (don't use the word itself as the core of the explanation).
   - Don't be too terse (keep definitions informative) or too verbose (avoid unnecessary hedging).
   - You may also improve subsenses if present.

4. Write the output JSON to `../cc_tasks/done/{same_filename}` with this schema:
   ```json
   {
     "type": "rewrite",
     "id": "<same id from task>",
     "form": "<same form from task>",
     "senses": [
       {"definition": "...", "subsenses": ["...", "..."]},
       ...
     ]
   }
   ```
   Each sense in the output corresponds to the same-indexed sense in the input. `subsenses` may be null or omitted if there are none.

5. Delete the pending file after writing the output.

Process ALL matching pending files, not just the first one.
