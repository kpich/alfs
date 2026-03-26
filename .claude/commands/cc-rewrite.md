You are a lexicographer improving dictionary entries. Process all pending rewrite task files.

## Steps

1. Find all task files: use Glob with path `../cc_tasks/pending/rewrite` and pattern `*.json`, then read each with the Read tool.

   **IMPORTANT: Never use Bash for file operations. Use Read to read files, Write to write files, and Bash only to delete files (`rm`).**

2. For each rewrite task file, you will see:
   - `form`: the word being defined
   - `senses`: list of current sense objects, each with `id`, `definition`, and `pos`

3. Improve any definitions that could be clearer or more precise. Rules:
   - Only include senses you are actually changing — omit any you are leaving unchanged.
   - Preserve the meaning of each sense; improve only the phrasing.
   - Definitions must not be self-referential (don't use the word itself as the core of the explanation).
   - Don't be too terse (keep definitions informative) or too verbose (avoid unnecessary hedging).
4. Write the output JSON to `../cc_tasks/done/rewrite/{same_filename}` with this schema:
   ```json
   {
     "type": "rewrite",
     "id": "<same id from task>",
     "form": "<same form from task>",
     "rewrites": [
       {"sense_num": 1, "definition": "..."},
       ...
     ]
   }
   ```
   `sense_num` is the 1-based index of the sense being changed. Use an empty list if no definitions need improvement.

5. Delete the pending file after writing the output.

Process ALL matching pending files, not just the first one.
