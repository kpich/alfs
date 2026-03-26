You are a lexicographer reviewing dictionary senses for redundancy. Process all pending trim_sense task files.

## Steps

1. Find all task files: use Glob with path `../cc_tasks/pending/trim_sense` and pattern `*.json`, then read each with the Read tool.

   **IMPORTANT: Never use Bash for file operations. Use Read to read files, Write to write files, and Bash only to delete files (`rm`).**

2. For each trim_sense task file, you will see:
   - `form`: the word being reviewed
   - `senses`: list of sense objects, each with `id`, `definition`, and `pos`
   - `examples`: list of lists of example sentences, one list per sense (showing how each sense is used in context). Note: some senses may have no examples — this just means the corpus didn't surface good instances; it is NOT grounds for deletion on its own.

3. Decide whether any sense should be deleted. Delete a sense if:
   - Two senses cover the same concept and one is weaker/redundant.
   - The form is a parsing artifact rather than a real word or expression.
   - The form is a foreign word that would not appear in an English dictionary (occurring almost entirely in non-English text, not as a loanword or expression commonly used in English).
   - Do NOT delete a sense merely because it is a more specific application of a broader sense — if the narrower sense has distinct semantic content (a different domain, connotation, or conceptual focus) that would merit its own entry in a standard dictionary, keep it.
   - Do NOT delete a sense merely because the form is a proper noun — proper names are valid dictionary entries.
   - Do NOT delete a sense merely because its examples list is empty.
   - If all senses are worth keeping, set `sense_num` to null.

4. If no sense should be deleted: delete the pending file and move to the next task — do NOT write an output file.

   Otherwise, write the output JSON to `../cc_tasks/done/trim_sense/{same_filename}` with this schema:
   ```json
   {
     "type": "trim_sense",
     "id": "<same id from task>",
     "form": "<same form from task>",
     "sense_num": 2,
     "reason": "brief explanation"
   }
   ```
   `sense_num` is 1-based (sense 1 = first sense). Then delete the pending file.

Process ALL matching pending files, not just the first one.
