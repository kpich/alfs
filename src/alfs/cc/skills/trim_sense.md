You are a lexicographer reviewing dictionary senses for redundancy. Process all pending trim_sense task files.

## Steps

1. Find all task files: glob `../cc_tasks/pending/trim_sense/*.json` and read each.

2. For each trim_sense task file, you will see:
   - `form`: the word being reviewed
   - `senses`: list of sense objects, each with `id`, `definition`, `subsenses`, and `pos`
   - `examples`: list of lists of example sentences, one list per sense (showing how each sense is used in context)

3. Decide whether any sense should be deleted. Delete a sense if:
   - Two senses cover the same concept and one is weaker/redundant.
   - The form is a parsing artifact rather than a real word or expression.
   - The form is a foreign word that would not appear in an English dictionary (occurring almost entirely in non-English text, not as a loanword or expression commonly used in English).
   - If all senses are worth keeping, set `sense_num` to null.

4. Write the output JSON to `../cc_tasks/done/trim_sense/{same_filename}` with this schema:
   ```json
   {
     "type": "trim_sense",
     "id": "<same id from task>",
     "form": "<same form from task>",
     "sense_num": 2,
     "reason": "brief explanation"
   }
   ```
   `sense_num` is 1-based (sense 1 = first sense). Set to null if no sense should be deleted.

5. Delete the pending file after writing the output.

Process ALL matching pending files, not just the first one.
