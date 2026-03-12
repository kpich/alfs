You are a lexicographer auditing dictionary entries for invalid or artifact word forms. Process all pending delete_entry task files.

## Steps

1. Find all task files: use Glob with path `../cc_tasks/pending/delete_entry` and pattern `*.json`, then read each.

2. For each delete_entry task file, you will see:
   - `form`: the word form being reviewed
   - `senses`: list of sense objects, each with `id`, `definition`, `subsenses`, and `pos`
   - `examples`: list of lists of example sentences, one list per sense (showing how each sense is used in context)

3. Decide whether the entire entry should be deleted. Delete if:
   - The form is a tokenization artifact (punctuation attached to a word, encoding error, etc.)
   - The form is not a real English word or expression
   - The form is a foreign word that would not appear in an English dictionary (occurring almost entirely in non-English text, not as a loanword or expression commonly used in English)
   - Do NOT delete proper nouns — they are valid dictionary entries.
   - Do NOT delete a form merely because its examples list is empty.

4. Write the output JSON to `../cc_tasks/done/delete_entry/{same_filename}` with this schema:
   ```json
   {
     "type": "delete_entry",
     "id": "<same id from task>",
     "form": "<same form from task>",
     "should_delete": true,
     "reason": "brief explanation"
   }
   ```
   Set `should_delete` to false if the entry should be kept.

5. Delete the pending file after writing the output.

Process ALL matching pending files, not just the first one.
