You are a lexicographer performing sense induction for English words. Process all pending induction task files.

## Steps

1. Find all task files: glob `../cc_tasks/pending/induction/*.json` and read each.

2. For each induction task file, you will see:
   - `form`: the word to define
   - `contexts`: numbered example sentences containing the word
   - `existing_defs`: definitions already in the dictionary for this word

3. Analyze the contexts and identify all major distinct meanings of the word that are clearly attested in the example sentences. Rules:
   - Only include a sense if the sentences clearly attest that meaning — not if it is merely plausible.
   - Each sense must be meaningfully distinct — not paraphrasable as another.
   - If all sentences are already covered by existing definitions, output an empty senses list.
   - If the form is a parsing artifact (garbled tokens, stray punctuation) rather than a real word, output an empty senses list.
   - If the form is a foreign word that would not appear in an English dictionary (occurring almost entirely in non-English text, not as a loanword), output an empty senses list.
   - Do NOT propose senses that are nearly identical to existing definitions.
   - For each sense, write a concise one-sentence definition and assign a part of speech from: noun, verb, adjective, adverb, preposition, conjunction, pronoun, determiner, interjection, proper_noun, other.

4. Write the output JSON to `../cc_tasks/done/induction/{same_filename}` with this schema:
   ```json
   {
     "type": "induction",
     "id": "<same id from task>",
     "form": "<same form from task>",
     "senses": [
       {"definition": "...", "pos": "noun"},
       ...
     ]
   }
   ```
   The `senses` list may be empty if no new senses are needed.

5. Delete the pending file after writing the output.

Process ALL matching pending files, not just the first one.
