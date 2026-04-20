"""Reserved values for the ``sense_key`` column in ``labeled.db``.

Both values are rating=0 markers — neither is a real sense in ``senses.db``
(which holds UUIDs). They distinguish *why* an occurrence couldn't be
assigned to a defined sense:

- ``SKIP_SENSE_KEY`` — noise / OCR garbage / parsing artifact / proper
  noun / truly one-off usage. Should never be re-examined for sense
  assignment or trigger re-induction for the form.

- ``UNCOVERED_SENSE_KEY`` — the word is being used in some real meaning
  none of the currently-defined senses covers. Counted as poor coverage
  and used to trigger re-induction for the form.
"""

SKIP_SENSE_KEY = "0"
UNCOVERED_SENSE_KEY = "_none"
RESERVED_SENSE_KEYS: frozenset[str] = frozenset({SKIP_SENSE_KEY, UNCOVERED_SENSE_KEY})
