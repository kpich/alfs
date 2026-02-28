from enum import Enum


class PartOfSpeech(str, Enum):
    noun = "noun"
    verb = "verb"
    adjective = "adjective"
    adverb = "adverb"
    preposition = "preposition"
    conjunction = "conjunction"
    pronoun = "pronoun"
    determiner = "determiner"
    interjection = "interjection"
    other = "other"
