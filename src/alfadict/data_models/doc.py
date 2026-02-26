from pydantic import BaseModel


class Doc(BaseModel):
    title: str
    author: str  # first editor's username (from oldest revision)
    year: int  # year of first revision
    text: str  # plain text (mwparserfromhell-stripped wikitext)
    source_url: str  # e.g. "https://en.wikibooks.org/wiki/Python_Programming"
