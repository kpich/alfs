from pydantic import BaseModel, ConfigDict


class Doc(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str  # 8-char hex SHA256 prefix of text
    text: str  # plain text (mwparserfromhell-stripped wikitext)
    title: str | None = None
    author: str | None = None  # first editor's username (from oldest revision)
    year: int | None = None  # year of first revision
    source_url: str | None = (
        None  # e.g. "https://en.wikibooks.org/wiki/Python_Programming"
    )
    source: str | None = None  # "wikibooks" or "wikisource"
