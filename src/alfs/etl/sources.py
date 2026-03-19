"""Source registry for MediaWiki corpus dumps."""

from dataclasses import dataclass, field


@dataclass
class Source:
    name: str
    type: str
    dump_url: str
    dump_filename: str
    base_url: str
    hf_dataset: str = field(default="")


SOURCES: dict[str, Source] = {
    "wikibooks": Source(
        name="wikibooks",
        type="mediawiki",
        dump_url="https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2",
        dump_filename="enwikibooks-latest-pages-articles.xml.bz2",
        base_url="https://en.wikibooks.org/wiki/",
    ),
    "wikisource": Source(
        name="wikisource",
        type="mediawiki",
        dump_url="https://dumps.wikimedia.org/enwikisource/latest/enwikisource-latest-pages-articles.xml.bz2",
        dump_filename="enwikisource-latest-pages-articles.xml.bz2",
        base_url="https://en.wikisource.org/wiki/",
    ),
    "wikipedia": Source(
        name="wikipedia",
        type="mediawiki",
        dump_url="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
        dump_filename="enwiki-latest-pages-articles.xml.bz2",
        base_url="https://en.wikipedia.org/wiki/",
    ),
    "gutenberg": Source(
        name="gutenberg",
        type="gutenberg",
        dump_url="https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2",
        dump_filename="gutenberg-catalog.tar.bz2",
        base_url="https://www.gutenberg.org/ebooks/",
    ),
    "cc_news": Source(
        name="cc_news",
        type="hf",
        dump_url="",
        dump_filename="",
        base_url="",
        hf_dataset="cc_news",
    ),
}
