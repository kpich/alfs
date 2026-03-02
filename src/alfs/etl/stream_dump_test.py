import re


def extract_wikisource_year(wikitext: str, timestamp_year: int) -> int:
    """Replicate the year-extraction logic from stream_dump.main()."""
    m = re.search(r"\|\s*year\s*=\s*(\d{4})", wikitext)
    if m:
        return int(m.group(1))
    return timestamp_year


def test_year_from_header_template():
    wikitext = "{{header\n| title = Moby Dick\n| year = 1851\n| author = Melville\n}}"
    assert extract_wikisource_year(wikitext, 2007) == 1851


def test_year_from_header_template_no_spaces():
    wikitext = "{{header|title=Foo|year=1900|author=Bar}}"
    assert extract_wikisource_year(wikitext, 2010) == 1900


def test_year_falls_back_to_timestamp_when_no_header():
    wikitext = "Some page text without a header template."
    assert extract_wikisource_year(wikitext, 2009) == 2009


def test_year_falls_back_to_timestamp_when_year_missing_from_header():
    wikitext = "{{header\n| title = Something\n| author = Someone\n}}"
    assert extract_wikisource_year(wikitext, 2012) == 2012
