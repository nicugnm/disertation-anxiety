from src.preprocessing.clean import clean_text


def test_clean_strips_urls_and_markdown():
    title = "Help"
    body = "Read [this](https://example.com) and visit www.foo.com please. &amp; thanks."
    out = clean_text(title, body)
    assert "[URL]" in out
    assert "https://" not in out
    assert "&amp;" not in out
    assert "Read this" in out


def test_clean_handles_none():
    out = clean_text(None, None)  # type: ignore[arg-type]
    assert out == ""


def test_clean_normalizes_whitespace():
    out = clean_text("Title", "Hello\n\n\n\nworld    foo")
    assert "\n\n\n" not in out
    assert "    " not in out
