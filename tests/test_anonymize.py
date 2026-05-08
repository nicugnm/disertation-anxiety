from src.preprocessing.anonymize import _hash_username, regex_redact


def test_redacts_email_phone_user():
    text = "contact me at jane.doe@example.com or 555-123-4567 or u/foo_bar."
    out = regex_redact(text)
    assert "[EMAIL]" in out
    assert "[PHONE]" in out
    assert "[USER]" in out
    assert "jane.doe" not in out


def test_redacts_subreddit_mention():
    text = "go to r/Anxiety and /r/Foo"
    out = regex_redact(text)
    assert "[SUB]" in out


def test_hash_username_stable():
    assert _hash_username("alice") == _hash_username("alice")
    assert _hash_username("alice") != _hash_username("bob")
    assert _hash_username(None) is None
    assert _hash_username("alice").startswith("u_")
