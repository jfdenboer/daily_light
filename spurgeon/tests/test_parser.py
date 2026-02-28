import pytest

from spurgeon.core.parser import Parser


def test_parse_supports_unicode_line_separators() -> None:
    raw = (
        "JANUARY 1 MORNING\u2028\u2028"
        "Morning body text\u2028"
        "JANUARY 1 EVENING\u2028\u2028"
        "Evening body text"
    )

    readings = Parser().parse(raw, year=2027)

    assert len(readings) == 2
    assert readings[0].text == "Morning body text"
    assert readings[1].text == "Evening body text"


def test_parse_error_surfaces_header_like_lines() -> None:
    raw = "JANUARY MORNING\n\nBody"

    with pytest.raises(ValueError) as exc:
        Parser().parse(raw, year=2027)

    message = str(exc.value)
    assert "Header-like lines seen" in message
    assert "JANUARY MORNING" in message
