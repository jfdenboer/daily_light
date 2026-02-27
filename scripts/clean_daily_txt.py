from __future__ import annotations

import argparse
import re
from pathlib import Path

SEPARATOR_RE = re.compile(r"^\s*_+\s*$", re.MULTILINE)
DOT_ELLIPSIS_RE = re.compile(r"\.(?:\s*\.\s*\.)")
REFERENCE_LINE_RE = re.compile(r"^\s*(?:[1-3]|I{1,3})?\s*[A-Za-z][A-Za-z .']*(?:\d+:\d+|\d+,\d+)", re.IGNORECASE)

BOOK_ABBREVIATIONS = {
    "GEN.": "Genesis",
    "GN.": "Genesis",
    "EX.": "Exodus",
    "EXOD.": "Exodus",
    "EXO.": "Exodus",
    "LEV.": "Leviticus",
    "LV.": "Leviticus",
    "NUM.": "Numbers",
    "NM.": "Numbers",
    "DEUT.": "Deuteronomy",
    "DT.": "Deuteronomy",
    "JOSH.": "Joshua",
    "JOS.": "Joshua",
    "JUDG.": "Judges",
    "JDG.": "Judges",
    "RUTH.": "Ruth",
    "RU.": "Ruth",
    "1 SAM.": "1 Samuel",
    "1 SA.": "1 Samuel",
    "2 SAM.": "2 Samuel",
    "2 SA.": "2 Samuel",
    "1 KGS.": "1 Kings",
    "1 KI.": "1 Kings",
    "2 KGS.": "2 Kings",
    "2 KI.": "2 Kings",
    "1 CHRON.": "1 Chronicles",
    "1 CHR.": "1 Chronicles",
    "2 CHRON.": "2 Chronicles",
    "2 CHR.": "2 Chronicles",
    "EZRA.": "Ezra",
    "EZR.": "Ezra",
    "NEH.": "Nehemiah",
    "NE.": "Nehemiah",
    "EST.": "Esther",
    "ESTH.": "Esther",
    "JOB.": "Job",
    "JB.": "Job",
    "PS.": "Psalms",
    "PA.": "Psalms",
    "PSA.": "Psalms",
    "PROV.": "Proverbs",
    "PR.": "Proverbs",
    "ECCLES.": "Ecclesiastes",
    "ECCL.": "Ecclesiastes",
    "EC.": "Ecclesiastes",
    "SONG.": "Song of Solomon",
    "CANT.": "Song of Solomon",
    "ISA.": "Isaiah",
    "IS.": "Isaiah",
    "JER.": "Jeremiah",
    "JR.": "Jeremiah",
    "LAM.": "Lamentations",
    "LA.": "Lamentations",
    "EZEK.": "Ezekiel",
    "EZE.": "Ezekiel",
    "DAN.": "Daniel",
    "DN.": "Daniel",
    "HOS.": "Hosea",
    "HO.": "Hosea",
    "JOEL.": "Joel",
    "JL.": "Joel",
    "AMOS.": "Amos",
    "AM.": "Amos",
    "OBAD.": "Obadiah",
    "OB.": "Obadiah",
    "JONAH.": "Jonah",
    "JON.": "Jonah",
    "MIC.": "Micah",
    "MI.": "Micah",
    "NAH.": "Nahum",
    "NA.": "Nahum",
    "HAB.": "Habakkuk",
    "HB.": "Habakkuk",
    "ZEPH.": "Zephaniah",
    "ZEP.": "Zephaniah",
    "HAG.": "Haggai",
    "HG.": "Haggai",
    "ZECH.": "Zechariah",
    "ZEC.": "Zechariah",
    "MAL.": "Malachi",
    "ML.": "Malachi",
    "MATT.": "Matthew",
    "MT.": "Matthew",
    "MARK.": "Mark",
    "MK.": "Mark",
    "LUKE.": "Luke",
    "LK.": "Luke",
    "JOHN.": "John",
    "JN.": "John",
    "ACTS.": "Acts",
    "AC.": "Acts",
    "ROM.": "Romans",
    "RM.": "Romans",
    "1 COR.": "1 Corinthians",
    "1 CO.": "1 Corinthians",
    "2 COR.": "2 Corinthians",
    "2 CO.": "2 Corinthians",
    "GAL.": "Galatians",
    "GA.": "Galatians",
    "EPH.": "Ephesians",
    "EPHES.": "Ephesians",
    "PHIL.": "Philippians",
    "PHI.": "Philippians",
    "PHP.": "Philippians",
    "COL.": "Colossians",
    "COLO.": "Colossians",
    "1 THESS.": "1 Thessalonians",
    "1 THES.": "1 Thessalonians",
    "1 TH.": "1 Thessalonians",
    "2 THESS.": "2 Thessalonians",
    "2 THES.": "2 Thessalonians",
    "2 TH.": "2 Thessalonians",
    "1 TIM.": "1 Timothy",
    "1 TI.": "1 Timothy",
    "2 TIM.": "2 Timothy",
    "2 TI.": "2 Timothy",
    "TITUS.": "Titus",
    "TIT.": "Titus",
    "PHILEM.": "Philemon",
    "PHLM.": "Philemon",
    "HEB.": "Hebrews",
    "HE.": "Hebrews",
    "JAS.": "James",
    "JM.": "James",
    "1 PET.": "1 Peter",
    "1 PE.": "1 Peter",
    "2 PET.": "2 Peter",
    "2 PE.": "2 Peter",
    "1 JOHN.": "1 John",
    "1 JN.": "1 John",
    "2 JOHN.": "2 John",
    "2 JN.": "2 John",
    "3 JOHN.": "3 John",
    "3 JN.": "3 John",
    "JUDE.": "Jude",
    "JD.": "Jude",
    "REV.": "Revelation",
    "RV.": "Revelation",
}

for alias, full_name in list(BOOK_ABBREVIATIONS.items()):
    if alias.startswith("1 "):
        BOOK_ABBREVIATIONS[f"I {alias[2:]}"] = full_name
    elif alias.startswith("2 "):
        BOOK_ABBREVIATIONS[f"II {alias[2:]}"] = full_name
    elif alias.startswith("3 "):
        BOOK_ABBREVIATIONS[f"III {alias[2:]}"] = full_name

BOOK_ABBREVIATIONS_CANONICAL = {
    _key: value for _key, value in ((re.sub(r"\.", "", k.upper()), v) for k, v in BOOK_ABBREVIATIONS.items())
}


def _canonical_book_token(token: str) -> str:
    token = re.sub(r"\.", "", token.upper())
    token = re.sub(r"\s+", " ", token).strip()
    return token


def _book_alias_pattern(alias: str) -> str:
    parts = alias.split()
    part_patterns: list[str] = []
    for part in parts:
        if part.isdigit():
            part_patterns.append(re.escape(part))
        else:
            part_clean = part.rstrip(".")
            part_patterns.append(rf"{re.escape(part_clean)}\.?")
    return r"\s+".join(part_patterns)


BOOK_CITATION_RE = re.compile(
    r"(?<!\w)("
    + "|".join(_book_alias_pattern(alias) for alias in sorted(BOOK_ABBREVIATIONS, key=len, reverse=True))
    + r")(?=\s*\d+:\d+)",
    flags=re.IGNORECASE,
)


def _expand_book_abbreviations(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        canonical = _canonical_book_token(raw)
        return BOOK_ABBREVIATIONS_CANONICAL.get(canonical, raw)

    return BOOK_CITATION_RE.sub(replace, text)


def _normalize_text(text: str) -> str:
    """Apply global text replacements requested for daily.txt."""
    text = DOT_ELLIPSIS_RE.sub("...", text)
    text = text.replace("--", " ")
    text = _expand_book_abbreviations(text)
    return text


def _strip_pdf_indentation(text: str) -> str:
    """Strip a leading 3-space PDF indentation from each line."""
    return re.sub(r"^ {3}", "", text, flags=re.MULTILINE)


def _unwrap_pdf_linebreaks(text: str) -> str:
    """Join hard-wrapped lines while preserving blank lines and separators."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    output_lines: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            output_lines.append(" ".join(buffer))
            buffer.clear()

    for raw_line in normalized.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            flush_buffer()
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            elif not output_lines:
                output_lines.append("")
            continue

        if SEPARATOR_RE.match(stripped):
            flush_buffer()
            output_lines.append(stripped)
            continue

        buffer.append(stripped)

    flush_buffer()
    return "\n".join(output_lines)


def _looks_like_reference_start(line: str) -> bool:
    candidate = line.strip()
    candidate = re.sub(r"^\([^)]*\)\s*", "", candidate)
    candidate = re.sub(r"^[-–—.;:]+\s*", "", candidate)
    return REFERENCE_LINE_RE.match(candidate) is not None


def _drop_trailing_reference_line(block: str) -> str:
    """Remove trailing Bible-reference paragraph if present."""
    lines = block.splitlines()
    if not lines:
        return block

    end = len(lines) - 1
    while end >= 0 and not lines[end].strip():
        end -= 1
    if end < 0:
        return block

    start = end
    while start >= 0 and lines[start].strip():
        start -= 1
    para_start = start + 1
    paragraph = " ".join(lines[para_start : end + 1])

    has_many_refs = len(re.findall(r"\d+:\d+|\d+,\d+", paragraph)) >= 2
    starts_like_ref = _looks_like_reference_start(lines[para_start] or "")
    if has_many_refs and starts_like_ref:
        del lines[para_start : end + 1]

    return "\n".join(lines)


def clean_daily_text(raw: str) -> str:
    """Clean readings while preserving day separators."""
    raw = _strip_pdf_indentation(raw)
    raw = _unwrap_pdf_linebreaks(raw)
    raw = _normalize_text(raw)

    pieces = re.split(r"(^\s*_+\s*$)", raw, flags=re.MULTILINE)
    cleaned: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if SEPARATOR_RE.match(piece):
            cleaned.append(piece.strip())
        else:
            cleaned.append(_drop_trailing_reference_line(piece))

    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result).rstrip() + "\n"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean daily.txt readings.")
    parser.add_argument("--input", type=Path, default=Path("input/daily.txt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("input/daily_cleaned.txt"),
        help="Output path (ignored with --in-place).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file.",
    )
    args = parser.parse_args()

    raw = args.input.read_text(encoding="utf-8")
    cleaned = clean_daily_text(raw)

    target = args.input if args.in_place else args.output
    target.write_text(cleaned, encoding="utf-8")
    print(f"Cleaned file written to: {target}")


if __name__ == "__main__":
    main()
