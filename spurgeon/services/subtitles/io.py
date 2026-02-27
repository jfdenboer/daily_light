from __future__ import annotations

"""spurgeon.services.subtitles.io
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure I/O‑laag voor het ondertitel‑sub‑package.
Bevat uitsluitend functies die *bestanden* lezen of schrijven en heeft
*geen* domeinlogica. Dit voorkomt circulaire dependencies en houdt
unit‑testen eenvoudig met een `tmp_path` fixture.

Publieke API
============

- ``load_rev_json(path)`` → dict
- ``write_srt_file(subtitle_lines, path)``
- ``convert_srt_to_ass(srt_path, ass_path)``

Beide helpers accepteren *strings* of :class:`~pathlib.Path` objecten.
Alle paden worden intern geconverteerd naar ``Path`` zodat call‑sites
flexibel blijven.
"""

from pathlib import Path
import json
import re
from typing import Sequence, Any

from .caption_models import SubtitleLine

__all__ = [
    "load_rev_json",
    "write_srt_file",
    "convert_srt_to_ass",
]


def _to_path(path: str | Path) -> Path:
    """Helper: converteer *path‑like* naar :class:`Path`."""
    return path if isinstance(path, Path) else Path(path)


# ---------------------------------------------------------------------------
# Rev.ai JSON ----------------------------------------------------------------
# ---------------------------------------------------------------------------


def load_rev_json(path: str | Path) -> dict[str, Any]:
    """Lees een Rev.ai‑transcript JSON en retourneer de *parse tree* als dict.

    Parameters
    ----------
    path
        Bestands‑ of pathlib‑pad naar een geldige ``.json`` waarin een
        Rev.ai ASR‑/Transcription‑response staat.

    Raises
    ------
    FileNotFoundError
        Als het pad niet bestaat.
    json.JSONDecodeError
        Als de inhoud geen geldige JSON is.

    Notes
    -----
    Deze functie doet bewust *geen* schema‑validatie; dat is de
    verantwoordelijkheid van hogere lagen (bv. ``tokenizer.iter_tokens``).
    """
    p = _to_path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    with p.open("rt", encoding="utf-8") as fp:
        data: dict[str, Any] = json.load(fp)

    return data


# ---------------------------------------------------------------------------
# SRT output -----------------------------------------------------------------
# ---------------------------------------------------------------------------

_SRT_SEP = "\n\n"  # lege regel tussen cues

_SRT_BLOCK_RE = re.compile(r"\r?\n\r?\n", re.MULTILINE)


def _srt_timestamp_to_ass(timestamp: str) -> str:
    """Zet een SRT tijdstempel om naar het ASS-formaat (HH:MM:SS.cc)."""

    hours_str, minutes_str, rest = timestamp.split(":", 2)
    seconds_str, millis_str = rest.split(",", 1)

    total_millis = (
        int(hours_str) * 3_600_000
        + int(minutes_str) * 60_000
        + int(seconds_str) * 1_000
        + int(millis_str)
    )
    # ASS gebruikt centiseconden. Rond af op basis van de SRT milliseconden zodat
    # ``00:00:00,005`` → ``0:00:00.01`` in plaats van ``0:00:00.00``.
    centiseconds = (total_millis + 5) // 10

    hours, rem = divmod(centiseconds, 3_600 * 100)
    minutes, rem = divmod(rem, 60 * 100)
    seconds, centiseconds = divmod(rem, 100)

    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _escape_ass_text(text: str) -> str:
    """Maak tekst veilig voor ASS door speciale tekens te escapen."""

    escaped = text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    return escaped


def _ensure_parent_dir(path: Path) -> None:
    """Maak de bovenliggende directory aan (``mkdir -p``)."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)


def write_srt_file(
    subtitle_lines: Sequence[SubtitleLine],
    path: str | Path,
    *,
    overwrite: bool = True,
    encoding: str = "utf-8",
) -> Path:
    """Schrijf een lijst :class:`~caption_models.SubtitleLine` naar een SRT‑file.

    Parameters
    ----------
    subtitle_lines
        Een volgorde van ondertitelregels die reeds gemerged/gechunked zijn.
    path
        Doelbestand.
    overwrite
        Als ``False`` en het bestand bestaat al → ``FileExistsError``.
    encoding
        Tekst‑encoding om te gebruiken (default ``utf-8``).

    Returns
    -------
    Path
        Het *absolute* pad naar het geschreven bestand (handig in logging tests).
    """

    p = _to_path(path).expanduser().resolve()
    if p.exists() and not overwrite:
        raise FileExistsError(p)

    _ensure_parent_dir(p)

    with p.open("wt", encoding=encoding, newline="\n") as fp:
        for idx, line in enumerate(subtitle_lines, start=1):
            fp.write(f"{idx}\n")  # cue index
            fp.write(
                f"{line.start_srt} --> {line.end_srt}\n{line.text.strip()}"  # tijdstempels + tekst
            )
            # Maak geen trailing whitespace in de laatste regel; voeg *altijd*
            # een lege regel toe behalve voor de allerlaatste cue → compliant
            # met SRT‑spec & Rev‑import.
            if idx < len(subtitle_lines):
                fp.write(_SRT_SEP)

    return p

def convert_srt_to_ass(
    srt_path: str | Path,
    ass_path: str | Path,
    *,
    overwrite: bool = True,
    encoding: str = "utf-8",
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    wrap_style: int = 2,
    fontname: str = "Inter",
    fontsize: int = 48,
    primary: str = "&H00FFFFFF",
    outline_col: str = "&HAA000000",
    border_style: int = 1,
    outline: float = 2.6,
    shadow: int = 0,
    alignment: int = 5,
    margin_l: int = 230,
    margin_r: int = 230,
    margin_v: int = 40,
) -> Path:
    """Converteer een bestaand SRT-bestand naar een ASS-bestand.

    Parameters
    ----------
    srt_path
        Bronbestand met SRT-ondertitels.
    ass_path
        Doelbestand dat de geconverteerde ASS-ondertitels zal bevatten.
    overwrite
        Als ``False`` en het doelbestand bestaat al → ``FileExistsError``.
    encoding
        Encoding voor lezen/schrijven (default ``utf-8``).
    play_res_x/play_res_y
        Resolutieparameters die in de ASS-header worden gezet.
    wrap_style
        Bepaalt de ASS "WrapStyle" (0..3). Default ``2`` gebruikt slimme
        automatische regelafbreking.
    """

    if wrap_style not in {0, 1, 2, 3}:
        raise ValueError("wrap_style moet een waarde tussen 0 en 3 hebben")
    if border_style not in {1, 3}:
        raise ValueError("border_style moet 1 of 3 zijn")
    if alignment < 1 or alignment > 9:
        raise ValueError("alignment moet tussen 1 en 9 liggen")
    if fontsize <= 0:
        raise ValueError("fontsize moet groter dan 0 zijn")
    if outline < 0:
        raise ValueError("outline moet >= 0 zijn")
    if shadow < 0:
        raise ValueError("shadow moet >= 0 zijn")
    if margin_l < 0 or margin_r < 0 or margin_v < 0:
        raise ValueError("marges moeten >= 0 zijn")
    if not primary.startswith("&H"):
        raise ValueError("primary kleur moet beginnen met '&H'")
    if not outline_col.startswith("&H"):
        raise ValueError("outline kleur moet beginnen met '&H'")

    src = _to_path(srt_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    dest = _to_path(ass_path).expanduser().resolve()
    if dest.exists() and not overwrite:
        raise FileExistsError(dest)

    _ensure_parent_dir(dest)

    with src.open("rt", encoding=encoding) as fp:
        raw = fp.read().strip()

    blocks = [block for block in _SRT_BLOCK_RE.split(raw) if block.strip()]

    dialogue_lines: list[str] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        timing_line = lines[1]
        try:
            start_str, end_str = [part.strip() for part in timing_line.split("-->")]
        except ValueError as exc:  # pragma: no cover - defensief
            raise ValueError(f"Ongeldig SRT timing-formaat: {timing_line!r}") from exc

        start = _srt_timestamp_to_ass(start_str)
        end = _srt_timestamp_to_ass(end_str)
        text_lines = [
            _escape_ass_text(line)
            for line in lines[2:]
        ] or [""]
        text = r"\N".join(text_lines)

        dialogue_lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )

    style_outline = format(outline, "g")

    header = [
        "[Script Info]",
        "; Script generated by spurgeon",
        "ScriptType: v4.00+",
        "Collisions: Normal",
        f"WrapStyle: {wrap_style}",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {play_res_x}",
        f"PlayResY: {play_res_y}",
        "Timer: 100.0000",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            "Style: Default,"
            f"{fontname},"
            f"{fontsize},"
            f"{primary},"
            "&H000000FF,"
            f"{outline_col},"
            "&H64000000,"
            "0,0,0,0,100,100,0,0,"
            f"{border_style},"
            f"{style_outline},"
            f"{shadow},"
            f"{alignment},"
            f"{margin_l},"
            f"{margin_r},"
            f"{margin_v},"
            "1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    content_lines = header + dialogue_lines
    content = "\n".join(content_lines)
    if not content.endswith("\n"):
        content += "\n"

    with dest.open("wt", encoding=encoding, newline="\n") as fp:
        fp.write(content)

    return dest