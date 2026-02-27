"""Typer CLI entrypoint for de Daily Light-pipeline."""

from __future__ import annotations

import calendar
from datetime import date
from typing import Optional, TYPE_CHECKING

import typer
from typer import BadParameter, Option, Typer

from spurgeon.config.settings import load_settings
from spurgeon.utils.logging_setup import init_logging

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from spurgeon.core.pipeline import run_pipeline as _run_pipeline_type

__all__ = ["app"]

ISO_DATE_EXAMPLE = "YYYY-MM-DD"


app = Typer(help="Spurgeon: automatische Daily Light-video pipeline")


def _parse_date(value: str, param_name: str) -> date:
    """Parse an ISO date string with a leap-year fallback."""

    normalized = value.strip()
    if not normalized:
        raise BadParameter(
            "Optie '{param}' vereist een datum in formaat {example}, maar er is niets opgegeven.".format(
                param=param_name, example=ISO_DATE_EXAMPLE
            )
        )

    try:
        return date.fromisoformat(normalized)
    except ValueError as exc:
        parts = normalized.split("-")
        if len(parts) == 3:
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
            except ValueError:
                pass
            else:
                if month == 2 and day == 29:
                    mapped = _map_to_next_leap_day(year)
                    typer.secho(
                        (
                            "Niet-schrikkeljaar {year}: datum '{value}' wordt gemapt naar {mapped}"
                        ).format(year=year, value=normalized, mapped=mapped.isoformat()),
                        fg=typer.colors.YELLOW,
                        err=True,
                    )
                    return mapped
        raise BadParameter(
            (
                "Optie '{param}' heeft een ongeldige datum: '{value}'. "
                "Gebruik formaat {example}."
            ).format(param=param_name, value=value, example=ISO_DATE_EXAMPLE)
        ) from exc


def _map_to_next_leap_day(year: int) -> date:
    """Return February 29th in the next leap year greater or equal to *year*."""

    next_leap = year
    while not calendar.isleap(next_leap):
        next_leap += 1
    return date(next_leap, 2, 29)


def _ensure_valid_range(
    start: Optional[date], end: Optional[date]
) -> None:
    """Validate the requested date range."""

    if start and end and start > end:
        raise BadParameter(
            (
                "Optie 'start_date' ({start}) mag niet later zijn dan 'end_date' ({end})."
            ).format(start=start.isoformat(), end=end.isoformat()),
            param_hint="start_date,end_date",
        )


@app.callback()
def main() -> None:
    """Root command for the CLI – reserved for future expansion."""


@app.command(name="build")
def build(
    start_date: Optional[str] = Option(
        None,
        help="Verwerk readings vanaf deze datum (YYYY-MM-DD)",
        metavar="YYYY-MM-DD",
    ),
    end_date: Optional[str] = Option(
        None,
        help="Verwerk readings t/m deze datum (YYYY-MM-DD)",
        metavar="YYYY-MM-DD",
    ),
) -> None:
    """Voer de volledige pipeline uit voor alle devotional readings."""

    parsed_start = _parse_date(start_date, "start_date") if start_date else None
    parsed_end = _parse_date(end_date, "end_date") if end_date else None

    _ensure_valid_range(parsed_start, parsed_end)


    settings = load_settings()
    init_logging(settings)

    typer.echo(
        "🔧 Instellingen geladen uit {src} → {dest}".format(
            src=settings.input_dir, dest=settings.output_dir
        ),
        err=True,
    )

    from spurgeon.core.pipeline import run_pipeline as run_pipeline_impl

    run_pipeline_impl(
        settings=settings,
        start_date=parsed_start,
        end_date=parsed_end,
    )

    typer.secho("✅ Pipeline voltooid 🎉", fg=typer.colors.GREEN)


@app.command(name="gui")
def gui() -> None:
    """Start een eenvoudige Tkinter GUI om één datum te verwerken."""

    from spurgeon.gui import launch_gui

    launch_gui()


if __name__ == "__main__":  # pragma: no cover - CLI hook
    app()
