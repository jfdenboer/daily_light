"""Hulpfuncties voor het genereren van OAuth-credentials voor YouTube uploads."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from google.auth.exceptions import GoogleAuthError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

__all__ = [
    "AuthSetupError",
    "DEFAULT_SCOPES",
    "setup_youtube_token",
]

logger = logging.getLogger(__name__)

DEFAULT_SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/youtube.upload",
)


class AuthSetupError(RuntimeError):
    """Fout die aangeeft dat het verkrijgen van OAuth-credentials is mislukt."""


def setup_youtube_token(
    client_secrets: Path | str = Path("client_secrets.json"),
    token_output: Path | str = Path("token.json"),
    *,
    scopes: Sequence[str] = DEFAULT_SCOPES,
    use_local_server: bool = True,
    console_fallback: bool = True,
    port: int = 8080,
    prompt: str = "consent",
    overwrite: bool = False,
) -> Path:
    """Start de OAuth-flow en sla het resulterende token op schijf op.

    Parameters
    ----------
    client_secrets:
        Pad naar het JSON-bestand met client secrets van het Google Cloud-project.
    token_output:
        Pad waar de nieuw verkregen token als JSON wordt opgeslagen.
    scopes:
        Lijst met OAuth-scopes die gevraagd worden tijdens autorisatie.
    use_local_server:
        Gebruik de lokale webserver-flow (met automatische browser open).
    console_fallback:
        Wanneer de lokale webserver niet start, val terug op de console-flow.
    port:
        Poort waarop de lokale server luistert. Alleen relevant wanneer
        ``use_local_server`` ``True`` is.
    prompt:
        Prompt-parameter die doorgegeven wordt aan de OAuth-flow.
    overwrite:
        Overschrijf een bestaand tokenbestand wanneer ``True``.

    Returns
    -------
    pathlib.Path
        Het pad waar het token-bestand is weggeschreven.
    """

    normalized_scopes = _normalize_scopes(scopes)
    secrets_path = Path(client_secrets).expanduser()
    token_path = Path(token_output).expanduser()

    _ensure_token_target(token_path, overwrite)

    flow = _initialise_flow(secrets_path, normalized_scopes)
    creds = _run_authorisation_flow(
        flow,
        use_local_server=use_local_server,
        console_fallback=console_fallback,
        port=port,
        prompt=prompt,
    )
    _write_token(token_path, creds)

    logger.info("OAuth-token opgeslagen op %s", token_path)
    return token_path


def _normalize_scopes(scopes: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(scope.strip() for scope in scopes if isinstance(scope, str) and scope.strip())
    if not normalized:
        raise AuthSetupError("Minimaal één geldige OAuth-scope is vereist")
    return normalized


def _ensure_token_target(token_path: Path, overwrite: bool) -> None:
    if token_path.exists() and not overwrite:
        raise AuthSetupError(
            (
                "Token-bestand {path} bestaat al. "
                "Gebruik overwrite=True of verwijder het bestaande bestand."
            ).format(path=token_path)
        )

    parent = token_path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - zeldzame OS-fout
            raise AuthSetupError(f"Kon map {parent} niet aanmaken: {exc}") from exc
    elif not parent.is_dir():
        raise AuthSetupError(f"Pad {parent} is geen map. Kan token niet wegschrijven.")


def _initialise_flow(client_secrets_path: Path, scopes: Sequence[str]) -> InstalledAppFlow:
    if not client_secrets_path.exists():
        raise AuthSetupError(f"Client secrets-bestand niet gevonden: {client_secrets_path}")
    if client_secrets_path.is_dir():
        raise AuthSetupError(
            f"Client secrets-pad verwijst naar een map, verwacht bestand: {client_secrets_path}"
        )

    try:
        return InstalledAppFlow.from_client_secrets_file(str(client_secrets_path), scopes)
    except FileNotFoundError as exc:
        raise AuthSetupError(f"Client secrets-bestand niet gevonden: {client_secrets_path}") from exc
    except ValueError as exc:
        raise AuthSetupError(
            f"Client secrets-bestand bevat ongeldige JSON of configuratie: {client_secrets_path}"
        ) from exc
    except (OSError, GoogleAuthError) as exc:
        raise AuthSetupError(f"Kon OAuth-flow niet initialiseren: {exc}") from exc


def _run_authorisation_flow(
    flow: InstalledAppFlow,
    *,
    use_local_server: bool,
    console_fallback: bool,
    port: int,
    prompt: str,
) -> Credentials:
    try:
        if use_local_server:
            try:
                return flow.run_local_server(port=port, prompt=prompt)
            except OSError as exc:
                if not console_fallback:
                    raise AuthSetupError(
                        f"Kon lokale OAuth-server niet starten op poort {port}: {exc}"
                    ) from exc
                logger.warning(
                    "Lokale OAuth-server kon niet starten (%s). Val terug op console-flow.",
                    exc,
                )
        return flow.run_console(prompt=prompt)
    except (OSError, ValueError, GoogleAuthError) as exc:
        raise AuthSetupError(f"OAuth-flow is mislukt: {exc}") from exc


def _write_token(token_path: Path, creds: Credentials) -> None:
    try:
        token_path.write_text(creds.to_json(), encoding="utf-8")
    except OSError as exc:
        raise AuthSetupError(f"Kon token niet wegschrijven naar {token_path}: {exc}") from exc