"""CLI-helper om lokaal een YouTube OAuth-token te genereren."""

from __future__ import annotations

import sys

from spurgeon.auth_setup import AuthSetupError, setup_youtube_token


def main() -> None:
    """Genereer een OAuth-token en schrijf het naar ``token.json``."""

    try:
        token_path = setup_youtube_token()
    except AuthSetupError as exc:  # pragma: no cover - CLI script
        print(f"❌ OAuth-configuratie mislukt: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"✅ Token opgeslagen naar {token_path.resolve()}")


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main()