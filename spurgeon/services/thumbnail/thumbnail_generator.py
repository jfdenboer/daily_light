"""Integration with Bannerbear to produce YouTube-ready thumbnails."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from urllib.parse import urlparse

import requests

from spurgeon.config.settings import Settings
from spurgeon.models import Reading

logger = logging.getLogger(__name__)


class BannerbearAPIError(RuntimeError):
    """Raised when the Bannerbear API returns an error response."""


class BannerbearTimeoutError(TimeoutError):
    """Raised when polling Bannerbear exceeds the configured timeout."""


@dataclass(slots=True)
class _BannerbearJob:
    uid: str
    status: str
    image_url: str | None = None


class BannerbearClient:
    """Minimal wrapper around the Bannerbear REST API."""

    def __init__(self, api_key: str, *, base_url: str = "https://api.bannerbear.com/v2") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_image(
        self,
        template_id: str,
        modifications: Iterable[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
        webhook_url: str | None = None,
    ) -> _BannerbearJob:
        payload: Dict[str, Any] = {
            "template": template_id,
            "modifications": list(modifications),
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        if webhook_url:
            payload["webhook_url"] = webhook_url

        try:
            response = self._session.post(
                f"{self.base_url}/images",
                json=payload,
                headers=self._headers,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise BannerbearAPIError(f"Failed to call Bannerbear images endpoint: {exc}") from exc
        data = self._handle_response(response)
        uid = data.get("uid")
        if not isinstance(uid, str) or not uid:
            raise BannerbearAPIError("Bannerbear response missing job uid")
        status = data.get("status", "queued")
        image_url = data.get("image_url") or data.get("image_url_png")
        return _BannerbearJob(uid=uid, status=status, image_url=image_url)

    def retrieve_image(self, uid: str) -> _BannerbearJob:
        try:
            response = self._session.get(
                f"{self.base_url}/images/{uid}",
                headers=self._headers,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise BannerbearAPIError(f"Failed to retrieve Bannerbear image {uid}: {exc}") from exc
        data = self._handle_response(response)
        status = data.get("status", "")
        image_url = data.get("image_url") or data.get("image_url_png")
        return _BannerbearJob(uid=uid, status=status, image_url=image_url)

    def download(self, url: str, destination: Path) -> None:
        try:
            with self._session.get(url, timeout=60, stream=True) as response:
                if not response.ok:
                    raise BannerbearAPIError(
                        f"Failed to download thumbnail from Bannerbear: HTTP {response.status_code}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
        except requests.RequestException as exc:
            raise BannerbearAPIError(f"Failed to download thumbnail from Bannerbear: {exc}") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _handle_response(response: requests.Response) -> Dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise BannerbearAPIError("Invalid JSON response from Bannerbear") from exc

        if response.status_code >= 400:
            message = data.get("message") or data.get("error") or "Bannerbear API error"
            raise BannerbearAPIError(f"{message} (status={response.status_code})")

        if not isinstance(data, dict):
            raise BannerbearAPIError("Unexpected payload from Bannerbear")
        return data


class ThumbnailGenerationError(RuntimeError):
    """Raised when thumbnail generation fails in a recoverable way."""


class ThumbnailGenerator:
    """Create thumbnails for readings using Bannerbear templates."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.output_dir = Path(settings.output_dir) / "thumbnails"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not settings.bannerbear_api_key:
            logger.info("Bannerbear API key missing – thumbnail generation disabled")
            self.client = None
        else:
            base_url = (
                "https://sync.api.bannerbear.com/v2"
                if settings.bannerbear_use_sync_api
                else "https://api.bannerbear.com/v2"
            )
            if settings.bannerbear_use_sync_api:
                logger.debug("Using synchronous Bannerbear API endpoint")
            self.client = BannerbearClient(settings.bannerbear_api_key, base_url=base_url)

    def generate_thumbnail(
        self,
        reading: Reading,
        *,
        title: str,
        hero_image: Path | None = None,
        thumbnail_text: str | None = None,
    ) -> Path | None:
        if not self.client:
            return None

        template_id = self.settings.bannerbear_template_id
        if not template_id:
            logger.warning("Bannerbear template id missing – skipping thumbnail for %s", reading.slug)
            return None

        cached = self._find_existing_thumbnail(reading.slug)
        if cached:
            logger.debug("Reusing existing thumbnail %s", cached.name)
            return cached

        modifications = self._build_modifications(
            reading,
            title,
            hero_image,
            thumbnail_text,
        )
        metadata = {
            "slug": reading.slug,
            "reading_type": reading.reading_type.value,
            "date": reading.date.isoformat(),
            "project_name": self.settings.bannerbear_project_name,
            "template_id": template_id,
        }

        try:
            job = self.client.create_image(
                template_id,
                modifications,
                metadata=metadata,
                webhook_url=self.settings.bannerbear_webhook_url,
            )
        except BannerbearAPIError as exc:
            raise ThumbnailGenerationError(str(exc)) from exc

        if job.status == "completed" and job.image_url:
            completed = job
        else:
            try:
                completed = self._wait_for_completion(job.uid)
            except (BannerbearAPIError, BannerbearTimeoutError) as exc:
                raise ThumbnailGenerationError(str(exc)) from exc

        if not completed.image_url:
            raise ThumbnailGenerationError("Bannerbear job completed without an image URL")

        thumbnail_path = self._build_thumbnail_path(reading.slug, completed.image_url)
        try:
            self.client.download(completed.image_url, thumbnail_path)
        except BannerbearAPIError as exc:
            raise ThumbnailGenerationError(str(exc)) from exc

        logger.info("Saved thumbnail: %s", thumbnail_path.name)
        return thumbnail_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _wait_for_completion(self, uid: str) -> _BannerbearJob:
        assert self.client is not None  # appease type checkers

        timeout = float(self.settings.bannerbear_timeout_seconds)
        poll_interval = float(self.settings.bannerbear_poll_interval)
        deadline = time.monotonic() + timeout

        last_status = "queued"
        while True:
            if time.monotonic() > deadline:
                raise BannerbearTimeoutError(
                    f"Timeout waiting for Bannerbear image {uid} (last status={last_status})"
                )

            job = self.client.retrieve_image(uid)
            last_status = job.status
            if job.status == "completed":
                return job
            if job.status in {"failed", "errored", "error", "cancelled"}:
                raise BannerbearAPIError(
                    f"Bannerbear image {uid} failed with status '{job.status}'"
                )

            if job.status not in {"pending", "processing", "queued"}:
                logger.debug(
                    "Bannerbear image %s reported status %s – treating as pending",
                    uid,
                    job.status,
                )

            logger.debug("Bannerbear image %s status=%s – waiting", uid, job.status)
            time.sleep(poll_interval)

    def _find_existing_thumbnail(self, slug: str) -> Path | None:
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = self.output_dir / f"{slug}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _build_thumbnail_path(self, slug: str, image_url: str) -> Path:
        suffix = Path(urlparse(image_url).path).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            suffix = ".png"
        return self.output_dir / f"{slug}{suffix}"

    def _build_modifications(
            self,
            reading: Reading,
            title: str,
            hero_image: Path | None,
            thumbnail_text: str | None,
    ) -> List[Dict[str, Any]]:
        config_mods = self.settings.bannerbear_modifications or []
        if not config_mods:
            logger.warning(
                "No Bannerbear modifications configured – using fallback values"
            )
            config_mods = [
                {"name": "title", "text": "{thumbnail_text}"},
                {"name": "date_text", "text": "{date_text}"},
            ]

        context = self._build_context(reading, title, hero_image, thumbnail_text)
        resolved: List[Dict[str, Any]] = []
        for raw_mod in config_mods:
            mod: Dict[str, Any] = {}
            for key, value in raw_mod.items():
                if isinstance(value, str):
                    mod[key] = self._safe_format(value, context)
                else:
                    mod[key] = value
            resolved.append(mod)

        return resolved

    def _build_context(
            self,
            reading: Reading,
            title: str,
            hero_image: Path | None,
            thumbnail_text: str | None,
    ) -> Dict[str, Any]:
        date_long = reading.date.strftime("%B %d, %Y")
        month_name = reading.date.strftime("%B").upper()
        date_text = f"{month_name} {reading.date.day}"
        context: Dict[str, Any] = {
            "title": thumbnail_text or title,
            "thumbnail_text": thumbnail_text or title,
            "video_title": title,
            "date_iso": reading.date.isoformat(),
            "date_long": date_long,
            "date_text": date_text,
            "reading_type": reading.reading_type.value,
            "slug": reading.slug,
        }
        if hero_image:
            context["hero_image_path"] = str(hero_image)
            context["hero_image_name"] = hero_image.name
        return context

    @staticmethod
    def _safe_format(template: str, context: Mapping[str, Any]) -> str:
        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:  # pragma: no cover - simple helper
                return "{" + key + "}"

        return template.format_map(_SafeDict(context))

__all__ = [
    "ThumbnailGenerator",
    "ThumbnailGenerationError",
    "BannerbearClient",
    "BannerbearAPIError",
    "BannerbearTimeoutError",
]