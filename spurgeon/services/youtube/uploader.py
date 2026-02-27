# uploader.py – YouTube video uploader (OAuth 2.0 + resumable upload)

from __future__ import annotations

import json
import mimetypes
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Set

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError, ResumableUploadError
from googleapiclient.http import MediaFileUpload

from spurgeon.config.settings import Settings
from spurgeon.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)

# OAuth scope voor video upload
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
_RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
_RETRYABLE_RATE_REASONS = {"quotaExceeded", "userRateLimitExceeded", "rateLimitExceeded"}


class YouTubeUploadError(Exception):
    """Onherstelbare fout tijdens upload naar YouTube."""


class TransientYouTubeError(Exception):
    """Interne helper-exceptie voor retry-with-backoff."""


class YouTubeUploader:
    def __init__(
        self,
        client_secrets_path: Path,
        token_path: Path,
        settings: Optional[Settings] = None,
        *,
        max_retries: Optional[int] = None,
        backoff: Optional[float] = None,
        chunk_size: Optional[int] = None,
    ) -> None:
        self.client_secrets_path = client_secrets_path
        self.token_path = token_path
        self.settings = settings
        self.contains_synthetic_media = (
            settings.youtube_contains_synthetic_media if settings else None
        )

        default_retries = settings.youtube_max_retries if settings else 3
        default_backoff = settings.youtube_retry_backoff if settings else 1.0
        default_chunk_size = settings.youtube_chunk_size if settings else 8 * 1024 * 1024

        self.max_retries = max_retries if max_retries is not None else default_retries
        self.backoff = backoff if backoff is not None else default_backoff
        self.chunk_size = chunk_size if chunk_size is not None else default_chunk_size

        if self.chunk_size % (256 * 1024) != 0:
            raise ValueError("YouTube chunk size must be a multiple of 256 KB")

        self.service = self._authenticate()

    def _authenticate(self):
        creds = None
        if self.token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Kon bestaand OAuth-tokenbestand %s niet laden (%s). Nieuwe autorisatie vereist.",
                    self.token_path,
                    exc,
                )
                creds = None

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError as exc:
                logger.warning(
                    "OAuth-token in %s is verlopen of ingetrokken. Start nieuwe autorisatie (%s).",
                    self.token_path,
                    exc,
                )
                creds = None
                try:
                    self.token_path.unlink(missing_ok=True)
                except OSError as unlink_exc:  # pragma: no cover - zeldzame OS-fout
                    logger.debug(
                        "Kon ongeldig tokenbestand %s niet verwijderen: %s",
                        self.token_path,
                        unlink_exc,
                    )

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(self.client_secrets_path), SCOPES)
            creds = flow.run_local_server(port=8080, prompt="consent")
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            with self.token_path.open("w", encoding="utf-8") as f:
                f.write(creds.to_json())

        return build("youtube", "v3", credentials=creds)

    def upload_video(
        self,
        video_path: Path,
        title: str,
        description: str,
        tags: list[str],
        category_id: str = "22",  # "People & Blogs"
        privacy: str = "public",  # "private", "unlisted", "public"
        publish_at: Optional[datetime] = None,  # geplande publicatie (UTC datetime)
        contains_synthetic_media: Optional[bool] = None,
        thumbnail_path: Path | None = None,
    ) -> str:
        if not video_path.exists():
            raise YouTubeUploadError(f"Videobestand niet gevonden: {video_path}")

        logger.info("Uploaden van video: %s", video_path.name)

        status = {
            "privacyStatus": "private" if publish_at else privacy,
        }

        synthetic_flag = (
            contains_synthetic_media
            if contains_synthetic_media is not None
            else self.contains_synthetic_media
        )
        if synthetic_flag is not None:
            status["containsSyntheticMedia"] = synthetic_flag

        if publish_at:
            publish_at_utc = self._format_publish_time(publish_at)
            status["publishAt"] = publish_at_utc
            status["selfDeclaredMadeForKids"] = False  # verplicht veld bij geplande uploads
            logger.info("Video wordt gepland voor publicatie op %s", publish_at_utc)

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": category_id,
            },
            "status": status,
        }

        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=self.chunk_size,
        )

        request = self.service.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media,
        )

        try:
            response = self._resumable_upload(request, video_path.name)
        except HttpError as exc:  # fatal API error
            status_code = getattr(exc.resp, "status", "?")
            logger.error(
                "YouTube API-fout tijdens upload van %s (status %s): %s",
                video_path.name,
                status_code,
                exc,
            )
            raise YouTubeUploadError("YouTube API-fout tijdens upload") from exc
        except Exception as exc:
            logger.error("Upload van %s mislukt: %s", video_path.name, exc)
            raise YouTubeUploadError(f"Upload van {video_path.name} mislukt") from exc

        if not isinstance(response, dict) or "id" not in response:
            raise YouTubeUploadError("YouTube antwoord bevat geen video-id")

        video_id = response["id"]
        logger.info("✅ Video geüpload: https://youtu.be/%s", video_id)

        if thumbnail_path:
            try:
                self._set_thumbnail(video_id, thumbnail_path)
            except Exception as exc:
                logger.warning(
                    "Het uploaden van de thumbnail voor %s mislukte: %s",
                    video_id,
                    exc,
                )

        return video_id

    def _format_publish_time(self, publish_at: datetime) -> str:
        if publish_at.tzinfo is None:
            logger.warning("Ontvangen naive publish_at – veronderstel UTC")
            publish_at = publish_at.replace(tzinfo=timezone.utc)

        utc_time = publish_at.astimezone(timezone.utc).replace(microsecond=0)
        return utc_time.isoformat().replace("+00:00", "Z")

    def _resumable_upload(self, request, video_name: str):
        response = None
        context = f"YouTube upload chunk ({video_name})"
        while response is None:
            status, response = retry_with_backoff(
                func=lambda: self._next_chunk(request, video_name),
                max_retries=self.max_retries,
                backoff=self.backoff,
                error_types=(TransientYouTubeError,),
                context=context,
            )

            if status and hasattr(status, "progress"):
                progress = int(status.progress() * 100)
                logger.debug("Uploadvoortgang %d%% voor %s", progress, video_name)

        return response

    def _next_chunk(self, request, video_name: str):
        try:
            return request.next_chunk()
        except HttpError as exc:
            if self._is_retryable_http_error(exc):
                status_code = getattr(exc.resp, "status", "?")
                logger.warning(
                    "Tijdelijke YouTube HTTP-fout (%s) tijdens upload van %s – retry",
                    status_code,
                    video_name,
                )
                raise TransientYouTubeError(f"HTTP {status_code}: {exc}") from exc
            raise
        except ResumableUploadError as exc:
            logger.warning("Resumable upload error voor %s: %s – retry", video_name, exc)
            raise TransientYouTubeError(str(exc)) from exc
        except OSError as exc:
            logger.warning("Netwerkfout tijdens upload van %s: %s – retry", video_name, exc)
            raise TransientYouTubeError(str(exc)) from exc

    def _is_retryable_http_error(self, exc: HttpError) -> bool:
        status_code = getattr(exc.resp, "status", None)
        if status_code in _RETRYABLE_HTTP_STATUSES:
            return True
        if status_code == 403:
            reasons = self._extract_error_reasons(exc)
            return bool(reasons & _RETRYABLE_RATE_REASONS)
        return False

    def _extract_error_reasons(self, exc: HttpError) -> Set[str]:
        raw_content = getattr(exc, "content", None)
        if raw_content is None:
            return set()
        if isinstance(raw_content, bytes):
            content = raw_content.decode("utf-8", errors="ignore")
        else:
            content = str(raw_content)

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return set()

        reasons: Set[str] = set()
        error = payload.get("error")
        if not isinstance(error, dict):
            return reasons

        legacy_errors = error.get("errors")
        if isinstance(legacy_errors, list):
            for item in legacy_errors:
                if not isinstance(item, dict):
                    continue
                reason = item.get("reason")
                if isinstance(reason, str):
                    reasons.add(reason)

        details = error.get("details")
        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                reason = detail.get("reason")
                if isinstance(reason, str):
                    reasons.add(reason)
                metadata = detail.get("metadata")
                if isinstance(metadata, dict):
                    meta_reason = metadata.get("reason")
                    if isinstance(meta_reason, str):
                        reasons.add(meta_reason)

        status_reason = error.get("status")
        if isinstance(status_reason, str):
            reasons.add(status_reason)

        return reasons

    def _set_thumbnail(self, video_id: str, thumbnail_path: Path) -> None:
        if not thumbnail_path.exists():
            raise FileNotFoundError(f"Thumbnailbestand ontbreekt: {thumbnail_path}")

        mime_type, _ = mimetypes.guess_type(str(thumbnail_path))
        if not mime_type:
            mime_type = "image/png"

        logger.info("Thumbnail uploaden voor video %s (%s)", video_id, thumbnail_path.name)
        context = f"YouTube thumbnail upload ({video_id})"

        def _upload_thumbnail() -> None:
            media = MediaFileUpload(
                str(thumbnail_path),
                mimetype=mime_type,
                resumable=False,
            )

            request = self.service.thumbnails().set(
                videoId=video_id,
                media_body=media,
            )

            try:
                request.execute()
            except HttpError as exc:
                status_code = getattr(exc.resp, "status", "?")
                if self._is_transient_thumbnail_http_error(exc):
                    raise TransientYouTubeError(
                        f"HTTP {status_code}: {exc}"
                    ) from exc
                logger.warning(
                    "YouTube thumbnail upload faalde (status %s) voor %s: %s",
                    status_code,
                    video_id,
                    exc,
                )
                raise

        retry_with_backoff(
            _upload_thumbnail,
            max_retries=self.max_retries,
            backoff=self.backoff,
            error_types=(TransientYouTubeError,),
            context=context,
        )

    def _is_transient_thumbnail_http_error(self, exc: HttpError) -> bool:
        status_code = getattr(exc.resp, "status", None)
        if status_code in _RETRYABLE_HTTP_STATUSES or status_code == 409:
            return True
        if status_code == 403:
            reasons = self._extract_error_reasons(exc)
            return any("processing" in reason.lower() for reason in reasons)
        return False


__all__ = ["YouTubeUploader", "YouTubeUploadError"]
