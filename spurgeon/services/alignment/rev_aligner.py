# rev_aligner.py

from __future__ import annotations

"""RevAligner – forced-alignment via Rev.ai (upload → poll → JSON→SRT)."""

import logging
import time
import json
from pathlib import Path
from typing import Any, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from spurgeon.config.settings import Settings
from spurgeon.utils.gcs_uploader import GCSUploader
from spurgeon.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)


class RevAligner:
    """Force-alignment service wrapper around the Rev.ai REST API."""

    BASE_URL = "https://api.rev.ai/alignment/v1"
    DEFAULT_POLL_INTERVAL: float = 5.0
    DEFAULT_MAX_ATTEMPTS: int = 120

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.token = settings.rev_ai_token
        self.uploader = GCSUploader(settings)

        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        total_sec = max(0.0, seconds)
        hrs, rem = divmod(int(total_sec), 3600)
        mins, secs = divmod(rem, 60)
        ms = int((total_sec - int(total_sec)) * 1000)
        ms = min(max(ms, 0), 999)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

    def _json_to_srt(self, transcript: Dict[str, Any], srt_path: Path) -> int:
        def collect_words() -> list[tuple[float, float, str]]:
            out: list[tuple[float, float, str]] = []
            if "words" in transcript:
                for w in transcript["words"]:
                    start = w.get("start") or w.get("ts")
                    end = w.get("end") or w.get("end_ts")
                    text = w.get("alignedWord") or w.get("value") or w.get("text")
                    if start is not None and end is not None and text:
                        out.append((float(start), float(end), str(text)))
            else:
                for mono in transcript.get("monologues", []):
                    for elem in mono.get("elements", []):
                        if elem.get("type") != "text":
                            continue
                        start = (
                            elem.get("ts")
                            or elem.get("start_ts")
                            or elem.get("start")
                            or elem.get("timestamp")
                        )
                        end = (
                            elem.get("end_ts")
                            or elem.get("end")
                            or elem.get("end_time")
                        )
                        text = elem.get("value") or elem.get("text")
                        if start is not None and end is not None and text:
                            out.append((float(start), float(end), str(text)))
            return out

        entries = collect_words()
        if not entries:
            raise ValueError("No words found in Rev.ai transcript JSON")

        with srt_path.open("w", encoding="utf-8") as fh:
            for idx, (start, end, text) in enumerate(entries, start=1):
                fh.write(f"{idx}\n")
                fh.write(f"{self._fmt_ts(start)} --> {self._fmt_ts(end)}\n")
                fh.write(f"{text}\n\n")
        return len(entries)

    def align(
        self,
        reading_slug: str,
        transcript_text: str,
        audio_path: Path,
    ) -> Path:
        srt_dir = Path(self.settings.output_dir) / "subtitles"
        srt_dir.mkdir(parents=True, exist_ok=True)
        srt_path = srt_dir / f"{reading_slug}.srt"

        blob_name = f"audio/{reading_slug}{audio_path.suffix}"
        media_url = self.uploader.upload_file(
            local_path=audio_path,
            destination_blob_name=blob_name,
            make_public=False,
        )
        logger.debug("Uploaded audio to GCS: %s", media_url)

        payload: Dict[str, Any] = {
            "source_config": {"url": media_url},
            "transcript_text": transcript_text,
            "language": self.settings.rev_ai_language,
            "metadata": reading_slug,
        }

        def submit_job() -> requests.Response:
            return self.session.post(
                f"{self.BASE_URL}/jobs", json=payload, timeout=30
            )

        resp = retry_with_backoff(
            func=submit_job,
            max_retries=3,
            backoff=2.0,
            error_types=(requests.RequestException,),
            context=f"submit alignment job for {reading_slug}",
        )

        resp.raise_for_status()
        job_id = resp.json().get("id")
        if not job_id:
            raise RuntimeError("Failed to obtain job ID from Rev.ai response")
        logger.info("Submitted Rev.ai alignment job %s", job_id)

        poll_interval = float(
            getattr(self.settings, "alignment_poll_interval", self.DEFAULT_POLL_INTERVAL)
        )
        max_attempts = int(
            getattr(self.settings, "alignment_max_attempts", self.DEFAULT_MAX_ATTEMPTS)
        )
        status_url = f"{self.BASE_URL}/jobs/{job_id}"

        for attempt in range(1, max_attempts + 1):
            time.sleep(poll_interval)

            def poll_status() -> requests.Response:
                return self.session.get(status_url, timeout=10)

            try:
                status_resp = retry_with_backoff(
                    func=poll_status,
                    max_retries=1,
                    backoff=1.0,
                    error_types=(requests.RequestException,),
                    context=f"polling job {job_id}",
                )
                status_resp.raise_for_status()
                status = status_resp.json().get("status", "").lower()
            except Exception as e:
                logger.warning("Poll %d/%d failed for job %s: %s", attempt, max_attempts, job_id, e)
                continue

            logger.debug("Job %s status: %s", job_id, status)
            if status == "completed":
                break
            if status == "failed":
                raise RuntimeError(f"Rev.ai alignment job {job_id} failed")
        else:
            raise TimeoutError(f"Alignment job {job_id} did not complete after {poll_interval * max_attempts}s")

        transcript_url = f"{self.BASE_URL}/jobs/{job_id}/transcript"
        headers = {"Accept": "application/vnd.rev.transcript.v1.0+json"}
        transcript_resp = self.session.get(transcript_url, headers=headers, timeout=15)
        transcript_resp.raise_for_status()
        transcript_json = transcript_resp.json()
        logger.info("Fetched transcript JSON for job %s", job_id)

        # Nieuw: bewaar word-based SRT-bestand
        words_srt_path = srt_dir / "words" / f"{reading_slug}.words.srt"
        words_srt_path.parent.mkdir(parents=True, exist_ok=True)
        count = self._json_to_srt(transcript_json, words_srt_path)
        logger.info("Wrote %d word-level captions to %s", count, words_srt_path)

        # Optioneel: bewaar originele transcript JSON
        json_path = srt_dir / "json" / f"{reading_slug}.rev.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(transcript_json, indent=2))
        logger.info("Saved Rev.ai transcript JSON to %s", json_path)

        return words_srt_path
