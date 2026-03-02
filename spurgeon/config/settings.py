"""Central application configuration powered by :mod:`pydantic-settings`.

The goal of this module is to provide a single, well-validated source of truth
for all configuration that the application relies on. Settings are resolved
from environment variables (optionally loaded from a ``.env`` file) and are
validated to catch configuration issues as early as possible.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Spurgeon pipeline."""

    # API keys uit .env
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    elevenlabs_api_key: str = Field(..., env="ELEVENLABS_API_KEY")
    rev_ai_token: str = Field(..., env="REV_AI_TOKEN")
    gcs_credentials_path: Path = Field(..., env="GCS_CREDENTIALS_PATH")

    # Bannerbear thumbnail generation
    bannerbear_api_key: str | None = Field(default=None, env="BANNERBEAR_API_KEY")
    bannerbear_project_name: str = Field(
        "spurgeon",
        min_length=1,
        env="BANNERBEAR_PROJECT_NAME",
        description="Naam van het Bannerbear-project waarin onze templates leven.",
    )
    bannerbear_template_id: str | None = Field(
        default="RnxGpW5l7NXyZEXrJ1",
        min_length=1,
        env="BANNERBEAR_TEMPLATE_ID",
        description="UID van het Bannerbear-image-template dat als basis dient voor thumbnails.",
    )
    bannerbear_webhook_url: str | None = Field(
        default=None,
        env="BANNERBEAR_WEBHOOK",
        description="Optional webhook URL configured on the Bannerbear template.",
    )
    bannerbear_use_sync_api: bool = Field(
        False,
        env="BANNERBEAR_USE_SYNC_API",
        description="Switch between synchronous and asynchronous Bannerbear API endpoints.",
    )
    bannerbear_poll_interval: float = Field(2.0, gt=0.0)
    bannerbear_timeout_seconds: int = Field(90, gt=0)
    bannerbear_modifications: list[dict[str, str]] = Field(
        default_factory=lambda: [
            {"name": "title", "text": "{thumbnail_text}"},
            {"name": "date_text", "text": "{date_text}"},
        ]
    )

    # Bucket name
    gcs_bucket_name: str = Field("spurgeon_bucket", min_length=1)

    # Prompt
    prompt_model: Literal["gpt-4o"] = "gpt-4o"
    prompt_generator_model: Literal["gpt-5-chat-latest", "gpt-5", "gpt-4o"] = (
        "gpt-5-chat-latest"
    )
    prompt_temperature: float = Field(0.5, ge=0.0, le=2.0)
    prompt_generator_temperature: float = Field(0.7, ge=0.0, le=2.0)
    # Optional: determinisme voor promptmodel (gebruikt door prompt_generator indien ingesteld)
    prompt_seed: int | None = Field(default=None, ge=0, env="PROMPT_SEED")
    prompt_subject_tokens: int = Field(90, gt=0)

    # ElevenLabs
    elevenlabs_enable_v3: bool = Field(True)
    elevenlabs_voice_id: str = Field("onwK4e9ZLuTAKqWW03F9", min_length=1)
    elevenlabs_model_id: str = Field("eleven_v3", min_length=1)
    elevenlabs_voice_speed: float = Field(0.95, gt=0.0)
    elevenlabs_output_format: str = Field("mp3_44100_128", min_length=1)
    elevenlabs_max_retries: int = Field(2, ge=0)
    elevenlabs_retry_backoff: float = Field(1.0, gt=0.0)
    elevenlabs_v3_max_chars: int = Field(3000, gt=0)
    elevenlabs_language_code: str | None = Field(default="en")
    elevenlabs_dialogue_seed: int | None = Field(default=None)
    elevenlabs_allow_pcm_chunk_join: bool = Field(True)
    elevenlabs_streaming: bool = Field(False)

    # Intro audio
    intro_enabled: bool = Field(True)
    intro_cache_enabled: bool = Field(True)
    hook_generator_model: str = Field("gpt-5.2", env="HOOK_GENERATOR_MODEL")
    hook_judge_model: str = Field("gpt-5.2", env="HOOK_JUDGE_MODEL")
    hook_generator_temperature: float = Field(0.9, ge=0.0, le=2.0)
    hook_judge_temperature: float = Field(0.1, ge=0.0, le=2.0)
    hook_num_candidates: int = Field(10, ge=8, le=12)
    intro_pause_pre_intro_ms: int = Field(120, ge=0)
    intro_pause_between_ms: int = Field(550, ge=0)
    intro_pause_after_credit_ms: int = Field(550, ge=0)
    intro_fail_open: bool = Field(True)
    intro_hook_fail_strategy: Literal["credit_only", "skip_intro", "raise"] = Field(
        "credit_only"
    )

    elevenlabs_supported_output_formats: ClassVar[set[str]] = {
        "mp3_22050_32",
        "mp3_44100_32",
        "mp3_44100_64",
        "mp3_44100_96",
        "mp3_44100_128",
        "mp3_44100_192",
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
        "pcm_44100",
        "ulaw_8000",
        "alaw_8000",
        "opus_48000",
    }

    @field_validator("elevenlabs_output_format")
    @classmethod
    def _validate_elevenlabs_output_format(cls, value: str) -> str:
        fmt = value.strip()
        if fmt not in cls.elevenlabs_supported_output_formats:
            raise ValueError(
                "Unsupported ElevenLabs output format: "
                f"{fmt}. Supported values: {sorted(cls.elevenlabs_supported_output_formats)}"
            )
        return fmt

    @field_validator("video_visualizer_mode")
    @classmethod
    def _validate_visualizer_mode(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in {"bar", "line", "dot"}:
            raise ValueError("video_visualizer_mode must be one of: bar, line, dot")
        return mode

    @property
    def elevenlabs_audio_extension(self) -> str:
        fmt = self.elevenlabs_output_format
        if fmt.startswith("mp3"):
            return ".mp3"
        if fmt.startswith("pcm"):
            return ".wav"
        if fmt.startswith("ulaw"):
            return ".wav"
        if fmt.startswith("alaw"):
            return ".wav"
        if fmt.startswith("opus"):
            return ".opus"
        return ".bin"

    # Rev.ai
    rev_ai_language: str = Field("en", min_length=2)

    # GCS
    input_dir: Path = Field(default_factory=lambda: Path("input"))
    output_dir: Path = Field(default_factory=lambda: Path("output"))

    # YouTube uploads
    youtube_max_retries: int = Field(3, ge=0)
    youtube_retry_backoff: float = Field(1.0, gt=0.0)
    youtube_chunk_size: int = Field(8 * 1024 * 1024, gt=0)  # 8 MB chunks
    youtube_contains_synthetic_media: bool = Field(
        True,
        description=(
            "Mark uploaded videos as containing altered or synthetic media so that "
            "YouTube adds the required disclosure."
        ),
    )

    # Visual strategy
    visual_mode: Literal["single_image"] = "single_image"


    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: Path = Field(
        default_factory=lambda: Path("logs") / "daily_light.log",
        description="Location of the rotating log file.",
    )
    log_file_max_bytes: int = Field(10 * 1024 * 1024, gt=0)
    log_file_backup_count: int = Field(5, ge=0)

    # Video rendering
    video_parallel_workers: int = Field(1, gt=0)
    video_timeout: int = Field(300, gt=0)
    ffmpeg_cmd: str = Field("ffmpeg", min_length=1)
    video_fps: int = Field(30, gt=0)
    video_codec: str = Field("libx264", min_length=1)
    audio_bitrate: str = Field("192k", min_length=1)
    video_zoom_wide_enabled: bool = Field(True)
    video_zoom_wide_start: float = Field(1.0, gt=0.0)
    video_zoom_wide_end: float = Field(1.10, gt=0.0)
    video_visualizer_enabled: bool = Field(False)
    video_visualizer_mode: str = Field("bar", min_length=1)
    video_visualizer_alpha: float = Field(0.70, ge=0.0, le=1.0)
    video_visualizer_width_wide: int = Field(640, gt=0)
    video_visualizer_height_wide: int = Field(180, gt=0)
    video_visualizer_margin_top: int = Field(40, gt=0)
    video_visualizer_margin_right: int = Field(40, gt=0)
    video_visualizer_ascale: str = Field("log", min_length=1)
    video_visualizer_fscale: str = Field("log", min_length=1)

    # Subtitles (legacy defaults)
    srt_max_chars: int = Field(22, gt=0)
    srt_hard_max_chars: int = Field(27, gt=0)

    subtitle_font: str = Field("Inter", min_length=1)
    subtitle_size: int = Field(84, gt=0)  # fallback; varianten hieronder zijn leidend
    subtitle_style_extra: str = ""  # fallback; varianten hieronder zijn leidend
    subtitle_line_spacing: int = Field(12)
    subtitle_line_spacing_wide: int = Field(12)
    subtitle_line_spacing_short: int = Field(12)

    # Variant-specific subtitle styling
    subtitle_font_wide: str = Field("Inter", min_length=1)
    subtitle_size_wide: int = Field(84, gt=0)
    subtitle_style_extra_wide: str = (
        "WrapStyle=2,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"  # volledig opaak zwart voor maximale contrast
        "BorderStyle=1,"
        "Outline=2,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=0,"
        "MarginL=120,"
        "MarginR=120"
    )

    subtitle_font_short: str = Field("Inter", min_length=1)
    subtitle_size_short: int = Field(84, gt=0)
    subtitle_style_extra_short: str = (
        "WrapStyle=2,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=2,"
        "Shadow=0,"
        "Alignment=2,"
        "MarginV=0,"
        "MarginL=100,"
        "MarginR=100"
    )

    # Opschonen
    temp_retention_hours: int = Field(24, gt=0)

    # Image generation (flattened)
    image_provider: str = Field("openai", min_length=1)
    image_model: Literal["gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini"] = (
        "gpt-image-1.5"
    )
    image_size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1536x1024"
    image_quality: Literal["low", "medium", "high", "auto"] = "low"
    image_background: Literal["transparent", "opaque", "auto"] = "opaque"
    image_max_retries: int = Field(3, ge=0)
    image_retry_backoff: float = Field(1.5, gt=0.0)
    image_max_input_chars: int = Field(700, gt=0)
    image_max_prompt_tokens: int = Field(150, gt=0)
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def image_file_extension(self) -> str:
        """Return the file extension used for generated images."""

        return "png"

    @field_validator("gcs_credentials_path", "input_dir", "output_dir", "log_file", mode="before")
    @classmethod
    def _coerce_path(cls, value: str | Path) -> Path:
        """Ensure path-based fields are stored as :class:`~pathlib.Path` objects."""

        if isinstance(value, Path):
            return value.expanduser().resolve()
        return Path(value).expanduser().resolve()

    @field_validator("bannerbear_project_name")
    @classmethod
    def _normalise_project_name(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("bannerbear_project_name cannot be empty")
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("bannerbear_project_name cannot be empty")
        return cleaned

    @field_validator("bannerbear_api_key", "bannerbear_template_id", mode="before")
    @classmethod
    def _empty_to_none(cls, value: str | None):
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return value

    @field_validator("bannerbear_modifications", mode="before")
    @classmethod
    def _parse_modifications(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                import json

                parsed = json.loads(value)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    "BANNERBEAR_MODIFICATIONS must be valid JSON"
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError("BANNERBEAR_MODIFICATIONS must decode to a list")
            value = parsed

        if not isinstance(value, list):
            raise TypeError("bannerbear_modifications must be a list of mappings")

        cleaned: list[dict[str, str]] = []
        for entry in value:
            if not isinstance(entry, dict):
                raise TypeError("Each Bannerbear modification must be a mapping")
            if "name" not in entry or not str(entry["name"]).strip():
                raise ValueError("Each Bannerbear modification requires a non-empty name")
            cleaned.append({str(k): str(v) for k, v in entry.items()})
        return cleaned

    @model_validator(mode="after")
    def _post_validate(self) -> Settings:
        """Perform validations that depend on multiple fields."""


        if not self.gcs_credentials_path.exists():
            raise FileNotFoundError(
                f"GCS credentials file not found: {self.gcs_credentials_path}"
            )

        for directory in (self.input_dir, self.output_dir, self.log_file.parent):
            directory.mkdir(parents=True, exist_ok=True)

        zoom_delta = abs(self.video_zoom_wide_end - self.video_zoom_wide_start)
        if zoom_delta - 0.12 > 1e-9:
            raise ValueError("video_zoom_wide range must stay within 0.12 to prevent extreme motion")

        if self.bannerbear_api_key and not self.bannerbear_template_id:
            raise ValueError(
                "bannerbear_template_id must be configured when bannerbear_api_key is provided"
            )

        return self


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Return a cached instance of :class:`Settings`."""

    return Settings()


__all__ = ["Settings", "load_settings"]