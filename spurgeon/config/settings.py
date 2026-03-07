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

    # Thumbnail generation
    thumbnail_enabled: bool = Field(True, env="THUMBNAIL_ENABLED")
    thumbnail_image_model: Literal["gpt-image-1.5", "gpt-image-1", "gpt-image-1-mini"] = (
        "gpt-image-1.5"
    )
    thumbnail_image_size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1536x1024"
    thumbnail_image_quality: Literal["low", "medium", "high", "auto"] = "medium"
    thumbnail_image_background: Literal["transparent", "opaque", "auto"] = "opaque"
    thumbnail_intent_card_model: Literal["gpt-5.2", "gpt-5", "gpt-4o"] = "gpt-5.2"
    thumbnail_intent_card_temperature: float = Field(0.2, ge=0.0, le=2.0)
    thumbnail_max_retries: int = Field(3, ge=0)
    thumbnail_retry_backoff: float = Field(1.0, gt=0.0)
    thumbnail_font_path: str | None = Field(default=None, env="THUMBNAIL_FONT_PATH")
    thumbnail_prompt_version: str = Field("v1", env="THUMBNAIL_PROMPT_VERSION", min_length=1)
    thumbnail_cache_by_fingerprint: bool = Field(True, env="THUMBNAIL_CACHE_BY_FINGERPRINT")
    thumbnail_quality_checks_enabled: bool = Field(True, env="THUMBNAIL_QUALITY_CHECKS_ENABLED")
    thumbnail_quality_min_luma_stddev: float = Field(12.0, ge=0.0, env="THUMBNAIL_QUALITY_MIN_LUMA_STDDEV")

    # Bucket name
    gcs_bucket_name: str = Field("spurgeon_bucket", min_length=1)

    # Prompt
    prompt_model: Literal["gpt-4o"] = "gpt-4o"
    prompt_generator_model: Literal["gpt-5-chat-latest", "gpt-5", "gpt-4o"] = (
        "gpt-5-chat-latest"
    )
    prompt_temperature: float = Field(0.5, ge=0.0, le=2.0)
    prompt_generator_temperature: float = Field(0.7, ge=0.0, le=2.0)
    thumbnail_text_generator_model: str = Field("gpt-5.2", min_length=1)
    thumbnail_text_selector_model: str = Field("gpt-5.2", min_length=1)
    thumbnail_text_generator_temperature: float = Field(0.9, ge=0.0, le=2.0)
    thumbnail_text_selector_temperature: float = Field(0.1, ge=0.0, le=2.0)
    thumbnail_text_num_candidates: int = Field(8, ge=6, le=10)
    # Optional: determinisme voor promptmodel (gebruikt door prompt_generator indien ingesteld)
    prompt_seed: int | None = Field(default=None, ge=0, env="PROMPT_SEED")
    prompt_subject_tokens: int = Field(90, gt=0)

    # ElevenLabs
    elevenlabs_enable_v3: bool = Field(True)
    elevenlabs_voice_id: str = Field("pNInz6obpgDQGcFmaJgB", min_length=1)
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
    hook_tweaker_model: str = Field("gpt-5.2", env="HOOK_TWEAKER_MODEL")
    hook_generator_temperature: float = Field(0.9, ge=0.0, le=2.0)
    hook_judge_temperature: float = Field(0.1, ge=0.0, le=2.0)
    hook_tweaker_temperature: float = Field(0.45, ge=0.0, le=2.0, env="HOOK_TWEAKER_TEMPERATURE")
    hook_num_candidates: int = Field(10, ge=8, le=12)
    hook_tweaker_num_variants: int = Field(4, ge=3, le=6, env="HOOK_TWEAKER_NUM_VARIANTS")
    hook_style_profile: Literal["control", "curiosity", "consequence"] = Field(
        "control",
        env="HOOK_STYLE_PROFILE",
    )
    intro_timing_profile: Literal["balanced", "snappy", "spacious"] = Field(
        "balanced",
        env="INTRO_TIMING_PROFILE",
    )
    intro_telemetry_path: Path = Field(
        default_factory=lambda: Path("output") / "intro_telemetry" / "hook_events.jsonl",
        env="INTRO_TELEMETRY_PATH",
    )
    intro_telemetry_enabled: bool = Field(True, env="INTRO_TELEMETRY_ENABLED")
    intro_pause_pre_intro_ms: int = Field(200, ge=0)
    intro_pause_between_ms: int = Field(420, ge=0)
    intro_pause_after_credit_ms: int = Field(320, ge=0)
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
    # Subtitles (legacy defaults)
    srt_max_chars: int = Field(22, gt=0)
    srt_hard_max_chars: int = Field(27, gt=0)

    subtitle_font: str = Field("Inter", min_length=1)
    subtitle_size: int = Field(86, gt=0)  # fallback; varianten hieronder zijn leidend
    subtitle_style_extra: str = ""  # fallback; varianten hieronder zijn leidend
    subtitle_line_spacing: int = Field(12)
    subtitle_line_spacing_wide: int = Field(12)
    subtitle_line_spacing_short: int = Field(12)

    # Variant-specific subtitle styling
    subtitle_font_wide: str = Field("Inter", min_length=1)
    subtitle_size_wide: int = Field(86, gt=0)
    subtitle_style_extra_wide: str = (
        "WrapStyle=2,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"  # volledig opaak zwart voor maximale contrast
        "BorderStyle=1,"
        "Outline=2,"
        "Shadow=0,"
        "Alignment=5,"
        "MarginV=0,"
        "MarginL=120,"
        "MarginR=120"
    )

    subtitle_font_short: str = Field("Inter", min_length=1)
    subtitle_size_short: int = Field(86, gt=0)
    subtitle_style_extra_short: str = (
        "WrapStyle=2,"
        "PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,"
        "BorderStyle=1,"
        "Outline=2,"
        "Shadow=0,"
        "Alignment=5,"
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

    @field_validator("thumbnail_font_path", mode="before")
    @classmethod
    def _empty_to_none(cls, value: str | None):
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return value

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

        return self


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Return a cached instance of :class:`Settings`."""

    return Settings()


__all__ = ["Settings", "load_settings"]