"""Application-facing thumbnail generator wrapper around :class:`ThumbnailService`."""

from __future__ import annotations

import logging
from pathlib import Path

from openai import OpenAI

from spurgeon.config.settings import Settings
from spurgeon.models import Reading

from .thumbnail_adapters import (
    FilesystemThumbnailRepository,
    OpenAIImageProvider,
    PillowThumbnailRenderer,
)
from .thumbnail_contracts import (
    ImageProvider,
    ThumbnailRenderer,
    ThumbnailRepository,
)
from .thumbnail_errors import ThumbnailGenerationError
from .thumbnail_copy_intent_card import ThumbnailCopyIntentCard, ThumbnailCopyIntentCardBuilder
from .thumbnail_bucketed_candidate_generator import (
    ThumbnailBucketedCandidateGenerator,
    ThumbnailCandidateBuckets,
)
from .thumbnail_candidate_critic import ThumbnailCandidateCritic, ThumbnailCandidateCriticResult
from .thumbnail_pairwise_tournament import (
    ThumbnailPairwiseTournament,
    ThumbnailPairwiseTournamentResult,
)
from .thumbnail_final_selector import ThumbnailFinalSelection, ThumbnailFinalSelector
from .thumbnail_image_intent_card import ThumbnailImageIntentCard, ThumbnailImageIntentCardBuilder
from .thumbnail_prompting import (
    THUMBNAIL_BACKGROUND_LINE,
    THUMBNAIL_COMPOSITION_LINE,
    THUMBNAIL_CONSTRAINTS_LINE,
    THUMBNAIL_LIGHTING_LINE,
    THUMBNAIL_PALETTE_LINE,
    THUMBNAIL_STYLE_LINE,
    THUMBNAIL_SUBJECT_LINE,
)
from .thumbnail_reading_diagnoser import ThumbnailReadingDiagnosis, ThumbnailReadingDiagnoser
from .thumbnail_service import ThumbnailService
from .thumbnail_source_reader import ThumbnailSourceRead, ThumbnailSourceReader

logger = logging.getLogger(__name__)


class ThumbnailGenerator:
    """Public thumbnail API that delegates image rendering work to :class:`ThumbnailService`."""

    def __init__(
        self,
        settings: Settings,
        *,
        image_provider: ImageProvider | None = None,
        renderer: ThumbnailRenderer | None = None,
        repository: ThumbnailRepository | None = None,
    ) -> None:
        self.settings = settings

        provider_client = None
        if image_provider is None:
            provider_client = OpenAI(api_key=settings.openai_api_key)

        output_dir = Path(settings.output_dir) / "thumbnails"
        self.source_reader = ThumbnailSourceReader(settings, provider_client)
        self.reading_diagnoser = ThumbnailReadingDiagnoser(settings, provider_client)
        self.image_intent_card_builder = ThumbnailImageIntentCardBuilder(settings, provider_client)
        self.copy_intent_card_builder = ThumbnailCopyIntentCardBuilder(settings, provider_client)
        self.bucketed_candidate_generator = ThumbnailBucketedCandidateGenerator(settings, provider_client)
        self.candidate_critic = ThumbnailCandidateCritic(settings, provider_client)
        self.pairwise_tournament = ThumbnailPairwiseTournament(settings, provider_client)
        self.final_selector = ThumbnailFinalSelector(settings, provider_client)
        self.service = ThumbnailService(
            settings=settings,
            image_provider=image_provider or OpenAIImageProvider(provider_client, settings),
            renderer=renderer or PillowThumbnailRenderer(settings),
            repository=repository or FilesystemThumbnailRepository(output_dir),
            source_reader=self.source_reader,
            reading_diagnoser=self.reading_diagnoser,
            image_intent_card_builder=self.image_intent_card_builder,
        )

    def read_thumbnail_source(self, reading: Reading) -> ThumbnailSourceRead:
        """Run step 0 source-read as a standalone thumbnail pipeline step."""

        return self.source_reader.read(reading)

    def diagnose_thumbnail_reading(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
    ) -> ThumbnailReadingDiagnosis:
        """Run step 1 reading diagnosis from reading + step-0 source read."""

        return self.reading_diagnoser.diagnose(reading, source_read)


    def build_thumbnail_image_intent_card(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
    ) -> ThumbnailImageIntentCard:
        """Run step 2A image intent card from reading + step-0 + step-1."""

        return self.image_intent_card_builder.build(reading, source_read, diagnosis)

    def build_thumbnail_copy_intent_card(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
    ) -> ThumbnailCopyIntentCard:
        """Run step 2B copy intent card from reading + step-0 + step-1."""

        return self.copy_intent_card_builder.build(reading, source_read, diagnosis)

    def generate_thumbnail_candidate_buckets(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
    ) -> ThumbnailCandidateBuckets:
        """Run step 3 bucketed candidate generation for v3 text hooks."""

        return self.bucketed_candidate_generator.generate(
            reading,
            source_read,
            diagnosis,
            copy_intent_card,
        )


    def critique_thumbnail_candidates(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
    ) -> ThumbnailCandidateCriticResult:
        """Run step 4 independent critic pass for bucketed v3 candidates."""

        return self.candidate_critic.critique(
            reading,
            source_read,
            diagnosis,
            copy_intent_card,
            candidate_buckets,
        )

    def run_thumbnail_pairwise_tournament(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
    ) -> ThumbnailPairwiseTournamentResult:
        """Run step 5 pairwise tournament over v3 candidates."""

        return self.pairwise_tournament.run(
            reading,
            source_read,
            diagnosis,
            copy_intent_card,
            candidate_buckets,
            critic_result,
        )


    def select_thumbnail_final_candidate(
        self,
        reading: Reading,
        source_read: ThumbnailSourceRead,
        diagnosis: ThumbnailReadingDiagnosis,
        copy_intent_card: ThumbnailCopyIntentCard,
        candidate_buckets: ThumbnailCandidateBuckets,
        critic_result: ThumbnailCandidateCriticResult,
        pairwise_result: ThumbnailPairwiseTournamentResult,
    ) -> ThumbnailFinalSelection:
        """Run step 6 final selector over full v3 context."""

        return self.final_selector.select(
            reading,
            source_read,
            diagnosis,
            copy_intent_card,
            candidate_buckets,
            critic_result,
            pairwise_result,
        )

    def generate_thumbnail(
        self,
        reading: Reading,
        *,
        hero_image: Path | None = None,
        thumbnail_text: str | None = None,
    ) -> Path | None:
        del hero_image  # reserved for future use

        try:
            return self.service.generate(reading, thumbnail_text=thumbnail_text or "spurgeon")
        except Exception as exc:
            if isinstance(exc, ThumbnailGenerationError):
                raise
            raise ThumbnailGenerationError(str(exc)) from exc


__all__ = [
    "ThumbnailGenerator",
    "ThumbnailGenerationError",
    "ThumbnailImageIntentCard",
    "ThumbnailCopyIntentCard",
    "ThumbnailCandidateBuckets",
    "ThumbnailCandidateCriticResult",
    "ThumbnailPairwiseTournamentResult",
    "ThumbnailFinalSelection",
    "THUMBNAIL_STYLE_LINE",
    "THUMBNAIL_SUBJECT_LINE",
    "THUMBNAIL_BACKGROUND_LINE",
    "THUMBNAIL_COMPOSITION_LINE",
    "THUMBNAIL_CONSTRAINTS_LINE",
    "THUMBNAIL_PALETTE_LINE",
    "THUMBNAIL_LIGHTING_LINE",
]
