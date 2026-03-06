"""Generate a spoken hook from a reading via a generator+judge pipeline."""

from __future__ import annotations

import hashlib
import logging
import re

from openai import OpenAI

from spurgeon.config.settings import Settings
from spurgeon.services.intro.hook_judge_scorecard import parse_hook_judge_scorecard
from spurgeon.services.intro.hook_selector import RankedHookCandidate, select_by_scorecard
from spurgeon.services.intro.hook_pipeline.openai_utils import (
    create_response_with_transport_retries,
    extract_response_text,
)
from spurgeon.services.intro.hook_pipeline.prompts import (
    HOOK_EDITORIAL_SELECTOR_DEVMSG,
    HOOK_FINAL_RESELECT_DEVMSG,
    HOOK_GENERATOR_DEVMSG,
    HOOK_INTENT_DEVMSG,
    HOOK_JUDGE_DEVMSG,
    HOOK_TWEAKER_DEVMSG,
    get_hook_style_instruction,
)
from spurgeon.services.intro.hook_pipeline.rules import (
    WORD_PATTERN,
    CandidateCheck,
    HookOutcome,
    IntentCard,
    build_editorial_selector_user_input,
    build_final_reselection_user_input,
    build_generator_user_input,
    build_intent_user_input,
    build_judge_user_input,
    normalize_hook_punctuation,
    parse_intent_card,
    parse_numbered_candidates,
    parse_tweaker_variants,
    prepare_reading,
    strip_angle_tag,
    validate_candidate,
)
from spurgeon.services.intro.hook_pipeline.telemetry import append_hook_event

logger = logging.getLogger(__name__)
SCORE_JUDGE_MAX_OUTPUT_TOKENS = 360
SELECTOR_WINNER_PATTERN = re.compile(r"^winner\s*=\s*(\d+)\s*$", re.IGNORECASE)
SELECTOR_RATIONALE_PATTERN = re.compile(r"^rationale\s*=\s*(.+)$", re.IGNORECASE)


class SpokenHookValidationError(ValueError):
    """Raised when generated spoken hook violates output constraints."""


def _parse_selection_output(raw_output: str, *, max_winner: int) -> tuple[int, str] | None:
    winner: int | None = None
    rationale: str | None = None

    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        winner_match = SELECTOR_WINNER_PATTERN.match(stripped)
        if winner_match:
            winner = int(winner_match.group(1))
            continue

        rationale_match = SELECTOR_RATIONALE_PATTERN.match(stripped)
        if rationale_match:
            rationale = " ".join(rationale_match.group(1).split())

    if winner is None or rationale is None:
        return None
    if not 1 <= winner <= max_winner:
        return None
    return winner, rationale


def _pick_editorial_winner(
    client: OpenAI,
    *,
    settings: Settings,
    style_profile: str,
    intent_card: IntentCard,
    shortlist_top4: list[CandidateCheck],
) -> tuple[CandidateCheck, str, bool]:
    response = create_response_with_transport_retries(
        client,
        model=settings.hook_judge_model,
        temperature=min(settings.hook_judge_temperature, 0.20),
        top_p=1.0,
        max_output_tokens=200,
        input=[
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": HOOK_EDITORIAL_SELECTOR_DEVMSG}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_editorial_selector_user_input(
                            style_profile=style_profile,
                            intent=intent_card,
                            candidates=shortlist_top4,
                        ),
                    }
                ],
            },
        ],
    )

    parsed = _parse_selection_output(
        extract_response_text(response),
        max_winner=len(shortlist_top4),
    )
    if parsed is None:
        return shortlist_top4[0], "Editorial parse fallback kept first shortlist option.", False

    winner_idx, rationale = parsed
    return shortlist_top4[winner_idx - 1], rationale, True


def _tweak_winner(client: OpenAI, *, winner: str, settings: Settings) -> list[str]:
    response = create_response_with_transport_retries(
        client,
        model=settings.hook_tweaker_model,
        temperature=settings.hook_tweaker_temperature,
        top_p=1.0,
        max_output_tokens=200,
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": HOOK_TWEAKER_DEVMSG.format(
                            num_variants=settings.hook_tweaker_num_variants
                        ),
                    }
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": winner}]},
        ],
    )

    return parse_tweaker_variants(
        extract_response_text(response), settings.hook_tweaker_num_variants
    )


def _build_tweak_pool(winner: str, variants: list[str]) -> list[str]:
    """Return a de-duplicated tweak pool with the original winner as first option."""

    normalized_winner = normalize_hook_punctuation(winner)
    pool: list[str] = [normalized_winner]
    seen = {normalized_winner.lower()}

    for variant in variants:
        normalized_variant = normalize_hook_punctuation(variant)
        key = normalized_variant.lower()
        if key in seen:
            continue
        seen.add(key)
        pool.append(normalized_variant)

    return pool


def _select_final_variant(
    client: OpenAI,
    *,
    settings: Settings,
    candidates: list[CandidateCheck],
) -> tuple[str, str, bool]:
    response = create_response_with_transport_retries(
        client,
        model=settings.hook_judge_model,
        temperature=min(settings.hook_judge_temperature, 0.10),
        top_p=1.0,
        max_output_tokens=200,
        input=[
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": HOOK_FINAL_RESELECT_DEVMSG}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_final_reselection_user_input(candidates),
                    }
                ],
            },
        ],
    )

    parsed = _parse_selection_output(extract_response_text(response), max_winner=len(candidates))
    if parsed is None:
        return candidates[0].candidate, "Final selection parse fallback kept original winner.", False

    winner_idx, rationale = parsed
    return candidates[winner_idx - 1].candidate, rationale, True


def _select_best_effort_candidate(candidates: list[CandidateCheck]) -> CandidateCheck:
    """Choose candidate with fewest validator failures as best-effort fallback."""

    return sorted(candidates, key=lambda item: (len(item.reasons), item.candidate))[0]


def generate_spoken_hook(reading: str, settings: Settings) -> str:
    """Return a validated spoken hook string for the supplied reading text."""

    client = OpenAI(api_key=settings.openai_api_key)
    prepared_reading = prepare_reading(reading)
    reading_hash = hashlib.sha256(prepared_reading.encode("utf-8")).hexdigest()
    style_profile = settings.hook_style_profile

    logger.info(
        "hook_pipeline.start generator_temperature=%.2f judge_temperature=%.2f tweaker_temperature=%.2f tweaker_variants=%d style_profile=%s",
        settings.hook_generator_temperature,
        settings.hook_judge_temperature,
        settings.hook_tweaker_temperature,
        settings.hook_tweaker_num_variants,
        style_profile,
    )

    intent_response = create_response_with_transport_retries(
        client,
        model=settings.hook_generator_model,
        temperature=0.2,
        top_p=1.0,
        max_output_tokens=140,
        input=[
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": HOOK_INTENT_DEVMSG}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_intent_user_input(prepared_reading),
                    }
                ],
            },
        ],
    )
    intent_card = parse_intent_card(extract_response_text(intent_response))

    logger.info(
        "hook_pipeline.intent_card core_tension=%s implicit_choice=%s likely_consequence=%s emotional_tone=%s",
        intent_card.core_tension,
        intent_card.implicit_choice,
        intent_card.likely_consequence,
        intent_card.emotional_tone,
    )

    generator_response = create_response_with_transport_retries(
        client,
        model=settings.hook_generator_model,
        temperature=settings.hook_generator_temperature,
        top_p=0.95,
        max_output_tokens=200,
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            HOOK_GENERATOR_DEVMSG.format(
                                num_candidates=settings.hook_num_candidates
                            )
                            + "\n\nStyle profile instruction:\n"
                            + get_hook_style_instruction(style_profile)
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_generator_user_input(
                            prepared_reading,
                            intent_card,
                            num_candidates=settings.hook_num_candidates,
                        ),
                    }
                ],
            },
        ],
    )

    parsed_candidates = [
        normalize_hook_punctuation(candidate)
        for candidate in parse_numbered_candidates(extract_response_text(generator_response))
    ]
    angled_candidates: list[tuple[str, str]] = []
    for raw_candidate in parsed_candidates:
        angle, candidate_text = strip_angle_tag(raw_candidate)
        angled_candidates.append((angle, normalize_hook_punctuation(candidate_text)))

    if len(parsed_candidates) < settings.hook_num_candidates:
        logger.warning(
            "hook_pipeline.generator_candidates_underflow parsed=%d expected=%d",
            len(parsed_candidates),
            settings.hook_num_candidates,
        )

    checked = [
        CandidateCheck(candidate=c, reasons=validate_candidate(c), angle=a)
        for a, c in angled_candidates
    ]
    valid = [item for item in checked if not item.reasons]
    candidate_stats = [
        {
            "candidate": item.candidate,
            "angle": item.angle,
            "word_count": len(WORD_PATTERN.findall(item.candidate)),
            "reasons": item.reasons,
            "is_valid": not item.reasons,
        }
        for item in checked
    ]
    logger.info(
        "hook_pipeline.candidate_stats total=%d valid=%d stats=%s",
        len(checked),
        len(valid),
        candidate_stats,
    )

    if not checked:
        _log_hook_event(
            settings,
            reading_hash=reading_hash,
            intent_card={
                "core_tension": intent_card.core_tension,
                "implicit_choice": intent_card.implicit_choice,
                "likely_consequence": intent_card.likely_consequence,
                "emotional_tone": intent_card.emotional_tone,
            },
            candidate_stats=candidate_stats,
            outcome_status="NARRATION_ONLY",
            selected_source="failure",
            selected_candidate=None,
            selected_angle="unknown",
        )
        raise SpokenHookValidationError("no_candidates_generated")

    scoring_pool = valid if valid else [_select_best_effort_candidate(checked)]

    judge_response = create_response_with_transport_retries(
        client,
        model=settings.hook_judge_model,
        temperature=settings.hook_judge_temperature,
        top_p=1.0,
        max_output_tokens=SCORE_JUDGE_MAX_OUTPUT_TOKENS,
        input=[
            {
                "role": "developer",
                "content": [{"type": "input_text", "text": HOOK_JUDGE_DEVMSG}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_judge_user_input(
                            style_profile=style_profile,
                            intent=intent_card,
                            candidates=scoring_pool,
                        ),
                    }
                ],
            },
        ],
    )

    judge_raw_output = extract_response_text(judge_response)
    score_rows = parse_hook_judge_scorecard(
        judge_raw_output,
        expected_count=len(scoring_pool),
    )

    ranked_for_shortlist: list[RankedHookCandidate] = []
    if score_rows is None:
        logger.warning(
            "hook_pipeline.scorecard_parse_fallback shortlist_count=%d",
            len(scoring_pool),
        )
        for idx, item in enumerate(scoring_pool[:4], start=1):
            logger.info(
                "hook_pipeline.shortlist_top4 candidate_idx=%d weighted_total=%s subscores=%s angle=%s candidate=%s",
                idx,
                "n/a",
                "n/a",
                item.angle,
                item.candidate,
            )
        shortlist_top4 = scoring_pool[:4]
        selected_source_prefix = "mechanical_shortlist_parse_fallback"
    else:
        selection = select_by_scorecard(scoring_pool, score_rows)
        ranked_for_shortlist = selection.ranked
        for ranked in ranked_for_shortlist:
            logger.info(
                "hook_pipeline.scorecard rank=%d candidate_idx=%d weighted_total=%d subscores=t:%d,o:%d,v:%d,f:%d,s:%d,i:%d angle=%s word_count=%d candidate=%s",
                ranked.rank,
                ranked.candidate_idx,
                ranked.weighted_total,
                ranked.score.tension,
                ranked.score.open_loop,
                ranked.score.viewer,
                ranked.score.fluency,
                ranked.score.tone,
                ranked.score.intent,
                ranked.angle,
                ranked.word_count,
                ranked.candidate,
            )
        shortlist_ranked = ranked_for_shortlist[:4]
        shortlist_top4 = [
            CandidateCheck(candidate=item.candidate, reasons=[], angle=item.angle)
            for item in shortlist_ranked
        ]
        for ranked in shortlist_ranked:
            logger.info(
                "hook_pipeline.shortlist_top4 candidate_idx=%d weighted_total=%d subscores=t:%d,o:%d,v:%d,f:%d,s:%d,i:%d angle=%s candidate=%s",
                ranked.candidate_idx,
                ranked.weighted_total,
                ranked.score.tension,
                ranked.score.open_loop,
                ranked.score.viewer,
                ranked.score.fluency,
                ranked.score.tone,
                ranked.score.intent,
                ranked.angle,
                ranked.candidate,
            )
        selected_source_prefix = "mechanical_shortlist"

    editorial_winner, editorial_rationale, editorial_used = _pick_editorial_winner(
        client,
        settings=settings,
        style_profile=style_profile,
        intent_card=intent_card,
        shortlist_top4=shortlist_top4,
    )
    logger.info(
        "hook_pipeline.editorial_winner selected_candidate_idx=%d selected_candidate=%s selected_angle=%s rationale=%s",
        shortlist_top4.index(editorial_winner) + 1,
        editorial_winner.candidate,
        editorial_winner.angle,
        editorial_rationale,
    )

    winner_before_tweak = editorial_winner.candidate
    tweak_variants = _tweak_winner(client, winner=winner_before_tweak, settings=settings)
    tweak_pool = _build_tweak_pool(winner_before_tweak, tweak_variants)
    tweak_checked = [
        CandidateCheck(candidate=c, reasons=validate_candidate(c), angle=editorial_winner.angle)
        for c in tweak_pool
    ]
    tweak_valid = [item for item in tweak_checked if not item.reasons][:5]
    logger.info(
        "hook_pipeline.tweak_stats total=%d valid=%d stats=%s",
        len(tweak_checked),
        len(tweak_valid),
        [
            {
                "candidate": item.candidate,
                "word_count": len(WORD_PATTERN.findall(item.candidate)),
                "reasons": item.reasons,
                "is_valid": not item.reasons,
            }
            for item in tweak_checked
        ],
    )

    if not tweak_valid:
        selected_candidate = winner_before_tweak
        _log_hook_event(
            settings,
            reading_hash=reading_hash,
            intent_card={
                "core_tension": intent_card.core_tension,
                "implicit_choice": intent_card.implicit_choice,
                "likely_consequence": intent_card.likely_consequence,
                "emotional_tone": intent_card.emotional_tone,
            },
            candidate_stats=candidate_stats,
            outcome_status="FULL_INTRO_OK",
            selected_source=f"{selected_source_prefix}+editorial_pick+tweaker_fallback",
            selected_candidate=selected_candidate,
            selected_angle=editorial_winner.angle,
        )
        logger.warning(
            "hook_pipeline.final_selected_source winner_before_tweak=%s final_selected_candidate=%s tweaker_used=%s rejudge_used=%s final_selection_reason=%s",
            winner_before_tweak,
            winner_before_tweak,
            True,
            False,
            "No valid tweak variants; retained editorial winner.",
        )
        return normalize_hook_punctuation(winner_before_tweak)

    final_selected_candidate, final_reason, rejudge_used = _select_final_variant(
        client,
        settings=settings,
        candidates=tweak_valid,
    )

    for rank, item in enumerate(tweak_valid, start=1):
        logger.info(
            "hook_pipeline.final_variant_scorecard rank=%d candidate=%s score=%s rationale=%s",
            rank,
            item.candidate,
            "winner" if item.candidate == final_selected_candidate else "not_selected",
            final_reason if item.candidate == final_selected_candidate else "",
        )

    _log_hook_event(
        settings,
        reading_hash=reading_hash,
        intent_card={
            "core_tension": intent_card.core_tension,
            "implicit_choice": intent_card.implicit_choice,
            "likely_consequence": intent_card.likely_consequence,
            "emotional_tone": intent_card.emotional_tone,
        },
        candidate_stats=candidate_stats,
        outcome_status="FULL_INTRO_OK",
        selected_source=f"{selected_source_prefix}+editorial_pick+tweaker_selector",
        selected_candidate=final_selected_candidate,
        selected_angle=editorial_winner.angle,
    )
    logger.info(
        "hook_pipeline.final_selected_source winner_before_tweak=%s final_selected_candidate=%s tweaker_used=%s rejudge_used=%s final_selection_reason=%s",
        winner_before_tweak,
        final_selected_candidate,
        True,
        rejudge_used,
        final_reason,
    )
    return normalize_hook_punctuation(final_selected_candidate)


def _log_hook_event(
    settings: Settings,
    *,
    reading_hash: str,
    intent_card: dict[str, str],
    candidate_stats: list[dict[str, object]],
    outcome_status: str,
    selected_source: str,
    selected_candidate: str | None,
    selected_angle: str | None,
) -> None:
    if not getattr(settings, "intro_telemetry_enabled", True):
        return

    outcome = HookOutcome(
        status=outcome_status,
        selected_source=selected_source,
        selected_candidate=selected_candidate,
        selected_angle=selected_angle,
        style_profile=settings.hook_style_profile,
        model_generator=settings.hook_generator_model,
        model_judge=settings.hook_judge_model,
    )
    try:
        append_hook_event(
            settings.intro_telemetry_path,
            reading_hash=reading_hash,
            intent_card=intent_card,
            candidate_stats=candidate_stats,
            outcome=outcome,
        )
    except OSError as exc:
        logger.warning(
            "hook_pipeline.telemetry_write_failed path=%s error=%s",
            settings.intro_telemetry_path,
            exc,
        )


_parse_tweaker_variants = parse_tweaker_variants


__all__ = [
    "generate_spoken_hook",
    "normalize_hook_punctuation",
    "SpokenHookValidationError",
    "validate_candidate",
    "_parse_tweaker_variants",
]
