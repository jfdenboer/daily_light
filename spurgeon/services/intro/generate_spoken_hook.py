"""Generate a spoken hook from a reading via a generator+judge pipeline."""

from __future__ import annotations

import logging

from openai import OpenAI

from spurgeon.config.settings import Settings
from spurgeon.services.intro.hook_pipeline.openai_utils import (
    create_response_with_transport_retries,
    extract_response_text,
)
from spurgeon.services.intro.hook_pipeline.prompts import (
    HOOK_GENERATOR_DEVMSG,
    HOOK_JUDGE_DEVMSG,
    HOOK_REPAIR_DEVMSG,
    HOOK_TWEAKER_DEVMSG,
)
from spurgeon.services.intro.hook_pipeline.rules import (
    WORD_PATTERN,
    CandidateCheck,
    build_generator_user_input,
    build_judge_user_input,
    normalize_hook_punctuation,
    normalize_judge_output,
    parse_numbered_candidates,
    parse_tweaker_variants,
    prepare_reading,
    validate_candidate,
)

logger = logging.getLogger(__name__)


class SpokenHookValidationError(ValueError):
    """Raised when generated spoken hook violates output constraints."""


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
                        "text": HOOK_TWEAKER_DEVMSG.format(num_variants=settings.hook_tweaker_num_variants),
                    }
                ],
            },
            {"role": "user", "content": [{"type": "input_text", "text": winner}]},
        ],
    )

    return parse_tweaker_variants(extract_response_text(response), settings.hook_tweaker_num_variants)


def _repair_hook(client: OpenAI, settings: Settings, hook: str) -> str:
    response = create_response_with_transport_retries(
        client,
        model=settings.hook_judge_model,
        temperature=0.2,
        top_p=1.0,
        max_output_tokens=60,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": HOOK_REPAIR_DEVMSG}]},
            {"role": "user", "content": [{"type": "input_text", "text": hook}]},
        ],
    )
    return normalize_judge_output(extract_response_text(response))


def generate_spoken_hook(reading: str, settings: Settings) -> str:
    """Return a validated spoken hook string for the supplied reading text."""

    client = OpenAI(api_key=settings.openai_api_key)
    prepared_reading = prepare_reading(reading)
    selected_source = "fallback"

    logger.info(
        "hook_pipeline.start generator_temperature=%.2f judge_temperature=%.2f tweaker_temperature=%.2f tweaker_enabled=%s tweaker_variants=%d",
        settings.hook_generator_temperature,
        settings.hook_judge_temperature,
        settings.hook_tweaker_temperature,
        settings.hook_tweaker_enabled,
        settings.hook_tweaker_num_variants,
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
                        "text": HOOK_GENERATOR_DEVMSG.format(num_candidates=settings.hook_num_candidates),
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": build_generator_user_input(prepared_reading)}],
            },
        ],
    )

    raw_candidates_text = extract_response_text(generator_response)
    parsed_candidates = [normalize_hook_punctuation(candidate) for candidate in parse_numbered_candidates(raw_candidates_text)]

    if len(parsed_candidates) < settings.hook_num_candidates:
        logger.warning(
            "hook_pipeline.generator_candidates_underflow parsed=%d expected=%d",
            len(parsed_candidates),
            settings.hook_num_candidates,
        )

    checked = [CandidateCheck(candidate=c, reasons=validate_candidate(c)) for c in parsed_candidates]
    valid = [item for item in checked if not item.reasons]

    candidate_stats = [
        {"candidate": item.candidate, "word_count": len(WORD_PATTERN.findall(item.candidate)), "reasons": item.reasons}
        for item in checked
    ]
    logger.info(
        "hook_pipeline.candidate_stats total=%d valid=%d stats=%s",
        len(checked),
        len(valid),
        candidate_stats,
    )

    if not checked:
        raise SpokenHookValidationError("no_candidates_generated")

    if valid:
        shortlist = valid
        include_reasons = False
    else:
        shortlist = sorted(checked, key=lambda item: len(item.reasons))[: min(3, len(checked))]
        include_reasons = True

    judge_response = create_response_with_transport_retries(
        client,
        model=settings.hook_judge_model,
        temperature=settings.hook_judge_temperature,
        top_p=1.0,
        max_output_tokens=60,
        input=[
            {"role": "developer", "content": [{"type": "input_text", "text": HOOK_JUDGE_DEVMSG}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_judge_user_input(prepared_reading, shortlist, include_reasons),
                    }
                ],
            },
        ],
    )

    judged = normalize_hook_punctuation(normalize_judge_output(extract_response_text(judge_response)))
    judged_reasons = validate_candidate(judged)

    if not judged_reasons:
        winner_before_tweak = judged
        if not settings.hook_tweaker_enabled:
            selected_source = "judge"
            logger.info(
                "hook_pipeline.selected_source=%s selected_candidate=%s tweaker_used=%s rejudge_used=%s",
                selected_source,
                judged,
                False,
                False,
            )
            return normalize_hook_punctuation(judged)

        tweak_variants = [normalize_hook_punctuation(v) for v in _tweak_winner(client, winner=winner_before_tweak, settings=settings)]
        tweak_pool = [winner_before_tweak, *tweak_variants]
        tweak_checked = [CandidateCheck(candidate=c, reasons=validate_candidate(c)) for c in tweak_pool]
        tweak_valid = [item for item in tweak_checked if not item.reasons]
        tweak_stats = [
            {"candidate": item.candidate, "word_count": len(WORD_PATTERN.findall(item.candidate)), "reasons": item.reasons}
            for item in tweak_checked
        ]
        logger.info(
            "hook_pipeline.tweak_stats total=%d valid=%d stats=%s",
            len(tweak_checked),
            len(tweak_valid),
            tweak_stats,
        )

        if not tweak_valid:
            selected_source = "judge"
            logger.info(
                "hook_pipeline.selected_source=%s selected_candidate=%s winner_before_tweak=%s tweaker_used=%s rejudge_used=%s",
                selected_source,
                winner_before_tweak,
                winner_before_tweak,
                True,
                False,
            )
            return normalize_hook_punctuation(winner_before_tweak)

        if len(tweak_valid) == 1:
            only_candidate = tweak_valid[0].candidate
            selected_source = "judge+tweaker_single"
            logger.info(
                "hook_pipeline.selected_source=%s selected_candidate=%s winner_before_tweak=%s tweaker_used=%s rejudge_used=%s",
                selected_source,
                only_candidate,
                winner_before_tweak,
                True,
                False,
            )
            return normalize_hook_punctuation(only_candidate)

        rejudge_temperature = min(settings.hook_judge_temperature, 0.10)
        rejudge_response = create_response_with_transport_retries(
            client,
            model=settings.hook_judge_model,
            temperature=rejudge_temperature,
            top_p=1.0,
            max_output_tokens=60,
            input=[
                {"role": "developer", "content": [{"type": "input_text", "text": HOOK_JUDGE_DEVMSG}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": build_judge_user_input(prepared_reading, tweak_valid, False),
                        }
                    ],
                },
            ],
        )
        rejudged = normalize_hook_punctuation(normalize_judge_output(extract_response_text(rejudge_response)))
        rejudged_reasons = validate_candidate(rejudged)

        if not rejudged_reasons:
            selected_source = "judge+tweaker_judge"
            logger.info(
                "hook_pipeline.selected_source=%s selected_candidate=%s winner_before_tweak=%s tweaker_used=%s rejudge_used=%s",
                selected_source,
                rejudged,
                winner_before_tweak,
                True,
                True,
            )
            return normalize_hook_punctuation(rejudged)

        selected_source = "judge+tweaker_fallback"
        logger.warning(
            "hook_pipeline.selected_source=%s selected_candidate=%s winner_before_tweak=%s rejudge_reasons=%s tweaker_used=%s rejudge_used=%s",
            selected_source,
            winner_before_tweak,
            winner_before_tweak,
            rejudged_reasons,
            True,
            True,
        )
        return normalize_hook_punctuation(winner_before_tweak)

    repaired = normalize_hook_punctuation(_repair_hook(client, settings, judged))
    repaired_reasons = validate_candidate(repaired)
    if not repaired_reasons:
        selected_source = "repair"
        logger.info("hook_pipeline.selected_source=%s", selected_source)
        return normalize_hook_punctuation(repaired)

    fallback_item = sorted(checked, key=lambda item: len(item.reasons))[0]
    selected_source = "failure"
    logger.warning(
        "hook_pipeline.selected_source=%s judge_reasons=%s repair_reasons=%s fallback_reasons=%s",
        selected_source,
        judged_reasons,
        repaired_reasons,
        fallback_item.reasons,
    )
    raise SpokenHookValidationError("no_valid_hook_after_judge_and_repair")


_parse_tweaker_variants = parse_tweaker_variants


__all__ = [
    "generate_spoken_hook",
    "normalize_hook_punctuation",
    "SpokenHookValidationError",
    "validate_candidate",
    "_parse_tweaker_variants",
]
