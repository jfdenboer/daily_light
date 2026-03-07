"""Prompt policies and builders for thumbnail image generation."""

from __future__ import annotations

from spurgeon.models import Reading

from .thumbnail_intent_card import ThumbnailIntentCard

THUMBNAIL_STYLE_LINE = (
    "Style: thumbnail-first cinematic realism; calm, contemplative, emotionally immediate, and premium; "
    "naturalistic detail with large readable shapes, clean silhouette separation, and strong visual hierarchy; "
    "subtle highlight bloom/halation without overall softness; the main subject must remain clear, distinct, and instantly legible at small size; "
    "fine restrained film-like grain; matte finish; believable materials, surfaces, and natural imperfections; "
    "favor subject clarity, shape readability, and compositional simplicity over spectacle, atmosphere, or fine texture; "
    "avoid hyper-clarity, micro-contrast, crunchy edges, sharpening halos, HDR, smeary detail, heavy diffusion, and muddy softness; "
    "no illustration, paper, paint, stylized concept-art, or CGI/3D/glossy rendering."
)

THUMBNAIL_SUBJECT_LINE = (
    "Subject: exactly one dominant visual anchor only; prefer one clearly readable person, object, gesture, or symbolic form with immediate emotional clarity; "
    "any secondary elements must remain unmistakably subordinate in scale, contrast, and attention; "
    "the image should communicate one idea, not several equal ideas at once; "
    "if a person appears, favor a readable silhouette, pose, or partial profile with presence and emotional weight; "
    "avoid vague, fragmented, or weakly defined subjects, and avoid compositions where the supposed subject is too small, too distant, or visually overpowered by the setting."
)

THUMBNAIL_BACKGROUND_LINE = (
    "Setting: believable, simple environment with depth, atmosphere, and natural scale; may be interior or exterior depending on the reading, but always restrained and secondary to the main subject; "
    "prefer uncluttered, human-scale settings that support mood without competing for attention; "
    "background elements should reinforce the emotional tone and visual motif while remaining calm, coherent, and low-drama; "
    "subtle haze may be used only for depth separation, never as a flat wash that weakens readability; "
    "keep the text side calm, even-toned, and low-detail; "
    "avoid scenic excess, symbolic overload, and environments that become more memorable than the subject itself."
)

THUMBNAIL_COMPOSITION_LINE = (
    "Composition: 16:9 wide; exactly one dominant focal anchor placed on the right or center-right, with enough visual weight to read instantly on small screens; "
    "use clear foreground, midground, and background with large simple masses rather than many small details; "
    "keep the left and center-left broadly readable for large overlaid text (up to roughly the left 60 to 65 percent), with natural text overlap permitted and no dedicated contrast panel; "
    "keep the single most critical focal detail on the right or right-center rather than directly under the text block; "
    "maintain strong visual hierarchy, clean balance, and immediate comprehension; "
    "avoid competing focal points, scattered storytelling, cramped framing, and compositions that feel like a cinematic still rather than a thumbnail."
)

THUMBNAIL_CONSTRAINTS_LINE = (
    "Constraints: no visible text, captions, lettering, numbers, pseudo-text, watermarks, logos, readable signage, icons, emblems, frames, or UI overlays; "
    "prefer simple, unified scenes over multi-part narratives; "
    "human presence, if used, should feel natural, non-identifiable, and integrated rather than posed or theatrical; "
    "preserve tonal cleanliness and separation around the subject and across left-to-center areas likely to carry text; "
    "avoid visual noise, artificial emphasis, attention fragmentation, and any detail pattern that weakens small-size readability."
)

THUMBNAIL_PALETTE_LINE = (
    "Color grade: warm-neutral cinematic palette with restrained saturation and clean tonal separation; natural earth, stone, sky, water, wood, fabric, and muted vegetation tones; "
    "gentle contrast with clean highlights, soft shadows, and controlled mids; subtle warm highlight accents without orange cast; "
    "preserve a calm, readable value structure across the text side and a clear tonal anchor around the subject; "
    "avoid heavy color casts, gimmicky contrast, postcard prettiness, or monochrome murkiness."
)

THUMBNAIL_LIGHTING_LINE = (
    "Lighting: soft overcast daylight or gentle natural directional light, with calm luminous highlights and subtle atmosphere; "
    "maintain clean local contrast around the main subject and quieter, more even lighting across the text side; "
    "soft shadows and controlled highlights; gentle subject-background separation without extreme backlight or overpowering glow; "
    "preserve calm shape definition on the focal subject and keep the lighting emotionally resonant but visually disciplined; "
    "avoid theatrical light effects, sensational drama, and lighting that overwhelms the subject or disrupts copy readability."
)


def build_thumbnail_prompt(
    reading: Reading,
    thumbnail_text: str,
    intent_card: ThumbnailIntentCard,
) -> str:
    return "\n".join(
        [
            THUMBNAIL_STYLE_LINE,
            THUMBNAIL_SUBJECT_LINE,
            THUMBNAIL_BACKGROUND_LINE,
            THUMBNAIL_COMPOSITION_LINE,
            THUMBNAIL_CONSTRAINTS_LINE,
            THUMBNAIL_PALETTE_LINE,
            THUMBNAIL_LIGHTING_LINE,
            "",
            f"Devotional type: {reading.reading_type.value}.",
            f"Visual theme: {thumbnail_text}.",
            f"Core tension: {intent_card.core_tension}.",
            f"Emotional tone: {intent_card.emotional_tone}.",
            f"Visual motif: {intent_card.visual_motif}.",
            f"Scene direction: {intent_card.scene_direction}.",
            f"Avoid: {intent_card.avoid}.",
        ]
    )
