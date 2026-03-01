# Ontwerp: intro (hook + credit line) voor Daily Light video

## Doel
We voegen een **spraak-intro** toe aan elke video, bestaande uit:
1. `hook` uit `spurgeon/services/intro/generate_spoken_hook.py`
2. `credit_line` uit `spurgeon/services/intro/generate_credit_line.py`

Vereisten:
- Introtekst wordt omgezet naar audio via **ElevenLabs TTS**.
- Audio-opbouw intro: `pause_audio0 + hook_audio + pause_audio1 + credit_audio + pause_audio2`.
- Tijdens de intro blijft het beeld **dezelfde image** die al voor de volledige video wordt gebruikt.
- Intro gebruikt **dezelfde voice** als de reading.
- **Geen subtitles tijdens intro**.

---

## Gewenste eindflow
Per reading:
1. Genereer `hook` op basis van reading-tekst.
2. Haal `credit_line` op (vaste zin).
3. Syntheseer twee spraaksegmenten met ElevenLabs in exact dezelfde voice/config als main narration:
   - `hook_audio`
   - `credit_audio`
4. Genereer drie stiltesegmenten:
   - `pause_audio0` (micro-pause aan het begin van de intro)
   - `pause_audio1` (tussen hook en credit)
   - `pause_audio2` (na credit, vóór hoofdtekst)
5. Concateneer: `pause_audio0 + hook_audio + pause_audio1 + credit_audio + pause_audio2` -> `intro_audio`.
6. Syntheseer bestaande hoofdnarratie (`main_audio`) van `reading.text`.
7. Concateneer `intro_audio + main_audio` -> `final_audio`.
8. Gebruik `final_audio` voor alignment en videorendering.
9. Gebruik dezelfde single image over de volledige duur (intro + hoofdtekst).

---

## Componentontwerp

### 1) IntroBuilder (orchestrator)
Conceptueel nieuwe service die:
- `hook` ophaalt via `generate_spoken_hook(reading.text)`.
- `credit_line` ophaalt via `generate_credit_line()`.
- TTS uitvoert voor hook + credit.
- `pause_audio0`, `pause_audio1` en `pause_audio2` genereert.
- Introsegmenten concateneert naar `intro_audio`.
- Metadata teruggeeft, incl. totale `intro_duration`.

### 2) Voice-keuze: exact gelijk aan reading
Er wordt **geen aparte intro-voice** gebruikt.
- Zelfde ElevenLabs `voice_id`, `model_id`, `output_format` en relevante voice settings.
- Doel: consistente klankkleur tussen intro en hoofdtekst.

### 3) Pauzes: deterministisch en configureerbaar
Pauzes worden expliciet als stiltebestanden gemaakt (FFmpeg `anullsrc`), niet via interpunctie.
- `intro_pause_pre_intro_ms` voor `pause_audio0` (micro-pause, kort).
- `intro_pause_between_ms` voor `pause_audio1` (iets langer dan voorheen).
- `intro_pause_after_credit_ms` voor `pause_audio2`.

### 4) Audio samenvoegen
Twee concat-stappen:
1. Intro concat: `pause_audio0 + hook_audio + pause_audio1 + credit_audio + pause_audio2` -> `intro_audio`.
2. Eindconcat: `intro_audio + main_audio` -> `final_audio`.

Randvoorwaarde:
- Segmenten eerst normaliseren naar één audioformaat indien nodig.

### 5) Pipeline-integratie
In `_prepare_render_artifacts` conceptueel:
- Vervang “alleen main narration” door opbouw van `final_audio`.
- Geef `final_audio` door aan aligner en video compiler.

### 6) Subtitles: geen intro-subtitles
Voor v1:
- Alleen subtitles voor de hoofdtekst.
- Alle cues van hoofdtekst krijgen een offset gelijk aan `intro_duration`.
- Intro (hook, credits, pauzes) krijgt **geen** subtitle-cues.

### 7) Visuals tijdens intro
Geen extra intro-scene:
- Bestaande single background image blijft actief tijdens intro én hoofdtekst.
- Videoduur volgt automatisch de totale audiolengte.

---

## Configuratievoorstel
- `intro_enabled: bool = true`
- `intro_pause_pre_intro_ms: int = 120`
- `intro_pause_between_ms: int = 550`
- `intro_pause_after_credit_ms: int = 350`
- `intro_cache_enabled: bool = true`
- `intro_fail_open: bool = true`

Let op:
- Geen `intro_voice_id` nodig zolang intro altijd reading-voice hergebruikt.

---

## Foutafhandeling & fallback
- Hook-generatie faalt -> fallback naar alleen `credit_audio + pause_audio2` of volledige intro skip (config).
- Intro-TTS faalt -> bij `intro_fail_open=true` doorgaan met alleen `main_audio`.
- Concat faalt -> hard fail als `final_audio` niet valide kan worden opgebouwd.

---

## Acceptatiecriteria
1. Intro is hoorbaar vóór de hoofdtekst.
2. Intro start met een korte micro-pause (`pause0`) vóór de hook.
3. Introvolgorde is exact: `pause0 -> hook -> pause1 -> credit -> pause2`.
4. Pauze tussen hook en credit is merkbaar langer dan in het vorige ontwerp.
5. Intro-voice is identiek aan reading-voice.
6. Tijdens intro blijft exact dezelfde achtergrondimage zichtbaar.
7. Geen subtitles tijdens intro; hoofdtekstsubtitles starten na intro-offset.
8. Pipeline blijft bruikbaar bij introfouten volgens fail-open instelling.
