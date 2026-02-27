# Herontwerp: van chunk-gebaseerde beeldpipeline naar één hoofdbeeld per video

## 1) Doel

We willen de huidige workflow vereenvoudigen:

- **Van:** meerdere tekstchunks met elk een eigen prompt en image.
- **Naar:** **één** centrale image voor de volledige video.

Dit document beschrijft een functioneel en technisch herontwerp (zonder implementatiecode), inclusief impact op domeinmodel, pipeline, configuratie, foutafhandeling en migratie.

---

## 2) Huidige situatie (samengevat)

De bestaande pipeline werkt chunk-gebaseerd:

1. Lezing wordt gesegmenteerd in tekstblokken/chunks.
2. Per chunk wordt een prompt gemaakt.
3. Per chunk wordt een image gegenereerd.
4. Video-compiler plaatst/afwisselt images over de tijdlijn.

Dit geeft flexibiliteit, maar introduceert complexiteit in promptlogica, timing, bestandsbeheer en foutscenario’s.

---

## 3) Nieuwe productkeuze

### Kernbeslissing

Per video wordt precies **één visual concept** gekozen en precies **één image** gegenereerd.

### Productgevolgen

- Consistente visuele stijl over de hele video.
- Snellere doorlooptijd en lagere kosten voor image generatie.
- Minder variatie binnen één video (bewuste trade-off).

---

## 4) Nieuwe architectuur op hoog niveau

```mermaid
graph TD
    A[Input tekst] --> B[Parser]
    B --> C[Reading]
    C --> D[TTS]
    C --> E[Prompt Generation (reading-level)]
    E --> F[Image Generator (1x)]
    D --> G[Audio]
    G --> H[Alignment + Subtitles]
    F --> I[Single Image Asset]
    H --> J[Video Compiler (single-background mode)]
    I --> J
    J --> K[Final video]
```

### Wat verdwijnt

- Chunking als **driver voor beeldgeneratie**.
- Prompt-per-chunk en image-per-chunk opslag/iteratie.

### Wat blijft

- Parser, TTS, alignment, subtitlebouw en video-export.
- Optionele upload/publicatiepaden.

---

## 5) Domeinmodel (conceptueel)

### Oude concepten (te reduceren)

- `SegmentBlock` als verplicht object voor visual generation.
- Collectie van `image_assets[]` gekoppeld aan chunk-indexen.


**Ontwerpprincipe:** segmentatie/chunking wordt volledig verwijderd uit het domeinmodel om de keten eenduidig op reading-niveau te houden.

---

## 6) Pipeline-herontwerp per fase

### Fase A — Inname & tekstvoorbewerking

- Parser levert één `Reading`.
- Geen segmentatie- of chunkstap meer in deze pipeline.

### Fase B — Prompt generatie op video-niveau

- Behoud de bestaande `prompt_generation`-laag.
- Vervang “prompt per chunk” door één prompt op reading-/video-niveau.


### Fase C — Image generatie

- Gebruik de output van `prompt_generation` als enige input voor beeldgeneratie.
- Genereer **exact één image** per video.
- Sla op als primair video-asset.
- Registreer prompt + metadata voor reproduceerbaarheid.

### Fase D — Audio & subtitles

- TTS, alignment en ondertitels blijven lineair gekoppeld aan audio, direct op basis van de volledige reading.

### Fase E — Video compositing (single-background mode)

- Compiler gebruikt één achtergrondbeeld voor de volledige timeline.


---

## 7) Configuratieherziening

### Te verwijderen

- `chunk_max_words`
- segmentatie/chunking toggles
- instellingen die aantal beelden per chunk sturen

### Nieuw/te behouden

- `visual_mode=single_image`
- `single_image_resolution`
- `single_image_style_profile`
- `subtitle_readability_overlay` (true/false)

**Migratiestrategie:** verwijder segmentatievariabelen direct in dezelfde release als `single_image`-only gedrag; geen dual-path meer onderhouden.

---

## 8) Kwaliteit en acceptatiecriteria

### Functionele criteria

1. Voor elke video wordt exact **1 image request** gedaan.
2. Video compileert succesvol met één image over de totale duur.
3. Subtitles blijven synchroon met audio.

### Niet-functionele criteria

- Lagere gemiddelde buildtijd.
- Minder API-calls naar image provider.
- Minder tijdelijke artefacten en eenvoudiger opslagstructuur.

### UX/inhoudelijke criteria

- Visuele stijl blijft passend bij de dagtekst.
- Geen storende overgangen tussen beelden (want één beeld).

---

## 9) Foutafhandeling en fallback-ontwerp

### Scenario: image generatie faalt

Fallbackvolgorde:

1. Retry met dezelfde prompt (beperkt aantal pogingen).
2. Retry met vereenvoudigde stijlconstraints.
3. Gebruik curated fallback image (neutraal, project-breed).
4. Markeer run als “degraded success” met duidelijke audit-log.

### Scenario: beeld is inhoudelijk ongeschikt

- Voeg kwaliteitsgate toe op metadata/regels (bijv. banned motifs).
- Bij afkeur: één automatische regeneratie met aangescherpte constraints.

---

## 10) Impact op observability

Nieuwe kernmetrics:

- `image_requests_per_video` (verwacht: 1)
- `single_image_generation_latency_ms`
- `single_image_fallback_used` (bool/counter)
- `video_compile_single_image_mode` (bool)

Doel: snel detecteren of legacy chunk-gedrag nog onbedoeld actief is.

---

## 11) Gefaseerde migratie

### Fase 1 — Verwijderen segmentatiepad

- Verwijder segmentatie/chunking code en configuratie uit de beeldpipeline.
- Houd tijdelijke mappinglaag alleen voor backward-compatible inputcontracten (zonder segmentatie-uitvoering).

### Fase 2 — Single-image-only afdwingen

- Verwijder eventuele legacy flags voor visual mode.
- Forceer exact één prompt + één image per run.

### Fase 3 — Opschonen

- Verwijder overgebleven segmentatiehelpers in services, tests en docs.
- Verwijder backward-compatible inputmapping zodra alle producers gemigreerd zijn.
- Vereenvoudig documentatie en operationele runbooks.

---

## 12) Risico’s en trade-offs

### Voordelen

- Sterk vereenvoudigde pipeline.
- Lagere kosten en minder foutoppervlak.
- Betere voorspelbaarheid van output.

### Nadelen

- Minder visuele variatie binnen één video.
- Zwaardere afhankelijkheid van kwaliteit van één enkele videoprompt.

### Mitigatie

- Zorgvuldige prompt-generatie met duidelijke stijlprofielen.
- Optionele lichte camera-beweging in compositing voor dynamiek.

---

## 13) Besluitvoorstel

**Aanbeveling:** migreer naar `single_image` als standaardarchitectuur.

Deze richting sluit aan op de wens “nog maar 1 image voor de hele video” en reduceert complexiteit zonder de kernwaarde van de pipeline (goede audio, leesbare subtitles, consistente visuele ondersteuning) te verliezen.




