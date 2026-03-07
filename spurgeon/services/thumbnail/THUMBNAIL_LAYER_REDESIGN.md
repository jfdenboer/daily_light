# Thumbnail-layer audit en herontwerp

## Context
De huidige `thumbnail_generator.py` combineert meerdere verantwoordelijkheden in één klasse van 762 regels:

1. Prompt-engineering voor zowel intent-card als image generation.
2. OpenAI transport/API-afhandeling inclusief retries.
3. Parsing/validatie van LLM-output.
4. Image compositing (Pillow): crop, text layout, font fallback, rendering.
5. Bestands-I/O/caching (output-map, bestaand bestand zoeken).
6. Logging en pipeline-observability.

Hierdoor is de module lastig te testen, lastig uit te breiden (bijv. ander model/renderer), en foutgevoelig bij wijzigingen.

## Audit: belangrijkste bevindingen

## 1) Sterke punten
- De pipeline is expliciet en logisch (`intent_card -> prompt -> image -> compose -> save`).
- Error handling rond OpenAI-calls is aanwezig en gebruikt retries.
- Text-layout heeft redelijke fallback-strategieën (preferred + emergency size).
- Parsing van intent-card is strict; dit voorkomt stilzwijgende kwaliteitsdegradatie.

## 2) Problemen en risico's

### A. Te veel verantwoordelijkheden in één klasse
`ThumbnailGenerator` is tegelijk orchestrator, domeinservice, infrastructuurclient, parser en renderer. Dit schaadt SRP en maakt unit-testing duur (veel mocking nodig).

### B. Hard-coded promptbeleid in code
Grote promptblokken als module-constanten maken iteratie lastig (A/B-testen, versies, per reading-type variatie). Promptbeheer hoort configureerbaar/versioneerbaar te zijn buiten de core-orchestrator.

### C. Fragiele koppeling tussen parsing en promptformat
De intent-card parser is streng (goed), maar nauw gekoppeld aan exacte line-formatting. Kleine modelvariaties veroorzaken pipeline-fails i.p.v. gecontroleerde degradatie.

### D. Gemengde foutsemantiek
Bijna alles wordt op hogere niveaus `ThumbnailGenerationError(str(exc))`. Daardoor verlies je typed oorzaak-informatie (bijv. parse vs transport vs rendering), wat troubleshooting en herstelbeleid beperkt.

### E. Performance/efficiency details
- `_load_font` wordt herhaaldelijk aangeroepen tijdens meten/renderen; zonder cache kan dit duur zijn.
- Bij iedere generatie wordt volledig image + text opnieuw gedaan; geen idempotency-key op promptniveau.

### F. Beperkte uitbreidbaarheid
Wil je later:
- meerdere render-stijlen,
- andere providers (bijv. lokaal model),
- varianten (A/B-thumbnails),
- quality gates (contrast checks),
dan moet je ingrijpend in één bestand refactoren.

### G. Domeinlogica en infrastructuurlogica zijn verweven
`Reading`-domeingegevens lopen direct de API en renderer in. Een expliciet domeinmodel voor `ThumbnailSpec` ontbreekt.

## Doelarchitectuur

Ontwerp de thumbnail-layer als **modulaire pipeline** met heldere contracten:

1. **Application layer (orchestratie)**
   - `ThumbnailService.generate(reading, title, overrides)`
   - Coördineert stappen, beleid en retries op use-case-niveau.

2. **Domain layer (zuivere modellen + regels)**
   - `ThumbnailIntentCard`
   - `ThumbnailSpec` (promptable scene + text policy + layout policy)
   - `RenderPlan` (exacte parameters voor compositing)
   - Domeinvalidatie (max lijnen, veilige defaults, enz.)

3. **Infrastructure adapters**
   - `IntentCardProvider` (OpenAI chat implementatie)
   - `ImageProvider` (OpenAI image implementatie)
   - `ThumbnailRenderer` (Pillow implementatie)
   - `ThumbnailRepository` (filesystem/caching)

4. **Policy/config layer**
   - Prompt templates (versioned YAML/TOML/Jinja)
   - Layout presets
   - Retry/circuit breaker instellingen

## Voorgestelde mappenstructuur

```text
spurgeon/services/thumbnail/
  application/
    thumbnail_service.py
    dto.py
  domain/
    models.py
    errors.py
    policies.py
  providers/
    intent_card_provider.py
    image_provider.py
    openai_intent_card_provider.py
    openai_image_provider.py
  rendering/
    renderer.py
    pillow_renderer.py
    layout_engine.py
    font_resolver.py
  storage/
    thumbnail_repository.py
    filesystem_repository.py
  prompts/
    thumbnail_intent_card.v1.txt
    thumbnail_image.v1.txt
  observability/
    events.py
    metrics.py
  __init__.py
```

## Belangrijkste componenten en contracten

### 1) `ThumbnailService` (orchestrator)
Verantwoordelijkheden:
- input normaliseren,
- cache check,
- intent-card ophalen,
- prompt bouwen,
- image aanvragen,
- renderen,
- opslaan,
- domein-specifieke fouten doorgeven.

Geen OpenAI/Pillow details binnen deze klasse.

### 2) Providers via interfaces
- `IntentCardProvider.generate(reading_context) -> ThumbnailIntentCard`
- `ImageProvider.generate(prompt, user_id) -> bytes`

Voordeel: eenvoudig mocken in tests en provider-swap zonder domeinwijzigingen.

### 3) `PromptBuilder` met versioning
- Promptregels in templatebestanden (`v1`, later `v2`).
- Runtime selecteerbare promptversie via settings/feature flag.
- Betere governance: promptwijzigingen zonder codevervuiling.

### 4) `LayoutEngine` + `Renderer`
Splits compositing in:
- layoutberekening (pure functies, testbaar zonder pixels),
- rendering (Pillow calls).

Extra: cache loaded fonts per `(font_path, size)`.

### 5) Typed fouten
Introduceer fout-hiërarchie:
- `ThumbnailError` (base)
- `IntentCardError`
- `PromptBuildError`
- `ImageProviderError`
- `RenderError`
- `StorageError`

Hiermee kan orchestratie gericht fallbacken (bijv. intent-card fail -> simpele prompt fallback).

### 6) Repository + cachebeleid
`ThumbnailRepository` met methodes:
- `get_existing(slug) -> Path | None`
- `save(slug, image_bytes|path)`
- optioneel: `get_by_fingerprint(fingerprint)`

Fingerprint kan bestaan uit promptversie + titel + reading hash + render preset.

## Pipeline-gedrag (herontwerp)

1. `ThumbnailService` ontvangt request.
2. Compute `thumbnail_fingerprint`.
3. Check repository cache op fingerprint/slug.
4. Vraag intent-card op (met retries in provider-adapter).
5. Bouw image prompt via `PromptBuilder`.
6. Vraag image bytes op.
7. Bepaal `RenderPlan` via `LayoutEngine`.
8. Render met `Renderer`.
9. Valideer output (afmetingen, formaat, min contrast indien gewenst).
10. Persist + return path.

## Fallback- en resiliencybeleid

- **Intent-card faalt:** terugvallen op minimal prompt op basis van `title + reading_type`.
- **Image provider faalt tijdelijk:** retry + exponential backoff + jitter.
- **Renderer faalt door font:** fallback fontset met caching en waarschuwing.
- **Output validatie faalt:** markeer als hard fail met duidelijke foutcode.

## Teststrategie

1. **Unit tests (domain/layout)**
   - intent-card parsingvarianten,
   - line-break en font-size selectie,
   - text-box positionering grenzen.

2. **Contract tests (providers)**
   - mocked OpenAI response schema,
   - foutmapping naar typed errors.

3. **Golden image tests (renderer)**
   - vaste input -> verwachte pixel-snapshot (met tolerantie).

4. **Integration tests (service)**
   - end-to-end met fake providers en tmp repository.

5. **Observability tests**
   - verifieer aanwezigheid van gestructureerde events en foutcodes.

## Migratieplan (stapsgewijs, laag risico)

### Fase 1 — Extract without behavior change
- Verplaats parser, promptbuilder en layoutlogica naar aparte modules.
- Houd publieke API van huidige generator intact.

### Fase 2 — Introduce interfaces
- Voeg `IntentCardProvider`, `ImageProvider`, `Renderer`, `Repository` toe.
- Maak huidige implementaties default adapters.

### Fase 3 — Typed errors + observability
- Vervang generieke wrapping door typed errors.
- Voeg consistente eventnamen/velden toe.

### Fase 4 — Prompt externalization
- Zet prompttekst naar versioned templates.
- Voeg setting `thumbnail_prompt_version` toe.

### Fase 5 — Cache/fingerprint + quality gates
- Introduceer fingerprint-based cache.
- Voeg optionele quality checks toe.

## Definition of Done voor het herontwerp

- `thumbnail_generator.py` is gereduceerd tot dunne orchestrator of vervangen door `ThumbnailService` (<150 regels).
- Promptteksten/versioning buiten core code.
- Alle externe afhankelijkheden via interfaces.
- Typed errors aanwezig en gedocumenteerd.
- Minimaal:
  - 90% dekking op domain + layout modules,
  - integration suite voor service-flow,
  - 3 golden image snapshots.

## Verwachte winst

- Sneller itereren op promptkwaliteit zonder regressies.
- Hogere testbaarheid en betrouwbaarheid.
- Lagere verandering-risico's bij nieuwe providers/styles.
- Betere observability voor productieproblemen.

## Update na uitvoering Fase 1 (extract zonder gedragswijziging)

Fase 1 is nu uitgevoerd met behoud van de bestaande publieke `ThumbnailGenerator`-API.

### Wat is concreet geëxtraheerd
- **Intent-card model + parser** naar `thumbnail_intent_card.py`.
- **Promptbeleid + promptbuilder** naar `thumbnail_prompting.py`.
- **Text-layout logica** naar `thumbnail_layout.py`.
- `thumbnail_generator.py` is teruggebracht naar orchestration + provider-call + rendering-aansturing.

### Nieuwe inzichten uit de extractie
1. **`ThumbnailGenerator` blijft nog steeds deels infrastructuur-gedreven**
   - OpenAI client-initialisatie en API-calls zitten nog direct in de orchestrator.
   - In fase 2 moeten provider-interfaces prioriteit krijgen om testisolatie echt te verbeteren.

2. **Font loading is nog niet gecachet**
   - De layoutmodule maakt de call-frequentie explicieter zichtbaar.
   - Een `FontResolver` met memoization per `(path, size)` is een snelle winst voor fase 2/3.

3. **Fouttypen zijn functioneel nog geaggregeerd**
   - Parserfouten worden nu intern onderscheiden, maar aan de buitenkant nog omgezet naar `ThumbnailGenerationError` om gedrag te behouden.
   - Dat bevestigt dat typed error-propagatie een expliciete fase-3 stap moet blijven.

4. **Promptversiebeheer is nog code-gebonden**
   - Nu opgesplitst in eigen module (betere scheiding), maar nog niet extern in templates.
   - Dit maakt fase 4 (externalization) mechanisch eenvoudiger.

### Effect van fase 1
- Minder cognitieve belasting in `thumbnail_generator.py`.
- Parser/layout/prompt onderdelen zijn nu afzonderlijk unit-testbaar.
- Vervolgstappen (interfaces, typed errors, template versioning) kunnen incrementeel zonder grote rewrite.

## Update na uitvoering Fase 2 (interfaces + default adapters)

Fase 2 is uitgevoerd met behoud van de bestaande publieke `ThumbnailGenerator`-API.

### Wat is concreet toegevoegd
- **Interfaces (contracts)** in `thumbnail_contracts.py`:
  - `IntentCardProvider`
  - `ImageProvider`
  - `ThumbnailRenderer`
  - `ThumbnailRepository`
- **Default adapters** in `thumbnail_adapters.py`:
  - `OpenAIIntentCardProvider`
  - `OpenAIImageProvider`
  - `PillowThumbnailRenderer`
  - `FilesystemThumbnailRepository`
- **Orchestrator wiring** in `thumbnail_generator.py`:
  - dependency injection via optionele constructor-parameters,
  - retries blijven op orchestratie-niveau,
  - bestaande flow en logging behouden.

### Effect op testbaarheid en uitbreidbaarheid
- Providers/renderer/repository zijn nu los te mocken zonder OpenAI/Pillow/filesysteem direct in de test te initialiseren.
- Toekomstige provider-swap (ander LLM/image model) kan via interface-implementatie zonder orchestratorlogica te wijzigen.
- De generator is nu duidelijker application-layer georiënteerd; infrastructuurdetails zijn verplaatst naar adapters.

## Update na uitvoering Fase 4 (prompt externalization + versioning)

Fase 4 is uitgevoerd met versiebeheer op prompt-niveau, zonder breuk in de publieke `ThumbnailGenerator`-API.

### Wat is concreet opgeleverd
- **Versioned prompt templates op schijf** in `spurgeon/services/thumbnail/prompts/`:
  - `thumbnail_image.v1.txt`
  - `thumbnail_intent_card.v1.txt`
- **Prompt loading met caching**:
  - image prompt templates via `thumbnail_prompting.py` (`@lru_cache`),
  - intent-card prompt template geladen via promptversie in de provider.
- **Nieuwe setting** `thumbnail_prompt_version` (env: `THUMBNAIL_PROMPT_VERSION`) toegevoegd aan `Settings`.
- **Orchestratie gebruikt expliciet promptversie**:
  - `ThumbnailGenerator` geeft de geconfigureerde promptversie door aan de promptbuilder,
  - `OpenAIIntentCardProvider` gebruikt dezelfde versie voor de system prompt.

### Nieuwe inzichten na fase 4
1. **Backward compatibility blijft een aandachtspunt**
   - De bestaande promptlijn-constanten blijven beschikbaar, maar zijn nu afgeleid van template `v1`.
   - Dit houdt huidige imports stabiel terwijl de bron van waarheid naar templatebestanden is verplaatst.

2. **Één versieknop voor beide promptstappen werkt goed**
   - Door dezelfde `thumbnail_prompt_version` te gebruiken voor intent-card én image prompt blijft de promptketen coherent.
   - Bij toekomstige experimenten (bijv. `v2`) kan een complete promptset als pakket worden uitgerold.

3. **Fase 5 kan nu zuiver op cache/kwaliteit focussen**
   - Prompt-externalization is afgerond; vervolgstappen hoeven geen promptstrings meer in Python-code te muteren.

## Update na uitvoering Fase 5 (cache/fingerprint + quality gates)

Fase 5 is nu uitgevoerd met behoud van de bestaande `ThumbnailGenerator` entrypoint.

### Wat is concreet toegevoegd
- **Fingerprint-based cache lookup** in de orchestrator:
  - fingerprint wordt deterministisch berekend uit promptversie, reading-slug/type, titel en hash van de reading-tekst;
  - cache-check gebeurt eerst op fingerprint, daarna op legacy slug-cache.
- **Repository-ondersteuning voor fingerprint cache**:
  - `ThumbnailRepository` contract uitgebreid met `get_by_fingerprint(...)`;
  - filesystem-adapter bewaart fingerprint-indexbestanden onder `.fingerprints/`.
- **Quality gates in de pipeline**:
  - na renderen valideert de pipeline standaard op canvas-grootte en minimale luminantie-variatie (`stddev`);
  - configurabel via settings (`thumbnail_quality_checks_enabled`, `thumbnail_quality_min_luma_stddev`).
- **Typed error uitgebreid** met `QualityGateError` en observability event voor geslaagde quality gate.

### Nieuwe inzichten uit Fase 5
1. **Fingerprinting vraagt expliciete invalidatiestrategie**
   - Nu de cache semantisch op input/fingerprint werkt, moeten modelwijzigingen (image model, renderer-stijl, fontbeleid) expliciet in de fingerprint worden opgenomen zodra gedrag wijzigt.

2. **Quality gates werken het best als “policy knobs”**
   - Een universele harde threshold is contextafhankelijk; daarom is de minimale contrast/variatie-check als instelbare policy geïmplementeerd i.p.v. hardcoded kwaliteitsregel.

3. **Backwards compatibility blijft relevant tijdens migratie**
   - Slug-cache fallback voorkomt regressies voor bestaande output-bestanden zonder fingerprint-index.
