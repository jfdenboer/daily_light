# Herontwerp Intro-laag (concept + voortgang)

## 1) Doel van het herontwerp

De intro-laag moet in één consistente flow drie dingen doen:
1. **Aandacht grijpen** (sterke hook in spreektaal).
2. **Context geven** (korte credit/positionering zonder tempoverlies).
3. **Naadloos overgaan** naar de hoofdtekst (ritme + audiokwaliteit).

Het oude systeem werkte functioneel, maar de hook-logica was te afhankelijk van één generatie/judge-pass met beperkte structurele semantiek.

---

## 2) Ontwerpprincipes

- **Hook-first architectuur**: de hook bepaalt de intro.
- **Kwaliteit meetbaar maken**: elke stap moet scorebaar en uitlegbaar zijn.
- **Fail-soft i.p.v. fail-hard**: degradeer netjes (full intro → credit-only → no intro).
- **Losse verantwoordelijkheden**: generatie, beoordeling, repair en audio-assemblage gescheiden.
- **Prompts als assets**: prompt/policybeheer expliciet en versioneerbaar.

---

## 3) Architectuur (doelbeeld)

### 3.1 Componenten

1. **Intro Orchestrator**
2. **Hook Ideation Engine**
3. **Hook Evaluator**
4. **Hook Refiner (Repair/Tweak)**
5. **Credit Composer**
6. **Intro Audio Assembler**
7. **Observability & Experiment Layer**

---

## 4) Hook generatie: drie-fasen model

### Fase A — Intent extraction
- Maak eerst een compacte `reading intent card`:
  - `core_tension`
  - `implicit_choice`
  - `likely_consequence`
  - `emotional_tone`

### Fase B — Multi-angle generation
- Genereer kandidaten per vaste angle-set (`risk`, `choice`, `blindspot`, `reveal`, `cost`).
- Kandidaten dragen een angle-tag in de generator-output en worden daarna opgeschoond voor validatie/judge.

### Fase C — Judge + constrained rewrite
- Harde filter (lengte, tekenset, banned terms, etc.).
- Scoremodel voor ranking en diagnostics.
- Judge + repair/tweak blijven het finalisatiepad.

### 4.1 Scoremodel (nu geïmplementeerd)

Per kandidaat wordt nu gelogd:
- compliance (0/1)
- curiosity_tension (0–5)
- concreteness (0–5)
- viewer_relevance (0–5)
- spoken_fluency (0–5)
- novelty (0–5)
- total (gewogen som)

---

## 5) Intro-flow en fallbacks

### 5.1 Statusmachine (doel)
- `FULL_INTRO_OK`
- `HOOK_WEAK_REPAIRED`
- `CREDIT_ONLY`
- `NARRATION_ONLY`

### 5.2 Beslisregels
- Top-hook onder drempel na repair-pass → credit-only.
- TTS/concat stukloopt maar narratie bestaat → narration-only.
- Statusgedreven afronding i.p.v. exception-gedreven default.

---

## 6) Prompt- en policybeheer

Promptset uitgebreid met intent-stap:
- `hook_intent.prompt` (nu als prompt-constant)
- `hook_generate.prompt`
- `hook_judge.prompt`
- `hook_repair.prompt`
- `hook_tweaker.prompt`

Volgende stap blijft: prompts fysiek als externe versieerde assets plaatsen.

---

## 7) Data-contracten

In code zitten nu expliciete structuren voor hook-evaluatie:
- `CandidateCheck` (incl. `angle`)
- `IntentCard`
- `HookScoreCard`

Doel blijft om door te groeien naar volledige objecten zoals `HookDecision`, `IntroPlan`, `IntroResult`.

---

## 8) Monoliet vs modulair

**Keuze blijft: modulair.**

De nieuwe intent + angle + scorelaag laat zien dat iteratie sneller kan zonder audioflow te herschrijven.

---

## 9) Gefaseerd implementatieplan (bijgewerkt)

1. ✅ **Fase 1 — Functionele ontvlechting**
   - Gereed.

2. ✅ **Fase 2 — Nieuwe hookflow**
   - Intent extraction toegevoegd.
   - Angle-gedreven generatie toegevoegd.
   - Conceptueel scoremodel toegevoegd en aangesloten op ranking/observability.
   - Basis-tests toegevoegd voor parser/score/input-builders.

3. ✅ **Fase 3 — Telemetrie & experimenten**
   - Promptversies nu opgenomen in hook-metadata en logging.
   - Structurele opslag toegevoegd via JSONL-eventlog met intent card, candidate stats en outcome.
   - A/B-profielen toegevoegd voor hookstijlen (`control`, `curiosity`, `consequence`).

4. ⏳ **Fase 4 — Kwaliteitsoptimalisatie**
   - Drempels/gewichten kalibreren op productiedata.
   - Timingprofielen en fallback tuning.

---

## 10) Afbakening en bijstelling

Bijstelling op oorspronkelijk plan:
- In plaats van direct volledig semantisch scorende LLM-judge per dimensie, eerst een **lichte deterministische scorelaag** ingevoerd voor transparantie, snelheid en lage kosten.
- Deze scorelaag is bedoeld als **tussenstap** richting uitgebreidere telemetrie in fase 3.

Resultaat: fase 2 is functioneel afgerond met lage implementatierisico’s en duidelijke basis voor data-gedreven iteratie.

Resultaat fase 3: basis-telemetrie en experimenteerlaag staan nu operationeel, waardoor hookkwaliteit per profiel en promptversie vergelijkbaar is over runs heen.
