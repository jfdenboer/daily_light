# Herontwerp Intro-laag (concept, geen code)

## 1) Doel van het herontwerp

De intro-laag moet in één consistente flow drie dingen doen:
1. **Aandacht grijpen** (sterke hook in spreektaal).
2. **Context geven** (korte credit/positionering zonder tempoverlies).
3. **Naadloos overgaan** naar de hoofdtekst (ritme + audiokwaliteit).

Het huidige systeem werkt functioneel, maar de hook-logica is zwaar gecentraliseerd in één groot script. Dat maakt iteratie op prompt-strategie, evaluatie en fallback-gedrag relatief traag en foutgevoeliger.

---

## 2) Ontwerpprincipes

- **Hook-first architectuur**: de hook bepaalt de intro, niet andersom.
- **Kwaliteit meetbaar maken**: elke stap moet scorebaar en uitlegbaar zijn.
- **Fail-soft i.p.v. fail-hard**: degradeer netjes (full intro → credit-only → no intro).
- **Losse verantwoordelijkheden**: generatie, beoordeling, repair, audio-assemblage gescheiden.
- **Prompten als productonderdeel**: prompts/versioning als beheerde assets, niet als inline strings.

---

## 3) Voorgestelde architectuur

Ik zou **niet** één groot script houden. In plaats daarvan een modulaire intro-pipeline met duidelijke contracten.

### 3.1 Componenten

1. **Intro Orchestrator**
   - Stuurt de volledige intro-flow aan.
   - Beslist op basis van statuscodes en quality gates over volgende stap.

2. **Hook Ideation Engine**
   - Genereert meerdere hook-kandidaten per reading.
   - Werkt met variatie-assen (conflict, keuze, risico, ironie, emotionele frictie).

3. **Hook Evaluator**
   - Scoort kandidaten op harde regels + zachte kwaliteitsscores.
   - Geeft ranking + reden per kandidaat.

4. **Hook Refiner (Repair/Tweak)**
   - Repareert near-miss hooks met minimale semantische wijziging.
   - Produceert micro-varianten op de winnaar voor fine-tuning.

5. **Credit Composer**
   - Genereert/kiest korte creditzin in vast sjabloon.
   - Houdt stijl en lengte consistent met het kanaal.

6. **Intro Audio Assembler**
   - Maakt hook-audio, credit-audio, pauzes en concat.
   - Beheert timingprofielen (sneller/langzamer intro-stijl).

7. **Observability & Experiment Layer**
   - Logging, score-historie, win-rate per promptversie.
   - Ondersteunt A/B-profielen voor hookstijl.

---

## 4) Hook generatie: nieuw concept

### 4.1 Drie-fasen model

**Fase A — Intent extraction**
- Maak eerst een compacte “reading intent card”:
  - kernspanning
  - impliciete keuze
  - potentiële consequentie
  - emotionele toon
- Doel: generator voedt op inhoudelijke spanning i.p.v. losse tekstoppervlakken.

**Fase B — Multi-angle generation**
- Genereer kandidaten per hoek (bijv. “what you risk”, “what you ignore”, “what this reveals”).
- Minimaal 2 kandidaten per hoek om lexicale diversiteit te behouden.

**Fase C — Judge + constrained rewrite**
- Eerst harde filter (lengte, 1 zin, verboden termen, etc.).
- Daarna semantische ranking (curiosity, concreetheid, relevantie, spoilerveilig).
- Daarna één constrained rewrite op top-1 en top-2, opnieuw beoordelen, dan winnaar.

### 4.2 Hook-scoremodel (conceptueel)

Per kandidaat een scorekaart:
- **Compliance (0/1 gate)**
- **Curiosity tension (0–5)**
- **Concreteness (0–5)**
- **Viewer relevance (0–5)**
- **Spoken fluency (0–5)**
- **Novelty t.o.v. andere kandidaten (0–5)**

Totale score = gewogen som, maar alleen als compliance=1.

### 4.3 Waarom dit beter is

- Minder afhankelijk van één “perfecte” prompt.
- Betere reproduceerbaarheid door expliciete fases.
- Hogere kans op sterke hooks door angle-gedreven generatie.

---

## 5) Intro-flow en fallbacks

### 5.1 Gewenste statusmachine

- `FULL_INTRO_OK`: hook + credit + pauzes.
- `HOOK_WEAK_REPAIRED`: gerepareerde hook gebruikt.
- `CREDIT_ONLY`: hook niet betrouwbaar genoeg.
- `NARRATION_ONLY`: intro uitgezet of alle introstappen gefaald.

### 5.2 Beslisregels

- Als top-hook onder kwaliteitsdrempel blijft na 1 repair-pass → credit-only.
- Als TTS/concat stukloopt maar narratie bestaat → narration-only.
- Geen exception als default pad; statusgedreven afronding wel.

---

## 6) Prompt- en policybeheer

- Prompts buiten code opslaan als versieerde templates.
- Per component eigen promptbestand:
  - `hook_intent.prompt`
  - `hook_generate.prompt`
  - `hook_judge.prompt`
  - `hook_repair.prompt`
- Policylijsten (verboden woorden, stijlregels) centraal beheren.
- Promptversie loggen in outputmetadata voor analyse achteraf.

---

## 7) Data-contracten (zonder code)

Voorstel om met expliciete objecten te werken:
- `HookCandidate` (tekst, angle, scorekaart, violations)
- `HookDecision` (winner, runner-up, rationale, prompt_versions)
- `IntroPlan` (hook_text, credit_text, timing_profile)
- `IntroResult` (status, intro_duration, assets, diagnostics)

Dit maakt debugging en experimenteren veel eenvoudiger dan ad-hoc strings doorgeven.

---

## 8) Monoliet vs modulair: expliciete keuze

**Advies: opsplitsen in modules.**

### Waarom niet één groot script houden?
- Te veel verantwoordelijkheden op één plek.
- Lastiger testen per stap (generator/judge/repair/audio).
- Promptwijzigingen hebben nu grote blast radius.

### Waarom wel modulair?
- Sneller itereren op hookkwaliteit zonder audioflow te breken.
- Betere unit/integratietesten per component.
- Eenvoudiger toekomstige extensies (bijv. meerdere hookstijlen per kanaal).

---

## 9) Gefaseerd implementatieplan (hoog niveau)

1. **Fase 1 — Functionele ontvlechting**
   - Logica opdelen in bovengenoemde componenten, gedrag inhoudelijk gelijk houden.

2. **Fase 2 — Nieuwe hookflow**
   - Intent extraction + angle generation + scoremodel invoeren.

3. **Fase 3 — Telemetrie & experimenten**
   - Promptversies, scorecards en outcome tracking toevoegen.

4. **Fase 4 — Kwaliteitsoptimalisatie**
   - Drempels, gewichten, timingprofielen en fallback-tuning op basis van data.

---

## 10) Concrete deliverables van dit herontwerp

- Nieuwe intro-architectuur met heldere componentgrenzen.
- Hookgeneratie als multi-fase systeem met scorekaart.
- Statusgedreven fallbackmodel i.p.v. exception-gedreven gedrag.
- Prompt/policybeheer als versieerbare assets.
- Duidelijke keuze om het huidige grote script op te delen.

