# Spurgeon

🎥 **Automatische pipeline voor het omzetten van dagelijkse Spurgeon-overdenkingen naar video’s**
📜 Gebaseerd op *Morning and Evening* van C.H. Spurgeon
🧠 AI-gegenereerde beelden, 🎙️ ElevenLabs TTS, 📋 ondertiteling via Rev.ai, en 🎬 volledige video-compilatie.

---

## ✨ Overzicht

Spurgeon is een Python-project dat een volledige *text-to-video* pipeline biedt voor christelijke dagboeklezingen. Het systeem verwerkt ruwe `.txt`-bestanden met dagteksten tot professioneel uitziende video’s, inclusief:

- 📖 Tekstanalyse en segmentatie
- 🗣️ Tekst-naar-spraak synthese via ElevenLabs
- 🧠 Beeldgeneratie met OpenAI `gpt-image-1` (Images API)
- ⏱️ Forced alignment en ondertiteling met Rev.ai
- 🎞️ Compilatie tot eindvideo met ondertiteling
- ☁️ Upload naar Google Cloud Storage (optioneel)

---

## 📦 Features

- ✅ CLI-interface via [`typer`](https://typer.tiangolo.com/)
- ✅ Modulair ontwerp per service (TTS, alignment, image, video)
- ✅ Configuratie via `.env` met validatie via `pydantic-settings`
- ✅ Retry-logica met exponential backoff en jitter
- ✅ Line-based én word-based `.srt` ondertitels
- ✅ Automatische leap-year mapping (29 februari)

---

## 🚀 Installatie

1. **Clone de repository**

   ```bash
   git clone https://github.com/jfdenboer/spurgeon.git
   cd spurgeon
   ```

2. **Installeer afhankelijkheden**
   Vereist: Python ≥ 3.11.9
   Aanbevolen: [Poetry](https://python-poetry.org/)

   ```bash
   poetry install
   ```

3. **Maak een `.env` bestand aan**

   Er is momenteel geen `.env.example` in de repository. Maak daarom handmatig een `.env` aan in de projectroot.

   Minimale variabelen:

   - `OPENAI_API_KEY`
   - `ELEVENLABS_API_KEY`
   - `ELEVENLABS_VOICE_ID`
   - `REV_AI_TOKEN`

   Optioneel (voor GCS upload):

   - `GCS_CREDENTIALS_PATH`
   - `GCS_BUCKET_NAME`

---

## 🖼️ Bannerbear-configuratie

1. **Projectnaam** – In Bannerbear gebruiken we het project `spurgeon`. Dit kun je aanpassen via `BANNERBEAR_PROJECT_NAME` in `.env`.
2. **Template** – Maak in het `spurgeon`-project een *Image Template* aan met YouTube-thumbnailformaat `1280 × 720`.
3. **Template-ID** – Noteer de UID van dit template en vul deze in als `BANNERBEAR_TEMPLATE_ID` in `.env`.
4. **Modifications** – Optioneel kun je eigen text-/image-layers configureren. Zonder extra configuratie gebruikt de code fallback-waarden voor titel en subtitel.

---

## 🧪 CLI-gebruik

Voer de volledige pipeline uit:

```bash
poetry run spurgeon build --start-date 2025-10-18 --end-date 2025-10-18
```

Beschikbare opties voor `build`:

| Optie               | Beschrijving                                        |
| ------------------- | --------------------------------------------------- |
| `--start-date`      | Startdatum (YYYY-MM-DD)                             |
| `--end-date`        | Einddatum (YYYY-MM-DD)                              |
| `--chunk-max-words` | Max woorden per segment (overschrijft `.env` value) |

---

## 📁 Bestandsstructuur

```text
spurgeon/
├── cli.py               # Typer CLI interface
├── config/
│   └── settings.py      # Pydantic settings
├── core/
│   ├── pipeline.py      # Main orchestration logic
│   ├── parser.py        # Raw .txt → Reading[]
│   └── segmenter.py     # Reading → SegmentBlocks
├── models.py            # Domeinmodellen (Reading, SegmentBlock, etc.)
├── services/
│   ├── tts/
│   │   ├── speech_synthesizer.py
│   │   └── elevenlabs_tts_client.py
│   ├── alignment/
│   │   └── rev_aligner.py
│   ├── image_gen/
│   │   └── image_generator.py
│   ├── prompt_generation/
│   │   └── prompt_generator.py
│   ├── subtitles/
│   │   └── builder.py
│   └── video_compile/
│       └── video_compiler.py
└── utils/
    ├── gcs_uploader.py
    ├── logging_setup.py
    └── retry_utils.py
```

---

## 🧠 Architectuur

```mermaid
graph TD
    A[.txt files (input/)] --> B[Parser]
    B --> C[Readings]
    C --> D[Segmenter]
    C --> E[ElevenLabs TTS]
    C --> F[PromptGenerator → OpenAI Images API]
    E --> G[Audio (mp3)]
    F --> H[Images (png)]
    G --> I[Rev.ai]
    I --> J[Word-based subtitles + JSON]
    J --> K[SubtitleBuilder (line-based)]
    H --> L[VideoCompiler]
    G --> L
    K --> L
    L --> M[Output/video.mp4]
```

---

## 🧼 Logging

Standaard logt Spurgeon naar console én (optioneel) naar `logs/spurgeon.log` met roterende logbestanden:

```env
LOG_LEVEL=INFO
LOG_FILE=logs/spurgeon.log
LOG_FILE_MAX_BYTES=10485760
LOG_FILE_BACKUP_COUNT=5
```

---

## 🌐 Cloud Upload (GCS)

Indien ingeschakeld in `.env`:

- Audio-bestanden worden geüpload naar `gs://<bucket>/audio/`
- Rev.ai gebruikt de GCS-URL voor alignments

Zorg voor een service-account met toegang tot GCS en configureer:

```bash
GCS_CREDENTIALS_PATH=/path/to/google_gcs.json
```

---

## ✅ TODO / Roadmap

- [x] Refactor naar `services/`-structuur
- [x] Ondersteuning voor `.srt` op woordniveau
- [x] Retry-logica geïntegreerd
- [ ] Whisper fallback bij afwezigheid ondertiteling
- [ ] Unit tests voor parser/subtitles
- [ ] YouTube upload integratie
- [ ] Web-based configuratie-interface

---

## 🤝 Contributie

Contributies zijn welkom. Maak een fork, open een PR, en houd code zoveel mogelijk PEP8-conform.

---

## 📄 Licentie

MIT License – vrij te gebruiken, wijzigen en delen. Zie `LICENSE`.

---

## ✝️ Waarom Spurgeon?

C.H. Spurgeon’s *Morning and Evening* is een tijdloos dagboek dat mensen inspireert tot dagelijkse overdenking. Met deze pipeline brengen we zijn woorden naar een nieuw publiek: visueel, auditief en toegankelijk via moderne technologie.
