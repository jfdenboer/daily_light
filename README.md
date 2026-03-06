# Daily Light

🎥 **Automatische pipeline voor het omzetten van dagelijkse *Daily Light*-overdenkingen naar video’s**
📜 Gebaseerd op *Daily Light on the Daily Path*
🧠 AI-gegenereerde beelden, 🎙️ ElevenLabs TTS, 📋 ondertiteling via Rev.ai, en 🎬 volledige video-compilatie.

---

## ✨ Overzicht

Daily Light is een Python-project dat een volledige *text-to-video* pipeline biedt voor christelijke dagboeklezingen. Het systeem verwerkt ruwe `.txt`-bestanden met dagteksten tot professioneel uitziende video’s, inclusief:

- 📖 Tekstanalyse op reading-niveau
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
   git clone https://github.com/jfdenboer/daily_light.git
   cd daily_light
   ```

2. **Installeer afhankelijkheden**
   Vereist: Python ≥ 3.11.9
   Aanbevolen: [Poetry](https://python-poetry.org/)

   ```bash
   poetry install
   ```

3. **Zorg dat `ffmpeg` geïnstalleerd is**

   Controleer lokaal:

   ```bash
   ffmpeg -version
   ```

4. **Maak een `.env` bestand aan**

   Er is momenteel geen `.env.example` in de repository. Maak daarom handmatig een `.env` aan in de projectroot.

   Minimale variabelen:

   - `OPENAI_API_KEY`
   - `ELEVENLABS_API_KEY`
   - `REV_AI_TOKEN`
   - `GCS_CREDENTIALS_PATH`

   Aanbevolen aanvullingen:

   - `ELEVENLABS_VOICE_ID`
   - `GCS_BUCKET_NAME`
   - `THUMBNAIL_ENABLED` (standaard `true`)
   - `THUMBNAIL_IMAGE_MODEL` (standaard `gpt-image-1.5`)
   - `THUMBNAIL_INTENT_CARD_MODEL` (standaard `gpt-5.2`)
   - `THUMBNAIL_INTENT_CARD_TEMPERATURE` (standaard `0.2`)
   - `THUMBNAIL_FONT_PATH` (optioneel, voor custom typografie)
   - `VIDEO_ZOOM_WIDE_START` (standaard `1.00`)
   - `VIDEO_ZOOM_WIDE_END` (standaard `1.10`, aanbevolen bereikverschil maximaal `0.12`)

   Voorbeeld:

   ```env
   OPENAI_API_KEY=...
   ELEVENLABS_API_KEY=...
   REV_AI_TOKEN=...
   GCS_CREDENTIALS_PATH=/abs/path/to/gcs-service-account.json
   GCS_BUCKET_NAME=spurgeon_bucket
   ```

---

## 🖼️ Thumbnail-configuratie

Thumbnails worden nu volledig in de pipeline gemaakt:

1. **Thumbnailtekst** via OpenAI chatmodel.
2. **Achtergrondafbeelding** via OpenAI `gpt-image-*` Images API.
3. **Compositie (image + tekst)** lokaal in Python.
4. **Upload** als YouTube thumbnail tijdens publicatie.

Optionele `.env` variabelen:

- `THUMBNAIL_ENABLED=true`
- `THUMBNAIL_IMAGE_MODEL=gpt-image-1.5`
- `THUMBNAIL_IMAGE_SIZE=1536x1024`
- `THUMBNAIL_IMAGE_QUALITY=medium`
- `THUMBNAIL_INTENT_CARD_MODEL=gpt-5.2`
- `THUMBNAIL_INTENT_CARD_TEMPERATURE=0.2`
- `THUMBNAIL_FONT_PATH=/abs/path/to/font.ttf`

---

## 🧪 CLI-gebruik

Voer de volledige pipeline uit:

```bash
poetry run daily build --start-date 2025-10-18 --end-date 2025-10-18
```

Start de lokale GUI voor een enkele datum:

```bash
poetry run daily gui
```

Genereer een YouTube OAuth-token (`token.json`):

```bash
poetry run python auth_setup.py
```

Beschikbare opties voor `build`:

| Optie               | Beschrijving                                        |
| ------------------- | --------------------------------------------------- |
| `--start-date`      | Startdatum (YYYY-MM-DD)                             |
| `--end-date`        | Einddatum (YYYY-MM-DD)                              |

---

## 📁 Bestandsstructuur

```text
spurgeon/
├── cli.py               # Typer CLI interface
├── config/
│   └── settings.py      # Pydantic settings
├── core/
│   ├── pipeline.py      # Main orchestration logic
│   └── parser.py        # Raw .txt → Reading[]
├── models.py            # Domeinmodellen (Reading, etc.)
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

Prompt generation now uses a two-message OpenAI payload: a full system prompt plus a minimal user excerpt wrapper (`EXCERPT` in triple quotes).

---

## 🧼 Logging

Standaard logt Spurgeon naar console én (optioneel) naar `logs/daily_light.log` met roterende logbestanden:

```env
LOG_LEVEL=INFO
LOG_FILE=logs/daily_light.log
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
