# Khmer Voice Studio

A desktop GUI front-end for `khmer_voice_pipeline.py` — a local Khmer (ខ្មែរ) text-to-speech and voice-cloning toolkit built on top of `facebook/mms-tts-khm` and XTTS v2.

The GUI wraps the pipeline's modes (dataset prep, zero-shot cloning, batch synthesis, MMS fine-tune / resume / inference, XTTS v2 fine-tune) in a single CustomTkinter app with a live log, training dashboard, hardware-profile picker and audio preview.

---

## Features

- **Dataset** — transcribe & segment a long `.wav` into training clips with Whisper.
- **Clone** — zero-shot voice clone from a short reference clip (XTTS v2).
- **Batch** — synthesize a list of Khmer sentences using a single reference voice.
- **Train** — fine-tune `facebook/mms-tts-khm` on `.wav` / `.txt` pairs.
- **Resume** — continue MMS training from the last saved checkpoint.
- **Infer** — synthesize Khmer speech from text using your fine-tuned MMS model.
- **Fine-tune** — advanced XTTS v2 fine-tuning on a prepared dataset.
- **GUI tools**: hardware profile picker (auto-detects GPU / VRAM), audio preview with waveform, live training dashboard with epoch-vs-loss curve, checkpoint manager, sentence editor and export (merge `.wav`, convert to `.mp3`).

---

## Requirements

- Python 3.10+
- Linux (tested on Arch); should work on macOS / Windows with Tk.
- `ffmpeg` on `PATH` for the export tools.
- `khmer_voice_pipeline.py` next to `khmer_voice_gui.py` (the GUI shells out to it per run).

### Python dependencies

```bash
pip install customtkinter psutil --break-system-packages
```

The pipeline itself pulls in `torch`, `transformers`, `TTS`, `whisper`, etc. The GUI imports the pipeline lazily and falls back to a defaults shim if the import fails, so you can launch the GUI even before the heavy ML stack is installed.

---

## Run

```bash
python khmer_voice_gui.py
```

Pick a mode in the left sidebar, fill the form, hit **Run**. Live output streams into the log panel at the bottom; the **Dashboard** tab parses `loss=...` lines into a chart.

---

## Hardware profiles

The **Hardware** tab exposes two presets (Low VRAM ≈ 4 GB, Mid VRAM ≈ 8 GB) and an **Auto Detect** button that reads `torch.cuda.get_device_properties(0)` and picks one for you. The selected profile is saved to `config.json` and re-applied on launch; `BATCH_SIZE` / `EPOCHS` are pushed into the matching mode panels automatically.

---

## Project layout

```
khmer_voice_gui.py        # this app
khmer_voice_pipeline.py   # underlying pipeline (separate)
config.json               # last-used hardware profile
data/                     # training data (.wav + .txt pairs)
khmer-voice/              # MMS checkpoints
```

---

## Notes

- The GUI does **not** import `torch` itself — every run is launched as a subprocess of `python -u -c "<bootstrap>"`, so a crash in the pipeline never takes down the UI.
- Closing the window terminates any in-flight subprocess and cancels the log-drain timer.
- All file I/O is UTF-8; Khmer text in `voice2.txt` and the sentence editor is preserved as-is.
