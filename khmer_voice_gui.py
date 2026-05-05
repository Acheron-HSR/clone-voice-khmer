"""
Khmer Voice Cloning & Fine-tuning — Desktop GUI
CustomTkinter front-end for khmer_voice_pipeline.py

Run:
  python khmer_voice_gui.py

Install:
  pip install customtkinter --break-system-packages
"""

import os
import sys
import re
import time
import json
import wave
import shutil
import struct
import threading
import subprocess
import queue
import math
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import customtkinter as ctk

try:
    import psutil
except ImportError:
    psutil = None

# The pipeline pulls in torch/transformers — heavy and sometimes broken at
# import time. The GUI itself doesn't need them (it shells out per run), so
# import lazily and fall back to a small defaults shim if it fails.
try:
    import khmer_voice_pipeline as pipeline
except Exception as _pipeline_err:  # noqa: BLE001
    print(f"[gui] pipeline import deferred: {_pipeline_err}", file=sys.stderr)

    class _PipelineDefaults:
        DATA_DIR          = "./data"
        OUTPUT_DIR        = "./khmer-voice"
        BASE_MODEL        = "facebook/mms-tts-khm"
        TRAINED_MODEL     = "./khmer-voice/final"
        RESUME_FROM       = "./khmer-voice/final"
        SAMPLE_RATE       = 16000
        BATCH_SIZE        = 1
        EPOCHS            = 300
        ADDITIONAL_EPOCHS = 300
        LEARNING_RATE     = 1e-4
        RESUME_LR         = 5e-5
        INPUT_VOICE       = "voice.wav"
        VOICE2            = "voice2.wav"
        CLONE_TEXT        = "សួស្តី! នេះជាការសាកល្បងសម្លេងមែរ"
        CLONE_OUTPUT      = "cloned_khmer.wav"
        CLONE_LANG        = "km"
        INFER_TEXT_FILE   = "voice2.txt"
        INFER_OUTPUT_WAV  = "khmer_output.wav"
        DATASET_DIR       = "dataset_khmer"

    pipeline = _PipelineDefaults()


# ─── Theme ───────────────────────────────────────────────────────
BG          = "#0f1117"
PANEL       = "#161922"
PANEL_ALT   = "#1c2030"
BORDER      = "#262b3d"
TEXT        = "#e6e8f0"
MUTED       = "#8b91a8"
ACCENT      = "#7c3aed"
ACCENT_HOV  = "#6d28d9"
OK          = "#22c55e"
WARN        = "#f59e0b"
ERR         = "#ef4444"
TERMBG      = "#0a0c12"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PIPELINE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "khmer_voice_pipeline.py")
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config.json")


# ─── Hardware profiles ───────────────────────────────────────────
PROFILES = {
    "low": {
        "icon": "🟡",
        "label": "Low VRAM",
        "subtitle": "4GB VRAM / 32GB RAM",
        "color": WARN,
        "settings": {
            "compute_type":   "int8",
            "xtts_device":    "cpu",
            "whisper_device": "cuda",
            "BATCH_SIZE":     1,
            "MAX_TEXT_LEN":   256,
            "whisper_model":  "small",
            "EPOCHS":         100,
            "amp_autocast":   True,
        },
        "warning": "XTTS may be slow, use MMS for training",
    },
    "mid": {
        "icon": "🟢",
        "label": "Mid VRAM",
        "subtitle": "8GB VRAM",
        "color": OK,
        "settings": {
            "compute_type":   "int8_float16",
            "xtts_device":    "cuda",
            "whisper_device": "cuda",
            "BATCH_SIZE":     1,
            "MAX_TEXT_LEN":   512,
            "whisper_model":  "medium",
            "EPOCHS":         300,
            "amp_autocast":   False,
        },
        "warning": "",
    },
}


# ─── Mode metadata ───────────────────────────────────────────────
MODES = [
    {
        "id": "dataset",
        "icon": "🎙️",
        "label": "Dataset",
        "title": "Dataset Preparation",
        "desc":  "Transcribe & segment a long audio file into training clips with Whisper.",
        "fields": [
            ("file",   "VOICE2",      "Source audio (long .wav)", "voice2.wav", [("Audio", "*.wav *.mp3 *.flac")]),
            ("dir",    "DATASET_DIR", "Output dataset folder",    "dataset_khmer", None),
        ],
        "waveform": False,
    },
    {
        "id": "clone",
        "icon": "🔊",
        "label": "Clone",
        "title": "Zero-shot Voice Clone",
        "desc":  "Clone a voice from a short reference clip with XTTS v2 (no training).",
        "fields": [
            ("file",   "INPUT_VOICE",  "Reference voice (.wav)",   "voice.wav", [("Audio", "*.wav *.mp3 *.flac")]),
            ("text",   "CLONE_TEXT",   "Khmer text to synthesize", pipeline.CLONE_TEXT, None),
            ("string", "CLONE_LANG",   "Language code",            "km", None),
            ("save",   "CLONE_OUTPUT", "Output .wav",              "cloned_khmer.wav", [("WAV", "*.wav")]),
        ],
        "waveform": True,
    },
    {
        "id": "batch",
        "icon": "📦",
        "label": "Batch",
        "title": "Batch Synthesis",
        "desc":  "Synthesize a list of Khmer sentences using a single reference voice.",
        "fields": [
            ("file",   "VOICE2",      "Reference voice (.wav)", "voice2.wav", [("Audio", "*.wav *.mp3 *.flac")]),
            ("string", "CLONE_LANG",  "Language code",          "km", None),
        ],
        "waveform": False,
    },
    {
        "id": "train",
        "icon": "🏋️",
        "label": "Train",
        "title": "Fine-tune MMS (fresh)",
        "desc":  "Fine-tune facebook/mms-tts-khm on .wav/.txt pairs in your data folder.",
        "fields": [
            ("dir",    "DATA_DIR",      "Training data folder", "./data", None),
            ("dir",    "OUTPUT_DIR",    "Checkpoint output",    "./khmer-voice", None),
            ("int",    "EPOCHS",        "Epochs",               300, None),
            ("float",  "LEARNING_RATE", "Learning rate",        1e-4, None),
            ("int",    "BATCH_SIZE",    "Batch size",           1, None),
        ],
        "waveform": False,
    },
    {
        "id": "resume",
        "icon": "▶️",
        "label": "Resume",
        "title": "Resume Training",
        "desc":  "Continue MMS training from the last saved checkpoint.",
        "fields": [
            ("dir",    "RESUME_FROM",        "Resume from checkpoint", "./khmer-voice/final", None),
            ("dir",    "OUTPUT_DIR",         "Checkpoint output",      "./khmer-voice", None),
            ("int",    "ADDITIONAL_EPOCHS",  "Additional epochs",      300, None),
            ("float",  "RESUME_LR",          "Learning rate",          5e-5, None),
        ],
        "waveform": False,
    },
    {
        "id": "infer",
        "icon": "🎧",
        "label": "Infer",
        "title": "MMS Inference",
        "desc":  "Synthesize Khmer speech from text using your fine-tuned MMS model.",
        "fields": [
            ("dir",    "TRAINED_MODEL",    "Fine-tuned model dir", "./khmer-voice/final", None),
            ("file",   "INFER_TEXT_FILE",  "Khmer text file",      "voice2.txt", [("Text", "*.txt")]),
            ("save",   "INFER_OUTPUT_WAV", "Output .wav",          "khmer_output.wav", [("WAV", "*.wav")]),
        ],
        "waveform": True,
    },
    {
        "id": "finetune",
        "icon": "⚙️",
        "label": "Fine-tune",
        "title": "XTTS v2 Fine-tuning",
        "desc":  "Advanced: fine-tune XTTS v2 on your prepared dataset (uses xtts_config.json).",
        "fields": [
            ("file",   "XTTS_CONFIG", "XTTS config json", "xtts_config.json", [("JSON", "*.json")]),
            ("dir",    "XTTS_BASE",   "XTTS base ckpt dir", "./xtts_v2_base/", None),
            ("dir",    "XTTS_OUT",    "Output folder",      "./khmer_xtts_finetuned/", None),
        ],
        "waveform": False,
    },
]


# GUI-only utilities (don't shell out to the pipeline)
GUI_MODES = [
    {"id": "hardware",    "icon": "🛠️", "label": "Hardware",
     "title": "Hardware Profile",
     "desc":  "Pick a VRAM preset or auto-detect your GPU. Settings push to the relevant mode panels and persist in config.json."},
    {"id": "preview",     "icon": "🎵", "label": "Preview",
     "title": "Audio Preview",
     "desc":  "Play and visualize any .wav output without leaving the app."},
    {"id": "dashboard",   "icon": "📊", "label": "Dashboard",
     "title": "Training Dashboard",
     "desc":  "Live epoch-vs-loss curve parsed from the training log."},
    {"id": "checkpoints", "icon": "🗂️", "label": "Checkpoints",
     "title": "Checkpoint Manager",
     "desc":  "Browse, load, and delete saved model checkpoints."},
    {"id": "editor",      "icon": "📋", "label": "Editor",
     "title": "Sentence Editor",
     "desc":  "Edit, import, and save the Khmer sentence list used by Batch mode."},
    {"id": "export",      "icon": "💾", "label": "Export",
     "title": "Audio Export",
     "desc":  "Merge generated .wav files and convert them to .mp3 (ffmpeg)."},
]


# ─── Reusable widgets ────────────────────────────────────────────

class SidebarButton(ctk.CTkButton):
    def __init__(self, master, icon, label, command):
        super().__init__(
            master,
            text=f"  {icon}   {label}",
            anchor="w",
            height=44,
            corner_radius=10,
            fg_color="transparent",
            hover_color=PANEL_ALT,
            text_color=TEXT,
            font=ctk.CTkFont(size=14),
            command=command,
        )

    def set_active(self, active: bool):
        if active:
            self.configure(fg_color=ACCENT, hover_color=ACCENT_HOV, text_color="#ffffff")
        else:
            self.configure(fg_color="transparent", hover_color=PANEL_ALT, text_color=TEXT)


class StatusBadge(ctk.CTkLabel):
    COLORS = {
        "idle":    ("#1f2433", MUTED,  "● idle"),
        "running": ("#1f1633", "#c4b5fd", "● running"),
        "done":    ("#0f2a1c", OK,    "● done"),
        "error":   ("#2a1212", ERR,   "● error"),
    }

    def __init__(self, master):
        super().__init__(
            master,
            text="● idle",
            corner_radius=999,
            fg_color=self.COLORS["idle"][0],
            text_color=self.COLORS["idle"][1],
            font=ctk.CTkFont(size=12, weight="bold"),
            padx=14, pady=4,
        )
        self._state = "idle"

    def set(self, state: str):
        bg, fg, txt = self.COLORS.get(state, self.COLORS["idle"])
        self.configure(fg_color=bg, text_color=fg, text=txt)
        self._state = state


class Waveform(ctk.CTkCanvas):
    """Static decorative waveform for clone/infer panels."""
    def __init__(self, master):
        super().__init__(master, height=84, bg=PANEL_ALT, highlightthickness=0)
        self.bind("<Configure>", lambda e: self._draw())

    def _draw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 4 or h < 4:
            return
        self.create_rectangle(0, 0, w, h, fill=PANEL_ALT, outline=BORDER)
        bars = max(40, w // 6)
        cx = h / 2
        for i in range(bars):
            x = (i + 0.5) * (w / bars)
            amp = (0.18 + 0.82 * abs(math.sin(i * 0.35) * math.cos(i * 0.13))) * (h * 0.42)
            self.create_line(x, cx - amp, x, cx + amp, fill=ACCENT, width=2, capstyle="round")


# ─── Mode panel ──────────────────────────────────────────────────

class ModePanel(ctk.CTkFrame):
    def __init__(self, master, app, mode):
        super().__init__(master, fg_color=BG)
        self.app   = app
        self.mode  = mode
        self.vars  = {}

        # Header card
        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
        header.pack(fill="x", padx=24, pady=(20, 14))

        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.pack(fill="x", padx=22, pady=(18, 4))
        ctk.CTkLabel(
            title_row, text=f"{mode['icon']}  {mode['title']}",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT,
        ).pack(side="left")

        ctk.CTkLabel(
            header, text=mode["desc"],
            font=ctk.CTkFont(size=13), text_color=MUTED, justify="left",
        ).pack(anchor="w", padx=22, pady=(0, 18))

        # Form card
        form = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
        form.pack(fill="both", expand=False, padx=24, pady=(0, 14))

        if mode.get("waveform"):
            wave_wrap = ctk.CTkFrame(form, fg_color=PANEL_ALT, corner_radius=10, border_width=1, border_color=BORDER)
            wave_wrap.pack(fill="x", padx=22, pady=(20, 8))
            ctk.CTkLabel(
                wave_wrap, text="audio preview",
                font=ctk.CTkFont(size=11), text_color=MUTED,
            ).pack(anchor="w", padx=12, pady=(8, 0))
            Waveform(wave_wrap).pack(fill="x", padx=10, pady=(2, 10))

        for spec in mode["fields"]:
            self._build_field(form, *spec)

        # Run row
        run_row = ctk.CTkFrame(form, fg_color="transparent")
        run_row.pack(fill="x", padx=22, pady=(14, 22))

        self.run_btn = ctk.CTkButton(
            run_row, text=f"▶  Run {mode['label']}",
            height=46, width=220, corner_radius=10,
            fg_color=ACCENT, hover_color=ACCENT_HOV,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.on_run,
        )
        self.run_btn.pack(side="left")

        self.badge = StatusBadge(run_row)
        self.badge.pack(side="left", padx=14)

        self.stop_btn = ctk.CTkButton(
            run_row, text="■ Stop",
            height=46, width=90, corner_radius=10,
            fg_color="transparent", border_width=1, border_color=BORDER,
            text_color=MUTED, hover_color=PANEL_ALT,
            command=self.on_stop, state="disabled",
        )
        self.stop_btn.pack(side="left")

    def _build_field(self, parent, kind, var_name, label, default, file_types):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="x", padx=22, pady=6)

        ctk.CTkLabel(
            wrap, text=label, anchor="w",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED,
        ).pack(anchor="w", pady=(4, 4))

        row = ctk.CTkFrame(wrap, fg_color="transparent")
        row.pack(fill="x")

        var = tk.StringVar(value=str(default))
        self.vars[var_name] = (kind, var)

        entry = ctk.CTkEntry(
            row, textvariable=var, height=38, corner_radius=8,
            fg_color=PANEL_ALT, border_color=BORDER, border_width=1,
            text_color=TEXT, font=ctk.CTkFont(size=13),
        )
        entry.pack(side="left", fill="x", expand=True)

        if kind in ("file", "dir", "save"):
            def browse(k=kind, v=var, ft=file_types):
                if k == "file":
                    p = filedialog.askopenfilename(filetypes=ft or [("All", "*.*")])
                elif k == "save":
                    p = filedialog.asksaveasfilename(filetypes=ft or [("All", "*.*")],
                                                    defaultextension=".wav")
                else:
                    p = filedialog.askdirectory()
                if p:
                    v.set(p)
            ctk.CTkButton(
                row, text="Browse", width=92, height=38, corner_radius=8,
                fg_color=PANEL_ALT, hover_color=BORDER,
                border_width=1, border_color=BORDER, text_color=TEXT,
                command=browse,
            ).pack(side="left", padx=(8, 0))

    # ── run / stop ──
    def on_run(self):
        if self.app.is_running():
            return
        overrides = self._collect_overrides()
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.badge.set("running")
        self.app.run_mode(self.mode["id"], overrides, on_done=self._on_done)

    def on_stop(self):
        self.app.stop_mode()

    def _on_done(self, ok: bool):
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.badge.set("done" if ok else "error")

    def _collect_overrides(self):
        out = {}
        for name, (kind, var) in self.vars.items():
            v = var.get().strip()
            if not v:
                continue
            try:
                if kind == "int":
                    out[name] = int(float(v))
                elif kind == "float":
                    out[name] = float(v)
                else:
                    out[name] = v
            except ValueError:
                out[name] = v
        return out


# ─── GUI-only panels ─────────────────────────────────────────────

class GuiPanel(ctk.CTkFrame):
    """Shared header/body shell for GUI-only mode panels."""
    def __init__(self, master, app, mode):
        super().__init__(master, fg_color=BG)
        self.app  = app
        self.mode = mode

        header = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=14,
                              border_width=1, border_color=BORDER)
        header.pack(fill="x", padx=24, pady=(20, 14))
        ctk.CTkLabel(
            header, text=f"{mode['icon']}  {mode['title']}",
            font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT,
        ).pack(anchor="w", padx=22, pady=(18, 4))
        ctk.CTkLabel(
            header, text=mode["desc"],
            font=ctk.CTkFont(size=13), text_color=MUTED, justify="left",
        ).pack(anchor="w", padx=22, pady=(0, 18))

        self.body = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=14,
                                 border_width=1, border_color=BORDER)
        self.body.pack(fill="both", expand=True, padx=24, pady=(0, 20))


class PreviewPanel(GuiPanel):
    def __init__(self, master, app, mode):
        super().__init__(master, app, mode)
        self.play_proc = None
        self._samples  = []

        ctk.CTkLabel(
            self.body, text="WAV file",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED,
        ).pack(anchor="w", padx=22, pady=(20, 4))

        path_row = ctk.CTkFrame(self.body, fg_color="transparent")
        path_row.pack(fill="x", padx=22)
        self.path_var = tk.StringVar(
            value=getattr(pipeline, "INFER_OUTPUT_WAV", "khmer_output.wav"))
        ctk.CTkEntry(
            path_row, textvariable=self.path_var, height=38, corner_radius=8,
            fg_color=PANEL_ALT, border_color=BORDER, border_width=1,
            text_color=TEXT, font=ctk.CTkFont(size=13),
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            path_row, text="Browse", width=92, height=38, corner_radius=8,
            fg_color=PANEL_ALT, hover_color=BORDER, border_width=1, border_color=BORDER,
            text_color=TEXT, command=self._browse,
        ).pack(side="left", padx=(8, 0))

        wave_wrap = ctk.CTkFrame(self.body, fg_color=PANEL_ALT, corner_radius=10,
                                 border_width=1, border_color=BORDER)
        wave_wrap.pack(fill="x", padx=22, pady=(14, 8))
        ctk.CTkLabel(
            wave_wrap, text="waveform",
            font=ctk.CTkFont(size=11), text_color=MUTED,
        ).pack(anchor="w", padx=12, pady=(8, 0))
        self.canvas = ctk.CTkCanvas(wave_wrap, height=160, bg=PANEL_ALT,
                                    highlightthickness=0)
        self.canvas.pack(fill="x", padx=10, pady=(2, 10))
        self.canvas.bind("<Configure>", lambda e: self._draw_wave())

        ctrl = ctk.CTkFrame(self.body, fg_color="transparent")
        ctrl.pack(fill="x", padx=22, pady=(8, 22))
        self.play_btn = ctk.CTkButton(
            ctrl, text="▶  Play", height=42, width=120, corner_radius=10,
            fg_color=ACCENT, hover_color=ACCENT_HOV,
            font=ctk.CTkFont(size=14, weight="bold"), command=self._play,
        )
        self.play_btn.pack(side="left")
        self.stop_btn = ctk.CTkButton(
            ctrl, text="■  Stop", height=42, width=100, corner_radius=10,
            fg_color="transparent", border_width=1, border_color=BORDER,
            text_color=MUTED, hover_color=PANEL_ALT, command=self._stop,
        )
        self.stop_btn.pack(side="left", padx=(10, 0))
        self.info_lbl = ctk.CTkLabel(
            ctrl, text="—", font=ctk.CTkFont(size=12), text_color=MUTED,
        )
        self.info_lbl.pack(side="left", padx=14)

    def _browse(self):
        p = filedialog.askopenfilename(filetypes=[("WAV", "*.wav")])
        if p:
            self.path_var.set(p)
            self._load(p)

    def _load(self, path):
        try:
            with wave.open(path, "rb") as wf:
                n     = wf.getnframes()
                sr    = wf.getframerate()
                ch    = wf.getnchannels()
                width = wf.getsampwidth()
                raw   = wf.readframes(min(n, sr * 30))  # cap to first 30s
            fmt = {1: "B", 2: "h", 4: "i"}.get(width, "h")
            count = len(raw) // width
            samples = list(struct.unpack(f"<{count}{fmt}", raw))
            if width == 1:
                # WAV 8-bit PCM is unsigned (0..255, midpoint 128) per spec —
                # recenter to signed so the symmetric waveform renderer works.
                samples = [s - 128 for s in samples]
            if ch > 1:
                samples = samples[::ch]
            target = 600
            step = max(1, len(samples) // target)
            self._samples = samples[::step]
            dur = n / sr if sr else 0
            self.info_lbl.configure(
                text=f"{os.path.basename(path)} · {sr} Hz · {ch}ch · {dur:.1f}s",
                text_color=OK,
            )
            self._draw_wave()
        except Exception as e:
            self.info_lbl.configure(text=f"load failed: {e}", text_color=ERR)
            self._samples = []
            self._draw_wave()

    def _draw_wave(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width(); h = c.winfo_height()
        if w < 4 or h < 4:
            return
        c.create_rectangle(0, 0, w, h, fill=PANEL_ALT, outline=BORDER)
        if not self._samples:
            return
        peak = max((abs(s) for s in self._samples), default=1) or 1
        cy = h / 2
        n = len(self._samples)
        step = w / max(1, n)
        for i, s in enumerate(self._samples):
            x = i * step
            amp = (s / peak) * (h * 0.45)
            c.create_line(x, cy, x, cy - amp, fill=ACCENT, width=1)

    def _play(self):
        path = self.path_var.get().strip()
        if not path or not os.path.isfile(path):
            self.info_lbl.configure(text="file not found", text_color=ERR)
            return
        self._load(path)
        self._stop()
        for cmd in (
            ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", path],
            ["aplay", "-q", path],
            ["paplay", path],
        ):
            if shutil.which(cmd[0]):
                try:
                    self.play_proc = subprocess.Popen(
                        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.info_lbl.configure(text=f"playing via {cmd[0]}", text_color=OK)
                    return
                except Exception as e:
                    self.info_lbl.configure(text=f"{cmd[0]} failed: {e}", text_color=ERR)
        self.info_lbl.configure(
            text="no audio player found (install ffmpeg or alsa-utils)",
            text_color=WARN,
        )

    def _stop(self):
        if self.play_proc and self.play_proc.poll() is None:
            try:
                self.play_proc.terminate()
            except Exception:
                pass
        self.play_proc = None


class DashboardPanel(GuiPanel):
    LOSS_RE = re.compile(r"loss[=:\s]+([0-9]+\.[0-9]+)", re.I)

    def __init__(self, master, app, mode):
        super().__init__(master, app, mode)
        self.points = []          # [(step, loss), ...]
        self.start_time = None

        info = ctk.CTkFrame(self.body, fg_color="transparent")
        info.pack(fill="x", padx=22, pady=(20, 0))
        self.elapsed_lbl = ctk.CTkLabel(
            info, text="elapsed: —",
            font=ctk.CTkFont(size=12), text_color=MUTED)
        self.elapsed_lbl.pack(side="left")
        self.last_lbl = ctk.CTkLabel(
            info, text="last loss: —",
            font=ctk.CTkFont(size=12), text_color=MUTED)
        self.last_lbl.pack(side="left", padx=20)
        self.eta_lbl = ctk.CTkLabel(
            info, text="eta: —",
            font=ctk.CTkFont(size=12), text_color=MUTED)
        self.eta_lbl.pack(side="left")

        self.canvas = ctk.CTkCanvas(self.body, height=320, bg=TERMBG,
                                    highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=22, pady=14)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        ctrl = ctk.CTkFrame(self.body, fg_color="transparent")
        ctrl.pack(fill="x", padx=22, pady=(0, 22))
        ctk.CTkButton(
            ctrl, text="Clear", height=38, width=100, corner_radius=8,
            fg_color=PANEL_ALT, hover_color=BORDER,
            border_width=1, border_color=BORDER,
            text_color=TEXT, command=self.clear,
        ).pack(side="left")

        self.app.add_log_subscriber(self.feed_line)
        self._tick_id = self.app.after(1000, self._tick)

    def clear(self):
        self.points.clear()
        self.start_time = None
        self.elapsed_lbl.configure(text="elapsed: —")
        self.last_lbl.configure(text="last loss: —")
        self.eta_lbl.configure(text="eta: —")
        self._redraw()

    def feed_line(self, line):
        m = self.LOSS_RE.search(line)
        if not m:
            return
        try:
            v = float(m.group(1))
        except ValueError:
            return
        if self.start_time is None:
            self.start_time = time.time()
        self.points.append((len(self.points) + 1, v))
        self.last_lbl.configure(text=f"last loss: {v:.4f}", text_color=OK)
        self._redraw()

    def _tick(self):
        if self.start_time is not None:
            elapsed = int(time.time() - self.start_time)
            self.elapsed_lbl.configure(
                text=f"elapsed: {elapsed//60}m {elapsed%60}s")
            if len(self.points) >= 2:
                rate = elapsed / max(1, len(self.points))
                target = getattr(pipeline, "EPOCHS", 300) or 300
                remaining = max(0, target - len(self.points))
                eta = int(rate * remaining)
                self.eta_lbl.configure(text=f"eta: {eta//60}m {eta%60}s")
        try:
            self._tick_id = self.app.after(1000, self._tick)
        except Exception:
            pass

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        w = c.winfo_width(); h = c.winfo_height()
        if w < 4 or h < 4:
            return
        pad = 30
        c.create_rectangle(pad, pad, w - 10, h - pad, outline=BORDER)
        if len(self.points) < 2:
            c.create_text(
                w / 2, h / 2, text="waiting for loss values…",
                fill=MUTED, font=("JetBrains Mono", 12),
            )
            return
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x_lo, x_hi = min(xs), max(xs)
        y_lo, y_hi = min(ys), max(ys)
        if y_hi == y_lo:
            y_hi = y_lo + 1e-6
        if x_hi == x_lo:
            x_hi = x_lo + 1
        plot_w = (w - 10) - pad
        plot_h = (h - pad) - pad

        def to_px(x, y):
            px = pad + (x - x_lo) / (x_hi - x_lo) * plot_w
            py = (h - pad) - (y - y_lo) / (y_hi - y_lo) * plot_h
            return px, py

        last = None
        for x, y in self.points:
            cur = to_px(x, y)
            if last is not None:
                c.create_line(last[0], last[1], cur[0], cur[1],
                              fill=ACCENT, width=2)
            last = cur
        c.create_text(pad - 4, pad + 4, text=f"{y_hi:.3f}", anchor="ne",
                      fill=MUTED, font=("JetBrains Mono", 10))
        c.create_text(pad - 4, h - pad - 4, text=f"{y_lo:.3f}", anchor="se",
                      fill=MUTED, font=("JetBrains Mono", 10))


class CheckpointsPanel(GuiPanel):
    def __init__(self, master, app, mode):
        super().__init__(master, app, mode)

        ctk.CTkLabel(
            self.body, text="Checkpoint root",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED,
        ).pack(anchor="w", padx=22, pady=(20, 4))

        path_row = ctk.CTkFrame(self.body, fg_color="transparent")
        path_row.pack(fill="x", padx=22)
        self.root_var = tk.StringVar(
            value=getattr(pipeline, "OUTPUT_DIR", "./khmer-voice"))
        ctk.CTkEntry(
            path_row, textvariable=self.root_var, height=38, corner_radius=8,
            fg_color=PANEL_ALT, border_color=BORDER, border_width=1,
            text_color=TEXT, font=ctk.CTkFont(size=13),
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            path_row, text="Browse", width=92, height=38, corner_radius=8,
            fg_color=PANEL_ALT, hover_color=BORDER,
            border_width=1, border_color=BORDER, text_color=TEXT,
            command=lambda: self._set_root(filedialog.askdirectory()),
        ).pack(side="left", padx=(8, 0))
        ctk.CTkButton(
            path_row, text="Refresh", width=92, height=38, corner_radius=8,
            fg_color=ACCENT, hover_color=ACCENT_HOV, text_color="#ffffff",
            command=self.refresh,
        ).pack(side="left", padx=(8, 0))

        self.list_frame = ctk.CTkScrollableFrame(
            self.body, fg_color=PANEL_ALT, corner_radius=10, height=380)
        self.list_frame.pack(fill="both", expand=True, padx=22, pady=(14, 22))

        self.refresh()

    def _set_root(self, p):
        if p:
            self.root_var.set(p)
            self.refresh()

    def _scan(self):
        root = self.root_var.get().strip()
        out = []
        if not root or not os.path.isdir(root):
            return out
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if not os.path.isdir(full):
                continue
            size = 0
            for d, _, fs in os.walk(full):
                for f in fs:
                    fp = os.path.join(d, f)
                    if os.path.isfile(fp):
                        try:
                            size += os.path.getsize(fp)
                        except OSError:
                            pass
            mtime = os.path.getmtime(full)
            out.append((name, full, size, mtime))
        return out

    def refresh(self):
        for w in self.list_frame.winfo_children():
            w.destroy()
        items = self._scan()
        if not items:
            ctk.CTkLabel(
                self.list_frame, text="no checkpoints found",
                text_color=MUTED,
            ).pack(pady=20)
            return
        for name, full, size, mtime in items:
            self._add_row(name, full, size, mtime)

    def _add_row(self, name, full, size, mtime):
        row = ctk.CTkFrame(self.list_frame, fg_color=PANEL, corner_radius=8,
                           border_width=1, border_color=BORDER)
        row.pack(fill="x", padx=8, pady=4)
        info = ctk.CTkFrame(row, fg_color="transparent")
        info.pack(side="left", fill="x", expand=True, padx=12, pady=10)
        ctk.CTkLabel(
            info, text=name, anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT,
        ).pack(anchor="w")
        size_mb = size / (1024 * 1024)
        date = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        ctk.CTkLabel(
            info, text=f"{size_mb:.1f} MB · {date}", anchor="w",
            font=ctk.CTkFont(size=11), text_color=MUTED,
        ).pack(anchor="w")

        ctk.CTkButton(
            row, text="Resume", width=80, height=32, corner_radius=8,
            fg_color=PANEL_ALT, hover_color=BORDER, text_color=TEXT,
            command=lambda p=full: self._load_into("resume", "RESUME_FROM", p),
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            row, text="Infer", width=80, height=32, corner_radius=8,
            fg_color=PANEL_ALT, hover_color=BORDER, text_color=TEXT,
            command=lambda p=full: self._load_into("infer", "TRAINED_MODEL", p),
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            row, text="Delete", width=80, height=32, corner_radius=8,
            fg_color="transparent", hover_color="#3a1818",
            border_width=1, border_color=ERR, text_color=ERR,
            command=lambda p=full: self._delete(p),
        ).pack(side="left", padx=(4, 12))

    def _load_into(self, mode_id, var_name, path):
        panel = self.app.panels.get(mode_id)
        if isinstance(panel, ModePanel) and var_name in panel.vars:
            _, var = panel.vars[var_name]
            var.set(path)
        self.app.show_mode(mode_id)

    def _delete(self, path):
        try:
            shutil.rmtree(path)
            self.app.log_write(f"[checkpoints] deleted {path}\n")
        except Exception as e:
            self.app.log_write(f"[checkpoints] delete failed: {e}\n")
        self.refresh()


class EditorPanel(GuiPanel):
    def __init__(self, master, app, mode):
        super().__init__(master, app, mode)

        head = ctk.CTkFrame(self.body, fg_color="transparent")
        head.pack(fill="x", padx=22, pady=(20, 8))
        ctk.CTkLabel(
            head, text="Khmer sentences (one per line)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED,
        ).pack(side="left")
        self.count_lbl = ctk.CTkLabel(
            head, text="0 lines",
            font=ctk.CTkFont(size=12), text_color=MUTED)
        self.count_lbl.pack(side="right")

        self.text = ctk.CTkTextbox(
            self.body, fg_color=PANEL_ALT, text_color=TEXT,
            font=ctk.CTkFont(family="JetBrains Mono", size=13),
            corner_radius=10, border_width=1, border_color=BORDER,
            wrap="word", height=380,
        )
        self.text.pack(fill="both", expand=True, padx=22, pady=(0, 14))
        self.text.bind("<KeyRelease>", lambda e: self._update_count())

        ctrl = ctk.CTkFrame(self.body, fg_color="transparent")
        ctrl.pack(fill="x", padx=22, pady=(0, 22))
        ctk.CTkButton(
            ctrl, text="Import .txt", height=42, width=130, corner_radius=10,
            fg_color=PANEL_ALT, hover_color=BORDER,
            border_width=1, border_color=BORDER, text_color=TEXT,
            command=self._import,
        ).pack(side="left")
        ctk.CTkButton(
            ctrl, text="Save .txt", height=42, width=130, corner_radius=10,
            fg_color=ACCENT, hover_color=ACCENT_HOV, text_color="#ffffff",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._save,
        ).pack(side="left", padx=(10, 0))
        self.status_lbl = ctk.CTkLabel(
            ctrl, text="—", font=ctk.CTkFont(size=12), text_color=MUTED)
        self.status_lbl.pack(side="left", padx=14)

    def _update_count(self):
        text = self.text.get("1.0", "end").rstrip("\n")
        n = len([ln for ln in text.split("\n") if ln.strip()]) if text else 0
        self.count_lbl.configure(text=f"{n} line{'s' if n != 1 else ''}")

    def _import(self):
        p = filedialog.askopenfilename(
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.status_lbl.configure(text=f"load failed: {e}", text_color=ERR)
            return
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.status_lbl.configure(
            text=f"loaded {os.path.basename(p)}", text_color=OK)
        self._update_count()

    def _save(self):
        p = filedialog.asksaveasfilename(
            filetypes=[("Text", "*.txt")], defaultextension=".txt")
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.text.get("1.0", "end").rstrip("\n") + "\n")
        except Exception as e:
            self.status_lbl.configure(text=f"save failed: {e}", text_color=ERR)
            return
        self.status_lbl.configure(
            text=f"saved → {os.path.basename(p)}", text_color=OK)


class ExportPanel(GuiPanel):
    def __init__(self, master, app, mode):
        super().__init__(master, app, mode)
        self.selected = set()

        ctk.CTkLabel(
            self.body, text="Source folder",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED,
        ).pack(anchor="w", padx=22, pady=(20, 4))

        sel_row = ctk.CTkFrame(self.body, fg_color="transparent")
        sel_row.pack(fill="x", padx=22)
        self.dir_var = tk.StringVar(value=".")
        ctk.CTkEntry(
            sel_row, textvariable=self.dir_var, height=38, corner_radius=8,
            fg_color=PANEL_ALT, border_color=BORDER, border_width=1,
            text_color=TEXT, font=ctk.CTkFont(size=13),
        ).pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            sel_row, text="Browse", width=92, height=38, corner_radius=8,
            fg_color=PANEL_ALT, hover_color=BORDER,
            border_width=1, border_color=BORDER, text_color=TEXT,
            command=lambda: self._set_dir(filedialog.askdirectory()),
        ).pack(side="left", padx=(8, 0))
        ctk.CTkButton(
            sel_row, text="Refresh", width=92, height=38, corner_radius=8,
            fg_color=ACCENT, hover_color=ACCENT_HOV, text_color="#ffffff",
            command=self.refresh,
        ).pack(side="left", padx=(8, 0))

        self.list_frame = ctk.CTkScrollableFrame(
            self.body, fg_color=PANEL_ALT, corner_radius=10, height=240)
        self.list_frame.pack(fill="both", expand=True, padx=22, pady=14)

        ctrl = ctk.CTkFrame(self.body, fg_color="transparent")
        ctrl.pack(fill="x", padx=22, pady=(0, 22))
        ctk.CTkButton(
            ctrl, text="Merge selected → .wav", height=42, width=200,
            corner_radius=10, fg_color=PANEL_ALT, hover_color=BORDER,
            border_width=1, border_color=BORDER, text_color=TEXT,
            command=self._merge,
        ).pack(side="left")
        ctk.CTkButton(
            ctrl, text="Convert selected → .mp3", height=42, width=200,
            corner_radius=10, fg_color=ACCENT, hover_color=ACCENT_HOV,
            text_color="#ffffff", font=ctk.CTkFont(size=13, weight="bold"),
            command=self._to_mp3,
        ).pack(side="left", padx=(10, 0))
        self.status_lbl = ctk.CTkLabel(
            ctrl, text="—", font=ctk.CTkFont(size=12), text_color=MUTED)
        self.status_lbl.pack(side="left", padx=14)

        self.refresh()

    def _set_dir(self, p):
        if p:
            self.dir_var.set(p)
            self.refresh()

    def refresh(self):
        for w in self.list_frame.winfo_children():
            w.destroy()
        self.selected.clear()
        d = self.dir_var.get().strip() or "."
        if not os.path.isdir(d):
            ctk.CTkLabel(self.list_frame, text="folder not found",
                         text_color=ERR).pack(pady=20)
            return
        wavs = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        if not wavs:
            ctk.CTkLabel(self.list_frame, text="no .wav files",
                         text_color=MUTED).pack(pady=20)
            return
        for name in wavs:
            full = os.path.join(d, name)
            try:
                size_kb = os.path.getsize(full) / 1024
            except OSError:
                size_kb = 0
            var = tk.BooleanVar(value=True)
            self.selected.add(full)
            ctk.CTkCheckBox(
                self.list_frame, text=f"{name}  ·  {size_kb:.1f} KB",
                variable=var, text_color=TEXT,
                command=lambda v=var, p=full: self._toggle(v, p),
            ).pack(anchor="w", padx=10, pady=3)

    def _toggle(self, var, path):
        if var.get():
            self.selected.add(path)
        else:
            self.selected.discard(path)

    def _ensure_ffmpeg(self):
        if not shutil.which("ffmpeg"):
            self.status_lbl.configure(
                text="ffmpeg not found in PATH", text_color=ERR)
            return False
        return True

    def _merge(self):
        if not self.selected:
            self.status_lbl.configure(text="nothing selected", text_color=WARN)
            return
        if not self._ensure_ffmpeg():
            return
        out = filedialog.asksaveasfilename(
            filetypes=[("WAV", "*.wav")], defaultextension=".wav",
            initialfile="merged.wav")
        if not out:
            return
        files = sorted(self.selected)
        listfile = out + ".concat.txt"
        try:
            with open(listfile, "w") as f:
                for p in files:
                    abs_p = os.path.abspath(p).replace("'", "'\\''")
                    f.write(f"file '{abs_p}'\n")
            r = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                 "-c", "copy", out],
                capture_output=True, text=True,
            )
        except Exception as e:
            self.status_lbl.configure(text=f"merge error: {e}", text_color=ERR)
            return
        finally:
            try:
                os.remove(listfile)
            except OSError:
                pass
        if r.returncode != 0:
            tail = r.stderr.splitlines()[-1] if r.stderr else "unknown"
            self.status_lbl.configure(text=f"ffmpeg failed: {tail}",
                                      text_color=ERR)
            return
        kb = os.path.getsize(out) / 1024
        self.status_lbl.configure(
            text=f"merged {len(files)} files → {os.path.basename(out)} "
                 f"({kb:.1f} KB)",
            text_color=OK,
        )

    def _to_mp3(self):
        if not self.selected:
            self.status_lbl.configure(text="nothing selected", text_color=WARN)
            return
        if not self._ensure_ffmpeg():
            return
        ok = 0
        for p in sorted(self.selected):
            mp3 = os.path.splitext(p)[0] + ".mp3"
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", p, "-codec:a", "libmp3lame",
                 "-qscale:a", "2", mp3],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                ok += 1
        self.status_lbl.configure(
            text=f"converted {ok}/{len(self.selected)} → .mp3",
            text_color=OK if ok else ERR,
        )


class HardwarePanel(GuiPanel):
    """Hardware preset picker with GPU auto-detect + live RAM display."""

    def __init__(self, master, app, mode):
        super().__init__(master, app, mode)
        self.active = None
        self.buttons = {}
        self._ram_after_id = None

        info = ctk.CTkFrame(self.body, fg_color="transparent")
        info.pack(fill="x", padx=22, pady=(20, 0))
        self.detect_lbl = ctk.CTkLabel(
            info, text="GPU: not scanned — click Auto Detect",
            font=ctk.CTkFont(size=12), text_color=MUTED,
        )
        self.detect_lbl.pack(anchor="w")
        self.ram_lbl = ctk.CTkLabel(
            info, text="RAM: —",
            font=ctk.CTkFont(size=12), text_color=MUTED,
        )
        self.ram_lbl.pack(anchor="w", pady=(4, 0))

        ctk.CTkLabel(
            self.body, text="Presets",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED,
        ).pack(anchor="w", padx=22, pady=(18, 6))

        grid = ctk.CTkFrame(self.body, fg_color="transparent")
        grid.pack(fill="x", padx=22)
        for pid, prof in PROFILES.items():
            self._build_preset_button(grid, pid, prof)
        self._build_auto_button(grid)

        self.status_lbl = ctk.CTkLabel(
            self.body, text="No profile applied",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=MUTED,
        )
        self.status_lbl.pack(anchor="w", padx=22, pady=(18, 4))

        self.warn_lbl = ctk.CTkLabel(
            self.body, text="", font=ctk.CTkFont(size=12), text_color=WARN,
        )
        self.warn_lbl.pack(anchor="w", padx=22, pady=(0, 10))

        ctk.CTkLabel(
            self.body, text="Active settings",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED,
        ).pack(anchor="w", padx=22, pady=(4, 6))

        self.preview_box = ctk.CTkTextbox(
            self.body, fg_color=PANEL_ALT, text_color=TEXT,
            font=ctk.CTkFont(family="JetBrains Mono", size=12),
            corner_radius=10, border_width=1, border_color=BORDER,
            wrap="none", height=200,
        )
        self.preview_box.pack(fill="both", expand=True,
                              padx=22, pady=(0, 22))
        self.preview_box.configure(state="disabled")

        self._load_saved()
        self._tick_ram()

    def _build_preset_button(self, parent, pid, prof):
        btn = ctk.CTkButton(
            parent,
            text=f"{prof['icon']}  {prof['label']}\n{prof['subtitle']}",
            height=72, width=210, corner_radius=10,
            fg_color=PANEL_ALT, hover_color=BORDER,
            border_width=1, border_color=BORDER,
            text_color=TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=lambda p=pid: self.apply(p),
        )
        btn.pack(side="left", padx=(0, 10))
        self.buttons[pid] = btn

    def _build_auto_button(self, parent):
        btn = ctk.CTkButton(
            parent,
            text="🔵  Auto Detect\nScan GPU",
            height=72, width=210, corner_radius=10,
            fg_color=ACCENT, hover_color=ACCENT_HOV,
            text_color="#ffffff",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._auto_detect,
        )
        btn.pack(side="left", padx=(0, 10))
        self.buttons["auto"] = btn

    def _auto_detect(self):
        try:
            import torch
        except Exception as e:
            self.detect_lbl.configure(
                text=f"GPU: torch not installed ({e})", text_color=ERR,
            )
            return
        try:
            if not torch.cuda.is_available():
                self.detect_lbl.configure(
                    text="GPU: no CUDA device — falling back to Low",
                    text_color=WARN,
                )
                self.apply("low")
                return
            props = torch.cuda.get_device_properties(0)
            name = props.name
            vram_gb = props.total_memory / (1024 ** 3)
        except Exception as e:
            self.detect_lbl.configure(
                text=f"GPU: detection failed ({e})", text_color=ERR,
            )
            return
        self.detect_lbl.configure(
            text=f"Detected: {name} — {vram_gb:.0f}GB",
            text_color=OK,
        )
        # 6GB threshold: most cards labelled "8GB" report ~7.7GB.
        chosen = "mid" if vram_gb >= 6 else "low"
        self.apply(chosen)

    def apply(self, pid):
        prof = PROFILES.get(pid)
        if not prof:
            return
        self.active = pid

        for bid, btn in self.buttons.items():
            if bid == pid:
                btn.configure(
                    fg_color=ACCENT, hover_color=ACCENT_HOV,
                    border_color=ACCENT, text_color="#ffffff",
                )
            elif bid != "auto":
                btn.configure(
                    fg_color=PANEL_ALT, hover_color=BORDER,
                    border_color=BORDER, text_color=TEXT,
                )

        self.status_lbl.configure(
            text=f"Profile applied: {prof['label']}",
            text_color=prof["color"],
        )
        self.warn_lbl.configure(text=prof.get("warning", ""))

        s = prof["settings"]
        # Push values into the matching ModePanel entries, where they exist.
        self._push_field("train",  "BATCH_SIZE", s["BATCH_SIZE"])
        self._push_field("train",  "EPOCHS",     s["EPOCHS"])
        self._push_field("resume", "ADDITIONAL_EPOCHS", s["EPOCHS"])

        self.preview_box.configure(state="normal")
        self.preview_box.delete("1.0", "end")
        for k, v in s.items():
            self.preview_box.insert("end", f"{k:<16} = {v}\n")
        self.preview_box.configure(state="disabled")

        self._save(pid)
        self.app.log_write(f"[hardware] profile '{pid}' applied\n")

    def _push_field(self, mode_id, var_name, value):
        panel = self.app.panels.get(mode_id)
        if isinstance(panel, ModePanel) and var_name in panel.vars:
            _, var = panel.vars[var_name]
            var.set(str(value))

    def _save(self, pid):
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({"profile": pid}, f, indent=2)
        except Exception as e:
            self.app.log_write(f"[hardware] save failed: {e}\n")

    def _load_saved(self):
        try:
            if os.path.isfile(CONFIG_PATH):
                with open(CONFIG_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                pid = data.get("profile")
                if pid in PROFILES:
                    self.apply(pid)
        except Exception:
            pass

    def _tick_ram(self):
        if psutil is not None:
            try:
                mem = psutil.virtual_memory()
                used = mem.used / (1024 ** 3)
                total = mem.total / (1024 ** 3)
                pct = mem.percent
                self.ram_lbl.configure(
                    text=f"RAM: {used:.1f} / {total:.1f} GB  ({pct:.0f}%)",
                    text_color=WARN if pct > 85 else MUTED,
                )
            except Exception as e:
                self.ram_lbl.configure(
                    text=f"RAM: read failed ({e})", text_color=ERR)
        else:
            self.ram_lbl.configure(
                text="RAM: psutil missing — pip install psutil",
                text_color=MUTED,
            )
        try:
            self._ram_after_id = self.after(2000, self._tick_ram)
        except Exception:
            self._ram_after_id = None


GUI_PANEL_CLASSES = {
    "hardware":    HardwarePanel,
    "preview":     PreviewPanel,
    "dashboard":   DashboardPanel,
    "checkpoints": CheckpointsPanel,
    "editor":      EditorPanel,
    "export":      ExportPanel,
}


# ─── Main app ────────────────────────────────────────────────────

class KhmerVoiceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Khmer Voice Studio")
        self.geometry("1360x860")
        self.minsize(1280, 800)
        self.configure(fg_color=BG)

        self.proc = None
        self.proc_thread = None
        self.log_queue: queue.Queue = queue.Queue()
        self.log_subscribers = []
        self._drain_after_id = None

        self._build_layout()
        self._show_mode(MODES[0]["id"])
        self._drain_after_id = self.after(80, self._drain_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def add_log_subscriber(self, fn):
        self.log_subscribers.append(fn)

    # ── layout ──
    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # top toolbar
        topbar = ctk.CTkFrame(self, fg_color=PANEL, height=58, corner_radius=0,
                              border_width=0)
        topbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        topbar.grid_propagate(False)

        ctk.CTkLabel(
            topbar, text="🎛  Khmer Voice Studio",
            font=ctk.CTkFont(size=15, weight="bold"), text_color=TEXT,
        ).pack(side="left", padx=22)

        ctk.CTkLabel(
            topbar, text="local · cuda-ready · mms + xtts v2",
            font=ctk.CTkFont(size=11), text_color=MUTED,
        ).pack(side="left")

        # sidebar
        sidebar = ctk.CTkFrame(self, fg_color=PANEL, width=240, corner_radius=0)
        sidebar.grid(row=1, column=0, rowspan=2, sticky="ns")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar, text="MODES",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=MUTED,
        ).pack(anchor="w", padx=22, pady=(22, 10))

        self.nav_buttons = {}
        for m in MODES:
            btn = SidebarButton(
                sidebar, m["icon"], m["label"],
                command=lambda mid=m["id"]: self._show_mode(mid),
            )
            btn.pack(fill="x", padx=12, pady=2)
            self.nav_buttons[m["id"]] = btn

        ctk.CTkLabel(
            sidebar, text="TOOLS",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=MUTED,
        ).pack(anchor="w", padx=22, pady=(18, 10))

        for m in GUI_MODES:
            btn = SidebarButton(
                sidebar, m["icon"], m["label"],
                command=lambda mid=m["id"]: self._show_mode(mid),
            )
            btn.pack(fill="x", padx=12, pady=2)
            self.nav_buttons[m["id"]] = btn

        # main column (panel + log)
        self.main_col = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self.main_col.grid(row=1, column=1, sticky="nsew")
        self.main_col.grid_columnconfigure(0, weight=1)
        self.main_col.grid_rowconfigure(0, weight=3)
        self.main_col.grid_rowconfigure(1, weight=2)

        # panel host
        self.panel_host = ctk.CTkScrollableFrame(self.main_col, fg_color=BG, corner_radius=0)
        self.panel_host.grid(row=0, column=0, sticky="nsew")
        self.panel_host.grid_columnconfigure(0, weight=1)

        self.panels = {}
        for m in MODES:
            p = ModePanel(self.panel_host, self, m)
            self.panels[m["id"]] = p
        for m in GUI_MODES:
            cls = GUI_PANEL_CLASSES[m["id"]]
            self.panels[m["id"]] = cls(self.panel_host, self, m)

        # log + progress
        log_card = ctk.CTkFrame(self.main_col, fg_color=PANEL, corner_radius=14,
                                border_width=1, border_color=BORDER)
        log_card.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 20))
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)

        log_head = ctk.CTkFrame(log_card, fg_color="transparent", height=42)
        log_head.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 0))
        ctk.CTkLabel(
            log_head, text="● live log",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=MUTED,
        ).pack(side="left")
        ctk.CTkButton(
            log_head, text="Clear", width=70, height=28, corner_radius=8,
            fg_color="transparent", border_width=1, border_color=BORDER,
            text_color=MUTED, hover_color=PANEL_ALT,
            command=lambda: self._clear_log(),
        ).pack(side="right")

        self.log_box = ctk.CTkTextbox(
            log_card, fg_color=TERMBG, text_color="#cdd2e5",
            font=ctk.CTkFont(family="JetBrains Mono", size=12),
            corner_radius=10, border_width=1, border_color=BORDER,
            wrap="word",
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=10)
        self.log_box.configure(state="disabled")

        self.progress = ctk.CTkProgressBar(
            log_card, height=8, corner_radius=4,
            fg_color=PANEL_ALT, progress_color=ACCENT,
        )
        self.progress.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.progress.set(0)

        self.log_write("Khmer Voice Studio ready. Pick a mode on the left to begin.\n")

    # ── navigation ──
    def _show_mode(self, mode_id):
        for mid, btn in self.nav_buttons.items():
            btn.set_active(mid == mode_id)
        for mid, panel in self.panels.items():
            panel.pack_forget()
        self.panels[mode_id].pack(fill="both", expand=True)
        self.active_mode = mode_id

    show_mode = _show_mode

    # ── log ──
    def log_write(self, text: str):
        # GUI panels (e.g. HardwarePanel._load_saved → apply) call log_write
        # during _build_layout, before log_box is constructed. Buffer to the
        # log_queue so _drain_log_queue picks it up after layout completes.
        if not hasattr(self, "log_box"):
            self.log_queue.put(text)
            return
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _drain_log_queue(self):
        # Bug fix: bail out if the window is gone, otherwise self.after()
        # raises _tkinter.TclError ("invalid command name") after destroy.
        try:
            if not self.winfo_exists():
                return
        except Exception:
            return
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item is None:
                    continue
                self.log_write(item)
                for sub in list(self.log_subscribers):
                    try:
                        sub(item)
                    except Exception:
                        pass
        except queue.Empty:
            pass
        try:
            self._drain_after_id = self.after(80, self._drain_log_queue)
        except Exception:
            self._drain_after_id = None

    # ── pipeline runner ──
    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def run_mode(self, mode_id, overrides, on_done):
        if self.is_running():
            return

        bootstrap = self._build_bootstrap(mode_id, overrides)

        self.log_write(f"\n──── ▶ running '{mode_id}' ────\n")
        self.progress.configure(mode="indeterminate")
        self.progress.start()

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", "-c", bootstrap],
                cwd=os.path.dirname(PIPELINE_PATH),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, env=env,
            )
        except Exception as e:
            self.log_write(f"[error] failed to launch: {e}\n")
            on_done(False)
            self._stop_progress()
            return

        def reader(p, on_finish):
            try:
                for line in iter(p.stdout.readline, ""):
                    if not line:
                        break
                    self.log_queue.put(line)
            finally:
                try:
                    p.stdout.close()
                except Exception:
                    pass
                rc = p.wait()
                self.log_queue.put(f"\n──── exit code {rc} ────\n")
                # Bug fix: if the user closed the window mid-run, self.after
                # raises TclError ("application has been destroyed") from this
                # daemon thread. Guard so the reader exits cleanly.
                try:
                    self.after(0, lambda: (self._stop_progress(),
                                           on_finish(rc == 0)))
                except Exception:
                    pass

        self.proc_thread = threading.Thread(
            target=reader, args=(self.proc, on_done), daemon=True,
        )
        self.proc_thread.start()

    def stop_mode(self):
        if self.is_running():
            self.log_write("[stop] terminating subprocess…\n")
            try:
                self.proc.terminate()
            except Exception as e:
                self.log_write(f"[stop] terminate failed: {e}\n")

    def _stop_progress(self):
        try:
            self.progress.stop()
        except Exception:
            pass
        self.progress.configure(mode="determinate")
        self.progress.set(1.0)

    def _build_bootstrap(self, mode_id, overrides):
        """Generate the inline Python that mutates pipeline globals & calls the mode."""
        lines = [
            "import sys, os",
            f"sys.path.insert(0, {os.path.dirname(PIPELINE_PATH)!r})",
            "import khmer_voice_pipeline as p",
        ]
        for key, val in overrides.items():
            if not hasattr(pipeline, key):
                continue
            lines.append(f"p.{key} = {val!r}")

        if mode_id == "finetune":
            xc = overrides.get("XTTS_CONFIG", "xtts_config.json")
            xb = overrides.get("XTTS_BASE",   "./xtts_v2_base/")
            xo = overrides.get("XTTS_OUT",    "./khmer_xtts_finetuned/")
            lines += [
                "from trainer import Trainer, TrainerArgs",
                "from TTS.tts.configs.xtts_config import XttsConfig",
                "from TTS.tts.models.xtts import Xtts",
                "config = XttsConfig()",
                f"config.load_json({xc!r})",
                "model = Xtts.init_from_config(config)",
                f"model.load_checkpoint(config, checkpoint_dir={xb!r})",
                f"trainer = Trainer(TrainerArgs(output_path={xo!r}), config, "
                f"output_path={xo!r}, model=model, train_samples=None, eval_samples=None)",
                "trainer.fit()",
            ]
        else:
            lines.append(f"p.MODES[{mode_id!r}]()")

        return "\n".join(lines)

    def _on_close(self):
        if self.is_running():
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self._drain_after_id is not None:
            try:
                self.after_cancel(self._drain_after_id)
            except Exception:
                pass
            self._drain_after_id = None
        self.destroy()


if __name__ == "__main__":
    app = KhmerVoiceApp()
    app.mainloop()
