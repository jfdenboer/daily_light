"""Eenvoudige Tkinter GUI voor het starten van een pipeline-run op een specifieke datum."""

from __future__ import annotations

import logging
from queue import Empty, Queue
import threading
from datetime import date
import tkinter as tk
from tkinter import messagebox, ttk

from spurgeon.config.settings import load_settings
from spurgeon.core.pipeline import run_pipeline
from spurgeon.utils.logging_setup import init_logging


class SpurgeonGui:
    """Kleine desktop-GUI om een run voor één datum te starten."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Spurgeon Runner")
        self.root.resizable(False, False)

        self.selected_date = tk.StringVar(value=date.today().isoformat())
        self.status_text = tk.StringVar(value="Kies een datum en klik op 'Run'.")
        self.is_running = False
        self.log_queue: Queue[str] = Queue()
        self.log_handler: GuiLogHandler | None = None

        self._build_layout()
        self._start_log_pump()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.grid(row=0, column=0, sticky="nsew")

        ttk.Label(container, text="Datum (YYYY-MM-DD):").grid(
            row=0, column=0, sticky="w"
        )

        self.date_entry = ttk.Entry(container, textvariable=self.selected_date, width=16)
        self.date_entry.grid(row=1, column=0, sticky="w", pady=(4, 12))

        self.run_button = ttk.Button(container, text="Run", command=self._start_run)
        self.run_button.grid(row=1, column=1, padx=(8, 0), pady=(4, 12))

        ttk.Label(container, textvariable=self.status_text, wraplength=340).grid(
            row=2, column=0, columnspan=2, sticky="w"
        )

        ttk.Label(container, text="Logging:").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(12, 4)
        )

        logs_frame = ttk.Frame(container)
        logs_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(4, weight=1)

        self.logs_text = tk.Text(
            logs_frame,
            width=90,
            height=20,
            state=tk.DISABLED,
            wrap="word",
        )
        self.logs_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(logs_frame, orient="vertical", command=self.logs_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.logs_text.configure(yscrollcommand=scrollbar.set)
        logs_frame.columnconfigure(0, weight=1)
        logs_frame.rowconfigure(0, weight=1)

    def _start_log_pump(self) -> None:
        self.root.after(100, self._drain_log_queue)

    def _drain_log_queue(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except Empty:
                break
            self._append_log(message)

        self.root.after(100, self._drain_log_queue)

    def _append_log(self, message: str) -> None:
        self.logs_text.configure(state=tk.NORMAL)
        self.logs_text.insert(tk.END, message + "\n")
        self.logs_text.see(tk.END)
        self.logs_text.configure(state=tk.DISABLED)

    def _attach_gui_log_handler(self) -> None:
        if self.log_handler is not None:
            return

        handler = GuiLogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
        logging.getLogger().addHandler(handler)
        self.log_handler = handler

    def _detach_gui_log_handler(self) -> None:
        if self.log_handler is None:
            return

        logging.getLogger().removeHandler(self.log_handler)
        self.log_handler = None

    def _start_run(self) -> None:
        if self.is_running:
            return

        date_string = self.selected_date.get().strip()
        if not date_string:
            messagebox.showerror("Ongeldige datum", "Vul een datum in.")
            return

        try:
            run_date = date.fromisoformat(date_string)
        except ValueError:
            messagebox.showerror(
                "Ongeldige datum",
                "Gebruik formaat YYYY-MM-DD, bijvoorbeeld 2026-02-14.",
            )
            return

        self.is_running = True
        self.logs_text.configure(state=tk.NORMAL)
        self.logs_text.delete("1.0", tk.END)
        self.logs_text.configure(state=tk.DISABLED)
        self.run_button.configure(state=tk.DISABLED)
        self.date_entry.configure(state=tk.DISABLED)
        self.status_text.set(f"Run gestart voor {run_date.isoformat()}...")

        worker = threading.Thread(target=self._run_pipeline_for_date, args=(run_date,))
        worker.daemon = True
        worker.start()

    def _run_pipeline_for_date(self, run_date: date) -> None:
        try:
            settings = load_settings()
            init_logging(settings)
            self._attach_gui_log_handler()
            run_pipeline(
                settings=settings,
                start_date=run_date,
                end_date=run_date,
            )
        except Exception as exc:  # pragma: no cover - GUI runtime path
            self.root.after(
                0,
                lambda: self._finish_run(
                    success=False,
                    message=f"Run gefaald voor {run_date.isoformat()}: {exc}",
                ),
            )
            return

        self.root.after(
            0,
            lambda: self._finish_run(
                success=True,
                message=f"Run voltooid voor {run_date.isoformat()}.",
            ),
        )

    def _finish_run(self, success: bool, message: str) -> None:
        self.is_running = False
        self._detach_gui_log_handler()
        self.run_button.configure(state=tk.NORMAL)
        self.date_entry.configure(state=tk.NORMAL)
        self.status_text.set(message)

        if success:
            messagebox.showinfo("Klaar", message)
        else:
            messagebox.showerror("Fout", message)

    def run(self) -> None:
        self.root.mainloop()


def launch_gui() -> None:
    """Start de Tkinter GUI."""

    app = SpurgeonGui()
    app.run()


class GuiLogHandler(logging.Handler):
    """Logging-handler die records doorsluist naar een thread-safe GUI-queue."""

    def __init__(self, log_queue: Queue[str]) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        self.log_queue.put(message)
