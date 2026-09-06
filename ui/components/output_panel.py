"""
Output Panel Component

The results "contact sheet": status pill, run/cancel actions, progress,
live log console and output files rendered as selectable cards.
"""

from nicegui import ui
from typing import List, Optional, Dict
from pathlib import Path
import re

from ..runner import open_folder, open_file
from .free_log import FreeLog
from .page_progress import PageProgress
from .result_previews import add_result_previews


# Label of a tqdm-style progress bar, e.g. "Building target profiles:" from
# "Building target profiles:  45%|████▍       | 1328/2972 [00:05<00:06, ...]"
# or "Processing paths:" from "Processing paths: 13295222path [00:28, ...]"
# (unit counters without a total). Used to refresh the same bar in place
# instead of appending a new line.
_PROGRESS_NAME_RE = re.compile(
    r"^\s*([^%]*?)\s*\d+%\||^\s*([^:]*?):\s*\d+(?:\.\d+)?(?:path|it|file)s?\s*\["
)

# Counter-bearing progress output from tqdm and the project's LineProgress,
# e.g. ``Building paths: 45%|...| 9/20 [...]`` or
# ``Deriving type-level paths: 120/500 (24.0%) [...]``.
_PROGRESS_FRACTION_RE = re.compile(
    r"^\s*(?P<label>.*?):\s*"
    r"(?:(?:\d+(?:\.\d+)?)%\|.*?\|\s*)?"
    r"(?P<current>[\d,]+)\s*/\s*(?P<total>[\d,]+)"
)

# Structured step-progress event emitted by backend pipelines, e.g.
# "[DROCAT][progress] 2/6 Discovering candidates (connection cache)".
# Drives the determinate progress bar + step label in the results panel;
# the line itself is a control event and never appears in the log.
_PROGRESS_EVENT_RE = re.compile(r"^\[DROCAT\]\[progress\]\s*(\d+)\s*/\s*(\d+)\s*(.*)$")


# Sentinel key for the file list inside a folder-tree node.
_FILES_KEY = "__files__"


def _progress_bar_name(message: str) -> str:
    """Extract the progress-bar label from a tqdm-style line ('' if none)."""
    fraction = _progress_bar_fraction(message)
    if fraction is not None:
        return fraction[2]
    match = _PROGRESS_NAME_RE.match(message)
    if not match:
        return ""
    return (match.group(1) or match.group(2) or "").rstrip()


def _progress_bar_fraction(message: str):
    """Return ``(current, total, label)`` for a counter-bearing bar."""
    match = _PROGRESS_FRACTION_RE.match(message)
    if not match:
        return None
    try:
        current = int(match.group("current").replace(",", ""))
        total = int(match.group("total").replace(",", ""))
    except ValueError:
        return None
    if total <= 0:
        return None
    return current, total, match.group("label").strip()


_STATUS_COLORS = {
    "idle": "grey-5",
    "running": "blue",
    "success": "green",
    "failed": "red",
    "completed": "green",
}


def _build_file_tree(files: List[dict], output_dir: Optional[str]) -> dict:
    """Nest files by their relative path under *output_dir*.

    Returns a nested dict mirroring the output folder structure:
    subdirectory name -> child dict, plus ``_FILES_KEY`` -> list of file
    entries at that level.
    """
    tree = {}
    root = Path(output_dir).resolve() if output_dir else None
    for f in files:
        rel = Path(f["name"])
        if root is not None:
            try:
                rel = Path(f["path"]).resolve().relative_to(root)
            except ValueError:
                rel = Path(f["name"])
        node = tree
        for part in rel.parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault(_FILES_KEY, []).append(f)
    return tree


def _count_tree_files(tree: dict) -> int:
    """Total number of files in a (sub)tree."""
    return len(tree.get(_FILES_KEY, [])) + sum(
        _count_tree_files(v) for k, v in tree.items() if k != _FILES_KEY
    )


# Client-side copy routine for the execution log. The __LOG_ID__ placeholder
# is substituted with the panel's log element id at call time. Lines are read
# from the log DOM (each pushed line is a child element) so the copied text
# matches exactly what is on screen, including in-place tqdm refreshes and
# the max_lines trim. The async Clipboard API needs a secure context, so
# plain-HTTP LAN sessions fall back to the hidden-textarea execCommand trick.
_COPY_LOG_JS = """
(() => {
  const notify = (message, type) => {
    try { Quasar.Notify.create({ message, type }); } catch (err) {}
  };
  const log = document.getElementById('__LOG_ID__');
  if (!log) {
    notify('Execution log is not available.', 'negative');
    return;
  }
  const text = Array.from(log.children)
    .map((child) => child.textContent || '')
    .join('\\n')
    .replace(/\\n+$/, '');
  if (!text) {
    notify('Execution log is empty.', 'info');
    return;
  }
  const fallbackCopy = () => {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
    document.body.removeChild(textarea);
    return ok;
  };
  const finish = (ok) => notify(
    ok ? 'Execution log copied to clipboard.' : 'Copying the execution log failed.',
    ok ? 'positive' : 'negative'
  );
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(
      () => finish(true),
      () => finish(fallbackCopy())
    );
  } else {
    finish(fallbackCopy());
  }
})();
"""


class OutputPanel:
    """Reusable panel for displaying logs and output files."""

    def __init__(self, title: str = "Output"):
        self.title = title
        self._dom_id = f"drocat-results-{id(self)}"
        self._log_dom_id = f"drocat-exec-log-{id(self)}"
        self.log_wrapper: Optional[ui.element] = None
        self.log_area: Optional[FreeLog] = None
        self.copy_log_button: Optional[ui.button] = None
        self.files_container = None
        self.previews_section = None
        self.previews_container = None
        self.status_label: Optional[ui.badge] = None
        self.progress_bar = None
        self.progress_label: Optional[ui.label] = None
        self.progress_row = None
        self.page_progress: Optional[PageProgress] = None
        self.run_button: Optional[ui.button] = None
        self.cancel_button: Optional[ui.button] = None
        self._files: List[dict] = []
        self._last_is_progress = False
        self._last_progress_name: Optional[str] = None
        # Streaming: polls the run's output folder while the run is active so
        # files appear as soon as they are written. Folder expansions are
        # tracked (by relative path) so in-place refreshes keep the user's
        # open/closed state.
        self._poll_timer = None
        self._file_expansions: Dict[str, ui.expansion] = {}

    def create(
        self,
        run_label: str = "Run",
        run_icon: str = "play_arrow",
    ):
        """Create the output panel UI (run/cancel actions + status + log + files)."""
        with ui.card().classes("w-full drocat-card drocat-results-card gap-0").props(
            f'id="{self._dom_id}"'
        ):
            # Header: title + status pill
            with ui.row().classes("w-full items-center justify-between drocat-results-head"):
                with ui.row().classes("items-center gap-2"):
                    with ui.element("div").classes("drocat-results-mark"):
                        ui.icon("receipt_long").classes("text-white")
                    ui.label(self.title).classes("drocat-card-title")
                self.status_label = ui.badge("Idle", color="grey-5").props("outline")

            # Action bar: Run / Cancel
            with ui.row().classes("w-full items-center gap-2 drocat-action-bar"):
                self.run_button = ui.button(
                    run_label,
                    icon=run_icon,
                    color="primary",
                ).classes("drocat-run-btn")
                self.cancel_button = ui.button(
                    "Cancel",
                    icon="stop",
                    color="negative",
                ).classes("drocat-cancel-btn")
                self.cancel_button.disable()

            # Persistent run notice (F6): a banner parked above the log —
            # the effective-threshold summary survives log streams and
            # file refreshes until the next run overwrites it.
            self.notice_label = ui.label("").classes(
                "w-full text-caption"
            ).style(
                "background: #fff8e1; border: 1px solid #ffd54f; "
                "border-radius: 6px; padding: 6px 10px; "
                "white-space: pre-wrap; color: #5d4037;"
            ).set_visibility(False)

            # Keep the tracker in the original progress-row position, directly
            # above the execution log. Its bar is intentionally 3x the old
            # 4px height (12px), while the compatibility attributes continue
            # to expose the current bar and label to existing callers.
            self.page_progress = PageProgress().create(compact=True, visible=False)
            self.progress_row = self.page_progress.container
            self.progress_label = self.page_progress.progress_label
            self.progress_bar = self.page_progress.progress_bar

            # Log console. The heading row carries the copy button so every
            # tab that embeds this panel gets log-to-clipboard support.
            with ui.row().classes("w-full items-center justify-between gap-2"):
                ui.label("Execution Log").classes("drocat-mini-label")
                self.copy_log_button = (
                    ui.button(
                        icon="content_copy",
                        on_click=self._copy_log_to_clipboard,
                    )
                    .props("flat dense round size=sm color=grey-6")
                    .tooltip("Copy the execution log to the clipboard")
                )
            # The log window is pointer-resizable (drag the bottom edge): the
            # wrapper owns the CSS resize handle and carries a definite
            # initial height, while the inner log fills it (h-full) so it
            # tracks every drag. FreeLog keeps streaming reliable like
            # ui.log but scrolls freely: it follows the newest line only
            # while the view already rests at the bottom, so output can be
            # read anywhere in the history without being yanked back down.
            self.log_wrapper = ui.element("div").classes("w-full").style(
                "resize: vertical; overflow: hidden; height: 400px; min-height: 100px; max-height: 1800px;"
            )
            with self.log_wrapper:
                # The DOM id lets the copy button locate this log's lines in
                # the browser (the copy handler runs fully client-side).
                self.log_area = FreeLog(max_lines=500).props(
                    f'id="{self._log_dom_id}"'
                ).classes(
                    "w-full h-full font-mono text-xs"
                ).style(
                    # Both axes scroll: long output stays on its own single
                    # line (white-space: pre is inherited by every pushed
                    # line) and is reached through the horizontal bar instead
                    # of wrapping onto the next visual line.
                    # overflow-anchor is disabled because free_log.js
                    # compensates for head trims itself (Safari has no
                    # native scroll anchoring, and double compensation in
                    # Chromium would jitter).
                    "overflow-y: auto; overflow-x: auto; overflow-anchor: none;"
                    " white-space: pre;"
                )
            ui.separator()
            ui.label("Output Files").classes("drocat-mini-label")
            self.files_container = ui.column().classes("w-full gap-2")
            with self.files_container:
                ui.label("No output files yet.").classes("drocat-empty")
            # Result previews sit directly under the output files. The
            # section stays hidden until a completed run registers previews
            # for its tool (tools without tabular results never show it).
            ui.separator()
            self.previews_section = ui.column().classes("w-full gap-2")
            with self.previews_section:
                ui.label("Result Previews").classes("drocat-mini-label")
                self.previews_container = ui.column().classes("w-full")
                with self.previews_container:
                    ui.label("Available after a completed run.").classes(
                        "drocat-empty")
            self.previews_section.set_visibility(False)

    def log(self, message: str, level: str = "stdout"):
        """Add a log message to the panel."""
        if not self.log_area:
            return
        try:
            # Drop trailing whitespace (tqdm clears lines with spaces) and
            # whitespace-only residue so the log stays tidy.
            message = message.rstrip()
            if not message:
                return
            # Structured step-progress events from the backend drive the
            # determinate bar + step label; they are control lines, not log
            # output, so they are consumed here and never pushed to the log.
            step_match = _PROGRESS_EVENT_RE.match(message)
            if step_match:
                self._last_is_progress = False
                self._last_progress_name = None
                step = int(step_match.group(1))
                total = int(step_match.group(2))
                label = step_match.group(3).strip()
                if self.progress_label is not None:
                    text = f"Step {step}/{total}:" + (f" {label}" if label else "")
                    self.progress_label.text = text
                if self.page_progress is not None:
                    # Single source of truth for the bar value and step
                    # label: update_step applies the completed-steps
                    # semantics ((step - 1) / total, 100% only at finish).
                    self.page_progress.update_step(step, total, label)
                return
            # Progress lines (tqdm-style \r updates) refresh the previous line
            # of the SAME progress bar in place, so long-running functions
            # show a live-updating status instead of flooding the log. A new
            # bar (different label) starts a fresh line.
            if level == "progress":
                fraction = _progress_bar_fraction(message)
                if fraction is not None:
                    current, total, label = fraction
                    if self.page_progress is not None:
                        self.page_progress.update_fraction(current, total, label)
                    elif self.progress_bar is not None:
                        self.progress_bar.props(
                            ":indeterminate='false'", remove="indeterminate"
                        )
                        self.progress_bar.value = min(1.0, current / total)
                children = self.log_area.default_slot.children
                name = _progress_bar_name(message)
                if (
                    self._last_is_progress
                    and children
                    and name == self._last_progress_name
                ):
                    last = children[-1]
                    if hasattr(last, "set_text"):
                        last.set_text(message)
                        last.update()
                        return
                self._last_is_progress = True
                self._last_progress_name = name
                self.log_area.push(message)
                self.log_area.update()
                return
            self._last_is_progress = False
            self._last_progress_name = None
            prefix_map = {
                "stdout": "",
                "stderr": "[WARN] ",
                "error": "[ERROR] ",
                "success": "[OK] ",
                "system": "[SYS] ",
            }
            self.log_area.push(prefix_map.get(level, "") + message)
            # Force an immediate flush to the browser
            self.log_area.update()
        except Exception:
            # The page may have been closed or navigated away while the run
            # was active; silently drop further log lines instead of crashing
            # the run handler.
            pass

    def _copy_log_to_clipboard(self) -> None:
        """Copy the log text currently shown in the browser to the clipboard.

        The copy runs entirely client-side: the pushed lines live in the
        browser DOM, so collecting them there reflects exactly what the user
        sees (in-place tqdm refreshes and the max_lines trim included).
        """
        if self.log_area is None:
            return
        ui.run_javascript(_COPY_LOG_JS.replace("__LOG_ID__", self._log_dom_id))

    def set_notice(self, message: str) -> None:
        """Show the persistent notice banner (F6 effective-threshold
        summary). Empty text clears it."""
        notice_label = getattr(self, "notice_label", None)
        if not notice_label:
            return
        text = (message or "").strip()
        notice_label.set_text(text)
        notice_label.set_visibility(bool(text))

    def clear_notice(self) -> None:
        """Hide the persistent notice banner."""
        self.set_notice("")

    def set_status(self, status: str, color: str = "grey"):
        """Update the status pill."""
        if self.status_label:
            self.status_label.text = status
            color = _STATUS_COLORS.get(status.lower(), color)
            self.status_label.props(f"color={color}")
        if self.page_progress is not None:
            normalized = status.lower()
            if normalized in {"completed", "success"}:
                self.page_progress.finish(True)
            elif normalized in {"failed", "error", "cancelled"}:
                self.page_progress.finish(False, status)
            elif normalized == "running":
                self.page_progress.set_status("Running", "blue")
            else:
                self.page_progress.set_status(status, color)

    def set_running(self, running: bool):
        """Update UI for running state."""
        if running:
            self.set_status("Running", "blue")
            if self.run_button:
                self.run_button.disable()
            if self.cancel_button:
                self.cancel_button.enable()
            if self.progress_row:
                self.progress_row.set_visibility(True)
            if self.progress_bar:
                self.progress_bar.props("indeterminate")
            if self.page_progress:
                self.page_progress.start()
                self.page_progress.container.set_visibility(True)
            # Make sure the results panel (with the log) is visible
            ui.run_javascript(
                f"const card = document.getElementById('{self._dom_id}');"
                "if (card) card.scrollIntoView({behavior:'smooth', block:'nearest'});"
            )
        else:
            self._stop_file_streaming()
            if self.run_button:
                self.run_button.enable()
            if self.cancel_button:
                self.cancel_button.disable()
            if self.progress_row:
                # Keep the final progress and result status visible after the
                # process stops; a later clear/new run can reset it.
                self.progress_row.set_visibility(True)
            if self.progress_bar:
                self.progress_bar.props(
                    ":indeterminate='false'", remove="indeterminate"
                )
            # ScriptRunner reports its collect phase before returning. Keep
            # that last real phase/value unchanged until the caller applies
            # the final Completed or Failed status.

    def _stop_file_streaming(self):
        """Stop the output-folder polling timer (run finished or cancelled)."""
        if self._poll_timer is not None:
            self._poll_timer.cancel()
            self._poll_timer = None

    def _poll_output_files(self, runner, output_dir: str):
        """Refresh the files panel with files created so far (streaming).

        Polls the current run's output folder while the subprocess is active;
        newly written files show up within one poll interval instead of only
        after the run completes. The panel refresh is skipped until the run
        folder is known, so files from older runs are never shown.
        """
        try:
            if not runner.is_running:
                return
            run_folder = runner._resolve_scan_dir(output_dir)
            if not run_folder:
                return
            files = runner._scan_output_files(run_folder)
            if files:
                self.show_files(files, run_folder)
        except Exception:
            # Page may have been closed or navigated away mid-run; stop
            # polling instead of crashing the run handler.
            self._stop_file_streaming()

    async def run(
        self,
        runner,
        tool_name: str,
        constructor_params: dict,
        method_name: str,
        method_params: Optional[dict] = None,
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Run a tool through the UI runner and always surface errors in the log.

        While the run is active, the output folder is polled so files appear
        in the panel as soon as they are created (streaming).

        Any exception is written into the execution log (instead of silently
        failing the handler and leaving an empty log with a stuck Run button).
        """
        try:
            if self.page_progress is not None:
                # Method-level flags (visualization toggles, report/summary
                # switches, export options) decide which steps a run has, so
                # the step checklist is built from the merged parameter set.
                context = dict(constructor_params or {})
                context.update(method_params or {})
                self.page_progress.start(
                    tool_name,
                    method_name=method_name,
                    context=context,
                )
                self.page_progress.container.set_visibility(True)
            # Stream output files during the run: poll every 1.5s. Started
            # even without a caller-provided dir: the run folder is resolved
            # from the backend's own output-folder marker in that case.
            self._stop_file_streaming()
            self._poll_timer = ui.timer(
                1.5, lambda: self._poll_output_files(runner, output_dir)
            )
            result = await runner.run(
                tool_name,
                constructor_params,
                method_name,
                method_params=method_params,
                log_callback=self.log,
                progress_callback=self._runner_progress,
                output_dir=output_dir,
            )
            self._show_result_previews(result, tool_name)
            return result
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.log(
                f"[DROCAT] Unexpected UI error: {type(exc).__name__}: {exc}", "error"
            )
            self.log(traceback.format_exc().rstrip(), "error")
            return {"returncode": -1, "files": [], "duration": 0, "cancelled": False}
        finally:
            self._stop_file_streaming()

    def _runner_progress(self, phase: str, label: str = "") -> None:
        """Forward generic subprocess lifecycle phases to the page tracker."""
        if self.page_progress is not None:
            self.page_progress.update_phase(phase, label)

    def _show_result_previews(self, result: dict, tool_name: str) -> None:
        """Render the tool's registered result-table previews after a run.

        Only successful runs with a real output folder replace the preview
        section; failed or cancelled runs leave it hidden. Rendering is
        guarded so a preview problem can never turn a successful run into
        an error result.
        """
        folder = result.get("output_folder")
        if (
            result.get("returncode") != 0
            or result.get("cancelled")
            or not folder
            or self.previews_container is None
        ):
            return
        try:
            rendered = add_result_previews(
                folder, self.previews_container, tool_name)
            if self.previews_section is not None:
                self.previews_section.set_visibility(bool(rendered))
        except Exception:  # noqa: BLE001 — the run result must survive this
            pass

    def show_files(self, files: List[dict], output_dir: Optional[str] = None):
        """Display output files mirroring the output folder structure.

        Files directly in the run folder are listed first; subfolders render
        as nested expansions (data_details/, images/, bodyId_visualization/,
        ...) exactly like the folder on disk. The panel may be refreshed
        repeatedly while a run is active (streaming); the rebuild preserves
        which folder expansions the user has open, so newly created files
        appear without collapsing anything.
        """
        self._files = files

        if not self.files_container:
            return

        # Remember which folder expansions are open so an in-place refresh
        # during streaming does not collapse them.
        expanded = {
            rel_path: expansion.value
            for rel_path, expansion in self._file_expansions.items()
        }

        self.files_container.clear()
        self._file_expansions = {}

        with self.files_container:
            if not files:
                ui.label("No output files generated.").classes("drocat-empty")
                return

            if output_dir:
                with ui.row().classes("items-center gap-2"):
                    ui.button(
                        "Open Output Folder",
                        icon="folder_open",
                        on_click=lambda: open_folder(output_dir),
                    ).props("flat dense color=primary")
                    ui.label(str(Path(output_dir))).classes("text-caption drocat-muted drocat-truncate")

            tree = _build_file_tree(files, output_dir)
            self._render_file_tree(tree, expanded)

    def _render_file_tree(self, tree: dict, expanded: Dict[str, bool], prefix: str = ""):
        """Render one level of the folder tree: files first, then subfolders."""
        with ui.element("div").classes("drocat-file-list"):
            for f in sorted(tree.get(_FILES_KEY, []), key=lambda x: x["name"].lower()):
                self._render_file_row(f)
        for subdir in sorted(k for k in tree if k != _FILES_KEY):
            rel_path = f"{prefix}/{subdir}" if prefix else subdir
            subtree = tree[subdir]
            with ui.expansion(
                f"{subdir}  ({_count_tree_files(subtree)})", icon="folder"
            ).classes("w-full drocat-expansion") as expansion:
                self._file_expansions[rel_path] = expansion
                self._render_file_tree(subtree, expanded, rel_path)
            if expanded.get(rel_path):
                expansion.value = True

    def _render_file_row(self, f: dict):
        """One clickable file row (icon + name + size + open button)."""
        with ui.row().classes(
            "drocat-file-row items-center gap-2"
        ).on("click", lambda path=f["path"]: open_file(path)):
            ui.icon("insert_drive_file").classes("drocat-file-icon")
            ui.label(f["name"]).classes("drocat-file-name flex-grow")
            ui.label(self._format_size(f.get("size", 0))).classes(
                "text-caption drocat-muted drocat-file-size"
            )
            ui.button(
                icon="open_in_new",
                on_click=lambda path=f["path"]: open_file(path),
            ).props("flat dense round").classes("drocat-file-open")

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable format."""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"

    def clear(self):
        """Clear the log and files."""
        self._last_is_progress = False
        self._last_progress_name = None
        self._stop_file_streaming()
        self._file_expansions = {}
        if self.log_area:
            self.log_area.clear()
        self._files = []
        if self.files_container:
            self.files_container.clear()
            with self.files_container:
                ui.label("No output files yet.").classes("drocat-empty")
        if self.previews_container is not None:
            self.previews_container.clear()
            with self.previews_container:
                ui.label("Available after a completed run.").classes(
                    "drocat-empty")
        if self.previews_section is not None:
            self.previews_section.set_visibility(False)
        if self.page_progress:
            self.page_progress.reset()
