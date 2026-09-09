"""Persistent in-page banner stack (download info and similar notices).

The old persistent banners were Quasar notifications (``ui.notify`` with
``timeout=0`` and a "Read" close label): each one stacked into Quasar's
notification container and they piled up after a few downloads.  This
module replaces them with a fixed in-page stack:

- one unified banner size and orange theming for light/dark (CSS lives
  in ``ui/app.py`` next to the other ``drocat-`` styles),
- a corner ``×`` button that dismisses a single banner (no "Read" or
  "OK" label buttons),
- when several banners are alive at once, only the newest stays fully
  visible and the older ones fold into a slim "N earlier notices" row
  that expands on click.

The stack is created once per page (``create_banner_stack`` in the page
builder); call sites use :func:`push_banner` from event handlers, which
routes to the current client's stack and falls back to a plain
``ui.notify`` when no stack exists (other pages, bare test clients).
"""

from __future__ import annotations

import logging
from typing import Dict, List

from nicegui import context, ui

logger = logging.getLogger(__name__)

_ENTRY_CAP = 20


class BannerStack:
    """Fixed-position stack of persistent banners for one client."""

    def __init__(self) -> None:
        self._entries: List[Dict] = []  # oldest first: {message, icon}
        self._expanded = False
        self._body = None

    def build(self) -> None:
        """Create the (initially empty) stack container on the page."""
        container = ui.element("div").classes("drocat-banner-stack")
        with container:
            self._body = ui.column().classes("drocat-banner-stack-body")

    def push(self, message: str, icon: str = "download_done") -> None:
        """Append a banner; it becomes the newest (fully visible) one."""
        self._entries.append({"message": str(message), "icon": icon})
        del self._entries[:-_ENTRY_CAP]
        # A new arrival re-folds the stack: the newest stays readable and
        # everything older hides behind the fold row again.
        self._expanded = False
        self._render()

    def dismiss(self, entry: Dict) -> None:
        """Remove one banner (identity-matched) and re-render."""
        for index, candidate in enumerate(self._entries):
            if candidate is entry:
                del self._entries[index]
                break
        self._render()

    def _toggle_fold(self, _event=None) -> None:
        self._expanded = not self._expanded
        self._render()

    def _render(self) -> None:
        if self._body is None:
            return
        self._body.clear()
        if not self._entries:
            return
        with self._body:
            if len(self._entries) > 1:
                with ui.element("div").classes(
                        "drocat-banner-fold").on("click", self._toggle_fold):
                    with ui.row().classes("items-center gap-1"):
                        ui.icon("expand_less" if self._expanded
                                else "expand_more").classes("drocat-banner-fold-icon")
                        ui.label(
                            f"{len(self._entries) - 1} earlier "
                            f"notice{'s' if len(self._entries) > 2 else ''}"
                        ).classes("drocat-banner-fold-label")
                if self._expanded:
                    with ui.column().classes("drocat-banner-folded-list w-full gap-2"):
                        for entry in self._entries[:-1]:
                            self._render_banner(entry)
            self._render_banner(self._entries[-1])

    def _render_banner(self, entry: Dict) -> None:
        with ui.element("div").classes("drocat-banner"):
            ui.icon(entry["icon"]).classes("drocat-banner-icon")
            ui.label(entry["message"]).classes("drocat-banner-text")
            ui.button(icon="close", on_click=lambda *_: self.dismiss(entry)) \
                .props("flat round dense").classes("drocat-banner-close") \
                .tooltip("Dismiss")


_STACKS: Dict[str, BannerStack] = {}


def create_banner_stack() -> None:
    """Create the current client's banner stack (once per page build)."""
    client = context.client
    if client.id in _STACKS:
        return
    stack = BannerStack()
    stack.build()
    _STACKS[client.id] = stack
    client.on_disconnect(lambda _c: _STACKS.pop(_c.id, None))


def push_banner(message: str, *, icon: str = "download_done") -> None:
    """Push a persistent banner onto the current client's stack.

    The stack is created on the page build, but its registration dies
    with the client's first websocket disconnect — and a socket drop
    that reconnects within NiceGUI's window keeps the SAME client (and
    its DOM) alive without re-running the page builder.  Every download
    after such a drop then fell back to a bare ``ui.notify`` and the old
    pile of un-themed notifications returned.  So when the client
    context is alive but has no stack anymore, rebuild it lazily: the
    stack container is ``position: fixed``, so attaching it under the
    client layout mid-session renders exactly like the page-build one.
    Falls back to a plain persistent ``ui.notify`` with an ``×`` close
    button only when there is no live client context at all (background
    threads, bare test clients).
    """
    try:
        client = context.client
        stack = _STACKS.get(client.id)
        if stack is None:
            with client.layout:
                create_banner_stack()
            stack = _STACKS.get(client.id)
    except RuntimeError:
        stack = None
    except Exception:
        # A pruned/reconnecting client cannot accept new elements; the
        # fallback still tells the user where the download went.
        logger.debug("lazy banner stack rebuild failed", exc_info=True)
        stack = None
    if stack is not None:
        stack.push(message, icon=icon)
        return
    logger.debug("no banner stack in context; falling back to ui.notify")
    ui.notify(message, type="info", multi_line=True, timeout=0,
              close_button="✕")
