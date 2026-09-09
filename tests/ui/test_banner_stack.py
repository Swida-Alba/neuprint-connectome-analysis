"""Tests for the persistent in-page banner stack (ui/components/banner.py).

Covers the fold behaviour (older banners hide behind a "N earlier
notices" row while the newest stays visible), re-folding on new
arrivals, corner-× dismissal via the rendered close button, the entry
cap, the lazy self-heal when a reconnecting client lost its stack, and
the plain ui.notify fallback when no client context exists at all.
"""

import sys
from pathlib import Path

import pytest
from nicegui import Client, ui
from nicegui.page import page

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.components import banner as banner_module
from ui.components.banner import BannerStack, create_banner_stack, push_banner


def _labels(element):
    return [e.text for e in element.descendants() if isinstance(e, ui.label)]


def _banner_cards(body):
    return [e for e in body.descendants()
            if "drocat-banner" in e._classes
            and "drocat-banner-fold" not in e._classes]


def _fold_row(body):
    rows = [c for c in body.default_slot.children
            if "drocat-banner-fold" in c._classes]
    return rows[0] if rows else None


def _fire_click(element):
    """Invoke the Python click handlers the element was built with."""
    for listener in list(element._event_listeners.values()):
        if listener.type == "click" and listener.handler is not None:
            listener.handler({"sender": element.id, "args": None})


@pytest.fixture
def stack_client():
    client = Client(page("/banner-stack-test"))
    with client:
        create_banner_stack()
    yield client
    banner_module._STACKS.pop(client.id, None)


def test_first_banner_shows_without_fold_row(stack_client):
    stack = banner_module._STACKS[stack_client.id]
    with stack_client:
        push_banner("first download info")
    assert len(_banner_cards(stack._body)) == 1
    assert _fold_row(stack._body) is None
    assert _labels(stack._body) == ["first download info"]


def test_multiple_banners_fold_behind_a_count_row(stack_client):
    stack = banner_module._STACKS[stack_client.id]
    with stack_client:
        push_banner("first download info")
        push_banner("second download info")
        push_banner("third download info")
    body = stack._body
    # Only the newest stays fully visible; the two older ones are folded.
    cards = _banner_cards(body)
    assert len(cards) == 1
    assert _labels(cards[0]) == ["third download info"]
    assert _labels(_fold_row(body)) == ["2 earlier notices"]


def test_expanding_the_fold_reveals_older_banners(stack_client):
    stack = banner_module._STACKS[stack_client.id]
    with stack_client:
        push_banner("first download info")
        push_banner("second download info")
        push_banner("third download info")
        _fire_click(_fold_row(stack._body))
    cards = _banner_cards(stack._body)
    assert len(cards) == 3
    assert _labels(_fold_row(stack._body)) == ["2 earlier notices"]


def test_new_arrival_refolds_the_stack(stack_client):
    stack = banner_module._STACKS[stack_client.id]
    with stack_client:
        push_banner("first download info")
        push_banner("second download info")
        _fire_click(_fold_row(stack._body))
        assert len(_banner_cards(stack._body)) == 2  # expanded
        push_banner("third download info")
    # The arrival collapses everything but the newest again.
    assert len(_banner_cards(stack._body)) == 1
    assert _labels(_fold_row(stack._body)) == ["2 earlier notices"]


def test_corner_close_button_dismisses_a_banner(stack_client):
    stack = banner_module._STACKS[stack_client.id]
    with stack_client:
        push_banner("first download info")
        push_banner("second download info")
        newest = _banner_cards(stack._body)[0]
        close_buttons = [e for e in newest.descendants()
                         if isinstance(e, ui.button)]
        assert len(close_buttons) == 1
        _fire_click(close_buttons[0])
    assert len(stack._entries) == 1
    assert stack._entries[0]["message"] == "first download info"
    # Down to a single banner: the fold row disappears with it.
    assert len(_banner_cards(stack._body)) == 1
    assert _fold_row(stack._body) is None


def test_entry_cap_keeps_only_the_newest_banners(stack_client):
    stack = banner_module._STACKS[stack_client.id]
    with stack_client:
        for index in range(banner_module._ENTRY_CAP + 3):
            push_banner(f"download info {index}")
    assert len(stack._entries) == banner_module._ENTRY_CAP
    assert stack._entries[0]["message"] == "download info 3"


def test_push_banner_self_heals_a_stack_lost_to_disconnect(
        stack_client, monkeypatch):
    """A websocket drop unregisters the stack while the browser keeps the
    same client alive; the next push must lazily rebuild the stack (the
    unified orange banner) instead of falling back to a bare
    notification."""
    calls = []
    monkeypatch.setattr(
        banner_module.ui, "notify",
        lambda *a, **kw: calls.append((a, kw)))
    stack = banner_module._STACKS[stack_client.id]
    # simulate the disconnect cleanup of a reconnecting client
    banner_module._STACKS.pop(stack_client.id, None)
    with stack_client:
        push_banner("download after reconnect")
    healed = banner_module._STACKS.get(stack_client.id)
    assert healed is not None and healed is not stack
    assert calls == []
    assert _labels(healed._body) == ["download after reconnect"]


def test_self_heal_reuses_an_existing_stack(stack_client):
    """create_banner_stack is idempotent per client: a second call (the
    lazy self-heal racing a healthy stack) must not reset the stack."""
    stack = banner_module._STACKS[stack_client.id]
    with stack_client:
        push_banner("first download info")
        create_banner_stack()
        push_banner("second download info")
    assert banner_module._STACKS[stack_client.id] is stack
    assert stack._entries == [
        {"message": "first download info", "icon": "download_done"},
        {"message": "second download info", "icon": "download_done"},
    ]


def test_push_banner_falls_back_to_notify_without_a_client(monkeypatch):
    calls = []
    monkeypatch.setattr(
        banner_module.ui, "notify",
        lambda *a, **kw: calls.append((a, kw)))

    # No live client context at all (background thread, bare test):
    # the fallback must carry the message with an × close button
    # instead of the old "Read" label.
    class _NoClient:
        @property
        def client(self):
            raise RuntimeError("no client context")

    monkeypatch.setattr(banner_module, "context", _NoClient())
    push_banner("fallback download info", icon="warning")

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "fallback download info"
    assert kwargs["close_button"] == "✕"
    assert kwargs["timeout"] == 0


def test_banner_stack_without_a_page_build_stays_safe():
    stack = BannerStack()
    # No build(): pushes accumulate but must not try to render.
    stack.push("offline download info")
    assert len(stack._entries) == 1
