"""Retrieval backend selection (Sude).

The regression this file exists for: `get_backend()` used to choose Track A on
importability alone. `RetrievalService` imports and constructs fine with no index
on disk, so on a fresh clone it was selected, and then every query raised. On a
machine with a built index the same code worked - which is why the failure was
invisible to the developer who had the index.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research_assistant.config_J import get_settings
from research_assistant.mcp_server import backend_S
from research_assistant.mcp_server.backend_S import StubRetrieval, get_backend, set_backend


class Serving:
    """A Track A service that binds and answers."""

    probes = 0

    def _bind(self):
        return self

    def retrieve(self, request):
        type(self).probes += 1
        return SimpleNamespace(results=[])

    def get_chunk(self, chunk_id):
        return None


class ImportsButCannotAnswer:
    """Constructs and binds, then fails on the first real query - exactly the
    behaviour of `RetrievalService` with no BM25 index built."""

    def _bind(self):
        return self

    def retrieve(self, request):
        raise FileNotFoundError("BM25 index not found; build it first")


class CannotImport:
    def _bind(self):
        raise ImportError("No module named 'chromadb'")


class MustNotBeConstructed:
    def __init__(self):
        raise AssertionError("Track A was touched although the stub was pinned")


@pytest.fixture
def select(monkeypatch):
    """Run `get_backend()` with a chosen Track A stand-in and backend setting."""

    def _select(track_a, setting: str = "auto"):
        monkeypatch.setenv("RA_RETRIEVAL_BACKEND", setting)
        get_settings.cache_clear()
        set_backend(None)
        monkeypatch.setattr(backend_S, "TrackARetrieval", track_a)
        return get_backend()

    yield _select
    set_backend(None)
    get_settings.cache_clear()  # the session pin is re-read from the environment


def test_a_service_that_answers_is_selected(select):
    assert isinstance(select(Serving), Serving)


def test_a_service_that_imports_but_cannot_answer_falls_back_to_the_stub(select):
    """The regression. Importing is not serving."""
    assert isinstance(select(ImportsButCannotAnswer), StubRetrieval)


def test_a_service_that_cannot_be_imported_falls_back_to_the_stub(select):
    assert isinstance(select(CannotImport), StubRetrieval)


def test_selection_probes_exactly_once_and_caches(select):
    Serving.probes = 0
    first = select(Serving)
    assert get_backend() is first
    assert get_backend() is first
    assert Serving.probes == 1


def test_pinning_the_stub_never_touches_track_a(select):
    """What the test suite relies on: a pinned stub must not even construct the
    real service, or a machine with an index could leak into a test."""
    assert isinstance(select(MustNotBeConstructed, setting="stub"), StubRetrieval)


def test_requiring_track_a_refuses_to_fall_back(select):
    """A deployment that must not silently serve the stub says so, and gets an
    error rather than a degraded service that looks healthy."""
    with pytest.raises(FileNotFoundError):
        select(ImportsButCannotAnswer, setting="track_a")


def test_the_test_session_is_pinned_to_the_stub():
    """conftest pins RA_RETRIEVAL_BACKEND, so this suite means the same thing on a
    laptop with a built index as on a fresh clone."""
    get_settings.cache_clear()
    assert get_settings().retrieval_backend == "stub"
    set_backend(None)
    assert isinstance(get_backend(), StubRetrieval)
