"""Streaming overrides reuse the shared engine cache and evict on switches."""
from __future__ import annotations

import asyncio
import os
import sys
from types import ModuleType
from pathlib import Path

import pytest

os.environ.setdefault("OMNIVOICE_MODEL", "test")
os.environ.setdefault("OMNIVOICE_DISABLE_FILE_LOG", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


@pytest.fixture()
def engines(monkeypatch):
    from services import engine_memory, tts_backend

    class First:
        id = "first-test"
        created = 0
        unloaded = 0

        def __init__(self):
            type(self).created += 1

        def unload(self):
            type(self).unloaded += 1

    class Second(First):
        id = "second-test"
        created = 0
        unloaded = 0

    monkeypatch.setitem(tts_backend._REGISTRY, First.id, First)
    monkeypatch.setitem(tts_backend._REGISTRY, Second.id, Second)
    monkeypatch.setattr(tts_backend, "_ENGINE_INSTANCES", {})
    fake_router = ModuleType("api.routers.engines")
    fake_router._ENGINE_INSTANCES = tts_backend._ENGINE_INSTANCES
    monkeypatch.setitem(sys.modules, "api.routers.engines", fake_router)
    monkeypatch.setattr(engine_memory, "single_engine_resident", lambda: True)
    monkeypatch.setattr("services.model_manager.unload_shared_model", lambda: False)
    yield First, Second, tts_backend._ENGINE_INSTANCES


def test_stream_overrides_reuse_and_switch(engines):
    from api.routers.tts_stream import _resolve_stream_backend

    first_cls, second_cls, cache = engines

    async def run():
        first = await _resolve_stream_backend("first-test")
        again = await _resolve_stream_backend("first-test")
        assert first is again
        assert first_cls.created == 1
        assert first_cls.unloaded == 0

        second = await _resolve_stream_backend("second-test")
        assert second_cls.created == 1
        assert first_cls.unloaded == 1
        assert first_cls not in cache
        assert cache[second_cls] is second

        back = await _resolve_stream_backend("first-test")
        assert back is not first
        assert first_cls.created == 2
        assert second_cls.unloaded == 1
        assert cache[first_cls] is back
        assert second_cls not in cache

    asyncio.run(run())


def test_stream_without_override_keeps_active_backend_path(engines, monkeypatch):
    from api.routers.tts_stream import _resolve_stream_backend
    from services import tts_backend

    first_cls, _, cache = engines
    active = object()
    monkeypatch.setattr(tts_backend, "active_backend_id", lambda: "first-test")
    monkeypatch.setattr(tts_backend, "get_active_tts_backend", lambda: active)

    assert asyncio.run(_resolve_stream_backend(None)) is active
    assert first_cls.created == 0  # no explicit override instance was made
    assert not cache
