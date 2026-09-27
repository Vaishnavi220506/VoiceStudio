"""MCP describe_voice / design_voice: agents can design a voice profile.

The tools wrap ``POST /design/describe`` and ``POST /profiles`` (kind=design).
A stub app stands in for the backend: the real describe router (pure CPU, no
model) plus a ``/profiles`` endpoint that records the form it receives.
"""
import asyncio
import json
import os
import sys

os.environ.setdefault("OMNIVOICE_MODEL", "test")
os.environ.setdefault("OMNIVOICE_DISABLE_FILE_LOG", "1")

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

pytest.importorskip("mcp")


def _stub_app(received: list):
    from fastapi import FastAPI, Form

    from api.routers.describe_voice import router as describe_router

    app = FastAPI()
    app.include_router(describe_router)

    @app.post("/profiles")
    async def create_profile(
        name: str = Form(...),
        kind: str = Form("clone"),
        vd_states: str = Form(""),
        instruct: str = Form(""),
        language: str = Form("Auto"),
        personality: str = Form(""),
    ):
        received.append({
            "name": name, "kind": kind, "vd_states": vd_states,
            "instruct": instruct, "language": language, "personality": personality,
        })
        return {"id": "abcd1234", "name": name, "kind": kind}

    return app


def _call(mcp, tool: str, args: dict) -> dict:
    result = asyncio.run(mcp.call_tool(tool, args))
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.delenv("OMNIVOICE_API_URL", raising=False)
    from mcp_server import create_mcp_server

    received: list = []
    return create_mcp_server(app=_stub_app(received)), received


def test_design_tools_are_registered(server):
    mcp, _ = server
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert {"describe_voice", "design_voice"} <= names


def test_describe_voice_previews_without_saving(server):
    mcp, received = server
    out = _call(mcp, "describe_voice", {"description": "an old man with a deep voice"})
    assert out["attrs"]["Gender"] == "male"
    assert out["attrs"]["Age"] == "elderly"
    assert received == []


def test_design_voice_saves_a_design_profile(server):
    mcp, received = server
    out = _call(mcp, "design_voice", {
        "name": "Innkeeper",
        "description": "an elderly woman, british accent, gravelly",
    })
    assert out["profile_id"] == "abcd1234"
    assert out["kind"] == "design"
    assert "gravelly" in " ".join(out["unmatched"])

    (form,) = received
    assert form["kind"] == "design"
    states = json.loads(form["vd_states"])
    assert states["Gender"] == "female"
    assert states["EnglishAccent"] == "british accent"
    assert "female" in form["instruct"]


def test_design_voice_refuses_a_description_with_no_attributes(server):
    mcp, received = server
    out = _call(mcp, "design_voice", {"name": "Nobody", "description": "gravelly"})
    assert "error" in out
    assert out["unmatched"] == ["gravelly"]
    assert received == []
