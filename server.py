"""
Combined server:
- Landing page at /home (and /)
- Chainlit app mounted at /app
- Local API and dashboard for pitch history/rankings

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pitch_db import get_pitch, get_pitch_messages, init_db, list_pitches

app = FastAPI(title="OpenQuant")

ROOT_DIR = Path(__file__).parent
HOME_DIR = ROOT_DIR / "home"
PUBLIC_DIR = ROOT_DIR / "public"
THEME_PATH = PUBLIC_DIR / "theme.json"


def _load_theme() -> dict[str, Any]:
    if not THEME_PATH.exists():
        return {}
    try:
        return json.loads(THEME_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Theme JSON is invalid at {THEME_PATH}: {exc}") from exc


def _dashboard_html(theme: dict[str, Any]) -> str:
    dark = (
        theme.get("variables", {})
        .get("dark", {})
        if isinstance(theme.get("variables", {}), dict)
        else {}
    )
    background = f"hsl({dark.get('--background', '0 0% 7%')})"
    card = f"hsl({dark.get('--card', '0 0% 10%')})"
    foreground = f"hsl({dark.get('--foreground', '0 0% 95%')})"
    muted = f"hsl({dark.get('--muted-foreground', '0 0% 60%')})"
    primary = f"hsl({dark.get('--primary', '213 68% 55%')})"
    border = f"hsl({dark.get('--border', '0 0% 18%')})"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenQuant Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: {background};
      --card: {card};
      --fg: {foreground};
      --muted: {muted};
      --primary: {primary};
      --border: {border};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 24px;
      background: var(--bg); color: var(--fg);
      font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    h1 {{ margin: 0 0 16px; font-size: 24px; }}
    .layout {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr:hover {{ background: rgba(255,255,255,0.03); cursor: pointer; }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-block; border: 1px solid var(--border); border-radius: 999px;
      padding: 2px 8px; font-size: 12px;
    }}
    .score {{ color: var(--primary); font-weight: 600; }}
    .messages {{ max-height: 70vh; overflow: auto; display: grid; gap: 8px; }}
    .msg {{ border: 1px solid var(--border); border-radius: 8px; padding: 8px; }}
    .msg-meta {{ font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
    a {{ color: var(--primary); text-decoration: none; }}
    .toolbar {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
    .btn {{
      background: transparent; color: var(--fg); border: 1px solid var(--border);
      padding: 6px 10px; border-radius: 8px; cursor: pointer;
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <h1>OpenQuant Pitch Dashboard</h1>
    <div>
      <a href="/home" class="btn">Landing</a>
      <a href="/app" class="btn">Open App</a>
    </div>
  </div>
  <div class="layout">
    <div class="card">
      <div class="muted" style="margin-bottom:10px;">Completed pitches ranked by score</div>
      <table id="pitchTable">
        <thead>
          <tr>
            <th>#</th><th>Pitch</th><th>Status</th><th>Score</th><th>Decision</th><th>Msgs</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <div id="detailTitle" class="muted" style="margin-bottom:10px;">Select a pitch to view details and chat history</div>
      <div id="detailMeta" class="muted" style="margin-bottom:10px;"></div>
      <div class="messages" id="messages"></div>
    </div>
  </div>

  <script>
    const tableBody = document.querySelector("#pitchTable tbody");
    const messages = document.getElementById("messages");
    const detailTitle = document.getElementById("detailTitle");
    const detailMeta = document.getElementById("detailMeta");

    async function loadPitches() {{
      const res = await fetch("/api/pitches?completed_only=true&limit=300");
      const data = await res.json();
      tableBody.innerHTML = "";
      for (const row of data.items) {{
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${{row.rank ?? "-"}}</td>
          <td><code>${{row.pitch_id}}</code></td>
          <td><span class="pill">${{row.status}}</span></td>
          <td class="score">${{row.overall_score ?? "-"}}</td>
          <td>${{row.decision ?? "-"}}</td>
          <td>${{row.message_count}}</td>
        `;
        tr.addEventListener("click", () => loadPitch(row.pitch_id));
        tableBody.appendChild(tr);
      }}
    }}

    async function loadPitch(pitchId) {{
      const [pitchRes, msgRes] = await Promise.all([
        fetch(`/api/pitches/${{pitchId}}`),
        fetch(`/api/pitches/${{pitchId}}/messages?limit=1000`)
      ]);
      if (!pitchRes.ok) return;
      const pitch = await pitchRes.json();
      const msgData = await msgRes.json();

      detailTitle.innerHTML = `Pitch <code>${{pitch.pitch_id}}</code>`;
      detailMeta.textContent =
        `status=${{pitch.status}} | score=${{pitch.overall_score ?? "-"}} | decision=${{pitch.decision ?? "-"}} | completed_at=${{pitch.completed_at ?? "-"}}`;
      messages.innerHTML = "";

      for (const m of msgData.items) {{
        const div = document.createElement("div");
        div.className = "msg";
        div.innerHTML = `
          <div class="msg-meta">${{m.timestamp_utc}} · ${{m.role}}</div>
          <div>${{(m.content || "").replaceAll("<", "&lt;")}}</div>
        `;
        messages.appendChild(div);
      }}
    }}

    loadPitches();
  </script>
</body>
</html>"""


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/home")


@app.get("/home")
async def home() -> FileResponse:
    return FileResponse(HOME_DIR / "index.html", media_type="text/html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_dashboard_html(_load_theme()))


@app.get("/api/theme")
async def theme() -> dict[str, Any]:
    return _load_theme()


@app.get("/api/pitches")
async def api_pitches(
    status: str | None = None,
    completed_only: bool = False,
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    items = list_pitches(limit=limit, status=status, completed_only=completed_only)
    ranked: list[dict[str, Any]] = []
    rank = 0
    for item in items:
        if item.get("overall_score") is not None:
            rank += 1
            item["rank"] = rank
        else:
            item["rank"] = None
        ranked.append(item)
    return {"items": ranked, "count": len(ranked)}


@app.get("/api/pitches/{pitch_id}")
async def api_pitch(pitch_id: str) -> dict[str, Any]:
    item = get_pitch(pitch_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Pitch not found")
    return item


@app.get("/api/pitches/{pitch_id}/messages")
async def api_pitch_messages(
    pitch_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
) -> dict[str, Any]:
    return {"items": get_pitch_messages(pitch_id, limit=limit)}


app.mount("/home/static", StaticFiles(directory=HOME_DIR), name="home-static")
if PUBLIC_DIR.exists():
    app.mount("/public", StaticFiles(directory=PUBLIC_DIR), name="public-static")

# Mount Chainlit app at /app (run from project root: uvicorn server:app --host 0.0.0.0 --port 8000)
try:
    from chainlit.utils import mount_chainlit

    mount_chainlit(app=app, target="app.py", path="/app")
except Exception as exc:
    raise RuntimeError(
        f"Failed to mount Chainlit app at /app: {exc.__class__.__name__}: {exc}"
    ) from exc
