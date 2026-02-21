"""
Combined server: serves the landing page at /home (and /)
and mounts the Chainlit app at /app.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="DeepBrain")

HOME_DIR = Path(__file__).parent / "home"


@app.get("/")
async def root():
    return RedirectResponse(url="/home")


@app.get("/home")
async def home():
    return FileResponse(HOME_DIR / "index.html", media_type="text/html")


app.mount("/home/static", StaticFiles(directory=HOME_DIR), name="home-static")

try:
    from chainlit.utils import mount_chainlit

    mount_chainlit(app=app, target="app.py", path="/app")
except Exception:
    pass
