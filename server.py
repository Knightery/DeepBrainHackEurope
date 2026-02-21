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


@app.get("/eva-api")
async def eva_api():
    return FileResponse(HOME_DIR / "eva-api.html", media_type="text/html")


@app.get("/demo")
async def demo():
    return FileResponse(HOME_DIR / "demo.html", media_type="text/html")


app.mount("/home/static", StaticFiles(directory=HOME_DIR), name="home-static")

# Mount Chainlit app at /app (run from project root: uvicorn server:app --host 0.0.0.0 --port 8000)
try:
    from chainlit.utils import mount_chainlit

    _app_path = Path(__file__).parent / "app.py"
    mount_chainlit(app=app, target=str(_app_path), path="/app")
except Exception as e:
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning("Chainlit mount failed: %s. App will not be available at /app.", e)
