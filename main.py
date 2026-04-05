from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import FastAPI

from src.api.routes import router
from src.config.settings import settings

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"

app = FastAPI(title=settings.app_name)
app.include_router(router)

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/app")
async def serve_app():
    return FileResponse(str(INDEX_FILE))


@app.get("//app")
async def serve_double_slash_app():
    return RedirectResponse(url="/app")