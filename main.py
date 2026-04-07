from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.config.settings import settings


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
INDEX_FILE = FRONTEND_DIR / "index.html"

app = FastAPI(title=settings.app_name)

app.include_router(router)

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/app", include_in_schema=False)
async def serve_app():
    return FileResponse(str(INDEX_FILE))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)