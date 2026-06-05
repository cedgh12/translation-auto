import os
import threading
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent / ".env")

from trad_pipeline_openai import (
    LOKALISE_PROJECT_ID,
    OUTPUT_DIR,
    JobCancelledError,
    build_languages_progress,
    fetch_project_languages,
    run_pipeline,
)

BASE_DIR = Path(__file__).parent
INDEX_HTML = BASE_DIR / "index.html"

app = FastAPI(title="MyVoltalis Translation API")

from fastapi.middleware.cors import CORSMiddleware

from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException, status
from fastapi import FastAPI, Depends
import secrets

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("API_KEY")
    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clé API invalide ou manquante"
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_job_lock = threading.Lock()
_job_thread: threading.Thread | None = None
_cancel_all = threading.Event()
_skipped_langs: set[str] = set()

_IDLE_STATUS: dict = {
    "status": "idle",
    "source_lang": None,
    "target_langs": [],
    "filter_lang": None,
    "total_keys": 0,
    "processed_keys": 0,
    "current_lang": None,
    "current_lang_name": None,
    "completed_langs": [],
    "languages_progress": [],
    "error": None,
}

_job_status: dict = dict(_IDLE_STATUS)


class TranslateRequest(BaseModel):
    source_lang: str = "fr"
    target_langs: list[str] = Field(..., min_length=1)
    filter_lang: str | None = None


class TranslateResponse(BaseModel):
    message: str
    status: Literal["started"]


class CancelResponse(BaseModel):
    message: str
    status: Literal["cancelling"]


class SkipResponse(BaseModel):
    message: str
    lang_code: str
    lang_status: Literal["skipped"]


class LangProgress(BaseModel):
    code: str
    name: str
    status: Literal["pending", "translating", "completed", "skipped"]
    processed_keys: int = 0
    total_keys: int = 0


class JobStatus(BaseModel):
    status: Literal["idle", "fetching", "translating", "cancelling", "completed", "error"]
    source_lang: str | None = None
    target_langs: list[str] = []
    filter_lang: str | None = None
    total_keys: int = 0
    processed_keys: int = 0
    current_lang: str | None = None
    current_lang_name: str | None = None
    completed_langs: list[str] = []
    languages_progress: list[LangProgress] = []
    error: str | None = None


def _reset_to_idle(**keep: object) -> None:
    with _job_lock:
        _job_status.clear()
        _job_status.update(_IDLE_STATUS)
        _job_status.update(keep)


def _update_status(**kwargs) -> None:
    with _job_lock:
        _job_status.update(kwargs)


def _get_status_snapshot() -> dict:
    with _job_lock:
        return dict(_job_status)


def _is_job_running() -> bool:
    with _job_lock:
        return _job_status["status"] in ("fetching", "translating", "cancelling")


def _is_cancelled() -> bool:
    return _cancel_all.is_set()


def _is_lang_skipped(lang_code: str) -> bool:
    with _job_lock:
        return lang_code in _skipped_langs


def _skip_language(lang_code: str) -> None:
    with _job_lock:
        _skipped_langs.add(lang_code)
        for entry in _job_status.get("languages_progress", []):
            if entry["code"] == lang_code and entry["status"] in ("pending", "translating"):
                entry["status"] = "skipped"
                entry["processed_keys"] = 0


def _finalize_progress_after_global_cancel(progress: list[dict]) -> list[dict]:
    """Marque les langues non terminées en skipped ; conserve les completed."""
    result = []
    for entry in progress:
        updated = dict(entry)
        if updated["status"] in ("pending", "translating"):
            updated["status"] = "skipped"
            updated["processed_keys"] = 0
        result.append(updated)
    return result


def _idle_after_job(snapshot: dict) -> None:
    """Repasse à idle en conservant le résumé (JSON déjà écrits sur disque)."""
    _reset_to_idle(
        completed_langs=snapshot.get("completed_langs", []),
        languages_progress=snapshot.get("languages_progress", []),
        total_keys=snapshot.get("total_keys", 0),
        processed_keys=snapshot.get("processed_keys", 0),
    )


def _run_translation_job(
    source_lang: str,
    target_langs: list[str],
    filter_lang: str | None,
    lokalise_token: str,
    openai_key: str,
    lang_names: dict[str, str],
) -> None:
    global _job_thread

    try:
        _update_status(
            status="fetching",
            source_lang=source_lang,
            target_langs=target_langs,
            filter_lang=filter_lang,
            total_keys=0,
            processed_keys=0,
            current_lang=None,
            current_lang_name=None,
            completed_langs=[],
            error=None,
            languages_progress=build_languages_progress(
                target_langs, lang_names, 0, skipped=list(_skipped_langs)
            ),
        )

        def on_status(update: dict) -> None:
            if not _cancel_all.is_set():
                _update_status(**update)

        run_pipeline(
            project_id=LOKALISE_PROJECT_ID,
            lokalise_token=lokalise_token,
            openai_api_key=openai_key,
            source_lang=source_lang,
            target_langs=target_langs,
            filter_lang=filter_lang,
            lang_names=lang_names,
            on_status=on_status,
            is_cancelled=_is_cancelled,
            is_lang_skipped=_is_lang_skipped,
        )

        if not _cancel_all.is_set():
            _idle_after_job(_get_status_snapshot())

    except JobCancelledError:
        snapshot = _get_status_snapshot()
        progress = _finalize_progress_after_global_cancel(
            snapshot.get("languages_progress", [])
        )
        _idle_after_job({**snapshot, "languages_progress": progress})

    except Exception as e:
        _update_status(status="error", error=str(e))
    finally:
        _cancel_all.clear()
        with _job_lock:
            _skipped_langs.clear()
            _job_thread = None


@app.get("/")
def index():
    """Interface web de traduction."""
    if not INDEX_HTML.is_file():
        raise HTTPException(status_code=404, detail="index.html introuvable.")
    return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")


@app.get("/download/{lang_code}", dependencies=[Depends(verify_api_key)])
def download_translation(lang_code: str):
    """Télécharge le JSON traduit pour une langue."""
    path = OUTPUT_DIR / f"{lang_code}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Aucun fichier pour la langue '{lang_code}'.")
    return FileResponse(
        path,
        media_type="application/json",
        filename=f"{lang_code}.json",
    )


@app.get("/languages")
def get_languages():
    """Liste les langues configurées dans le projet Lokalise."""
    token = os.getenv("LOKALISE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=500, detail="LOKALISE_API_TOKEN non définie.")

    try:
        languages = fetch_project_languages(LOKALISE_PROJECT_ID, token)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur Lokalise : {e}")

    return {"languages": languages}


@app.get("/status", response_model=JobStatus, dependencies=[Depends(verify_api_key)])
def get_status():
    """Retourne la progression du job. status=idle quand aucun job n'est actif."""
    return _get_status_snapshot()


@app.post("/translate", response_model=TranslateResponse, status_code=202, dependencies=[Depends(verify_api_key)])
def start_translation(body: TranslateRequest):
    """Lance le pipeline de traduction en arrière-plan."""
    global _job_thread

    if _is_job_running():
        raise HTTPException(status_code=409, detail="Une traduction est déjà en cours.")

    lokalise_token = os.getenv("LOKALISE_API_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not lokalise_token:
        raise HTTPException(status_code=500, detail="LOKALISE_API_TOKEN non définie.")
    if not openai_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY non définie.")

    try:
        languages = fetch_project_languages(LOKALISE_PROJECT_ID, lokalise_token)
        lang_names = {lang["code"]: lang["name"] for lang in languages}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur Lokalise : {e}")

    _cancel_all.clear()
    with _job_lock:
        _skipped_langs.clear()
    _reset_to_idle()

    _job_thread = threading.Thread(
        target=_run_translation_job,
        args=(
            body.source_lang,
            body.target_langs,
            body.filter_lang,
            lokalise_token,
            openai_key,
            lang_names,
        ),
        daemon=True,
    )
    _job_thread.start()

    return TranslateResponse(message="Traduction démarrée.", status="started")


@app.post("/cancel", response_model=CancelResponse, dependencies=[Depends(verify_api_key)])
def cancel_all():
    """Arrête tout le job. Les JSON déjà générés (langues completed) restent sur disque."""
    if not _is_job_running():
        raise HTTPException(status_code=409, detail="Aucune traduction en cours à annuler.")

    _cancel_all.set()
    _update_status(status="cancelling", current_lang=None, current_lang_name=None)

    return CancelResponse(
        message="Annulation en cours. Les traductions déjà terminées sont conservées.",
        status="cancelling",
    )


@app.post("/skip/{lang_code}", response_model=SkipResponse, dependencies=[Depends(verify_api_key)])
def skip_language(lang_code: str):
    """Passe une langue (interrompt si en cours) et continue sur la suivante."""
    if not _is_job_running():
        raise HTTPException(status_code=409, detail="Aucune traduction en cours.")

    with _job_lock:
        target_langs = _job_status.get("target_langs", [])
        progress = _job_status.get("languages_progress", [])

    if lang_code not in target_langs:
        raise HTTPException(
            status_code=404,
            detail=f"La langue '{lang_code}' ne fait pas partie du job en cours.",
        )

    entry = next((p for p in progress if p["code"] == lang_code), None)
    if entry and entry["status"] == "completed":
        raise HTTPException(
            status_code=409,
            detail=f"La langue '{lang_code}' est déjà terminée.",
        )
    if entry and entry["status"] == "skipped":
        return SkipResponse(
            message=f"La langue '{lang_code}' était déjà passée.",
            lang_code=lang_code,
            lang_status="skipped",
        )

    _skip_language(lang_code)

    return SkipResponse(
        message=f"Langue '{lang_code}' passée — passage à la suivante.",
        lang_code=lang_code,
        lang_status="skipped",
    )
