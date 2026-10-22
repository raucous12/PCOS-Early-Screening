"""FastAPI web application for PCOS early screening.

Run:  python -m app.main [--config config/default.yaml] [--model models/pcos_screen_model.joblib]
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pcos_screen import __version__
from pcos_screen.config import Config, ConfigError, load_config
from pcos_screen.predict import ModelNotAvailable, PCOSPredictor
from pcos_screen.screening import screen
from pcos_screen.storage import ScreeningStore

from .schemas import ScreeningRequest

log = logging.getLogger("pcos_app")
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_MODEL_FILE = "pcos_screen_model.joblib"

SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
                               "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}


def _public_metrics(path: Path) -> dict[str, Any]:
    """The subset of the training report that the 'About the model' page shows."""
    if not path.is_file():
        return {}
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return {}
    chosen = m.get("selected_model")
    rep = m.get("models", {}).get(chosen, {})
    return {
        "selected_model": chosen,
        "selection_reason": m.get("selection_reason"),
        "final_oof": m.get("final_oof"),
        "reliability": m.get("reliability"),
        "loso": {s: {"roc_auc": v.get("roc_auc"), "roc_auc_ci": v.get("roc_auc_ci"), "n": v.get("n")}
                 for s, v in rep.get("loso", {}).get("per_source", {}).items()},
        "models": {name: {"cv_auc": r["cv"]["fold_auc_mean"], "cv_auc_sd": r["cv"]["fold_auc_sd"],
                          "loso_auc": r["loso"]["mean_auc"]} for name, r in m.get("models", {}).items()},
        "clinical_benchmark": m.get("clinical_benchmark"),
        "odds_ratios": m.get("odds_ratios"),
        "importance": m.get("importance"),
        "data": {k: m.get("data", {}).get(k) for k in ("n", "prevalence", "by_source")},
        "thresholds": m.get("thresholds"),
        "created": m.get("created"),
    }


def create_app(cfg: Config | None = None, model_path: str | Path | None = None) -> FastAPI:
    cfg = cfg or load_config()
    model_path = Path(model_path) if model_path else cfg.path("model_dir") / DEFAULT_MODEL_FILE

    predictor: PCOSPredictor | None
    model_error: str | None = None
    try:
        predictor = PCOSPredictor.load(model_path)
    except ModelNotAvailable as exc:
        predictor, model_error = None, str(exc)
        log.warning("Running in guideline-only mode: %s", exc)

    store = None
    if cfg.get("storage.enabled", False):
        store = ScreeningStore(cfg.resolve(cfg.get("storage.sqlite_path", "data/screenings.sqlite3")))
    metrics = _public_metrics(cfg.path("report_dir") / "metrics.json")

    app = FastAPI(title="PCOS Early Screening API", version=__version__,
                  description="Questionnaire-based PCOS/PCOD risk screening. Educational use only; not a diagnosis.")
    app.state.predictor = predictor
    app.state.store = store

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            if k == "Content-Security-Policy" and request.url.path in ("/docs", "/redoc"):
                continue  # Swagger UI loads its own CDN assets
            response.headers.setdefault(k, v)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        errors = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body",
                   "message": str(e["msg"]).removeprefix("Value error, ")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "Some answers need checking", "errors": errors})

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "model_loaded": predictor is not None,
                "model_error": model_error, "storage_enabled": store is not None}

    @app.get("/api/v1/model")
    def model_info() -> dict[str, Any]:
        return {"model": predictor.info() if predictor else None, "metrics": metrics,
                "mode": "guideline+ml" if predictor else "guideline-only"}

    @app.post("/api/v1/screen")
    def screen_endpoint(req: ScreeningRequest) -> dict[str, Any]:
        answers = req.to_answers()
        result = screen(answers, predictor)
        result["stored"] = False
        if store is not None and req.store:
            ml = result.get("model") or {}
            row_id = store.save(answers.to_dict(), result["level"], ml.get("probability"),
                                (ml.get("model") or {}).get("version"))
            result["stored"] = row_id is not None
        return result

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve the PCOS early-screening web app")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--model", default=None, help="Path to the trained .joblib artifact")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config, args.overrides)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    import uvicorn

    app = create_app(cfg, args.model)
    uvicorn.run(app, host=args.host or cfg.get("app.host", "127.0.0.1"), port=args.port or cfg.get("app.port", 8000))
    return 0


if __name__ == "__main__":
    sys.exit(main())
