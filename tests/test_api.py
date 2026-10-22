import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from pcos_screen.config import load_config

VALID = {
    "age": 23, "height_cm": 162, "weight_kg": 70, "periods_started": True, "menarche_age": 12,
    "cycle_length": "46_90", "gap_over_90_days": "yes", "hormonal_contraception": False,
    "hair": {"upper_lip": 2, "chin": 2, "cheeks": 0, "chest": 0, "abdomen": 1, "thighs_back": 0},
    "acne": 1, "hair_loss": True, "skin_darkening": False, "weight_gain": True,
    "fast_food_per_week": 4, "exercise_days_per_week": 1, "fatigue": True, "mood_swings": None,
    "conception_difficulty": "not_applicable", "family_history": "yes", "rapid_virilization": False,
    "possibly_pregnant": "no", "thyroid_or_prolactin": False,
}


@pytest.fixture(scope="module")
def client(model_file, tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "s.sqlite3"
    cfg = load_config(overrides=["storage.enabled=true", f"storage.sqlite_path={db.as_posix()}"])
    return TestClient(create_app(cfg, model_file))


@pytest.fixture(scope="module")
def guideline_only_client(tmp_path_factory):
    return TestClient(create_app(load_config(), tmp_path_factory.mktemp("none") / "missing.joblib"))


def test_health(client):
    r = client.get("/api/v1/health").json()
    assert r["status"] == "ok" and r["model_loaded"] is True and r["storage_enabled"] is True


def test_screen_happy_path(client):
    r = client.post("/api/v1/screen", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["level"] == "high"
    assert body["guideline"]["pattern"] == "consistent"
    assert 0 <= body["model"]["probability"] <= 1
    assert body["stored"] is False  # no consent given


def test_minimal_payload(client):
    r = client.post("/api/v1/screen", json={"age": 30})
    assert r.status_code == 200 and r.json()["level"] in {"low", "moderate", "high"}


def test_storage_with_consent(client):
    r = client.post("/api/v1/screen", json={**VALID, "store": True}).json()
    assert r["stored"] is True
    assert client.app.state.store.count() >= 1


@pytest.mark.parametrize("patch,field", [
    ({"age": 5}, "age"),
    ({"menarche_age": 30, "age": 25}, "menarche_age"),
    ({"menarche_age": 15, "age": 13}, "body"),
    ({"height_cm": None}, "body"),
    ({"weight_kg": 30, "height_cm": 200}, "body"),
    ({"cycle_length": "weekly"}, "cycle_length"),
    ({"hair": {"chin": 7}}, "hair.chin"),
    ({"unexpected": 1}, "unexpected"),
    ({"periods_started": False}, "body"),  # cycle answers given although periods not started
])
def test_validation_errors(client, patch, field):
    r = client.post("/api/v1/screen", json={**VALID, **patch})
    assert r.status_code == 422
    body = r.json()
    assert body["detail"] == "Some answers need checking"
    assert any(e["field"].startswith(field.split(".")[0]) or field == "body" for e in body["errors"])


def test_not_started_payload_is_valid(client):
    r = client.post("/api/v1/screen", json={"age": 15, "periods_started": False})
    assert r.status_code == 200
    assert r.json()["guideline"]["red_flags"]


def test_security_headers_and_static(client):
    r = client.get("/")
    assert r.status_code == 200 and "PCOS Early Check" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/app.js").status_code == 200


def test_model_endpoint(client):
    r = client.get("/api/v1/model").json()
    assert r["mode"] == "guideline+ml" and r["model"]["features"]


def test_guideline_only_mode(guideline_only_client):
    h = guideline_only_client.get("/api/v1/health").json()
    assert h["model_loaded"] is False and "not found" in h["model_error"]
    r = guideline_only_client.post("/api/v1/screen", json=VALID).json()
    assert r["model"] is None and r["level"] == "high"
