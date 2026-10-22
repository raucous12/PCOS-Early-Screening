"""End-to-end smoke test of the training CLI on a tiny budget."""
import json

import joblib

from pcos_screen.predict import PCOSPredictor
from pcos_screen.questionnaire import Answers
from pcos_screen.train import main


def test_training_cli_produces_working_artifact(tmp_path, data):  # `data` skips without raw files
    rc = main([
        "--quick", "--skip-extras", "--models", "logreg", "lightgbm",
        "--set", f"paths.model_dir={(tmp_path / 'models').as_posix()}",
        "--set", f"paths.report_dir={(tmp_path / 'reports').as_posix()}",
        "--set", f"paths.processed_dir={(tmp_path / 'processed').as_posix()}",
        "--set", f"paths.docs_dir={(tmp_path / 'docs').as_posix()}",
        "--set", "training.n_jobs=2",
    ])
    assert rc == 0
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert metrics["selected_model"] in {"logreg", "lightgbm"}
    assert set(metrics["models"]["logreg"]["loso"]["per_source"]) == {"kerala", "survey_in_2018", "survey_mumbai",
                                                                      "survey_lifestyle"}
    assert metrics["final_oof"]["roc_auc"] > 0.75
    art = joblib.load(tmp_path / "models" / "pcos_screen_model.joblib")
    out = PCOSPredictor(art).predict(Answers(age=25, cycle_length="46_90", menarche_age=12))
    assert 0 <= out["probability"] <= 1
    assert (tmp_path / "reports" / "roc_curves.png").stat().st_size > 1000
    assert "Model card" in (tmp_path / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")


def test_training_cli_rejects_bad_input(tmp_path):
    assert main(["--models", "not_a_model", "--set", f"paths.model_dir={tmp_path.as_posix()}"]) == 2
    assert main(["--config", str(tmp_path / "missing.yaml")]) == 2
    assert main(["--set", "sources.kerala.file=nope.xlsx", "--set", f"paths.model_dir={tmp_path.as_posix()}"]) == 2
