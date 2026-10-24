"""Smoke test: the Streamlit app loads and returns a prediction.

Needs a trained model (python -m src.train); skipped otherwise.
"""
import pytest

from src.config import METADATA_PATH, MODEL_PATH, PROJECT_ROOT

pytestmark = pytest.mark.skipif(
    not (MODEL_PATH.exists() and METADATA_PATH.exists()),
    reason="Train the model first: python -m src.train",
)


def test_app_runs_and_predicts():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception

    # Click the form's submit button with the default inputs.
    app.button[0].click().run()
    assert not app.exception
    assert any("Estimate" in metric.label for metric in app.metric)
    assert any("estimate" in info.value.lower() for info in app.info)
