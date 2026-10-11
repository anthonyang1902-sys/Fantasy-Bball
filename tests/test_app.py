"""Headless smoke test of the Streamlit dashboard."""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_dashboard_renders_without_errors():
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Team A" in t.value for t in at.title)
    # evaluate a trade through the UI
    at.multiselect[1].select(at.multiselect[1].options[0])
    at.run()
    assert not at.exception
    assert at.info, "trade verdict should be shown"
