import json

import concept_probe.reporting as reporting


def test_generate_multi_probe_report_smoke(tmp_path, monkeypatch):
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    per_sample_path = analysis_dir / "per_sample.json"

    payload = {
        "items": [
            {
                "correct": True,
                "probe_names": ["focus", "planning"],
                "score_mean_by_probe": {"focus": 0.8, "planning": 0.4},
            },
            {
                "correct": False,
                "probe_names": ["focus", "planning"],
                "score_mean_by_probe": {"focus": -0.6, "planning": -0.3},
            },
            {
                "correct": True,
                "probe_names": ["focus", "planning"],
                "score_mean_by_probe": {"focus": 1.1, "planning": 0.7},
            },
            {
                "correct": False,
                "probe_names": ["focus", "planning"],
                "score_mean_by_probe": {"focus": -0.9, "planning": -0.5},
            },
        ]
    }
    per_sample_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        reporting,
        "_fit_model",
        lambda rows, probe_names, seed=123: {
            "train_n": len(rows),
            "test_n": 0,
            "coefficients": [],
            "metrics": {},
            "note": "model fit intentionally skipped in smoke test",
        },
    )

    report_path = reporting.generate_multi_probe_report(
        str(per_sample_path),
        title="Smoke Test Report",
    )
    report_html = report_path.read_text(encoding="utf-8")

    assert report_path.exists()
    assert "Smoke Test Report" in report_html
    assert "plotly" in report_html.lower()
