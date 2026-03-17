import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_items(path: Path) -> Tuple[List[Dict[str, object]], Path]:
    if path.is_dir():
        per_sample_path = path / "analysis" / "per_sample.json"
        analysis_dir = path / "analysis"
    else:
        per_sample_path = path
        analysis_dir = path.parent
    if not per_sample_path.exists():
        raise FileNotFoundError(f"per_sample.json not found at {per_sample_path}")
    data = json.loads(per_sample_path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not isinstance(items, list):
        raise ValueError("per_sample.json missing items list.")
    return [item for item in items if isinstance(item, dict)], analysis_dir


def _extract_probe_names(items: List[Dict[str, object]]) -> List[str]:
    for item in items:
        probe_names = item.get("probe_names")
        if isinstance(probe_names, list) and probe_names:
            return [str(p) for p in probe_names]
    if items:
        keys = items[0].get("score_mean_by_probe", {})
        if isinstance(keys, dict):
            return [str(k) for k in keys.keys()]
    return []


def _build_rows(items: List[Dict[str, object]], probe_names: List[str]) -> List[Dict[str, object]]:
    rows = []
    for item in items:
        correct = item.get("correct")
        if correct not in (True, False):
            continue
        scores = item.get("score_mean_by_probe", {})
        if not isinstance(scores, dict):
            continue
        row = {"correct": bool(correct), "scores": {}}
        ok = True
        for probe in probe_names:
            if probe not in scores:
                ok = False
                break
            row["scores"][probe] = float(scores[probe])
        if ok:
            rows.append(row)
    return rows


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _compute_summary(rows: List[Dict[str, object]], probe_names: List[str]):
    overall = {}
    correct = {}
    incorrect = {}
    for probe in probe_names:
        vals = [r["scores"][probe] for r in rows]
        overall[probe] = _mean(vals)
        correct_vals = [r["scores"][probe] for r in rows if r["correct"]]
        incorrect_vals = [r["scores"][probe] for r in rows if not r["correct"]]
        correct[probe] = _mean(correct_vals)
        incorrect[probe] = _mean(incorrect_vals)
    return overall, correct, incorrect


def _fit_model(rows: List[Dict[str, object]], probe_names: List[str], seed: int = 123):
    if not rows:
        return None
    X = np.array([[r["scores"][p] for p in probe_names] for r in rows], dtype=np.float64)
    y = np.array([1 if r["correct"] else 0 for r in rows], dtype=np.int32)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    if len(rows) < 2:
        return {
            "train_n": int(len(rows)),
            "test_n": 0,
            "coefficients": [],
            "metrics": {},
            "note": "Need at least 2 labeled rows for train/test modeling.",
        }

    split = int(0.75 * len(rows))
    split = max(1, min(len(rows) - 1, split))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train_z = (X_train - mean) / std
    X_test_z = (X_test - mean) / std

    result = {
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "coefficients": [],
        "metrics": {},
        "note": "",
    }

    if len(train_idx) == 0 or len(test_idx) == 0:
        result["note"] = "Train/test split is empty; skipping classifier fit."
        return result
    if np.unique(y_train).size < 2:
        result["note"] = "Training split has a single class; skipping classifier fit."
        return result

    try:
        import statsmodels.api as sm
    except Exception:
        sm = None

    if sm is not None:
        try:
            X_train_sm = sm.add_constant(X_train_z)
            X_test_sm = sm.add_constant(X_test_z)
            model = sm.Logit(y_train, X_train_sm).fit(disp=False)
            probs = model.predict(X_test_sm)
        except Exception as exc:
            result["note"] = f"Logistic model fit failed: {exc}"
            return result

        try:
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
        except Exception:
            accuracy_score = None
            f1_score = None
            roc_auc_score = None
            log_loss = None

        if accuracy_score is not None:
            preds = (probs >= 0.5).astype(int)
            metrics = {"accuracy": float(accuracy_score(y_test, preds))}
            try:
                metrics["f1"] = float(f1_score(y_test, preds))
            except Exception:
                pass
            try:
                if np.unique(y_test).size >= 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
            except Exception:
                pass
            try:
                metrics["log_loss"] = float(log_loss(y_test, probs))
            except Exception:
                pass
        else:
            metrics = {}

        conf_int = model.conf_int()
        pvals = model.pvalues
        params = model.params
        names = ["(intercept)"] + probe_names
        for i, name in enumerate(names):
            coef = float(params[i])
            ci_lo = float(conf_int[i, 0])
            ci_hi = float(conf_int[i, 1])
            pval = float(pvals[i])
            result["coefficients"].append(
                {
                    "feature": name,
                    "coef": coef,
                    "odds_ratio": float(np.exp(coef)),
                    "ci_low": ci_lo,
                    "ci_high": ci_hi,
                    "p_value": pval,
                }
            )
        result["metrics"] = metrics
        return result

    result["note"] = "statsmodels not available; p-values not computed."
    return result


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def generate_multi_probe_report(
    path: str,
    *,
    title: str = "Multi-Probe Report",
    output_name: str = "report.html",
    seed: int = 123,
) -> Path:
    """
    Build an interactive HTML report for a multi-probe eval batch.

    Args:
        path: Batch directory or analysis/per_sample.json path.
        title: Title shown in the HTML.
        output_name: Output HTML filename (written under the analysis dir).
        seed: Random seed for train/test split in the classifier.
    """
    items, analysis_dir = _load_items(Path(path))
    probe_names = _extract_probe_names(items)
    if not probe_names:
        raise ValueError("No probe names found in per_sample.json.")
    rows = _build_rows(items, probe_names)
    if not rows:
        raise ValueError("No valid rows with correct labels and scores.")

    overall, correct, incorrect = _compute_summary(rows, probe_names)
    model_result = _fit_model(rows, probe_names, seed=seed)

    data_blob = {
        "rows": rows,
        "probe_names": probe_names,
        "summary": {
            "overall": overall,
            "correct": correct,
            "incorrect": incorrect,
        },
        "model": model_result,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{_html_escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1 {{ margin-top: 0; }}
    .row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
    .panel {{ flex: 1; min-width: 420px; }}
    .chart {{ width: 100%; height: 420px; }}
    select {{ margin-right: 10px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .note {{ color: #666; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>{_html_escape(title)}</h1>
  <div class="row">
    <div class="panel">
      <h2>Mean score by probe</h2>
      <div id="mean-bar" class="chart"></div>
    </div>
    <div class="panel">
      <h2>Mean score by probe (correct vs incorrect)</h2>
      <div id="correct-bar" class="chart"></div>
    </div>
  </div>
  <div class="row">
    <div class="panel">
      <h2>2D scatter (probe scores)</h2>
      <div>
        X: <select id="x-probe"></select>
        Y: <select id="y-probe"></select>
      </div>
      <div id="scatter-2d" class="chart"></div>
    </div>
    <div class="panel">
      <h2>3D scatter (probe scores)</h2>
      <div>
        X: <select id="x3-probe"></select>
        Y: <select id="y3-probe"></select>
        Z: <select id="z3-probe"></select>
      </div>
      <div id="scatter-3d" class="chart"></div>
    </div>
  </div>

  <h2>Predicting correctness from probe scores</h2>
  <div id="model-metrics"></div>
  <div id="model-coefs"></div>

  <script>
    const payload = {json.dumps(data_blob)};
    const probeNames = payload.probe_names;
    const rows = payload.rows;

    function buildOptions(selectId, defaultIdx) {{
      const sel = document.getElementById(selectId);
      sel.innerHTML = "";
      probeNames.forEach((p, i) => {{
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        if (i === defaultIdx) opt.selected = true;
        sel.appendChild(opt);
      }});
    }}

    function scoresFor(probe) {{
      return rows.map(r => r.scores[probe]);
    }}

    function maskScores(values, mask) {{
      const out = [];
      for (let i = 0; i < values.length; i++) {{
        if (mask[i]) out.push(values[i]);
      }}
      return out;
    }}

    function buildBarCharts() {{
      const overall = payload.summary.overall;
      const correct = payload.summary.correct;
      const incorrect = payload.summary.incorrect;

      const meanTrace = {{
        x: probeNames,
        y: probeNames.map(p => overall[p]),
        type: "bar",
        marker: {{ color: "#2b6aa6" }}
      }};
      Plotly.newPlot("mean-bar", [meanTrace], {{
        margin: {{ t: 20 }},
        yaxis: {{ title: "Mean score" }}
      }});

      const correctTrace = {{
        x: probeNames,
        y: probeNames.map(p => correct[p]),
        name: "Correct",
        type: "bar",
        marker: {{ color: "#2c8d5b" }}
      }};
      const incorrectTrace = {{
        x: probeNames,
        y: probeNames.map(p => incorrect[p]),
        name: "Incorrect",
        type: "bar",
        marker: {{ color: "#b24a3b" }}
      }};
      Plotly.newPlot("correct-bar", [correctTrace, incorrectTrace], {{
        barmode: "group",
        margin: {{ t: 20 }},
        yaxis: {{ title: "Mean score" }}
      }});
    }}

    function updateScatter2D() {{
      const xProbe = document.getElementById("x-probe").value;
      const yProbe = document.getElementById("y-probe").value;
      const x = scoresFor(xProbe);
      const y = scoresFor(yProbe);
      const mask = rows.map(r => r.correct);
      const xCorrect = maskScores(x, mask);
      const yCorrect = maskScores(y, mask);
      const xIncorrect = maskScores(x, mask.map(v => !v));
      const yIncorrect = maskScores(y, mask.map(v => !v));

      const traces = [
        {{ x: xCorrect, y: yCorrect, mode: "markers", name: "Correct", marker: {{ color: "#2c8d5b" }} }},
        {{ x: xIncorrect, y: yIncorrect, mode: "markers", name: "Incorrect", marker: {{ color: "#b24a3b" }} }},
      ];
      Plotly.react("scatter-2d", traces, {{
        margin: {{ t: 20 }},
        xaxis: {{ title: xProbe }},
        yaxis: {{ title: yProbe }},
      }});
    }}

    function updateScatter3D() {{
      const xProbe = document.getElementById("x3-probe").value;
      const yProbe = document.getElementById("y3-probe").value;
      const zProbe = document.getElementById("z3-probe").value;
      const x = scoresFor(xProbe);
      const y = scoresFor(yProbe);
      const z = scoresFor(zProbe);
      const mask = rows.map(r => r.correct);
      const xCorrect = maskScores(x, mask);
      const yCorrect = maskScores(y, mask);
      const zCorrect = maskScores(z, mask);
      const xIncorrect = maskScores(x, mask.map(v => !v));
      const yIncorrect = maskScores(y, mask.map(v => !v));
      const zIncorrect = maskScores(z, mask.map(v => !v));

      const traces = [
        {{ x: xCorrect, y: yCorrect, z: zCorrect, mode: "markers", type: "scatter3d", name: "Correct", marker: {{ size: 4, color: "#2c8d5b" }} }},
        {{ x: xIncorrect, y: yIncorrect, z: zIncorrect, mode: "markers", type: "scatter3d", name: "Incorrect", marker: {{ size: 4, color: "#b24a3b" }} }},
      ];
      Plotly.react("scatter-3d", traces, {{
        margin: {{ t: 20 }},
        scene: {{
          xaxis: {{ title: xProbe }},
          yaxis: {{ title: yProbe }},
          zaxis: {{ title: zProbe }},
        }}
      }});
    }}

    function renderModelTable() {{
      const container = document.getElementById("model-metrics");
      const coefs = document.getElementById("model-coefs");
      const model = payload.model;
      if (!model) {{
        container.innerHTML = "<p>No model results.</p>";
        return;
      }}
      let metricsHtml = "<table><tr><th>Metric</th><th>Value</th></tr>";
      metricsHtml += `<tr><td>Train N</td><td>${{model.train_n}}</td></tr>`;
      metricsHtml += `<tr><td>Test N</td><td>${{model.test_n}}</td></tr>`;
      if (model.metrics) {{
        for (const [k, v] of Object.entries(model.metrics)) {{
          metricsHtml += `<tr><td>${{k}}</td><td>${{v.toFixed(4)}}</td></tr>`;
        }}
      }}
      metricsHtml += "</table>";
      if (model.note) {{
        metricsHtml += `<p class='note'>${{model.note}}</p>`;
      }}
      container.innerHTML = metricsHtml;

      if (!model.coefficients || model.coefficients.length === 0) {{
        coefs.innerHTML = "";
        return;
      }}
      let coefHtml = "<table><tr><th>Feature</th><th>Coef</th><th>Odds ratio</th><th>95% CI</th><th>p-value</th></tr>";
      model.coefficients.forEach(row => {{
        const ci = `[${{row.ci_low.toFixed(3)}}, ${{row.ci_high.toFixed(3)}}]`;
        const p = row.p_value !== undefined ? row.p_value.toExponential(3) : "n/a";
        coefHtml += `<tr><td>${{row.feature}}</td><td>${{row.coef.toFixed(3)}}</td><td>${{row.odds_ratio.toFixed(3)}}</td><td>${{ci}}</td><td>${{p}}</td></tr>`;
      }});
      coefHtml += "</table>";
      coefs.innerHTML = coefHtml;
    }}

    buildOptions("x-probe", 0);
    buildOptions("y-probe", Math.min(1, probeNames.length - 1));
    buildOptions("x3-probe", 0);
    buildOptions("y3-probe", Math.min(1, probeNames.length - 1));
    buildOptions("z3-probe", Math.min(2, probeNames.length - 1));

    document.getElementById("x-probe").addEventListener("change", updateScatter2D);
    document.getElementById("y-probe").addEventListener("change", updateScatter2D);
    document.getElementById("x3-probe").addEventListener("change", updateScatter3D);
    document.getElementById("y3-probe").addEventListener("change", updateScatter3D);
    document.getElementById("z3-probe").addEventListener("change", updateScatter3D);

    buildBarCharts();
    updateScatter2D();
    updateScatter3D();
    renderModelTable();
  </script>
</body>
</html>
"""
    out_path = analysis_dir / output_name
    out_path.write_text(html, encoding="utf-8")
    return out_path


__all__ = ["generate_multi_probe_report"]
