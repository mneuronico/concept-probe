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
        alpha = item.get("alpha")
        row = {
            "correct": bool(correct),
            "alpha": None if alpha is None else float(alpha),
            "item_index": item.get("item_index"),
            "scores": {},
        }
        ok = True
        for probe in probe_names:
            if probe not in scores:
                ok = False
                break
            try:
                val = float(scores[probe])
            except (TypeError, ValueError):
                ok = False
                break
            if not np.isfinite(val):
                ok = False
                break
            row["scores"][probe] = val
        if ok:
            rows.append(row)
    return rows


def _mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _alpha_key(alpha: Optional[float]) -> str:
    return "none" if alpha is None else f"{float(alpha):g}"


def _compute_summary(rows: List[Dict[str, object]], probe_names: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean score per probe, overall and by correctness, separately for each alpha.

    Alphas are never pooled: steering moves both the scores and the accuracy.
    """
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key in sorted({_alpha_key(r["alpha"]) for r in rows}):
        sub = [r for r in rows if _alpha_key(r["alpha"]) == key]
        summary[key] = {
            "overall": {p: _mean([r["scores"][p] for r in sub]) for p in probe_names},
            "correct": {p: _mean([r["scores"][p] for r in sub if r["correct"]]) for p in probe_names},
            "incorrect": {p: _mean([r["scores"][p] for r in sub if not r["correct"]]) for p in probe_names},
        }
    return summary


def _fit_model(rows: List[Dict[str, object]], probe_names: List[str], seed: int = 123, n_folds: int = 5):
    """Logistic model of correctness from probe scores, with alpha as a categorical covariate.

    Metrics are pooled out-of-fold predictions from K-fold CV in which all rows of an item
    (one per alpha) fall in the same fold, so no item is both trained and tested on; rows
    without an item index are stratified by class instead. Coefficients and p-values come
    from a single fit on all rows, with standard errors clustered by item when items repeat.
    """
    if not rows:
        return None
    alphas = sorted({r.get("alpha") for r in rows if r.get("alpha") is not None})
    feature_names = list(probe_names) + [f"alpha={a:g}" for a in alphas[1:]]
    X = np.array(
        [[r["scores"][p] for p in probe_names] + [1.0 if r.get("alpha") == a else 0.0 for a in alphas[1:]] for r in rows],
        dtype=np.float64,
    )
    y = np.array([1 if r["correct"] else 0 for r in rows], dtype=np.int32)
    groups = np.array(
        [r["item_index"] if r.get("item_index") is not None else f"row{i}" for i, r in enumerate(rows)],
        dtype=object,
    )
    clustered = len(set(groups.tolist())) < len(rows)

    result = {
        "n": int(len(rows)),
        "n_items": int(len(set(groups.tolist()))),
        "cv_folds": 0,
        "cv_grouped_by_item": bool(clustered),
        "alpha_covariate": len(alphas) > 1,
        "se_clustered_by_item": bool(clustered),
        "coefficients": [],
        "metrics": {},
        "metrics_source": "out_of_fold_cv",
        "note": "",
    }

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 2 or n_neg < 2:
        result["note"] = "Need at least 2 rows of each class for cross-validated modeling."
        return result

    try:
        import statsmodels.api as sm
    except Exception:
        result["note"] = "statsmodels not available; model not fit."
        return result

    rng = np.random.default_rng(seed)
    folds = np.zeros(len(y), dtype=np.int32)
    if clustered:
        items = np.array(sorted(set(groups.tolist()), key=str), dtype=object)
        rng.shuffle(items)
        k = int(max(2, min(n_folds, len(items))))
        fold_of_item = {item: j % k for j, item in enumerate(items.tolist())}
        folds = np.array([fold_of_item[g] for g in groups.tolist()], dtype=np.int32)
    else:
        # Stratified fold assignment: shuffle within each class, deal round-robin so every
        # fold's training split contains both classes.
        k = int(max(2, min(n_folds, n_pos, n_neg)))
        for cls in (0, 1):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            for j, i in enumerate(idx):
                folds[i] = j % k
    result["cv_folds"] = k

    n_scores = len(probe_names)

    def design(train_mask: np.ndarray, rows_mask: np.ndarray) -> np.ndarray:
        # z-score the probe scores with the training rows' moments; alpha dummies stay 0/1
        mean = X[train_mask, :n_scores].mean(axis=0)
        std = X[train_mask, :n_scores].std(axis=0)
        std[std == 0] = 1.0
        Z = np.concatenate([(X[rows_mask, :n_scores] - mean) / std, X[rows_mask, n_scores:]], axis=1)
        return sm.add_constant(Z, has_constant="add")

    oof_probs = np.full(len(y), np.nan, dtype=np.float64)
    failed_folds = 0
    for f in range(k):
        tr = folds != f
        te = folds == f
        try:
            if len(np.unique(y[tr])) < 2:
                raise ValueError("one class only in the training split")
            fold_model = sm.Logit(y[tr], design(tr, tr)).fit(disp=False)
            oof_probs[te] = np.asarray(fold_model.predict(design(tr, te)), dtype=np.float64)
        except Exception:
            failed_folds += 1
    if failed_folds:
        result["note"] = (
            f"{failed_folds}/{k} CV folds failed to fit (e.g. perfect separation); "
            "metrics use the remaining folds."
        )

    valid = np.isfinite(oof_probs)
    metrics: Dict[str, float] = {}
    if valid.any():
        y_v = y[valid]
        p_v = oof_probs[valid]
        preds = (p_v >= 0.5).astype(int)
        metrics["accuracy"] = float((preds == y_v).mean())
        tp = int(((preds == 1) & (y_v == 1)).sum())
        fp = int(((preds == 1) & (y_v == 0)).sum())
        fn = int(((preds == 0) & (y_v == 1)).sum())
        if (2 * tp + fp + fn) > 0:
            metrics["f1"] = float(2 * tp / (2 * tp + fp + fn))
        try:
            from sklearn.metrics import log_loss, roc_auc_score

            if np.unique(y_v).size >= 2:
                metrics["roc_auc"] = float(roc_auc_score(y_v, p_v))
            metrics["log_loss"] = float(
                log_loss(y_v, np.clip(p_v, 1e-12, 1 - 1e-12), labels=[0, 1])
            )
        except Exception:
            pass
    result["metrics"] = metrics

    # Coefficients and p-values from a fit on all rows (metrics above are out-of-fold).
    everything = np.ones(len(y), dtype=bool)
    try:
        if clustered:
            codes = {g: i for i, g in enumerate(sorted(set(groups.tolist()), key=str))}
            full_model = sm.Logit(y, design(everything, everything)).fit(
                disp=False, cov_type="cluster", cov_kwds={"groups": np.array([codes[g] for g in groups.tolist()])}
            )
        else:
            full_model = sm.Logit(y, design(everything, everything)).fit(disp=False)
    except Exception as exc:
        extra = f"Full-data coefficient fit failed: {exc}"
        result["note"] = f"{result['note']} {extra}".strip()
        return result

    conf_int = np.asarray(full_model.conf_int())
    pvals = np.asarray(full_model.pvalues)
    params = np.asarray(full_model.params)
    names = ["(intercept)"] + feature_names
    for i, name in enumerate(names):
        coef = float(params[i])
        result["coefficients"].append(
            {
                "feature": name,
                "coef": coef,
                "odds_ratio": float(np.exp(coef)),
                "ci_low": float(conf_int[i, 0]),
                "ci_high": float(conf_int[i, 1]),
                "p_value": float(pvals[i]),
            }
        )
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
        seed: Random seed for the cross-validation folds.

    Charts show one steering alpha at a time (a selector switches between them); the
    correctness model uses every alpha, with alpha as a covariate.
    """
    items, analysis_dir = _load_items(Path(path))
    probe_names = _extract_probe_names(items)
    if not probe_names:
        raise ValueError("No probe names found in per_sample.json.")
    rows = _build_rows(items, probe_names)
    if not rows:
        raise ValueError("No valid rows with correct labels and scores.")

    summary = _compute_summary(rows, probe_names)
    model_result = _fit_model(rows, probe_names, seed=seed)
    alpha_keys = list(summary.keys())
    default_alpha = "0" if "0" in alpha_keys else alpha_keys[0]

    data_blob = {
        "rows": [dict(r, alpha_key=_alpha_key(r["alpha"])) for r in rows],
        "probe_names": probe_names,
        "summary_by_alpha": summary,
        "alpha_keys": alpha_keys,
        "default_alpha": default_alpha,
        "model": model_result,
    }
    # "</" would end the <script> block early if a probe name contained "</script>".
    payload_json = json.dumps(data_blob).replace("</", "<" + chr(92) + "/")

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
  <div>Steering alpha: <select id="alpha-select"></select>
    <span class="note">Charts show one alpha at a time; steering moves both scores and accuracy.</span></div>
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
    const payload = {payload_json};
    const probeNames = payload.probe_names;
    const allRows = payload.rows;
    let rows = allRows;

    function selectedAlpha() {{
      return document.getElementById("alpha-select").value;
    }}

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
      const summary = payload.summary_by_alpha[selectedAlpha()];
      const overall = summary.overall;
      const correct = summary.correct;
      const incorrect = summary.incorrect;

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
      const info = [["N rows", model.n], ["N items", model.n_items], ["CV folds", model.cv_folds],
                    ["CV grouped by item", model.cv_grouped_by_item], ["Alpha covariate", model.alpha_covariate],
                    ["SE clustered by item", model.se_clustered_by_item]];
      for (const [k, v] of info) {{
        if (v !== undefined && v !== null) {{
          metricsHtml += `<tr><td>${{k}}</td><td>${{v}}</td></tr>`;
        }}
      }}
      if (model.metrics) {{
        const src = model.metrics_source ? ` (${{model.metrics_source}})` : "";
        for (const [k, v] of Object.entries(model.metrics)) {{
          metricsHtml += `<tr><td>${{k}}${{src}}</td><td>${{v.toFixed(4)}}</td></tr>`;
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

    const alphaSelect = document.getElementById("alpha-select");
    payload.alpha_keys.forEach(a => {{
      const opt = document.createElement("option");
      opt.value = a;
      opt.textContent = a;
      if (a === payload.default_alpha) opt.selected = true;
      alphaSelect.appendChild(opt);
    }});
    function refreshAlpha() {{
      rows = allRows.filter(r => r.alpha_key === selectedAlpha());
      buildBarCharts();
      updateScatter2D();
      updateScatter3D();
    }}
    alphaSelect.addEventListener("change", refreshAlpha);
    rows = allRows.filter(r => r.alpha_key === selectedAlpha());

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
