/* PCOS Early Check: frontend logic (no dependencies, CSP-friendly). */
(function () {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (p, d = 0) => `${(100 * p).toFixed(d)}%`;
  const state = { step: 1, steps: 5, units: { height: "cm", weight: "kg" }, health: null, modelLoaded: false };

  // ------------------------------------------------------------------ theme
  function storedTheme() { try { return localStorage.getItem("theme"); } catch (_) { return null; } }
  function applyTheme(t) { if (t) document.documentElement.setAttribute("data-theme", t); }
  applyTheme(storedTheme());
  function toggleTheme() {
    const current = document.documentElement.getAttribute("data-theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("theme", next); } catch (_) { /* private mode */ }
  }

  // ------------------------------------------------------------------ routing
  function route() {
    const name = (location.hash || "#home").slice(1).split("?")[0];
    const view = ["home", "check", "learn", "model"].includes(name) ? name : "home";
    $$("[data-view]").forEach((v) => v.classList.toggle("hidden", v.dataset.view !== view));
    $$(".nav-links a").forEach((a) => {
      if (a.dataset.route === view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    if (view === "model") loadModelPage();
    window.scrollTo({ top: 0 });
  }

  // ------------------------------------------------------------------ build dynamic inputs
  const HAIR_REGIONS = [
    ["upper_lip", "Upper lip"], ["chin", "Chin"], ["cheeks", "Cheeks / sideburns"],
    ["chest", "Chest / between breasts"], ["abdomen", "Lower belly (below the navel)"],
    ["thighs_back", "Inner thighs or back"],
  ];
  const HAIR_LEVELS = ["None", "A little", "Moderate", "Heavy"];

  function radio(name, value, label, small) {
    return `<label class="opt"><input type="radio" name="${name}" value="${value}"><span>${label}${small ? `<small>${small}</small>` : ""}</span></label>`;
  }

  function buildInputs() {
    $("#hair-grid").innerHTML = HAIR_REGIONS.map(([key, label]) =>
      `<div class="hair-row" role="radiogroup" aria-label="${label}"><div class="region">${label}</div>
       <div class="options">${HAIR_LEVELS.map((l, i) => radio(`hair_${key}`, i, l)).join("")}</div></div>`).join("");
    $$("[data-yns]").forEach((el) => {
      const n = el.dataset.yns;
      el.innerHTML = radio(n, "true", "Yes") + radio(n, "false", "No") + radio(n, "null", "Not sure");
    });
    $("#exercise-options").innerHTML = [0, 1, 2, 3, 4, 5, 6, 7].map((d) => radio("exercise_days_per_week", d, String(d))).join("");
  }

  // ------------------------------------------------------------------ helpers to read the form
  const radioVal = (name) => { const el = $(`input[name="${name}"]:checked`); return el ? el.value : undefined; };
  const numVal = (id) => { const v = $(`#${id}`).value.trim(); return v === "" ? null : Number(v); };
  const boolOrNull = (name) => { const v = radioVal(name); return v === undefined || v === "null" ? null : v === "true"; };

  function heightCm() {
    if (state.units.height === "cm") return numVal("height_cm");
    const ft = numVal("height_ft"), inch = numVal("height_in");
    if (ft === null && inch === null) return null;
    return Math.round(((ft || 0) * 30.48 + (inch || 0) * 2.54) * 10) / 10;
  }
  function weightKg() {
    const w = numVal("weight_val");
    if (w === null) return null;
    return state.units.weight === "kg" ? w : Math.round(w * 0.45359237 * 10) / 10;
  }
  function updateBmiPreview() {
    const h = heightCm(), w = weightKg(), out = $("#bmi-preview");
    if (h && w && h > 90 && w > 20) {
      const bmi = w / Math.pow(h / 100, 2);
      out.textContent = `Your BMI is about ${bmi.toFixed(1)}.`;
    } else out.textContent = "";
  }

  function periodsStarted() { return radioVal("periods_started") !== "false"; }

  function updateConditional() {
    const started = radioVal("periods_started");
    $("#cycle-questions").classList.toggle("hidden", started === "false");
    $("#menarche_age").disabled = $("#menarche_unknown").checked;
    const longGap = ["gt90", "46_90"].includes(radioVal("cycle_length")) || radioVal("gap_over_90_days") === "yes";
    $("#q-pregnant").classList.toggle("hidden", !(started === "true" && longGap));
  }

  // ------------------------------------------------------------------ validation per step
  function fail(q, msg, errors) { errors.push(msg); const el = $(`[data-q="${q}"]`); if (el) el.classList.add("invalid"); }

  function validateStep(step) {
    $$(".q.invalid").forEach((q) => q.classList.remove("invalid"));
    const errors = [];
    if (step === 1) {
      const age = numVal("age");
      if (age === null || !Number.isFinite(age)) fail("age", "Please enter your age.", errors);
      else if (age < 10 || age > 60) fail("age", "This check is designed for ages 10 to 60.", errors);
      const h = heightCm(), w = weightKg();
      if ((h === null) !== (w === null)) fail(h === null ? "height" : "weight", "Please enter both height and weight, or leave both empty.", errors);
      if (h !== null && (h < 100 || h > 220)) fail("height", "Height should be between 100 and 220 cm (3 ft 4 in to 7 ft 2 in).", errors);
      if (w !== null && (w < 25 || w > 250)) fail("weight", "Weight should be between 25 and 250 kg (55 to 550 lb).", errors);
      if (h && w) { const bmi = w / Math.pow(h / 100, 2); if (bmi < 12 || bmi > 70) fail("weight", "Height and weight don't look right together; please check them.", errors); }
    }
    if (step === 2) {
      if (!radioVal("periods_started")) fail("periods_started", "Please say whether your periods have started.", errors);
      if (radioVal("periods_started") === "true") {
        const m = numVal("menarche_age"), age = numVal("age");
        if (!$("#menarche_unknown").checked) {
          if (m === null) fail("menarche_age", "Enter your age at your first period, or tick “I don't remember”.", errors);
          else if (m < 7 || m > 20) fail("menarche_age", "Age at first period should be between 7 and 20.", errors);
          else if (age !== null && m > age) fail("menarche_age", "Age at first period can't be more than your current age.", errors);
        }
        if (!radioVal("cycle_length")) fail("cycle_length", "Please choose your usual cycle length (or “I don't know”).", errors);
        if (!radioVal("gap_over_90_days")) fail("gap_over_90_days", "Please answer the question about 3-month gaps.", errors);
      }
    }
    if (step === 4) {
      const ff = numVal("fast_food_per_week");
      if (ff !== null && (ff < 0 || ff > 21)) fail("fast_food_per_week", "Fast-food meals should be between 0 and 21 a week.", errors);
    }
    const box = $("#form-error");
    if (errors.length) {
      box.innerHTML = `<strong>Please check:</strong><ul class="clean">${errors.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>`;
      box.classList.remove("hidden");
      const first = $(".q.invalid input:not([disabled])");
      if (first) first.focus(); else window.scrollTo(0, 0);
      return false;
    }
    box.classList.add("hidden");
    return true;
  }

  // ------------------------------------------------------------------ wizard navigation
  function showStep(n) {
    state.step = n;
    $$(".step").forEach((s) => s.classList.toggle("hidden", Number(s.dataset.step) !== n));
    const cur = $(`.step[data-step="${n}"]`);
    $("#step-name").textContent = cur.dataset.title.replace("&amp;", "&");
    $("#step-count").textContent = `Step ${n} of ${state.steps}`;
    $("#progress-bar").style.width = `${(100 * (n - 1)) / state.steps + 100 / state.steps / 2}%`;
    $("#btn-back").classList.toggle("hidden", n === 1);
    $("#btn-next").classList.toggle("hidden", n === state.steps);
    $("#btn-submit").classList.toggle("hidden", n !== state.steps);
    updateConditional();
    const h = $("h2", cur);
    if (h) { h.setAttribute("tabindex", "-1"); h.focus({ preventScroll: true }); }
    // Instant (not smooth) scroll: a click during a smooth scroll can land on the wrong answer.
    window.scrollTo(0, 0);
  }

  function collect() {
    const started = periodsStarted();
    const hair = {};
    HAIR_REGIONS.forEach(([k]) => { const v = radioVal(`hair_${k}`); hair[k] = v === undefined ? 0 : Number(v); });
    const ex = radioVal("exercise_days_per_week");
    const payload = {
      age: numVal("age"),
      height_cm: heightCm(),
      weight_kg: weightKg(),
      periods_started: started,
      menarche_age: started && !$("#menarche_unknown").checked ? numVal("menarche_age") : null,
      cycle_length: started ? (radioVal("cycle_length") || "unknown") : "unknown",
      gap_over_90_days: started ? (radioVal("gap_over_90_days") || "unsure") : "unsure",
      hormonal_contraception: radioVal("hormonal_contraception") === "true",
      hair,
      acne: Number(radioVal("acne") || 0),
      hair_loss: boolOrNull("hair_loss"),
      skin_darkening: boolOrNull("skin_darkening"),
      weight_gain: boolOrNull("weight_gain"),
      fast_food_per_week: numVal("fast_food_per_week"),
      exercise_days_per_week: ex === undefined ? null : Number(ex),
      fatigue: boolOrNull("fatigue"),
      mood_swings: boolOrNull("mood_swings"),
      conception_difficulty: radioVal("conception_difficulty") || "not_applicable",
      family_history: radioVal("family_history") || "unsure",
      rapid_virilization: radioVal("rapid_virilization") === "true",
      possibly_pregnant: $("#q-pregnant").classList.contains("hidden") ? "not_applicable" : (radioVal("possibly_pregnant") || "not_applicable"),
      thyroid_or_prolactin: radioVal("thyroid_or_prolactin") === "true",
      store: !$("#q-store").classList.contains("hidden") && $("#store").checked,
    };
    return payload;
  }

  async function submit(ev) {
    ev.preventDefault();
    for (let s = 1; s <= state.steps; s++) {
      if (!validateStep(s)) { showStep(s); validateStep(s); return; }
    }
    const btn = $("#btn-submit");
    btn.disabled = true; btn.textContent = "Checking…";
    try {
      const res = await fetch("/api/v1/screen", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(collect()) });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msgs = (body.errors || []).map((e) => `${e.field}: ${e.message}`);
        throw new Error(msgs.length ? msgs.join("; ") : (body.detail || `Server error (${res.status})`));
      }
      renderResult(body);
    } catch (err) {
      const box = $("#form-error");
      box.innerHTML = `<strong>Sorry, we couldn't compute your result.</strong> ${esc(err.message || err)}`;
      box.classList.remove("hidden");
    } finally {
      btn.disabled = false; btn.textContent = "See my result";
    }
  }

  // ------------------------------------------------------------------ results
  const LEVEL_LABEL = { low: "Low likelihood", moderate: "Moderate: worth a check", high: "High: see a doctor" };
  const STATUS_ICON = { met: "!", not_met: "✓", indeterminate: "?" };
  const STATUS_TEXT = { met: "Present", not_met: "Not present", indeterminate: "Can't tell yet" };

  function criterion(title, c) {
    return `<li class="${c.status}"><span class="badge-icon" aria-hidden="true">${STATUS_ICON[c.status]}</span>
      <div><strong>${esc(title)}: ${STATUS_TEXT[c.status]}</strong><br><span class="muted small">${c.reasons.map(esc).join(" · ")}</span></div></li>`;
  }

  function renderResult(r) {
    const g = r.guideline, m = r.model;
    const parts = [];
    parts.push(`<div class="card level-card ${r.level}">
      <span class="level-badge"><span class="dot" aria-hidden="true"></span>${LEVEL_LABEL[r.level]}</span>
      <h2>${esc(r.headline)}</h2><p>${esc(r.action)}</p><p class="muted small">${esc(r.rationale)}</p></div>`);
    if (g.red_flags.length) {
      parts.push(`<div class="alert" role="alert"><strong>Please don't wait. Talk to a doctor about this:</strong><ul class="clean">${g.red_flags.map((f) => `<li>${esc(f)}</li>`).join("")}</ul></div>`);
    }
    const pathway = g.pathway === "adolescent"
      ? "Teen / early-years pathway (under 18, or within 8 years of your first period)"
      : "Adult pathway (8+ years since your first period)";
    parts.push(`<div class="grid-2">
      <div class="card"><h3>Guideline check</h3><p class="muted small">${esc(pathway)}</p>
        <ul class="criteria">${criterion("Irregular cycles", g.ovulatory_dysfunction)}${criterion("Signs of excess androgen", g.hyperandrogenism)}</ul>
        <p class="small">${esc(g.summary)}</p></div>
      <div class="card" id="score-card">${m ? scoreCard(m, g) :`<h3>Statistical score</h3><p class="muted">The statistical model is unavailable right now, so this result uses the guideline check only.</p>`}</div>
    </div>`);
    if (m && m.contributions.length) {
      const max = Math.max(...m.contributions.map((c) => Math.abs(c.log_odds)), 0.01);
      parts.push(`<div class="card"><h3>What influenced your score</h3>
        <p class="muted small">Compared with someone of the same age without that sign. Longer bars mattered more.</p>
        <ul class="contrib">${m.contributions.slice(0, 8).map((c) => `<li>
          <div class="row"><span>${esc(c.label)}${c.feature === "body_weight" && c.value ? ` (${Number(c.value).toFixed(1)})` : ""}</span>
          <span class="muted">${c.direction === "raises" ? "raises" : "lowers"} your score</span></div>
          <div class="bar-track"><div class="bar ${c.direction}" data-w="${(100 * Math.abs(c.log_odds)) / max}"></div></div></li>`).join("")}</ul>
        ${m.unanswered.length ? `<p class="muted small">Not answered, or too early to judge (an average value was used): ${m.unanswered.map(esc).join(", ")}.</p>` : ""}</div>`);
    }
    const notes = g.notes;
    parts.push(`<div class="grid-2">
      ${r.doctor_tests.length ? `<div class="card"><h3>Tests a doctor may suggest</h3><ul class="clean">${r.doctor_tests.map((t) => `<li>${esc(t)}</li>`).join("")}</ul></div>` : ""}
      <div class="card"><h3>Tips for you</h3><ul class="clean">${r.lifestyle_tips.map((t) => `<li>${esc(t)}</li>`).join("")}</ul></div>
      ${notes.length ? `<div class="card"><h3>Good to know</h3><ul class="clean">${notes.map((t) => `<li>${esc(t)}</li>`).join("")}</ul></div>` : ""}
    </div>`);
    parts.push(`<div class="card"><p class="notice">${esc(r.disclaimer)}</p>
      <div class="actions no-print"><button type="button" class="btn btn-ghost" id="btn-restart">Start again</button>
      <button type="button" class="btn btn-primary" id="btn-print">Print / save as PDF</button></div></div>`);

    const box = $("#results");
    box.innerHTML = parts.join("");
    $$(".bar[data-w]", box).forEach((b) => { b.style.width = `${Math.max(4, Number(b.dataset.w))}%`; });
    if (m) placeGauge(m);
    $("#wizard").classList.add("hidden");
    box.classList.remove("hidden");
    box.focus({ preventScroll: true });
    window.scrollTo(0, 0);
    $("#btn-print").addEventListener("click", () => window.print());
    $("#btn-restart").addEventListener("click", restart);
  }

  function scoreCard(m, g) {
    const caveats = [];
    if (g.pathway === "adolescent") caveats.push("Few teenagers were in the training data, so at your stage this score is less reliable and the guideline check leads.");
    if (m.outside_training_range && m.outside_training_range.length) caveats.push(`Some answers (${m.outside_training_range.map(esc).join(", ")}) were outside the range of the training data, so the nearest value inside it was used.`);
    return `<h3>Statistical score</h3>
      <div class="prob">${pct(m.probability)}</div>
      <p class="muted small">In the clinical reference data, about ${pct(m.probability)} of women whose answers matched yours had PCOS. With no symptoms at all, this would be about ${pct(m.reference_probability)}.</p>
      <div class="gauge" role="img" aria-label="Score ${pct(m.probability)}; low below ${pct(m.thresholds.low)}, high from ${pct(m.thresholds.high)}">
        <div class="gauge-marker-wrap"><div class="gauge-marker" id="gauge-marker"></div></div>
        <div class="gauge-track"><div class="gauge-zone z-low" id="z-low"></div><div class="gauge-zone z-mod" id="z-mod"></div><div class="gauge-zone z-high" id="z-high"></div></div>
        <div class="gauge-labels"><span>Low</span><span>Moderate</span><span>High</span></div>
      </div>
      <p class="muted small">Model band: <strong>${esc(m.band)}</strong>. This score supports the guideline check; it does not replace it.</p>
      ${caveats.map((c) => `<p class="small notice">${c}</p>`).join("")}`;
  }

  function placeGauge(m) {
    const lo = m.thresholds.low, hi = m.thresholds.high;
    $("#z-low").style.width = `${100 * lo}%`;
    $("#z-mod").style.width = `${100 * (hi - lo)}%`;
    $("#z-high").style.flex = "1";
    $("#gauge-marker").style.left = `${Math.min(99.5, Math.max(0.5, 100 * m.probability))}%`;
  }

  function restart() {
    $("#wizard").reset();
    $("#results").classList.add("hidden");
    $("#results").innerHTML = "";
    $("#wizard").classList.remove("hidden");
    $("#bmi-preview").textContent = "";
    showStep(1);
  }

  // ------------------------------------------------------------------ model page
  let modelLoaded = false;
  const SOURCE_LABEL = { kerala: "Kerala clinics (clinical diagnosis)", survey_in_2018: "India survey 2018", survey_lifestyle: "Lifestyle survey", survey_mumbai: "Mumbai survey" };
  const MODEL_LABEL = { logreg: "Logistic regression", random_forest: "Random forest", hist_gb: "HistGB", lightgbm: "LightGBM", xgboost: "XGBoost", ensemble: "Ensemble" };
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));

  function bars(rows, cls = "", min = 0.5) {
    return `<div class="bars">${rows.map((r) => `<div class="b"><span>${esc(r.label)}</span>
      <div class="track"><div class="fill ${r.cls || cls}" data-w="${(100 * (r.value - min)) / (1 - min)}"></div></div>
      <span class="val">${fmt(r.value)}</span></div>`).join("")}</div>`;
  }

  async function loadModelPage() {
    if (modelLoaded) return;
    const box = $("#model-content");
    try {
      const res = await fetch("/api/v1/model");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const info = await res.json();
      box.innerHTML = renderModel(info);
      $$(".fill[data-w]", box).forEach((f) => { f.style.width = `${Math.max(0, Math.min(100, Number(f.dataset.w)))}%`; });
      modelLoaded = true;
    } catch (err) {
      box.innerHTML = `<p class="error-box">Could not load model information (${esc(err.message)}).</p>`;
    }
  }

  function renderModel(info) {
    const mt = info.metrics || {}, model = info.model;
    const out = [];
    out.push(`<div class="card"><h2>How your result is decided</h2>
      <ol class="clean">
        <li><strong>Guideline check.</strong> The 2023 International Evidence-based Guideline rules for irregular cycles (adjusted for years since the first period) and clinical signs of excess androgen.</li>
        <li><strong>Statistical score.</strong> A machine-learning model, checked to be well calibrated, turns your answers into a probability.</li>
        <li><strong>Combination.</strong> Both clinical features always mean “high”. For teens, the guideline leads. For adults with neither feature, the statistical score is capped at “moderate”, because PCOS then needs lab or ultrasound findings a questionnaire can't see.</li>
      </ol>
      <p class="muted small">Mode: ${esc(info.mode)}${model ? ` · model: ${esc(MODEL_LABEL[model.name] || model.name)} v${esc(model.version)} (trained ${esc((model.created || "").slice(0, 10))}, calibration: ${esc(model.calibration)})` : ""}</p></div>`);
    if (!mt.final_oof) { out.push(`<p class="muted">Detailed metrics are not available.</p>`); return out.join(""); }
    const fo = mt.final_oof, th = mt.thresholds || {};
    const losoVals = Object.values(mt.loso || {}).map((v) => v.roc_auc).filter((v) => v !== null);
    const losoMean = losoVals.length ? losoVals.reduce((a, b) => a + b, 0) / losoVals.length : null;
    out.push(`<div class="grid-3">
      <div class="stat kpi"><span class="lbl">Accuracy (ROC-AUC, cross-validated)</span><span class="num">${fmt(fo.roc_auc)}</span><span class="muted small">95% CI ${fmt(fo.roc_auc_ci[0])}–${fmt(fo.roc_auc_ci[1])}</span></div>
      <div class="stat kpi"><span class="lbl">On a never-seen population (mean)</span><span class="num">${fmt(losoMean)}</span><span class="muted small">leave-one-source-out AUC</span></div>
      <div class="stat kpi"><span class="lbl">PCOS cases flagged moderate or high</span><span class="num">${pct(th.low_sensitivity || 0)}</span><span class="muted small">sensitivity at the “low” cut-off</span></div>
    </div>`);
    const src = (mt.data && mt.data.by_source) || {};
    out.push(`<div class="card"><h2>Training data</h2>
      <p>${mt.data.n} people from ${Object.keys(src).length} independent Indian sources. One is a clinical cohort from 10 hospitals in Kerala (doctor-confirmed diagnosis); three are questionnaire surveys (self-reported diagnosis). No synthetic data was used.</p>
      <div class="table-scroll"><table class="data"><thead><tr><th>Source</th><th class="num">People</th><th class="num">With PCOS</th><th class="num">AUC when held out</th></tr></thead><tbody>
      ${Object.entries(src).map(([s, v]) => `<tr><td>${esc(SOURCE_LABEL[s] || s)}</td><td class="num">${v.n}</td><td class="num">${pct(v.prevalence, 1)}</td><td class="num">${fmt((mt.loso[s] || {}).roc_auc)}</td></tr>`).join("")}
      </tbody></table></div>
      <p class="muted small">“Held out” means the model was trained on the other three sources and tested on this one. That is the most honest estimate of how it performs on a new population.</p></div>`);
    if (mt.models) {
      const rows = Object.entries(mt.models).map(([k, v]) => ({ label: `${MODEL_LABEL[k] || k}${k === mt.selected_model ? " ★" : ""}`, value: v.loso_auc }));
      const chosen = MODEL_LABEL[mt.selected_model] || mt.selected_model;
      out.push(`<div class="card"><h2>Models compared</h2><p class="muted small">Six model types were tuned and tested the same way. Bars show mean AUC on a population the model never saw; the scale starts at 0.5 (a coin toss). ★ <strong>${esc(chosen)}</strong> was chosen as the simplest, most explainable model whose accuracy on unseen populations is within 0.01 of the best.</p>${bars(rows)}</div>`);
    }
    if (mt.odds_ratios) {
      out.push(`<div class="card"><h2>Which answers matter most</h2><div class="table-scroll"><table class="data"><thead><tr><th>Answer</th><th class="num">Odds ratio</th><th>Per</th></tr></thead><tbody>
        ${mt.odds_ratios.slice().sort((a, b) => b.odds_ratio - a.odds_ratio).map((r) => `<tr><td>${esc(r.label)}</td><td class="num">${fmt(r.odds_ratio, 2)}</td><td class="muted small">${esc(r.unit)}</td></tr>`).join("")}
        </tbody></table></div><p class="muted small">An odds ratio of 2 means the answer roughly doubles the odds of PCOS, with the other answers held equal.</p></div>`);
    } else if (mt.importance) {
      const rows = mt.importance.map((r) => ({ label: r.label, value: r.importance }));
      const max = Math.max(...rows.map((r) => r.value), 0.001);
      out.push(`<div class="card"><h2>Which answers matter most</h2><p class="muted small">How much accuracy (AUC) drops when an answer is shuffled.</p>
        <div class="bars">${rows.map((r) => `<div class="b"><span>${esc(r.label)}</span><div class="track"><div class="fill" data-w="${(100 * Math.max(0, r.value)) / max}"></div></div><span class="val">${fmt(r.value)}</span></div>`).join("")}</div></div>`);
    }
    if (mt.clinical_benchmark && mt.clinical_benchmark.length) {
      const sets = [...new Set(mt.clinical_benchmark.map((r) => r.feature_set))];
      const nice = { questionnaire: "Questionnaire only", "questionnaire+anthropometry": "+ body measurements, BP", "questionnaire+anthropometry+labs+ultrasound": "+ blood tests & ultrasound" };
      const best = sets.map((s) => {
        const rs = mt.clinical_benchmark.filter((r) => r.feature_set === s);
        return rs.reduce((a, b) => (b.auc_mean > a.auc_mean ? b : a));
      });
      out.push(`<div class="card"><h2>What a doctor's tests add</h2><p class="muted small">Kerala clinical cohort, best model per input set, cross-validated AUC.</p>
        ${bars(best.map((r) => ({ label: nice[r.feature_set] || r.feature_set, value: r.auc_mean })))}
        <p class="small spaced">A questionnaire is a good first filter. Blood tests and ultrasound are what confirm PCOS, which is why a high result should lead to a clinic visit.</p></div>`);
    }
    out.push(`<div class="card"><h2>Limitations</h2><ul class="clean">
      <li>All the training data comes from India. Accuracy elsewhere is not yet known.</li>
      <li>Survey diagnoses are self-reported, and people already diagnosed may notice symptoms more.</li>
      <li>Few training records are from girls under 18. For them, the guideline check leads.</li>
      <li>Excess hair is self-rated. Doctors use a standard scoring chart (modified Ferriman-Gallwey).</li>
      <li>Lifestyle answers (fast food, exercise, tiredness, mood) did not predict PCOS consistently across datasets, so they shape advice only.</li>
      <li>This tool has not been tested in a prospective clinical study and is not a medical device.</li></ul></div>`);
    return out.join("");
  }

  // ------------------------------------------------------------------ init
  async function init() {
    buildInputs();
    $("#theme-toggle").addEventListener("click", toggleTheme);
    window.addEventListener("hashchange", route);
    route();
    showStep(1);
    $("#btn-next").addEventListener("click", () => { if (validateStep(state.step)) showStep(state.step + 1); });
    $("#btn-back").addEventListener("click", () => showStep(Math.max(1, state.step - 1)));
    $("#wizard").addEventListener("submit", submit);
    $("#wizard").addEventListener("change", updateConditional);
    $("#wizard").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && e.target.tagName === "INPUT" && e.target.type !== "checkbox") {
        e.preventDefault();
        if (state.step < state.steps) $("#btn-next").click(); else $("#btn-submit").click();
      }
    });
    ["height_cm", "height_ft", "height_in", "weight_val"].forEach((id) => $(`#${id}`).addEventListener("input", updateBmiPreview));
    $$("[data-unit]").forEach((b) => b.addEventListener("click", () => {
      const kind = b.dataset.unit, val = b.dataset.val;
      state.units[kind] = val;
      $$(`[data-unit="${kind}"]`).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      if (kind === "height") { $("#height-metric").classList.toggle("hidden", val !== "cm"); $("#height-imperial").classList.toggle("hidden", val === "cm"); }
      if (kind === "weight") $("#weight-unit-label").textContent = val;
      updateBmiPreview();
    }));
    try {
      const res = await fetch("/api/v1/health");
      state.health = await res.json();
      $("#q-store").classList.toggle("hidden", !state.health.storage_enabled);
    } catch (_) { /* offline: the form still works, submit will report the error */ }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
