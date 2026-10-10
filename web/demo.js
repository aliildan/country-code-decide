// CountryCode-Decide browser demo: the release bundle (int8 ONNX + tokenizer + calibration)
// served from ./model/, run with onnxruntime-web; tokenisation with transformers.js.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.min.mjs";
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js";

env.allowRemoteModels = false;
env.allowLocalModels = true;  // in a browser transformers.js defaults to false: the tokenizer would never load
// A path, not a full URL: transformers.js 4.2 skips its local-file check for http(s) URLs.
env.localModelPath = new URL("./", import.meta.url).pathname;
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/";

const EXAMPLES = [
  ["Deleware", "jurisdiction", "US"], ["Delawqre", "jurisdiction", "US"],
  ["England and Whales", "country", "unknown"], ["BRITHISH VIRGIN ISLANDS", "jurisdiction", "US"],
  ["CANANDA", "jurisdiction", "US"], ["Untied Kingdom", "country", "unknown"],
  ["Østerrike", "country", "NO"], ["Норвегия", "country", "unknown"], ["10250 THAILAND", "country", "US"],
  ["CA", "jurisdiction", "US"], ["CA", "country", "US"], ["Georgia", "unknown", "unknown"],
  ["Congo", "country", "unknown"], ["n/a", "country", "unknown"], ["Jugoslavija", "nationality", "unknown"],
  ["🇦🇹", "country", "unknown"],
];

const $ = (id) => document.getElementById(id);
let tok, session, meta, cal;

function softmax(a, t) {
  const m = Math.max(...a.map((x) => x / t));
  const e = a.map((x) => Math.exp(x / t - m));
  const s = e.reduce((p, q) => p + q, 0);
  return e.map((x) => x / s);
}

function combine(pc, ps) {
  const ci = Object.fromEntries(meta.country_classes.map((c, i) => [c, i]));
  const si = Object.fromEntries(meta.state_classes.map((c, i) => [c, i]));
  const pUS = pc[ci.US];
  return meta.targets.map((t) => {
    if (t.startsWith("US-")) return pUS * ps[si[t]];
    if (t === "US") return pUS * ps[si.NO_STATE];
    return pc[ci[t]];
  });
}

async function run() {
  const text = $("text").value;
  const field = $("field").value;
  const reg = $("register").value;
  const alpha = $("alpha").value;
  const ctx = `field: ${field} | register: ${reg}`;
  const enc = await tok(ctx, { text_pair: text.trim() ? text : "∅", truncation: true, max_length: meta.max_len });
  const ids = Array.from(enc.input_ids.data, Number);
  const feeds = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    attention_mask: new ort.Tensor("int64", BigInt64Array.from(ids.map(() => 1n)), [1, ids.length]),
  };
  const t0 = performance.now();
  const out = await session.run(feeds);
  const ms = performance.now() - t0;
  const p = combine(softmax(Array.from(out.country_logits.data), cal.temps[0]),
                    softmax(Array.from(out.state_logits.data), cal.temps[1]));
  const order = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  const [conf, j] = order[0];
  const lam = alpha === "none" ? 0 : cal.thresholds[alpha];
  const answer = lam !== null && lam !== undefined && conf >= lam;
  const cands = [];
  if (cal.candidate_rule === "min_p") {  // v1.2: codes with p ≥ min_p, at most max, never empty
    for (const [v, i] of order.slice(0, cal.candidate_max)) if (v >= cal.candidate_min_p) cands.push(meta.targets[i]);
    if (!cands.length) cands.push(meta.targets[j]);
  } else {  // v1/v1.1 conformal set
    let mass = 0;
    for (const [v, i] of order.slice(0, 5)) { cands.push(meta.targets[i]); mass += v; if (mass >= cal.conformal_q) break; }
  }
  render(answer ? meta.targets[j] : null, conf, cands, ms, lam);
}

function render(target, conf, cands, ms, lam) {
  const r = $("result");
  r.replaceChildren();
  const head = document.createElement("div");
  if (target) {
    head.innerHTML = `<span class="code"></span> <span class="name"></span>`;
    head.querySelector(".code").textContent = target;
    head.querySelector(".name").textContent = meta.names[target];
  } else {
    head.innerHTML = `<span class="code">—</span> <span class="name">not sure: sent for review</span>`;
  }
  const badge = document.createElement("span");
  badge.className = "badge " + (target ? "ok" : "warn");
  badge.textContent = target ? "answered" : (lam === null ? "this error rate is not certified" : "abstained");
  const meter = document.createElement("div");
  meter.className = "meter";
  meter.innerHTML = `<div style="width:${(100 * conf).toFixed(1)}%"></div>`;
  const info = document.createElement("div");
  info.className = "muted";
  info.textContent = `confidence ${(100 * conf).toFixed(1)} % · candidates: ${cands.map((c) => `${c} (${meta.names[c]})`).join(", ")} · ${ms.toFixed(1)} ms`;
  r.append(head, badge, meter, info);
}

async function main() {
  const base = new URL("./model/", import.meta.url);
  meta = await (await fetch(new URL("meta.json", base))).json();
  const cals = await (await fetch(new URL("calibration.json", base))).json();
  // The demo shows register-field strings: use the register policy when the bundle has one.
  cal = cals.domains ? (cals.domains.register || Object.values(cals.domains)[0]) : cals;
  tok = await AutoTokenizer.from_pretrained("model");
  session = await ort.InferenceSession.create(new URL("model.int8.onnx", base).href, { executionProviders: ["wasm"] });
  $("ver").textContent = meta.version || "dev";
  for (const [t, f, r] of EXAMPLES) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = `${t} · ${f}`;
    b.onclick = () => { $("text").value = t; $("field").value = f; $("register").value = r; run(); };
    $("examples").append(b);
  }
  for (const id of ["text", "field", "register", "alpha"]) $(id).addEventListener("input", run);
  await run();
}

main().catch((e) => { $("result").textContent = "Could not load the model: " + e.message; });
