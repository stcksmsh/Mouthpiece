import "./style.css";
import { Engine, parseCmu, tokenize } from "./engine.js";
import { render, wav, SR } from "./render.js";

const BASE = import.meta.env.BASE_URL + "bank/";
const reduce = matchMedia("(prefers-reduced-motion: reduce)");
const $ = (id) => document.getElementById(id);

const FONTS = ["f-archivo", "f-bebas", "f-serif", "f-serif-i", "f-mono", "f-elite", "f-inter"];
const PAPERS = ["p-news", "p-white", "p-yellow", "p-pink", "p-black", "p-blue", "p-kraft"];
const pick = (a) => a[Math.floor(Math.random() * a.length)];
const rot = (max) => (Math.random() * 2 - 1) * max;
const TIER_COLOR = { word: "#f2ead8", phrase: "#ffd84a", syllable: "#8fd3ff", phone: "#ff7eb0", sounded: "#c9a36b", missing: "#5a5046" };
const TIER_NAME = { word: "whole word", phrase: "phrase reuse", syllable: "syllable stitch", phone: "phoneme fallback", sounded: "sounded out", missing: "missing" };

/* ---------- ransom-note title ---------- */
function cutTitle() {
  const h = $("title");
  h.replaceChildren(
    ...[...h.getAttribute("aria-label").toUpperCase()].map((ch, i) => {
      const s = document.createElement("span");
      s.className = `ltr ${pick(FONTS)} ${pick(PAPERS)}`;
      s.setAttribute("aria-hidden", "true");
      s.textContent = ch;
      s.style.setProperty("--r", rot(7).toFixed(1) + "deg");
      s.style.setProperty("--y", rot(6).toFixed(0) + "px");
      s.style.setProperty("--d", i * 45 + "ms");
      return s;
    }),
  );
}
cutTitle();
$("title").addEventListener("click", cutTitle);

/* ---------- bank ---------- */
let engine = null;
let bankLoad = null;
function loadBank() {
  if (!bankLoad) {
    setStatus("loading the sound bank…");
    bankLoad = Promise.all([fetch(BASE + "index.json").then((r) => r.json()), fetch(BASE + "cmu.txt").then((r) => r.text())])
      .then(([bank, cmu]) => {
        engine = new Engine(bank, parseCmu(cmu));
        fillBankInfo(bank);
        updateCoverage();
        if (!result) setStatus("ready. press “Say it”");
        return engine;
      })
      .catch((e) => {
        bankLoad = null;
        setStatus("couldn't load the sound bank: " + e.message);
        throw e;
      });
  }
  return bankLoad;
}
const text = $("text");
const warm = () => loadBank().catch(() => {});
text.addEventListener("focus", warm, { once: true });
document.querySelector(".machine").addEventListener("pointerenter", warm, { once: true });

function setStatus(s) {
  $("status").textContent = s;
}

/* ---------- state ---------- */
let seed = 1;
let result = null; // { words, samples, timing, text }
let selected = null; // [wordIndex, fragIndex]
let ac = null;
let playing = null; // { src, t0, from, to }

const opts = () => ({ granularity: +$("gran").value, seed });

/* ---------- coverage (live, like `mouthpiece coverage`) ---------- */
let covTimer = 0;
text.addEventListener("input", () => {
  clearTimeout(covTimer);
  covTimer = setTimeout(updateCoverage, 200);
});
function updateCoverage() {
  const box = $("coverage");
  if (!engine) return;
  const words = engine.match(text.value, opts());
  box.replaceChildren(
    ...words.map((w) => {
      const s = document.createElement("span");
      s.className = "cov t-" + w.tier;
      s.textContent = w.token;
      s.title = TIER_NAME[w.tier] + (w.clips.length ? ` · ${w.clips.length} fragment${w.clips.length > 1 ? "s" : ""}` : "");
      return s;
    }),
  );
  const counts = {};
  words.forEach((w) => (counts[w.tier] = (counts[w.tier] || 0) + 1));
  box.setAttribute("aria-label", "Coverage: " + Object.entries(counts).map(([k, v]) => `${v} ${TIER_NAME[k]}`).join(", "));
}

/* ---------- say it ---------- */
let busy = 0;
async function say() {
  const value = text.value.trim();
  if (!tokenize(value).length) {
    setStatus("type some words first");
    return;
  }
  const my = ++busy;
  stop();
  try {
    await loadBank();
  } catch {
    return;
  }
  ac ??= new (window.AudioContext || window.webkitAudioContext)();
  ac.resume();
  setStatus("cutting…");
  const words = engine.match(value, opts());
  updateCoverage();
  let out;
  try {
    out = await render(words, engine.bank, BASE, { gapMs: +$("gap").value });
  } catch (e) {
    setStatus("couldn't load audio: " + e.message);
    return;
  }
  if (my !== busy) return; // a newer request superseded this one
  result = { words, ...out, text: value };
  selected = null;
  $("src-link")?.remove();
  const frags = words.reduce((n, w) => n + w.clips.length, 0);
  const sources = new Set(words.flatMap((w) => w.clips.map((c) => c.source))).size;
  setStatus(`${words.length} word${words.length === 1 ? "" : "s"} · ${frags} fragments · ${sources} recording${sources === 1 ? "" : "s"} · ${(out.samples.length / SR).toFixed(2)} s`);
  ["play", "dl-wav", "dl-man"].forEach((id) => ($(id).disabled = false));
  renderScraps();
  playRange(0, out.samples.length);
}
$("say").addEventListener("click", say);
text.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    say();
  }
});
document.querySelectorAll(".ex").forEach((b) =>
  b.addEventListener("click", () => {
    text.value = b.textContent;
    cutTitle();
    say();
  }),
);
$("reroll").addEventListener("click", () => {
  seed = (seed * 48271) % 2147483647 || 7;
  say();
});
$("gran").addEventListener("input", () => {
  clearTimeout(covTimer);
  covTimer = setTimeout(updateCoverage, 120);
});
$("gran").addEventListener("change", () => result && say());
$("gap").addEventListener("input", () => ($("gap-out").textContent = $("gap").value + " ms"));
$("gap").addEventListener("change", () => result && say());
$("play").addEventListener("click", () => result && playRange(0, result.samples.length));

/* ---------- playback ---------- */
function stop() {
  if (playing) {
    try {
      playing.src.stop();
    } catch {}
    playing = null;
  }
}
function playRange(from, to) {
  stop();
  const n = to - from;
  if (n <= 0) return;
  const buf = ac.createBuffer(1, n, SR);
  buf.copyToChannel(result.samples.subarray(from, to), 0);
  const src = ac.createBufferSource();
  src.buffer = buf;
  src.connect(ac.destination);
  const t0 = ac.currentTime + 0.03;
  src.start(t0);
  playing = { src, t0, from, to };
  src.onended = () => {
    if (playing?.src === src) playing = null;
  };
  loop();
}

/* ---------- download ---------- */
function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
const slug = () => tokenize(result.text).slice(0, 6).join("-").replace(/'/g, "") || "render";
$("dl-wav").addEventListener("click", () => download(wav(result.samples), `mouthpiece-${slug()}.wav`));
$("dl-man").addEventListener("click", () => download(new Blob([JSON.stringify(manifest(), null, 2)], { type: "application/json" }), `mouthpiece-${slug()}.manifest.json`));

function manifest() {
  const src = (id) => engine.src(id);
  return {
    rendered_at: new Date().toISOString(),
    seed,
    text: result.text,
    output_wav: `mouthpiece-${slug()}.wav`,
    sample_rate: SR,
    note: "Rendered in the browser from a slice of the Mouthpiece index (kostavukicevic.com/Mouthpiece/).",
    words: result.words.map((w, i) => {
      const t = result.timing[i];
      if (!w.clips.length) return { token: w.token, found: false, gap_s: 0.4 };
      return {
        token: w.token,
        found: true,
        tier: w.tier,
        ...(w.via ? { pronounced_via: w.via } : {}),
        start_s: +(t.start / SR).toFixed(3),
        end_s: +(t.end / SR).toFixed(3),
        fragments: w.clips.map((c) => ({
          unit: c.unit,
          cut_from: c.cutFrom,
          natural_run: !!c.natural,
          source: src(c.source).label,
          date: src(c.source).date || null,
          source_url: src(c.source).url,
        })),
      };
    }),
  };
}

/* ---------- scraps: every fragment, clickable ---------- */
function renderScraps() {
  const ol = $("scraps");
  ol.replaceChildren();
  let k = 0;
  result.words.forEach((w, wi) => {
    const li = document.createElement("li");
    li.className = "wgroup";
    const head = document.createElement("span");
    head.className = "wtok mono t-" + w.tier;
    head.textContent = w.token + (w.via && w.via !== "compound" && w.via !== "sounded out" ? ` (said as “${w.via}”)` : "");
    li.append(head);
    const row = document.createElement("div");
    row.className = "wfrags";
    if (!w.clips.length) {
      const m = document.createElement("span");
      m.className = "scrap missing mono";
      m.textContent = "400 ms of silence";
      row.append(m);
    }
    w.clips.forEach((c, ci) => {
      const b = document.createElement("button");
      b.type = "button";
      // paper colour = how the word was built (legend); whole words alternate two paper tones
      const paper = { word: k % 2 ? "p-white" : "p-news", phrase: "p-yellow", syllable: "p-blue", phone: "p-pink", sounded: "p-kraft" }[w.tier];
      b.className = `scrap ${paper}`;
      k++;
      b.style.setProperty("--r", rot(4).toFixed(1) + "deg");
      b.style.setProperty("--d", Math.min(k * 40, 900) + "ms");
      b.dataset.w = wi;
      b.dataset.f = ci;
      const s = engine.src(c.source);
      b.setAttribute("aria-label", `${c.unit}, cut from ${c.cutFrom}, ${s.label}. Play alone`);
      b.innerHTML = `<b class="lbl"></b><span class="from mono"></span><span class="who mono"></span>`;
      b.querySelector(".lbl").classList.add(FONTS[(k * 3) % FONTS.length]);
      b.querySelector(".lbl").textContent = c.unit;
      b.querySelector(".from").textContent = c.cutFrom === c.unit ? "whole" : `from “${c.cutFrom}”`;
      b.querySelector(".who").textContent = s.label;
      if (c.natural && c.kind !== "word") b.classList.add("natural");
      b.addEventListener("click", () => select(wi, ci, true));
      row.append(b);
    });
    li.append(row);
    ol.append(li);
  });
}

function select(wi, ci, play) {
  selected = [wi, ci];
  const f = result.timing[wi].frags[ci];
  document.querySelectorAll(".scrap.sel").forEach((x) => x.classList.remove("sel"));
  document.querySelector(`.scrap[data-w="${wi}"][data-f="${ci}"]`)?.classList.add("sel");
  const c = result.words[wi].clips[ci];
  const s = engine.src(c.source);
  setStatus(`“${c.unit}” · cut from “${c.cutFrom}” · ${s.label}${s.date ? " · " + s.date : ""}`);
  let a = $("src-link");
  if (!a) {
    a = document.createElement("a");
    a.id = "src-link";
    a.className = "src-link mono";
    $("status").after(a);
  }
  a.hidden = !s.url;
  a.href = s.url || "#";
  a.textContent = "source ↗";
  if (play) playRange(f.start, f.end);
  draw();
}

/* ---------- tape ---------- */
const canvas = $("wave");
const g = canvas.getContext("2d");
let raf = 0;
new ResizeObserver(() => {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}).observe(canvas);

function geometry() {
  const W = canvas.clientWidth;
  const pad = 10;
  return { W, H: canvas.clientHeight, pad, sx: (W - pad * 2) / Math.max(1, result.samples.length) };
}

function draw() {
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  g.clearRect(0, 0, W, H);
  const mid = H / 2 + 10;
  g.strokeStyle = "rgba(242,234,216,0.2)";
  g.beginPath();
  g.moveTo(0, mid);
  g.lineTo(W, mid);
  g.stroke();
  if (!result) return;
  const { pad, sx } = geometry();
  const s = result.samples;
  const amp = (H - 34) * 0.5;
  const tierAt = new Array(Math.ceil(W)).fill(null);
  result.timing.forEach((t, i) => {
    for (let x = Math.floor(pad + t.start * sx); x < pad + t.end * sx; x++) tierAt[x] = result.words[i].tier;
  });
  const per = 1 / sx;
  for (let x = pad; x < W - pad; x++) {
    const a = Math.floor((x - pad) * per);
    const b = Math.min(s.length, Math.floor((x - pad + 1) * per));
    let mn = 0, mx = 0;
    for (let k = a; k < b; k += 2) {
      if (s[k] < mn) mn = s[k];
      if (s[k] > mx) mx = s[k];
    }
    g.fillStyle = TIER_COLOR[tierAt[x]] || "#5a5046";
    const top = Math.max(-1, Math.min(1, mx * 1.8));
    const bot = Math.max(-1, Math.min(1, mn * 1.8));
    g.fillRect(x, mid - top * amp, 1, Math.max(1, (top - bot) * amp));
  }
  g.font = "11px 'Space Mono', monospace";
  g.textBaseline = "top";
  let lastLabelEnd = -Infinity;
  result.timing.forEach((t, i) => {
    t.frags.forEach((f, j) => {
      if (j > 0) {
        g.fillStyle = "#d12a6b";
        g.fillRect(pad + f.start * sx, 26, 2, H - 32);
      }
    });
    const x = pad + t.start * sx;
    const label = result.words[i].token;
    if (x > lastLabelEnd + 6) {
      g.fillStyle = TIER_COLOR[result.words[i].tier];
      g.fillText(label, x, 6);
      lastLabelEnd = x + g.measureText(label).width;
    }
  });
  if (selected) {
    const f = result.timing[selected[0]].frags[selected[1]];
    g.strokeStyle = "#fff";
    g.lineWidth = 1.5;
    g.strokeRect(pad + f.start * sx, 24, (f.end - f.start) * sx, H - 28);
  }
  const head = playing ? playing.from + (ac.currentTime - playing.t0) * SR : -1;
  if (playing && head >= playing.from && head <= playing.to) {
    g.fillStyle = "#fff";
    g.fillRect(pad + head * sx, 22, 2, H - 24);
  }
  document.querySelectorAll(".scrap[data-w]").forEach((el) => {
    const f = result.timing[+el.dataset.w].frags[+el.dataset.f];
    el.classList.toggle("playing", head >= f.start && head < f.end);
  });
}
function loop() {
  cancelAnimationFrame(raf);
  const tick = () => {
    draw();
    if (playing) raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}
canvas.addEventListener("click", (e) => {
  if (!result) return;
  const { pad, sx } = geometry();
  const x = (e.offsetX - pad) / sx;
  result.timing.forEach((t, wi) => t.frags.forEach((f, ci) => x >= f.start && x < f.end && select(wi, ci, true)));
});

/* ---------- bank-driven sections ---------- */
function fillBankInfo(bank) {
  const n = (x) => x.toLocaleString("en-US");
  const st = $("bank-stats").querySelectorAll("b");
  st[0].textContent = n(Object.keys(bank.words).length);
  st[1].textContent = n(Object.keys(bank.phrases).length);
  st[2].textContent = n(Object.keys(bank.sylls).length);
  st[3].textContent = n(Object.keys(bank.phones).length);
  renderSources(bank);
}
function renderSources(bank) {
  const count = {};
  const add = (e) => (count[e[3]] = (count[e[3]] || 0) + 1);
  Object.values(bank.words).forEach(add);
  Object.values(bank.phrases).forEach(add);
  Object.values(bank.sylls).forEach((l) => l.forEach(add));
  Object.values(bank.phones).forEach((l) => l.forEach(add));
  const groups = new Map(); // identical labels (e.g. the NASA recordings) share a line
  for (const [id, s] of Object.entries(bank.sources)) {
    if (!groups.has(s.label)) groups.set(s.label, { label: s.label, items: [], clips: 0 });
    const gr = groups.get(s.label);
    gr.items.push(s);
    gr.clips += count[id] || 0;
  }
  $("sources").replaceChildren(
    ...[...groups.values()]
      .sort((a, b) => b.clips - a.clips)
      .map((gr) => {
        const li = document.createElement("li");
        const name = document.createElement("span");
        name.className = "sname";
        name.textContent = gr.label + (gr.items.length > 1 ? ` · ${gr.items.length} recordings` : "");
        const c = document.createElement("span");
        c.className = "scount mono";
        c.textContent = gr.clips.toLocaleString("en-US") + " clips";
        const links = document.createElement("span");
        links.className = "slinks mono";
        const urls = [...new Set(gr.items.map((s) => s.url).filter(Boolean))];
        urls.slice(0, 3).forEach((u, i) => {
          const a = document.createElement("a");
          a.href = u;
          a.textContent = urls.length > 1 ? `source ${i + 1}` : "source";
          links.append(a);
        });
        if (urls.length > 3) links.append(` +${urls.length - 3}`);
        li.append(name, c, links);
        return li;
      }),
  );
}
if ("IntersectionObserver" in window) {
  const io = new IntersectionObserver(([en]) => en.isIntersecting && (io.disconnect(), warm()));
  io.observe($("sources"));
}

/* ---------- reveal ---------- */
if (!reduce.matches && "IntersectionObserver" in window) {
  const io = new IntersectionObserver(
    (es) => es.forEach((en) => en.isIntersecting && (en.target.classList.add("in"), io.unobserve(en.target))),
    { threshold: 0.1 },
  );
  document.querySelectorAll(".sec-head, .tiers > li, .stats, .run > li").forEach((el) => (el.classList.add("rv"), io.observe(el)));
}
