import "./style.css";

const BASE = import.meta.env.BASE_URL;
const reduce = matchMedia("(prefers-reduced-motion: reduce)");
const $ = (id) => document.getElementById(id);

// Same join as the engine's assemble.py (SYLL_CROSSFADE = 12 ms).
const CROSSFADE = 0.012;
const LEAD_IN = 0.05;

const FONTS = ["f-archivo", "f-bebas", "f-serif", "f-serif-i", "f-mono", "f-elite", "f-inter"];
const PAPERS = ["p-news", "p-white", "p-yellow", "p-pink", "p-black", "p-blue", "p-kraft"];
const pick = (a) => a[Math.floor(Math.random() * a.length)];
const rot = (max) => (Math.random() * 2 - 1) * max;

/* ---------- ransom-note title ---------- */
function cutTitle() {
  const h = $("title");
  const word = h.getAttribute("aria-label").toUpperCase();
  h.replaceChildren(
    ...[...word].map((ch, i) => {
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
$("say-name").addEventListener("click", () => {
  const e = entries.find((x) => x.text === "mouthpiece");
  if (!e) return;
  cutTitle();
  play(e);
  document.getElementById("demo").scrollIntoView({ behavior: reduce.matches ? "auto" : "smooth", block: "start" });
});

/* ---------- demo ---------- */
let entries = [];
let ctx = null;
const buffers = new Map();
let current = null; // { entry, segs:[{buf,start,dur,seg}], total, t0 }
let stopAll = () => {};

// archive.org metadata sometimes holds an uploader address in the speaker field.
const sourceName = (s) => (!s ? "unknown" : /@nasa\.gov$/i.test(s) ? "NASA audio" : s);

function audio() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  return ctx;
}
async function load(file) {
  if (buffers.has(file)) return buffers.get(file);
  const res = await fetch(BASE + "media/" + file);
  const buf = await audio().decodeAudioData(await res.arrayBuffer());
  buffers.set(file, buf);
  return buf;
}

fetch(BASE + "media/manifest.json")
  .then((r) => r.json())
  .then((d) => {
    entries = d;
    renderChips();
  })
  .catch(() => {
    $("tape-label").textContent = "demo clips failed to load";
  });

function renderChips() {
  const q = $("q").value.trim().toLowerCase();
  const shown = entries.filter((e) => !q || e.text.includes(q) || q.includes(e.text));
  $("chips").replaceChildren(
    ...shown.map((e) => {
      const b = document.createElement("button");
      b.type = "button";
      b.role = "listitem";
      b.className = "chip" + (current?.entry === e ? " on" : "");
      b.textContent = e.text;
      b.addEventListener("click", () => play(e));
      return b;
    }),
  );
  const miss = $("miss");
  miss.hidden = shown.length > 0;
  miss.textContent = `“${q}” isn't in the demo set. The real engine builds almost any word from its corpus, down to single phones (tier 04).`;
}
$("q").addEventListener("input", renderChips);
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter") playQuery();
});
$("go").addEventListener("click", playQuery);
function playQuery() {
  const q = $("q").value.trim().toLowerCase();
  const e = entries.find((x) => x.text === q) || entries.find((x) => !q || x.text.includes(q) || q.includes(x.text));
  if (e) play(e);
}

async function play(entry) {
  stopAll();
  const ac = audio();
  await ac.resume();
  $("tape-label").textContent = "cutting…";
  let bufs;
  try {
    bufs = await Promise.all(entry.segments.map((s) => load(s.audioFile)));
  } catch {
    $("tape-label").textContent = "couldn't load the clips";
    return;
  }
  let t = ac.currentTime + LEAD_IN;
  const t0 = t;
  const nodes = [];
  const segs = bufs.map((buf, i) => {
    const src = ac.createBufferSource();
    const g = ac.createGain();
    src.buffer = buf;
    src.connect(g).connect(ac.destination);
    const dur = buf.duration;
    const xf = Math.min(CROSSFADE, dur / 4);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(1, t + xf);
    g.gain.setValueAtTime(1, t + dur - xf);
    g.gain.linearRampToValueAtTime(0, t + dur);
    src.start(t);
    nodes.push(src);
    const seg = { buf, start: t - t0, dur, seg: entry.segments[i] };
    t += dur - xf;
    return seg;
  });
  const total = t + CROSSFADE - t0;
  current = { entry, segs, total, t0 };
  let stopped = false;
  stopAll = () => {
    if (stopped) return;
    stopped = true;
    nodes.forEach((n) => {
      try { n.stop(); } catch {}
    });
  };
  renderChips();
  renderScraps(entry);
  renderProv(entry);
  $("tape-label").textContent = `${entry.text} · ${entry.segments.length} fragment${entry.segments.length > 1 ? "s" : ""} · ${(total * 1000).toFixed(0)} ms`;
  loop();
}

function playOne(i) {
  if (!current) return;
  stopAll();
  const ac = audio();
  const s = current.segs[i];
  const src = ac.createBufferSource();
  src.buffer = s.buf;
  src.connect(ac.destination);
  const t = ac.currentTime + 0.02;
  src.start(t);
  stopAll = () => {
    try { src.stop(); } catch {}
  };
  current.solo = { i, t0: t, dur: s.dur };
  loop();
}

function renderScraps(entry) {
  const ol = $("scraps");
  ol.replaceChildren(
    ...entry.segments.map((s, i) => {
      const li = document.createElement("li");
      li.className = `scrap ${PAPERS[i % PAPERS.length]}` + (s.natural ? " natural" : "");
      li.style.setProperty("--r", rot(4).toFixed(1) + "deg");
      li.style.setProperty("--d", i * 70 + "ms");
      const b = document.createElement("button");
      b.type = "button";
      b.className = "lbl " + FONTS[(i * 3) % FONTS.length];
      b.textContent = s.label;
      b.setAttribute("aria-label", `Play fragment ${s.label} alone`);
      b.addEventListener("click", () => playOne(i));
      const from = document.createElement("span");
      from.className = "from mono";
      from.textContent = s.parentWord && s.parentWord !== s.label ? `cut from “${s.parentWord}”` : "whole word";
      const src = document.createElement(s.sourceUrl ? "a" : "span");
      src.className = "who mono";
      src.textContent = sourceName(s.speaker);
      if (s.sourceUrl) src.href = s.sourceUrl;
      li.append(b, from, src);
      if (s.natural) {
        const n = document.createElement("span");
        n.className = "nat mono";
        n.textContent = "natural run";
        n.title = "More than one unit taken as one continuous span of a real recording";
        li.append(n);
      }
      return li;
    }),
  );
}

function renderProv(entry) {
  const words = entry.segments.map((s) => ({
    unit: s.label,
    cut_from: s.parentWord,
    source: sourceName(s.speaker),
    source_url: s.sourceUrl,
    natural_run: s.natural,
    ms: s.durationMs,
  }));
  $("prov").textContent = JSON.stringify({ text: entry.text, fragments: words }, null, 2);
}

/* ---------- tape: real waveforms of each fragment, laid end to end ---------- */
const canvas = $("wave");
const g2 = canvas.getContext("2d");
// Same order as PAPERS, so each fragment's waveform matches its scrap (black paper drawn grey).
const COLORS = ["#e9e1cc", "#fbf8f1", "#ffd84a", "#ff7eb0", "#8a8070", "#8fd3ff", "#c9a36b"];
let raf = 0;

function size() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const r = canvas.getBoundingClientRect();
  canvas.width = r.width * dpr;
  canvas.height = r.height * dpr;
  g2.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
new ResizeObserver(size).observe(canvas);

function draw() {
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  g2.clearRect(0, 0, W, H);
  // idle: a flat line of tape
  g2.strokeStyle = "rgba(242,234,216,0.25)";
  g2.lineWidth = 1;
  g2.beginPath();
  g2.moveTo(0, H / 2);
  g2.lineTo(W, H / 2);
  g2.stroke();
  if (!current) return;
  const { segs, total } = current;
  const pad = 8;
  const sx = (W - pad * 2) / total;
  const now = ctx ? ctx.currentTime : 0;
  const head = current.solo ? null : now - current.t0;
  segs.forEach((s, i) => {
    const x0 = pad + s.start * sx;
    const w = s.dur * sx;
    const data = s.buf.getChannelData(0);
    const cols = Math.max(1, Math.floor(w));
    const per = data.length / cols;
    const on = current.solo ? current.solo.i === i : head != null && head >= s.start && head < s.start + s.dur;
    g2.fillStyle = COLORS[i % COLORS.length];
    g2.globalAlpha = on ? 1 : 0.55;
    for (let c = 0; c < cols; c++) {
      let mn = 1, mx = -1;
      const a = Math.floor(c * per), b = Math.floor((c + 1) * per);
      for (let k = a; k < b; k += 4) {
        const v = data[k];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      const y0 = H / 2 - mx * H * 0.45;
      const y1 = H / 2 - mn * H * 0.45;
      g2.fillRect(x0 + c, y0, 1, Math.max(1, y1 - y0));
    }
    // cut marks
    g2.globalAlpha = 0.8;
    g2.fillStyle = "#d12a6b";
    if (i > 0) g2.fillRect(x0, 4, 2, H - 8);
  });
  g2.globalAlpha = 1;
  let px = null;
  if (current.solo) {
    const s = segs[current.solo.i];
    const p = (now - current.solo.t0) / current.solo.dur;
    if (p >= 0 && p <= 1) px = pad + (s.start + p * s.dur) * sx;
  } else if (head >= 0 && head <= total) px = pad + head * sx;
  if (px != null) {
    g2.fillStyle = "#fff";
    g2.fillRect(px, 0, 2, H);
  }
  document.querySelectorAll("#scraps .scrap").forEach((li, i) => {
    const s = segs[i];
    const on = current.solo ? current.solo.i === i && px != null : head >= s.start && head < s.start + s.dur;
    li.classList.toggle("playing", !!on);
  });
}

function loop() {
  cancelAnimationFrame(raf);
  const tick = () => {
    draw();
    const now = ctx.currentTime;
    const done = current.solo ? now > current.solo.t0 + current.solo.dur : now - current.t0 > current.total;
    if (done) {
      if (current.solo) current.solo = null;
      draw();
      return;
    }
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

/* ---------- reveal ---------- */
if (!reduce.matches && "IntersectionObserver" in window) {
  const io = new IntersectionObserver(
    (es) => es.forEach((en) => en.isIntersecting && (en.target.classList.add("in"), io.unobserve(en.target))),
    { threshold: 0.1 },
  );
  document.querySelectorAll(".sec-head, .tiers > li, .stats, .manifest, .run > li").forEach((el) => (el.classList.add("rv"), io.observe(el)));
}
