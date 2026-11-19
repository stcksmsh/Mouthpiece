// Browser port of Mouthpiece's matcher (mouthpiece/match.py + phonemes.py),
// working against the exported sound bank (tools/export_bank.py) instead of
// the full SQLite index. Pure functions: no audio here, see render.js.

export const VOWELS = new Set(["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]);
const strip = (p) => p.replace(/\d$/, "");
const isVowel = (p) => /\d$/.test(p);
export const phoneKey = (phones) => phones.map(strip).join("-");

// phonemes.py _PHONE_WEIGHTS: rough relative durations used to split a word's
// window into syllables/phones.
const W = {
  B: 0.7, D: 0.7, G: 0.7, P: 0.7, T: 0.7, K: 0.7, CH: 0.9, JH: 0.9,
  F: 0.9, V: 0.8, TH: 0.9, DH: 0.7, S: 0.9, Z: 0.8, SH: 0.9, ZH: 0.8, HH: 0.7,
  M: 0.8, N: 0.8, NG: 0.8, L: 0.8, R: 0.8, W: 0.7, Y: 0.6,
  AA: 1.5, AE: 1.5, AH: 1.1, AO: 1.5, EH: 1.4, IH: 1.2, IY: 1.5, UH: 1.2, UW: 1.5, ER: 1.4,
  AY: 1.8, AW: 1.8, EY: 1.7, OW: 1.7, OY: 1.9,
};
const weight = (p) => W[strip(p)] ?? 1;

export const tokenize = (text) => (text.toLowerCase().match(/[a-z']+/g) || []).map((t) => t.replace(/^'+|'+$/g, "")).filter(Boolean);

export function parseCmu(text) {
  const m = new Map();
  for (const line of text.split("\n")) {
    const i = line.indexOf(" ");
    if (i > 0) m.set(line.slice(0, i), line.slice(i + 1).split(" "));
  }
  return m;
}

function suffixVariants(w) {
  const v = [];
  if (w.endsWith("ing") && w.length > 5) v.push(w.slice(0, -3), w.slice(0, -3) + "e");
  if (w.endsWith("tion") && w.length > 5) v.push(w.slice(0, -3) + "e");
  if (w.endsWith("ive") && w.length > 5) v.push(w.slice(0, -3) + "e");
  if (w.endsWith("ed") && w.length > 4) v.push(w.slice(0, -2), w.slice(0, -2) + "e");
  if (w.endsWith("er") && w.length > 4) v.push(w.slice(0, -2));
  if (w.endsWith("est") && w.length > 5) v.push(w.slice(0, -3));
  if (w.endsWith("ly") && w.length > 4) v.push(w.slice(0, -2));
  if (w.endsWith("s") && w.length > 3) v.push(w.slice(0, -1));
  return [...new Set(v)].filter((x) => x !== w);
}

function splitCompound(word, cmu, maxPieces, memo) {
  if (memo.has(word)) return memo.get(word);
  let result = null;
  if (maxPieces >= 2) {
    for (let i = 2; i < word.length - 1; i++) {
      const left = cmu.get(word.slice(0, i));
      if (!left) continue;
      const right = cmu.get(word.slice(i));
      if (right) { result = [...left, ...right]; break; }
      const rs = splitCompound(word.slice(i), cmu, maxPieces - 1, memo);
      if (rs) { result = [...left, ...rs]; break; }
    }
  }
  memo.set(word, result);
  return result;
}

// phonemes._cmu_entries: exact, then suffix-stripped, then compound split.
export function pronounce(word, cmu) {
  const w = word.toLowerCase();
  if (cmu.has(w)) return { phones: cmu.get(w), via: null };
  for (const v of suffixVariants(w)) if (cmu.has(v)) return { phones: cmu.get(v), via: v };
  if (w.length >= 4) {
    const c = splitCompound(w, cmu, 3, new Map());
    if (c) return { phones: c, via: "compound" };
  }
  return null;
}

// phonemes.syllabify: vowel nuclei, simplified maximal onset.
export function syllabify(phones) {
  const vi = phones.map((p, i) => (isVowel(p) ? i : -1)).filter((i) => i >= 0);
  if (!vi.length) return [phones.slice()];
  const s = vi.map((i) => [phones[i]]);
  s[0].unshift(...phones.slice(0, vi[0]));
  for (let n = 0; n < vi.length - 1; n++) {
    const run = phones.slice(vi[n] + 1, vi[n + 1]);
    if (!run.length) continue;
    if (run.length === 1) s[n + 1].unshift(run[0]);
    else { s[n].push(...run.slice(0, -1)); s[n + 1].unshift(run[run.length - 1]); }
  }
  s[s.length - 1].push(...phones.slice(vi[vi.length - 1] + 1));
  return s;
}

// Split a core duration across groups by phone weight (compute_syllable_rows).
function splitTimes(groups, dur) {
  const ws = groups.map((g) => g.reduce((a, p) => a + weight(p), 0));
  const tot = ws.reduce((a, b) => a + b, 0) || 1;
  let t = 0;
  return ws.map((w, i) => {
    const s = t;
    t = i === ws.length - 1 ? dur : t + (dur * w) / tot;
    return [s, t];
  });
}

export function mulberry(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SUB_PAD = 0.01;
const MAX_RUN = 6;

export class Engine {
  constructor(bank, cmu) {
    this.bank = bank;
    this.cmu = cmu;
    // Natural runs: contiguous syllable/phone sub-spans of whole words in the
    // bank, cut from that word's clip (the bank's analogue of match.py's
    // _find_natural_run over consecutive syllable/phone instances).
    this.runs = { syll: new Map(), phone: new Map() };
    for (const [word, e] of Object.entries(bank.words)) {
      const pr = cmu.get(word);
      if (!pr) continue;
      for (const [kind, groups] of [["syll", syllabify(pr)], ["phone", pr.map((p) => [p])]]) {
        const times = splitTimes(groups, e[5]);
        const keys = groups.map(phoneKey);
        for (let i = 0; i < keys.length; i++)
          for (let j = i + 2; j <= Math.min(keys.length, i + MAX_RUN); j++) {
            if (i === 0 && j === keys.length) continue; // that's the whole word
            const k = keys.slice(i, j).join("|");
            if (!this.runs[kind].has(k)) this.runs[kind].set(k, { word, e, from: times[i][0], to: times[j - 1][1], keys: keys.slice(i, j) });
          }
      }
    }
  }

  src(id) {
    return this.bank.sources[id] || { label: "unknown" };
  }

  // A clip descriptor: where in which sprite, and what it is.
  clip(e, extra) {
    return { chunk: e[0], off: e[1], len: e[2], source: e[3], ...extra };
  }

  wholeWord(token) {
    const e = this.bank.words[token];
    return e ? [this.clip(e, { unit: token, cutFrom: token, kind: "word" })] : null;
  }

  // _resolve_groups: tile the pronunciation into spans that exist in the bank,
  // preferring the largest span ending at each point.
  resolve(phones, table) {
    const n = phones.length;
    const reach = new Array(n + 1).fill(false);
    const back = new Array(n + 1).fill(null);
    reach[0] = true;
    for (let j = 1; j <= n; j++)
      for (let i = 0; i < j; i++)
        if (reach[i] && table[phoneKey(phones.slice(i, j))]) { reach[j] = true; back[j] = i; break; }
    if (!reach[n]) return null;
    const spans = [];
    for (let j = n; j > 0; j = back[j]) spans.unshift(phones.slice(back[j], j));
    return spans;
  }

  stitchAt(phones, kind, rng) {
    const table = kind === "syll" ? this.bank.sylls : this.bank.phones;
    const groups = this.resolve(phones, table);
    if (!groups) return null;
    const keys = groups.map(phoneKey);
    const picks = [];
    let segs = 0;
    for (let i = 0; i < keys.length; ) {
      let run = null;
      for (let j = Math.min(keys.length, i + MAX_RUN); j >= i + 2; j--) {
        run = this.runs[kind].get(keys.slice(i, j).join("|"));
        if (run) break;
      }
      if (run) {
        const [chunk, off, , source, pad] = run.e;
        const a = Math.max(0, pad + run.from - SUB_PAD);
        const b = pad + run.to + SUB_PAD;
        picks.push({ chunk, off: off + a, len: b - a, source, unit: run.keys.join(" · "), cutFrom: run.word, kind, natural: true });
        i += run.keys.length;
      } else {
        const opts = table[keys[i]];
        const e = opts[Math.floor(rng() * opts.length)];
        picks.push(this.clip(e, { unit: keys[i], cutFrom: e[4], kind }));
        i += 1;
      }
      segs++;
    }
    return { picks, segs };
  }

  // stitch_syllables: try both granularities, keep fewer splice points.
  stitch(phones, rng, forcePhones) {
    if (forcePhones) return this.stitchAt(phones, "phone", rng)?.picks || null;
    const s = this.stitchAt(phones, "syll", rng);
    const p = this.stitchAt(phones, "phone", rng);
    if (!s && !p) return null;
    if (!s) return p.picks;
    if (!p) return s.picks;
    return p.segs < s.segs ? p.picks : s.picks;
  }


  // match_text: phrase reuse (granularity > 0), whole word, stitch, spell, gap.
  match(text, { granularity = 0.5, seed = 1 } = {}) {
    const rng = mulberry(seed);
    const tokens = tokenize(text);
    const out = [];
    for (let i = 0; i < tokens.length; ) {
      if (granularity > 0 && rng() < granularity) {
        let hit = null;
        for (let n = Math.min(3, tokens.length - i); n >= 2; n--) {
          const k = tokens.slice(i, i + n).join(" ");
          if (this.bank.phrases[k]) { hit = [k, n]; break; }
        }
        if (hit) {
          const e = this.bank.phrases[hit[0]];
          out.push({ token: hit[0], tier: "phrase", clips: [this.clip(e, { unit: hit[0], cutFrom: hit[0], kind: "phrase", natural: true })] });
          i += hit[1];
          continue;
        }
      }
      const token = tokens[i];
      let res = null;
      if (granularity < 0 && rng() < Math.abs(granularity)) {
        const pr = pronounce(token, this.cmu);
        const st = pr && this.stitch(pr.phones, rng, granularity < -0.5);
        if (st) res = { tier: tierOf(st), clips: st, via: pr.via };
      }
      if (!res) {
        const w = this.wholeWord(token);
        if (w) res = { tier: "word", clips: w };
      }
      if (!res) {
        const pr = pronounce(token, this.cmu);
        const st = pr && this.stitch(pr.phones, rng, false);
        if (st) res = { tier: tierOf(st), clips: st, via: pr.via };
      }
      if (!res) {
        const ph = soundOut(token, this.cmu);
        const st = ph && this.stitch(ph, rng, granularity < -0.5);
        if (st) res = { tier: "sounded", clips: st, via: "sounded out" };
      }
      out.push({ token, ...(res || { tier: "missing", clips: [] }) });
      i += 1;
    }
    return out;
  }
}

function tierOf(clips) {
  return clips.some((c) => c.kind === "syll") ? "syllable" : "phone";
}

// Web-only fallback for words CMUdict can't place at all (the engine leaves a
// gap): sound the word out from known CMUdict pieces of 3+ letters, with
// single-letter rules in between, taking the split with the lowest cost.
const LETTER = {
  a: ["AE1"], b: ["B"], c: ["K"], d: ["D"], e: ["EH1"], f: ["F"], g: ["G"], h: ["HH"], i: ["IH1"],
  j: ["JH"], k: ["K"], l: ["L"], m: ["M"], n: ["N"], o: ["AA1"], p: ["P"], q: ["K"], r: ["R"],
  s: ["S"], t: ["T"], u: ["AH1"], v: ["V"], w: ["W"], x: ["K", "S"], y: ["IY1"], z: ["Z"],
};
export function soundOut(word, cmu) {
  const w = word.replace(/[^a-z]/g, "");
  const n = w.length;
  if (!n) return null;
  const cost = new Array(n + 1).fill(Infinity);
  const back = new Array(n + 1).fill(null);
  cost[0] = 0;
  for (let i = 0; i < n; i++) {
    if (cost[i] === Infinity) continue;
    if (LETTER[w[i]] && cost[i] + 2.5 < cost[i + 1]) { cost[i + 1] = cost[i] + 2.5; back[i + 1] = [i, LETTER[w[i]]]; }
    for (let j = i + 3; j <= Math.min(n, i + 12); j++) {
      const pr = cmu.get(w.slice(i, j));
      if (pr && cost[i] + 1 < cost[j]) { cost[j] = cost[i] + 1; back[j] = [i, pr]; }
    }
  }
  if (cost[n] === Infinity) return null;
  const phones = [];
  for (let j = n; j > 0; j = back[j][0]) phones.unshift(...back[j][1]);
  // collapse a doubled consonant from adjacent pieces ("ss" -> S)
  return phones.filter((p, i) => i === 0 || p !== phones[i - 1] || isVowel(p));
}
