// Assembly, following mouthpiece/assemble.py: 5 ms anti-click fades, a 12 ms
// crossfade between the fragments of one word, a gap between words, 400 ms of
// silence for a word that can't be built. Clips in the bank are already
// loudness-normalized (TARGET_RMS 0.08) by tools/export_bank.py.

export const SR = 16000;
const FADE = Math.round(0.005 * SR);
const SYLL_XF = Math.round(0.012 * SR);
const MISSING = Math.round(0.4 * SR);

const chunks = new Map();
let decoder = null;

function getDecoder() {
  // Decoding through a 16 kHz offline context keeps every sprite at the
  // corpus rate (a 45 s chunk is ~2.9 MB of floats instead of ~8.6 MB at 48 kHz).
  if (!decoder) decoder = new OfflineAudioContext(1, 1, SR);
  return decoder;
}

export function loadChunk(base, name, idx) {
  if (!chunks.has(idx)) {
    chunks.set(
      idx,
      fetch(base + name)
        .then((r) => {
          if (!r.ok) throw new Error(name + " " + r.status);
          return r.arrayBuffer();
        })
        .then((b) => getDecoder().decodeAudioData(b))
        .then((ab) => ab.getChannelData(0))
        .catch((e) => {
          chunks.delete(idx);
          throw e;
        }),
    );
  }
  return chunks.get(idx);
}

function fade(a) {
  const f = Math.min(FADE, Math.max(2, Math.floor(a.length / 10)));
  if (a.length < 2 * f) return a;
  for (let i = 0; i < f; i++) {
    const g = i / f;
    a[i] *= g;
    a[a.length - 1 - i] *= g;
  }
  return a;
}

function xfJoin(a, b) {
  const xf = Math.min(SYLL_XF, Math.floor(a.length / 8), Math.floor(b.length / 8));
  if (xf < 2) return concat([a, b]);
  const out = new Float32Array(a.length + b.length - xf);
  out.set(a.subarray(0, a.length - xf));
  for (let i = 0; i < xf; i++) {
    const r = 1 - i / (xf - 1);
    out[a.length - xf + i] = a[a.length - xf + i] * r + b[i] * (1 - r);
  }
  out.set(b.subarray(xf), a.length);
  return out;
}

function concat(parts) {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

// words: Engine.match() output. Returns samples + per-word/per-fragment timing.
export async function render(words, bank, base, { gapMs = 100 } = {}) {
  const need = new Set(words.flatMap((w) => w.clips.map((c) => c.chunk)));
  const data = new Map();
  await Promise.all([...need].map(async (i) => data.set(i, await loadChunk(base, bank.chunks[i], i))));
  const gap = new Float32Array(Math.round((gapMs / 1000) * SR));
  const parts = [];
  const timing = [];
  let t = 0;
  words.forEach((w, wi) => {
    if (wi > 0) {
      parts.push(gap);
      t += gap.length;
    }
    if (!w.clips.length) {
      parts.push(new Float32Array(MISSING));
      timing.push({ start: t, end: t + MISSING, frags: [] });
      t += MISSING;
      return;
    }
    // cut each fragment, then join fragments of one word with the 12 ms crossfade
    let joined = null;
    const frags = [];
    for (const c of w.clips) {
      const src = data.get(c.chunk);
      const a = Math.max(0, Math.round(c.off * SR));
      const b = Math.min(src.length, Math.round((c.off + c.len) * SR));
      const piece = fade(src.slice(a, b));
      const startInWord = joined ? joined.length - Math.min(SYLL_XF, Math.floor(joined.length / 8), Math.floor(piece.length / 8)) : 0;
      joined = joined ? xfJoin(joined, piece) : piece;
      frags.push({ start: t + Math.max(0, startInWord), end: t + Math.max(0, startInWord) + piece.length });
    }
    parts.push(joined);
    timing.push({ start: t, end: t + joined.length, frags });
    t += joined.length;
  });
  return { samples: concat(parts), timing };
}

export function wav(samples) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVEfmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, SR, true);
  v.setUint32(28, SR * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 0x7fff, true);
  return new Blob([buf], { type: "audio/wav" });
}
