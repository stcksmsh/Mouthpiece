# Mouthpiece — website branch

The source of Mouthpiece's website, <https://kostavukicevic.com/Mouthpiece/>. This is an orphan
branch: it shares no history with `master` and holds only the site.

- Plain Vite, no framework, one dev dependency. `npm install && npm run dev` to work on it,
  `npm run build` for `dist/`.
- `.github/workflows/deploy-website.yml` builds and deploys it to this repository's GitHub
  Pages on every push to `website` (Settings → Pages → Source: "GitHub Actions").
- Base path: `/Mouthpiece/` by default (`vite.config.js`). For a custom domain, set the
  repository variable `BASE_PATH` to `/`, add `public/CNAME`, and update `url` in
  `publication.json`.
- `publication.json` is the only file stcksmsh.github.io reads from this branch.
- Content comes from `master`'s README and the `io-page` branch's `page/` bundle (tiers, full-index
  stats).
- **The machine** runs Mouthpiece's matching rules in the browser: `src/engine.js` ports
  `match.py` + `phonemes.py` (CMUdict lookup with suffix/compound fallback, syllabify, the
  tiling DP, natural runs, syllable vs phone by fewest splices, phrase reuse and stitching by
  granularity); `src/render.js` follows `assemble.py` (5 ms fades, 12 ms crossfade inside a
  word, word gap, 400 ms for missing words). One deliberate difference: no whole-word lemma
  fallback (it would say "run" for "running" because the bank is small; stitching says the
  real word). Web-only addition: words CMUdict can't place are sounded out from dictionary
  pieces and letter rules.
- **The sound bank** (`public/bank/`) is generated, not hand-made:
  `../Mouthpiece/.venv/bin/python tools/export_bank.py --db ../Mouthpiece/db/mouthpiece.db --corpus ../Mouthpiece/corpus --out public/bank`
  It cuts the 3,000 most frequent words, 650 phrases, one clip per syllable shape and four per
  phone into ~45 s Opus sprites (`chunk-NNN.ogg`, fetched only when a render needs them), plus
  `index.json` and `cmu.txt`. Re-run it after re-indexing the corpus. The French LibriVox
  reading is excluded (it was transcribed as English).
