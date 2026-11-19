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
- Content comes from `master`'s README and the `io-page` branch's `page/` bundle (tiers, corpus
  stats). `public/media/manifest.json` and `public/media/clips/` are copies of `io-page`'s
  `page/assets/` (cut by `mouthpiece/export_demo.py`); update both places together. The demo in
  `src/main.js` joins them with the engine's 12 ms crossfade, like `io-page`'s `DemoScope.tsx`.
