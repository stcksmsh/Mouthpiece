import DemoScope from "./DemoScope";
import "./style.css";

/*
 * MOUTHPIECE's full custom project page — federated whole into the host
 * site (stcksmsh.github.io)'s build via its src/pages/projects/[slug].astro
 * generic template (see that repo's AGENTS.md, "custom project pages").
 * default export is mounted client-side in place of the site's standard
 * hero; the devlog stream below it is still the site's own generic,
 * server-rendered section — this only owns the hero + the "how it works"
 * explainer, not the whole page top to bottom.
 *
 * Props shape is the site's contract, not this repo's: `project` mirrors
 * this repo's own meta.json, `devlogs` is a plain summary (title/date/tags)
 * of what's federated from devlogs/ — no rendered body, that stays
 * server-rendered by the site itself below this component.
 */

interface ProjectMeta {
  slug: string;
  title: string;
  tagline: string;
  status: "active" | "shipped" | "archived";
  stack: string[];
  repo: string;
  demo?: string;
  accent?: [number, number, number];
}

interface DevlogSummary {
  id: string;
  title: string;
  date: string;
  tags: string[];
}

// Same padding scheme as the host site's own src/lib/ticker.ts — copied
// rather than imported, same reasoning as useScopeDraw.ts's header comment.
function padForTicker<T>(items: T[], minCount = 6): (T | null)[] {
  if (items.length === 0) return items;
  const reps = Math.max(1, Math.ceil(minCount / items.length));
  const out: (T | null)[] = [];
  for (let i = 0; i < reps; i++) out.push(...items, null);
  return out;
}

const STATS: [string, string][] = [
  ["words indexed", "165,425"],
  ["syllables", "195,011"],
  ["phones", "491,964"],
];

const TIERS = [
  {
    n: "01",
    title: "whole word",
    body: "The exact word was spoken somewhere in the corpus — use that clip directly. Most natural, when it's available.",
  },
  {
    n: "02",
    title: "phrase reuse",
    body: "Consecutive words were spoken together in one recording — reuse that whole span as one continuous clip. Real co-articulation, no seam.",
  },
  {
    n: "03",
    title: "syllable stitch",
    body: "Split the word into syllables, search for the same natural continuity between them where it exists, splice what's left from unrelated clips.",
  },
  {
    n: "04",
    title: "phoneme fallback",
    body: "No syllable shape exists anywhere in the corpus — fall all the way down to individual phones. Rougher, but it makes almost any word constructible.",
  },
];

export default function MouthpiecePage({ project }: { project: ProjectMeta; devlogs: DevlogSummary[] }) {
  const stackTicker = [...padForTicker(project.stack), ...padForTicker(project.stack)];

  return (
    <>
      <header className="mp-hero">
        <div className="mp-hero-scrim" />
        <div className="shell mp-hero-inner">
          <span className="coord">{project.status}</span>
          <h1>{project.title}</h1>
          <p className="mp-lede">{project.tagline}</p>
          <div className="ticker-viewport" aria-hidden="true">
            <ul className="mp-stack">
              {stackTicker.map((s, i) =>
                s === null ? <li key={i} className="ticker-gap" /> : <li key={i}>{s}</li>
              )}
            </ul>
          </div>
          <div className="mp-project-links">
            <a href={project.repo}>repo →</a>
            {project.demo && <a href={project.demo}>demo →</a>}
          </div>

          <div className="mp-demo-shell">
            <span className="coord mp-demo-kicker">type a word — hear it built from real people's real words</span>
            <DemoScope slug={project.slug} accent={project.accent} />
          </div>
        </div>
      </header>

      <section className="shell">
        <div className="mp-section-head">
          <span className="coord">how it works</span>
          <h2>Four ways to build a word</h2>
        </div>
        <ol className="mp-tiers">
          {TIERS.map((t) => (
            <li key={t.n} className="mp-tier">
              <span className="mp-tier-n coord">{t.n}</span>
              <div>
                <h3>{t.title}</h3>
                <p>{t.body}</p>
              </div>
            </li>
          ))}
        </ol>
        <ul className="mp-stat-line">
          {STATS.map(([label, value]) => (
            <li key={label}>
              <b>{value}</b>
              <span className="coord">{label}</span>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
