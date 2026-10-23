"""Movie Recommender - single-file web app (Flask + scikit-learn, TF-IDF + cosine similarity).
Setup: pip install flask pandas scikit-learn
Run:   python movie_recommender.py                 (uses the built-in sample movies)
       python movie_recommender.py --data my.csv   (CSV with title, genres, overview columns)
Then open http://127.0.0.1:5002. You can also upload a CSV from the page itself.
"""
import argparse
import io

import pandas as pd
from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

SAMPLE_CSV = r'''title,genres,overview
Inception,Sci-Fi Thriller,A thief enters dreams to steal secrets and plant an idea in a target's mind.
Interstellar,Sci-Fi Drama,Astronauts travel through a wormhole to find a new home for humanity.
The Matrix,Sci-Fi Action,A hacker discovers reality is a simulation controlled by machines.
The Dark Knight,Action Crime,Batman faces the Joker who spreads chaos across Gotham city.
Batman Begins,Action Crime,Bruce Wayne trains and becomes Batman to fight crime in Gotham.
Titanic,Romance Drama,A young couple falls in love aboard the doomed ocean liner.
The Notebook,Romance Drama,A couple's love story told across decades despite class differences.
3 Idiots,Comedy Drama,Three engineering students challenge the pressure of the education system.
Dangal,Drama Sports,A wrestler trains his daughters to become world champion wrestlers.
Toy Story,Animation Comedy,Toys come to life when humans are not around and go on adventures.
'''


def fit(df):
    """Return (vectorizer, tfidf_matrix) for a movies dataframe."""
    text = (df["genres"].fillna("") + " ") * 2 + df["overview"].fillna("")
    vec = TfidfVectorizer(stop_words="english")
    return vec, vec.fit_transform(text)


PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Movie Recommender</title>
<style>
:root{--bg:#f5f6fa;--card:#fff;--text:#1b2233;--muted:#6b7385;--accent:#4f46e5;--on-accent:#fff;--accent-2:#e0e7ff;--border:#e3e6ee;--danger:#d92d20;--ok:#067647}
@media (prefers-color-scheme:dark){:root{--bg:#0e1120;--card:#171b2e;--text:#e9ebf5;--muted:#9aa2b8;--accent:#8b8cf8;--on-accent:#0e1120;--accent-2:#262b52;--border:#2a3049;--danger:#f97066;--ok:#47cd89}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text)}
main{max-width:900px;margin:0 auto;padding:24px 16px 48px;display:grid;gap:16px}
h1{font-size:22px;margin:0}
h2{font-size:13px;margin:0 0 4px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.05em}
h3{font-size:16px;margin:0}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}
.top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.muted{color:var(--muted);font-size:13px}
.tabs{display:flex;gap:8px;margin-bottom:14px}
.tab{background:transparent;color:var(--muted)}
.tab.active{background:var(--accent-2);color:var(--text);font-weight:600}
.row{display:flex;gap:10px}
input,textarea,button,label.btn{font:inherit;padding:10px 12px;border-radius:8px;border:1px solid var(--border);background:var(--card);color:var(--text)}
input[type=text],textarea{flex:1;min-width:0}
input[type=range]{padding:0;accent-color:var(--accent);vertical-align:middle}
button,label.btn{cursor:pointer}
button.primary{background:var(--accent);color:var(--on-accent);border:0;font-weight:600}
.filters{margin-top:14px;display:grid;gap:10px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{padding:4px 10px;border-radius:999px;font-size:13px;background:transparent;color:var(--muted)}
.chip.active{background:var(--accent);color:var(--on-accent);border-color:var(--accent)}
.msg{min-height:1.2em;font-size:13px;color:var(--ok);margin:0}
.msg.err{color:var(--danger)}
#results{display:grid;gap:12px}
.movie{display:grid;gap:8px}
.movie-head{display:flex;justify-content:space-between;align-items:center;gap:10px}
.pill{font-size:12px;padding:2px 10px;border-radius:999px;background:var(--accent-2);white-space:nowrap}
.track{background:var(--accent-2);border-radius:6px;height:8px;overflow:hidden}
.fill{background:var(--accent);height:100%}
.tags{display:flex;flex-wrap:wrap;gap:6px}
.tag{font-size:12px;color:var(--muted);border:1px solid var(--border);border-radius:6px;padding:0 8px}
.movie p{margin:0;color:var(--muted)}
.movie button{justify-self:start;font-size:13px;padding:6px 10px}
[hidden]{display:none!important}
</style>
</head>
<body>
<main>
  <div class="top">
    <div><h1>🎬 Movie Recommender</h1><span id="count" class="muted"></span></div>
    <label class="btn">Upload your CSV<input id="upload" type="file" accept=".csv" hidden></label>
  </div>

  <section class="card">
    <div class="tabs">
      <button class="tab active" data-tab="similar" type="button">Similar to a movie</button>
      <button class="tab" data-tab="describe" type="button">Describe what you want</button>
    </div>

    <form id="similar-form" class="row">
      <input id="title" type="text" list="titles" placeholder="Start typing a movie title..." required autocomplete="off" aria-label="Movie title">
      <datalist id="titles"></datalist>
      <button class="primary" type="submit">Recommend</button>
    </form>
    <form id="describe-form" class="row" hidden>
      <textarea id="query" rows="2" maxlength="500" placeholder="e.g. a thrilling space adventure with an emotional story" required aria-label="Describe a movie"></textarea>
      <button class="primary" type="submit">Find movies</button>
    </form>

    <div class="filters">
      <label>Results: <b id="n-label">5</b> <input id="n" type="range" min="1" max="10" value="5"></label>
      <div><div class="muted" style="margin-bottom:6px">Filter by genre (optional)</div><div id="chips" class="chips"></div></div>
    </div>
    <p id="msg" class="msg" role="status"></p>
  </section>

  <section id="results"></section>
  <p id="hint" class="muted">Pick a movie or describe one to get recommendations.</p>
</main>

<script>
const $ = (s) => document.querySelector(s);
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  e.append(...kids.filter((k) => k != null && k !== false));
  return e;
}
const state = {genres: new Set(), last: null};

async function api(path, opts) {
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function status(text, isErr = false) { const m = $("#msg"); m.textContent = text; m.className = "msg" + (isErr ? " err" : ""); }
function qs(extra) {
  const p = new URLSearchParams(extra);
  p.set("n", $("#n").value);
  if (state.genres.size) p.set("genres", [...state.genres].join(","));
  return p.toString();
}

function card(r, top) {
  const rel = top > 0 ? (r.score / top * 100).toFixed(0) : 0;
  return el("article", {class: "movie card"},
    el("div", {class: "movie-head"}, el("h3", {}, r.title), el("span", {class: "pill"}, Math.round(r.score * 100) + "% match")),
    el("div", {class: "track"}, el("div", {class: "fill", style: `width:${rel}%`})),
    el("div", {class: "tags"}, ...r.genres.split(/\s+/).filter(Boolean).map((g) => el("span", {class: "tag"}, g))),
    el("p", {}, r.overview),
    el("button", {type: "button", onclick: () => { switchTab("similar"); $("#title").value = r.title; runSimilar(); }}, "More like this"));
}
function show(results, heading) {
  $("#hint").hidden = true;
  const head = el("h2", {}, heading);
  $("#results").replaceChildren(head, ...(results.length
    ? results.map((r) => card(r, results[0].score))
    : [el("p", {class: "muted"}, "No matches found. Try different wording or fewer genre filters.")]));
}

async function runSimilar() {
  state.last = runSimilar; status("");
  try {
    const d = await api("/api/similar?" + qs({title: $("#title").value}));
    show(d.results, "Because you picked " + d.query.title);
  } catch (e) { status(e.message, true); }
}
async function runSearch() {
  state.last = runSearch; status("");
  try {
    const d = await api("/api/search?" + qs({q: $("#query").value}));
    show(d.results, "Best matches for your description");
  } catch (e) { status(e.message, true); }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $("#similar-form").hidden = name !== "similar";
  $("#describe-form").hidden = name !== "describe";
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
$("#similar-form").addEventListener("submit", (e) => { e.preventDefault(); runSimilar(); });
$("#describe-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
$("#n").addEventListener("input", () => ($("#n-label").textContent = $("#n").value));
$("#n").addEventListener("change", () => state.last && state.last());

async function init() {
  const m = await api("/api/meta");
  $("#count").textContent = m.count + " movies loaded";
  $("#titles").replaceChildren(...m.titles.map((t) => el("option", {value: t})));
  $("#chips").replaceChildren(...m.genres.map((g) => el("button", {class: "chip", type: "button", onclick: (e) => {
    state.genres.has(g) ? state.genres.delete(g) : state.genres.add(g);
    e.currentTarget.classList.toggle("active");
    state.last && state.last();
  }}, g)));
}

$("#upload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const res = await fetch("/api/upload", {method: "POST", headers: {"Content-Type": "text/csv"}, body: await file.text()});
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || res.statusText);
    state.genres.clear(); state.last = null;
    $("#results").replaceChildren(); $("#hint").hidden = false; $("#title").value = "";
    await init();
    status("Loaded " + data.count + " movies from " + file.name + ".");
  } catch (err) { status(err.message, true); }
  e.target.value = "";
});

init().catch((e) => status(e.message, true));
</script>
</body>
</html>
'''


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit
REQUIRED = {"title", "genres", "overview"}
STATE = {}


def load_dataset(df):
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError("missing columns: " + ", ".join(sorted(missing)))
    df = df.dropna(subset=["title"]).copy()
    df["title"] = df["title"].astype(str)
    df = df.drop_duplicates("title").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("need at least 2 movies")
    df["genres"] = df["genres"].fillna("").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)
    vec, matrix = fit(df)
    STATE.update(df=df, vec=vec, matrix=matrix,
                 genres=sorted({g for s in df["genres"] for g in s.split()}))


def rank(scores, n, genres, exclude=None):
    out = STATE["df"].assign(score=scores)
    if exclude is not None:
        out = out.drop(index=exclude)
    if genres:
        out = out[out["genres"].apply(lambda s: any(g in s.split() for g in genres))]
    out = out[out["score"] > 0].sort_values("score", ascending=False).head(n)
    return [{"title": r.title, "genres": r.genres, "overview": r.overview, "score": round(float(r.score), 3)}
            for r in out.itertuples()]


def common_args():
    n = max(1, min(request.args.get("n", 5, type=int), 20))
    genres = [g for g in request.args.get("genres", "").split(",") if g]
    return n, genres


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File too large (limit is 5 MB)."), 413


@app.get("/")
def index():
    return PAGE


@app.get("/api/meta")
def meta():
    return jsonify(titles=STATE["df"]["title"].tolist(), genres=STATE["genres"], count=len(STATE["df"]))


@app.get("/api/similar")
def similar():
    n, genres = common_args()
    df = STATE["df"]
    idx = df.index[df["title"].str.lower() == request.args.get("title", "").strip().lower()]
    if len(idx) == 0:
        return jsonify(error="Movie not found. Pick a title from the list."), 404
    i = int(idx[0])
    scores = linear_kernel(STATE["matrix"][i], STATE["matrix"])[0]
    row = df.loc[i]
    return jsonify(query={"title": row["title"], "genres": row["genres"], "overview": row["overview"]},
                   results=rank(scores, n, genres, exclude=i))


@app.get("/api/search")
def search():
    n, genres = common_args()
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify(error="Describe what you want to watch first."), 400
    scores = linear_kernel(STATE["vec"].transform([q[:500]]), STATE["matrix"])[0]
    return jsonify(results=rank(scores, n, genres))


@app.post("/api/upload")
def upload():
    raw = request.get_data()
    if not raw:
        return jsonify(error="No file uploaded"), 400
    try:
        load_dataset(pd.read_csv(io.BytesIO(raw)))
    except Exception as e:
        return jsonify(error=f"Could not use that CSV ({e}). It needs title, genres and overview columns."), 400
    return jsonify(count=len(STATE["df"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="path to your own movies CSV")
    a = p.parse_args()
    load_dataset(pd.read_csv(a.data) if a.data else pd.read_csv(io.StringIO(SAMPLE_CSV)))
    app.run(debug=True, port=5002)
