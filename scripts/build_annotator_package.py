"""Build the standalone annotator-package folder + zip.

Output layout (under repo root, in `annotator_package/`):

    annotator_package/
        README.md
        annotate.html
        items.json          (63 test items, NO labels, shuffled is the
                             annotator's responsibility in JS)
        example_output.json

The annotator double-clicks annotate.html, rates each item, downloads a
JSON file from their browser at the end, and emails it back.

Run:  python scripts/build_annotator_package.py
Then zip annotator_package/ and send.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.domains.code_review import CodeReviewDomain  # noqa: E402
from src.domains.content_mod import ContentModerationDomain  # noqa: E402
from src.domains.meals import MealDomain  # noqa: E402

OUT_DIR = REPO_ROOT / "annotator_package"


DOMAIN_LABELS = {
    "code_review": {
        "title": "Code review",
        "question": "Is this code production-ready?",
        "note": "Production-ready means: safe, correct, follows good practice. Read the snippet, decide if you would merge it.",
    },
    "content_moderation": {
        "title": "Content moderation",
        "question": "Is this comment appropriate?",
        "note": "Appropriate means: would be allowed on a typical news-site / community forum without removal.",
    },
    "meals": {
        "title": "Nutrition",
        "question": "Is this a healthy choice?",
        "note": "Healthy means: a reasonable everyday eating choice for an average adult.",
    },
}


def collect_items() -> tuple[list[dict], dict[str, str]]:
    """Return (shipped_items, mapping anon_id -> original_id)."""
    items: list[dict] = []
    mapping: dict[str, str] = {}
    for cls in (CodeReviewDomain, ContentModerationDomain, MealDomain):
        d = cls()
        for it in d.get_test_items():
            anon = f"item_{len(items) + 1:03d}"
            mapping[anon] = it.id
            items.append({
                "item_id": anon,             # anonymised; original kept in mapping
                "domain": d.name,
                "domain_title": DOMAIN_LABELS[d.name]["title"],
                "question": DOMAIN_LABELS[d.name]["question"],
                "note": DOMAIN_LABELS[d.name]["note"],
                "text": it.text,
            })
    return items, mapping


README_MD = """# Annotation task — AMEL paper

Thank you for helping. This task takes about 60-90 minutes.

## Before you start

**Please read `codebook.md` first** (in this folder). It contains
the definitions for each of the three rating categories and
two-three concrete examples per category per domain. Annotators
who skip the codebook tend to give noisy ratings, which makes
the data unusable for the paper.

## What you are doing

You will rate 63 short items across three domains:

- **Code review** (21 items) — is each Python snippet production-ready?
- **Content moderation** (21 items) — is each comment appropriate?
- **Nutrition** (21 items) — is each meal a healthy choice?

For each item, pick one of three categories:

1. **clear_positive** — obviously yes (production-ready / appropriate / healthy)
2. **ambiguous** — could go either way; reasonable people would disagree
3. **clear_negative** — obviously no

The codebook has worked examples. There are no right answers,
but applying the codebook consistently is what makes your work
useful.

## How to run it

1. Unzip this folder somewhere on your machine.
2. **Read `codebook.md`** (5 minutes).
3. Double-click `annotate.html` to open it in your default browser.
   (Works fully offline; no internet needed.)
4. Fill in the short "About you" page (country, profession, fluency —
   all optional). This helps us report annotator demographics in the
   paper.
5. The page guides you through one item at a time. Order is shuffled.
6. At the end, the page will download a `results-<your-name>.json`
   file to your Downloads folder.
7. **Send that JSON file back to the person who hired you, via
   Upwork chat (paperclip icon to attach).**

If your browser blocks the download, allow it for this local file
and click the download button again.

## Time

About 1 minute per item × 63 items = ~60-90 minutes including reading.
You can take breaks; the page saves your progress in localStorage
so you can close the tab and come back. The page also records how
long you spent on each item — this is used only to detect copy-paste
patterns, not to grade you.

## No AI tools

Please do not use ChatGPT / Claude / Gemini / any other AI to help
you rate items. The paper specifically studies LLM behavior, so AI-
generated annotations would invalidate the research. Trust your own
judgment.

## Privacy

The items themselves are public research stimuli (some are
deliberately bad code or inappropriate comments — that is the point
of the task). The page does not send any data anywhere; the
downloaded JSON only contains your ratings, the (optional)
demographics, your chosen name, and any optional comments per item.

Your ratings will be aggregated with two other annotators' ratings
to compute inter-rater agreement (Krippendorff's α) for the paper.
By default, you will be pseudonymized as "Annotator A/B/C". If you
would like to be thanked by name in the Acknowledgments, let the
person who hired you know.

## Questions

Reply to the person who hired you with any questions before you
start labeling. Once you start, please complete all 63 items in
the same session if possible.
"""


CODEBOOK_MD = '''# Codebook — AMEL annotation task

This codebook defines the three rating categories and gives two-three
worked examples per category per domain. Read all of it (about 5
minutes) before you open `annotate.html`.

If you are unsure on any item during the task, come back to this
codebook for guidance.

---

## The three categories (apply to all domains)

### clear_positive — obviously YES
Most reasonable people would say YES without hesitation. The item is
unambiguously on the positive side of the question. If you find
yourself thinking "well, maybe, but…" then it is not clear_positive.

### ambiguous — could go either way
Reasonable people would disagree. The item is genuinely on the border,
or has both clear positives and clear negatives that roughly balance.
Pick this when your honest answer is "it depends" or "I could argue
either side."

### clear_negative — obviously NO
Most reasonable people would say NO without hesitation. The item is
unambiguously on the negative side of the question.

**Rule of thumb:** if you imagine 10 random thoughtful people from
the relevant field rating this item, how many would agree with you?
- 9+ agree → clear_positive or clear_negative (depending on direction)
- 4-8 agree → ambiguous

---

## Domain 1 — Code review

**Question:** Is this code production-ready?

"Production-ready" means: you would feel comfortable merging this
to a real codebase used by paying customers. Safe (no obvious
vulnerabilities), correct (does what it claims to do), and follows
typical good practice for its language.

### clear_positive examples

Example A.
```python
from pydantic import BaseModel, Field

class User(BaseModel):
    email: EmailStr
    age: int = Field(ge=18, le=120)
```
Why clear_positive: standard validation pattern, well-typed,
nothing obviously wrong.

Example B.
```python
def safe_divide(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b
```
Why clear_positive: handles edge case, typed, simple.

### ambiguous examples

Example A.
```python
def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d")
```
Why ambiguous: works for the happy path but raises on any
non-matching format with no error handling. A senior engineer
might want a try/except + return-None; another might say "let
the caller handle the exception."

Example B.
```python
def load_config(path):
    with open(path) as f:
        return json.load(f)
```
Why ambiguous: minimal but functional; no error handling, no
schema validation, no type hint, but also not actively dangerous.

### clear_negative examples

Example A.
```python
def login(user, pwd):
    query = f"SELECT * FROM users WHERE name='{user}' AND pass='{pwd}'"
    return db.execute(query)
```
Why clear_negative: textbook SQL injection vulnerability.

Example B.
```python
def hash_password(p):
    return hashlib.md5(p.encode()).hexdigest()
```
Why clear_negative: MD5 is cryptographically broken for password
hashing.

---

## Domain 2 — Content moderation

**Question:** Is this comment appropriate?

"Appropriate" means: this comment would be allowed on a typical
news-site comments section or community forum without being removed
by a human moderator. Free expression of an opinion (even a strong
or unpopular one) is appropriate; targeted abuse, hate speech, or
threats are not.

### clear_positive examples

Example A.
> "I disagree with the article's premise. The data on housing
>  shortages cited at the top is from 2018, and the picture has
>  shifted considerably since then."

Why clear_positive: substantive disagreement, civil tone, no
attacks on people.

Example B.
> "Looking forward to seeing how this turns out in the next
>  quarter. The early numbers look promising."

Why clear_positive: harmless, on-topic.

### ambiguous examples

Example A.
> "What a stupid idea. Who actually thought this was worth
>  publishing?"

Why ambiguous: dismissive and a bit rude, but criticism of a
piece of work, not a personal attack on a named individual. Some
mods would remove for tone; others would leave it.

Example B.
> "This is fake news, the author is clearly biased."

Why ambiguous: a charged claim, but not threatening or hateful.
Whether to remove depends heavily on the platform's policy.

### clear_negative examples

Example A.
> "People from [ethnic group] should be banned from this country.
>  They are all criminals."

Why clear_negative: blanket statement targeting an ethnic group;
qualifies as hate speech on essentially every platform.

Example B.
> "I know where you live. I'm coming for you next."

Why clear_negative: explicit threat.

---

## Domain 3 — Nutrition

**Question:** Is this a healthy choice?

"Healthy" means: a reasonable everyday eating choice for an
average adult, judged on the dish as a whole. Occasional indulgences
are not automatically unhealthy; deep-fried or heavily processed
foods generally are.

### clear_positive examples

Example A.
> "Grilled salmon with steamed broccoli and brown rice."

Why clear_positive: balanced, lean protein + vegetables + complex
carb.

Example B.
> "Greek yogurt with fresh berries and a sprinkle of granola."

Why clear_positive: protein + fruit + minimal added sugar.

### ambiguous examples

Example A.
> "Whole-wheat pasta with creamy alfredo sauce and grilled chicken."

Why ambiguous: lean protein + whole grain, but heavy cream sauce
adds significant saturated fat. Healthier than the white-flour
version but not unambiguously healthy.

Example B.
> "Vegetarian burrito with rice, beans, cheese, sour cream, and
>  guacamole."

Why ambiguous: plant-based and fiber-rich, but high-calorie due
to multiple high-fat toppings.

### clear_negative examples

Example A.
> "Large bacon double-cheeseburger with fries and a soda."

Why clear_negative: high in saturated fat, sodium, refined carbs,
added sugar.

Example B.
> "Deep-fried butter sticks with whipped cream and a chocolate
>  milkshake."

Why clear_negative: novelty food with essentially no nutritional
value.

---

## Common questions

**Q: I have technical knowledge in only one of the three domains —
can I still rate the others?**
A: Yes. For domains outside your expertise, apply the codebook
directly and use your honest judgment as an informed adult. The
"average reasonable person" is the right reference; you do not
need to be a nutritionist to judge a burger meal.

**Q: What if I think an item is funny / weird / unusual?**
A: Comment in the optional comment box. The author of the paper
will read all comments and may use your observation in the
discussion section.

**Q: What if I really cannot decide between two categories?**
A: Default to `ambiguous`. That is what `ambiguous` is for. But
do not pick `ambiguous` just to avoid commitment — only pick it
when you would genuinely defend "it depends."

**Q: How long should I spend per item?**
A: 30-90 seconds is normal. Less than 10 seconds suggests
guessing; more than 3 minutes suggests overthinking. The form
records timing for quality-control purposes only.
'''




HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>AMEL annotation task</title>
<style>
  :root {
    --bg: #fafaf7;
    --fg: #1c1c1c;
    --muted: #6a6a6a;
    --accent: #2a4d7a;
    --border: #d8d4cb;
    --positive: #2d6a4f;
    --ambiguous: #b08900;
    --negative: #9d2933;
  }
  * { box-sizing: border-box; }
  body {
    font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI",
          Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--fg);
    margin: 0;
    padding: 24px;
  }
  main { max-width: 760px; margin: 0 auto; }
  h1 { margin-top: 0; }
  .start, .demog, .item, .done {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 22px 26px;
    margin-bottom: 18px;
  }
  .meta {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .domain-title { color: var(--accent); font-weight: 600; }
  .question {
    font-size: 17px;
    font-weight: 600;
    margin: 4px 0 6px;
  }
  .note {
    color: var(--muted);
    font-size: 14px;
    margin: 0 0 18px;
  }
  pre, .item-text {
    background: #f4f1ea;
    border-left: 3px solid var(--accent);
    padding: 12px 14px;
    border-radius: 4px;
    white-space: pre-wrap;
    font: 14px/1.45 "SF Mono", Menlo, Consolas, monospace;
    overflow-x: auto;
    max-height: 360px;
    overflow-y: auto;
  }
  .choices {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin: 18px 0 12px;
  }
  .choice {
    display: block;
    border: 1.5px solid var(--border);
    background: #fff;
    border-radius: 6px;
    padding: 10px 12px;
    cursor: pointer;
    text-align: center;
    font-size: 14px;
    user-select: none;
  }
  .choice:hover { background: #f4f1ea; }
  .choice input { display: none; }
  .choice.positive  { border-color: var(--positive); }
  .choice.ambiguous { border-color: var(--ambiguous); }
  .choice.negative  { border-color: var(--negative); }
  .choice.selected.positive  { background: var(--positive); color: #fff; }
  .choice.selected.ambiguous { background: var(--ambiguous); color: #fff; }
  .choice.selected.negative  { background: var(--negative); color: #fff; }
  textarea, input[type=text], select {
    width: 100%;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    font: inherit;
  }
  textarea { min-height: 60px; resize: vertical; }
  .field { margin: 10px 0 14px; }
  .field label { display: block; font-size: 14px; color: var(--muted); margin-bottom: 4px; }
  .field .hint { font-size: 12px; color: var(--muted); margin-top: 3px; }
  .nav {
    display: flex;
    justify-content: space-between;
    margin-top: 16px;
  }
  button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 9px 18px;
    font-size: 15px;
    cursor: pointer;
  }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  button.secondary {
    background: #fff;
    color: var(--accent);
    border: 1.5px solid var(--accent);
  }
  .progress {
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin: 14px 0 18px;
  }
  .progress > div { height: 100%; background: var(--accent); transition: width 0.2s; }
  ul.rules { font-size: 14px; color: var(--muted); }
  ul.rules li { margin: 4px 0; }
  .consent {
    background: #f4f1ea;
    border-radius: 6px;
    padding: 12px 14px;
    font-size: 13px;
    color: var(--muted);
  }
  .consent label { display: flex; align-items: flex-start; gap: 8px; cursor: pointer; }
  .consent input { margin-top: 3px; }
</style>
</head>
<body>
<main>
  <h1>AMEL annotation task</h1>

  <!-- screen 0: intro / name -->
  <section id="screen-start" class="start">
    <p>Thank you for helping. There are 63 items across three domains
       (code review, content moderation, nutrition). For each one
       you pick:</p>
    <ul class="rules">
      <li><strong>clear_positive</strong> &mdash; obviously yes
          (production-ready / appropriate / healthy)</li>
      <li><strong>ambiguous</strong> &mdash; could go either way; reasonable
          people would disagree</li>
      <li><strong>clear_negative</strong> &mdash; obviously no</li>
    </ul>
    <p><strong>Please read <code>codebook.md</code> first</strong>
       (in the same folder as this HTML file). It has worked
       examples for each category.</p>
    <p>Items appear in random order. Your progress saves locally; you
       can close this tab and come back.</p>
    <div class="field">
      <label for="annotator-name">Your name or annotator ID
        (will be saved in the results file)</label>
      <input type="text" id="annotator-name" placeholder="e.g. Jane Smith"
             autocomplete="off" />
    </div>
    <div class="nav">
      <span></span>
      <button id="btn-start">Continue &rarr;</button>
    </div>
  </section>

  <!-- screen 0.5: optional demographics + consent -->
  <section id="screen-demog" class="demog" style="display:none">
    <h2 style="margin-top:0">About you (optional)</h2>
    <p style="font-size:14px;color:var(--muted)">
      All fields are optional and used only to report annotator
      demographics in the paper (e.g. "two annotators from
      Europe, one software engineer and one teacher"). No
      personally identifying information is collected.
    </p>
    <div class="field">
      <label for="d-country">Country of residence</label>
      <input type="text" id="d-country" placeholder="e.g. Brazil"
             autocomplete="off" />
    </div>
    <div class="field">
      <label for="d-profession">Profession / background</label>
      <select id="d-profession">
        <option value="">(prefer not to say)</option>
        <option>software engineer or developer</option>
        <option>data scientist or ML engineer</option>
        <option>academic researcher</option>
        <option>student</option>
        <option>teacher or instructor</option>
        <option>writer, editor, or journalist</option>
        <option>nutritionist or dietitian</option>
        <option>moderator or community manager</option>
        <option>generalist / other</option>
      </select>
    </div>
    <div class="field">
      <label for="d-fluency">English fluency</label>
      <select id="d-fluency">
        <option value="">(prefer not to say)</option>
        <option>native</option>
        <option>fluent (working proficiency)</option>
        <option>intermediate</option>
      </select>
    </div>
    <div class="consent">
      <label>
        <input type="checkbox" id="c-consent" />
        <span>I consent to my ratings being used in the
              published academic paper and in a public dataset
              release. I confirm I will rate items based on my
              own judgment, without help from AI tools
              (ChatGPT, Claude, etc.).</span>
      </label>
    </div>
    <div class="nav">
      <button id="btn-back-start" class="secondary">&larr; Back</button>
      <button id="btn-demog-next">Start rating &rarr;</button>
    </div>
  </section>

  <!-- screen 1: one item -->
  <section id="screen-item" class="item" style="display:none">
    <div class="progress"><div id="bar"></div></div>
    <div class="meta">
      <span><span class="domain-title" id="m-domain"></span>
            &middot; item <span id="m-ix"></span> of 63</span>
    </div>
    <div class="question" id="m-question"></div>
    <p class="note" id="m-note"></p>
    <div class="item-text" id="m-text"></div>
    <div class="choices">
      <label class="choice positive">
        <input type="radio" name="rating" value="clear_positive" />
        clear_positive
      </label>
      <label class="choice ambiguous">
        <input type="radio" name="rating" value="ambiguous" />
        ambiguous
      </label>
      <label class="choice negative">
        <input type="radio" name="rating" value="clear_negative" />
        clear_negative
      </label>
    </div>
    <p style="font-size:13px;color:var(--muted);margin:8px 0 4px">
      Optional comment:
    </p>
    <textarea id="m-comment"
              placeholder="Anything that influenced your choice (optional)."></textarea>
    <div class="nav">
      <button id="btn-prev" class="secondary">&larr; Previous</button>
      <button id="btn-next">Next &rarr;</button>
    </div>
  </section>

  <!-- screen 2: finish -->
  <section id="screen-done" class="done" style="display:none">
    <h2>All done. Thank you.</h2>
    <p>Click the button below to download a JSON file with your
       ratings, then send that file back to the person who hired
       you via Upwork chat (paperclip icon to attach).</p>
    <p id="done-summary" style="color:var(--muted);font-size:14px"></p>
    <button id="btn-download">Download results file</button>
    <p style="margin-top:18px;font-size:13px;color:var(--muted)">
      If the button does nothing, your browser may have blocked
      automatic downloads from a local file. Right-click the link
      and choose "save as".
    </p>
  </section>
</main>

<script type="application/json" id="amel-items-data">__ITEMS_INLINE_PLACEHOLDER__</script>
<script>
(async () => {
  const STORAGE_KEY = "amel_annotator_state_v2";

  // ---- items embedded at build time (avoids browser file:// fetch issues) ----
  const ITEMS = JSON.parse(
    document.getElementById("amel-items-data").textContent
  );

  // ---- load or initialise state ----
  let state = null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) state = JSON.parse(raw);
  } catch (e) { state = null; }

  function shuffleSeed(arr, seed) {
    const out = arr.slice();
    let r = seed;
    for (let i = out.length - 1; i > 0; i--) {
      r = (r * 9301 + 49297) % 233280;
      const j = Math.floor((r / 233280) * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }

  if (!state) {
    state = {
      annotator_name: "",
      started_at: null,
      demographics: { country: "", profession: "", fluency: "" },
      consent_given: false,
      order: null,
      ratings: {},           // {item_id: {label, comment, seconds_spent}}
      ix: 0,
      item_shown_at: null,   // timestamp (ms) when current item rendered
    };
  }

  const save = () => localStorage.setItem(STORAGE_KEY, JSON.stringify(state));

  const screens = {
    start: document.getElementById("screen-start"),
    demog: document.getElementById("screen-demog"),
    item:  document.getElementById("screen-item"),
    done:  document.getElementById("screen-done"),
  };
  const show = (which) => {
    for (const k in screens) screens[k].style.display = (k === which) ? "" : "none";
  };

  // ---- start -> demographics ----
  document.getElementById("btn-start").onclick = () => {
    const name = document.getElementById("annotator-name").value.trim();
    if (!name) {
      alert("Please enter a name or annotator ID first.");
      return;
    }
    state.annotator_name = name;
    if (!state.started_at) state.started_at = new Date().toISOString();
    save();
    // pre-fill demographics if already given
    document.getElementById("d-country").value = state.demographics.country || "";
    document.getElementById("d-profession").value = state.demographics.profession || "";
    document.getElementById("d-fluency").value = state.demographics.fluency || "";
    document.getElementById("c-consent").checked = state.consent_given;
    show("demog");
  };

  // ---- demographics -> items ----
  document.getElementById("btn-back-start").onclick = () => {
    document.getElementById("annotator-name").value = state.annotator_name;
    show("start");
  };
  document.getElementById("btn-demog-next").onclick = () => {
    if (!document.getElementById("c-consent").checked) {
      alert("Please tick the consent box before rating items.");
      return;
    }
    state.demographics = {
      country:    document.getElementById("d-country").value.trim(),
      profession: document.getElementById("d-profession").value,
      fluency:    document.getElementById("d-fluency").value,
    };
    state.consent_given = true;
    if (!state.order) {
      const seed = (state.annotator_name.split("")
                    .reduce((a,c)=>a+c.charCodeAt(0),0)) || 1;
      state.order = shuffleSeed(ITEMS.map((_, i) => i), seed);
      state.ix = 0;
    }
    save();
    renderItem();
    show("item");
  };

  function renderItem() {
    const itemIx = state.order[state.ix];
    const item = ITEMS[itemIx];
    document.getElementById("m-domain").textContent = item.domain_title;
    document.getElementById("m-ix").textContent = state.ix + 1;
    document.getElementById("m-question").textContent = item.question;
    document.getElementById("m-note").textContent = item.note;
    document.getElementById("m-text").textContent = item.text;
    document.getElementById("bar").style.width =
      `${((state.ix + 1) / state.order.length) * 100}%`;

    const existing = state.ratings[item.item_id] || {};
    document.querySelectorAll("input[name=rating]").forEach((r) => {
      r.checked = (r.value === existing.label);
    });
    document.querySelectorAll(".choice").forEach((c) => {
      c.classList.toggle("selected",
        c.querySelector("input").checked);
    });
    document.getElementById("m-comment").value = existing.comment || "";

    document.getElementById("btn-prev").disabled = (state.ix === 0);
    document.getElementById("btn-next").textContent =
      (state.ix + 1 === state.order.length) ? "Finish &rarr;" : "Next &rarr;";

    // start per-item timing
    state.item_shown_at = Date.now();
    save();
  }

  document.querySelectorAll("input[name=rating]").forEach((r) => {
    r.onchange = () => {
      document.querySelectorAll(".choice").forEach((c) => {
        c.classList.toggle("selected", c.querySelector("input").checked);
      });
    };
  });

  document.getElementById("btn-prev").onclick = () => {
    if (state.ix === 0) return;
    captureCurrent();
    state.ix--;
    save();
    renderItem();
  };
  document.getElementById("btn-next").onclick = () => {
    if (!captureCurrent({ requireLabel: true })) return;
    if (state.ix + 1 === state.order.length) {
      save();
      finish();
    } else {
      state.ix++;
      save();
      renderItem();
    }
  };

  function captureCurrent({ requireLabel = false } = {}) {
    const itemIx = state.order[state.ix];
    const item = ITEMS[itemIx];
    const picked = document.querySelector("input[name=rating]:checked");
    if (requireLabel && !picked) {
      alert("Please pick one of the three ratings before continuing.");
      return false;
    }
    const seconds = state.item_shown_at
      ? (Date.now() - state.item_shown_at) / 1000
      : null;
    const prior = state.ratings[item.item_id] || {};
    state.ratings[item.item_id] = {
      label: picked ? picked.value : null,
      comment: document.getElementById("m-comment").value.trim(),
      // keep the longer of prior + new (annotator may re-visit)
      seconds_spent: Math.max(prior.seconds_spent || 0, seconds || 0),
    };
    return true;
  }

  function finish() {
    const total = state.order.length;
    const labelled = Object.values(state.ratings)
      .filter(r => r.label).length;
    document.getElementById("done-summary").textContent =
      `${labelled} / ${total} items rated.`;
    show("done");
  }

  document.getElementById("btn-download").onclick = () => {
    const out = {
      annotator_name: state.annotator_name,
      annotator_id: state.annotator_name.toLowerCase().replace(/\s+/g, "_"),
      started_at: state.started_at,
      completed_at: new Date().toISOString(),
      n_items: ITEMS.length,
      demographics: state.demographics,
      consent_given: state.consent_given,
      ratings: ITEMS.map((it) => {
        const r = state.ratings[it.item_id] || {};
        return {
          item_id: it.item_id,
          domain: it.domain,
          label: r.label || null,
          comment: r.comment || "",
          seconds_spent: r.seconds_spent || null,
        };
      }),
    };
    const blob = new Blob([JSON.stringify(out, null, 2)],
                          { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const safe = (state.annotator_name || "annotator")
      .toLowerCase().replace(/[^a-z0-9]+/g, "_");
    a.download = `results-${safe}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // ---- resume? ----
  if (state.annotator_name && state.order && state.consent_given) {
    if (Object.keys(state.ratings).length >= state.order.length &&
        state.ix >= state.order.length - 1) {
      finish();
    } else {
      renderItem();
      show("item");
    }
  } else if (state.annotator_name && !state.consent_given) {
    document.getElementById("annotator-name").value = state.annotator_name;
    show("demog");
  } else {
    show("start");
  }
})();
</script>
</body>
</html>
"""


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    items, mapping = collect_items()
    print(f"collected {len(items)} items across {len({i['domain'] for i in items})} domains")

    (OUT_DIR / "items.json").write_text(json.dumps(items, indent=2, ensure_ascii=False))
    (OUT_DIR / "README.md").write_text(README_MD)
    # Embed items.json directly into the HTML so the form works on
    # file:// in Chrome/Edge (which block fetch() against local files).
    # We use a <script type="application/json"> tag so the JSON does not
    # need to be JS-string-escaped (no apostrophe/quote-escaping headaches).
    items_inline = json.dumps(items, ensure_ascii=False).replace(
        "</script>", "<\\/script>"
    )
    embedded_html = HTML.replace("__ITEMS_INLINE_PLACEHOLDER__", items_inline)
    (OUT_DIR / "annotate.html").write_text(embedded_html)
    (OUT_DIR / "codebook.md").write_text(CODEBOOK_MD)

    # Save the anon->original mapping LOCALLY (not in the shipped folder)
    map_path = REPO_ROOT / "data" / "annotator_id_mapping.json"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(mapping, indent=2))
    print(f"Wrote mapping (local only, NOT in zip): {map_path}")

    # Also produce a zip ready to send.
    zip_path = REPO_ROOT / "annotator_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", OUT_DIR.parent, OUT_DIR.name)
    print(f"\nWrote folder: {OUT_DIR}")
    print(f"Wrote zip:    {zip_path}  ({zip_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
