# Annotation task — AMEL paper

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
