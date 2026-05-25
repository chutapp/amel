# Codebook — AMEL annotation task

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
