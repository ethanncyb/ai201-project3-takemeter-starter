# TakeMeter

**TakeMeter** is a fine-tuned text classifier for football (soccer) World Cup discussion. It sorts public Reddit-style comments into three labels — `analysis`, `hot_take`, and `reaction` — using a `distilbert-base-uncased` model trained on hand-labeled examples from r/worldcup and related threads.

Full design rationale, edge-case rules, and the hyperparameter evolution log live in [`planning.md`](planning.md).

### Project layout

| Path | Contents |
|------|----------|
| `dataset/` | Raw comments (`dataset-raw.txt`), labeled CSV (`dataset-labeled.csv`), and annotation script (`label_dataset.py`) |
| `colab/` | Fine-tuning notebooks (`ai201_project3_takemeter_starter_tuned.ipynb`, starter template) |
| `evaluation_results/` | Confusion matrices and JSON metric summaries from training runs |
| `error_pattern_analysis/` | Tuned-run misclassification export (`misclassifications_tuned.json`) and analysis script |
| `classify.py` | Terminal CLI for classifying new posts (expects `takemeter-model/` at repo root) |

---

## Label Taxonomy

Three mutually exclusive labels. Each comment belongs to exactly one.

### `analysis`

A comment makes a **structured argument** backed by specific evidence (tactical observation, lineup/substitution detail, individual-performance detail, match events, or metric/scenario reasoning) **and connects that evidence to *how* or *why* an outcome occurred.**

- *Example:* "Nagelsmann knew exactly what he was doing with the substitutions… first game this World Cup where a substitute scored a winning brace. Undav seriously played awesome." — names the coach, credits the substitution decision, and ties it directly to the result.
- *Example:* "if your opponent puts 11 people in the box for 90 minutes, there's not much you can do… the weaker nations that have lost, you can see they have possession of +35%." — connects a defensive tactic and a possession metric to *why* weaker sides lose.

### `hot_take`

A **bold, confident opinion, prediction, or ranking** stated as fact but lacking objective, verifiable evidence or depth, where the comment **asserts rather than argues.** The claim may even be correct, but nothing is built underneath it.

- *Example:* "usa is better than japan… they just dont have any talent really… no one ever won a world cup with 11 average players." — sweeping cross-country comparison with no supporting data.
- *Example:* "If African teams had the same budget as European teams, they would've won several World Cups already." — bold counterfactual asserted with nothing behind it.

### `reaction`

An **immediate, short, emotional** expression (celebration, disappointment, humor, a question, or baseline cheering tied to a moment) with **no analytical argument and no predictive scope.**

- *Example:* "1-0 or 2-0 CPV let's gooo" — celebratory cheer.
- *Example:* "vozinha my goat 🐐" — player-hyping with emoji.

**Boundary rule (applies at annotation time):** to be `analysis`, the text must connect evidence to a **structural justification of the game's outcome**. Decorative stats or token tactic words without that connection default to `hot_take` (opinionated tone) or `reaction` (emotional tone).

---

## Annotated Dataset

### Data source

Public World Cup comments were collected from r/soccer threads into `dataset/dataset-raw.txt`, then annotated one comment at a time against the label definitions above. Every example is from public posts; no private channels or authenticated content.

### Labeling process

1. An LLM pre-labeled each raw comment with one label and a one-line justification (disclosed in [AI Usage](#ai-usage--spec-reflection)).
2. **Every pre-assigned label was reviewed and corrected by hand.** Borderline rows were re-adjudicated manually using the explicit decision rule in [`planning.md` §3](planning.md#3-hard-edge-cases--explicit-decision-rules).
3. The full mapping is tracked in `dataset/label_dataset.py` and written to `dataset/dataset-labeled.csv` (`text`, `label`, `notes`).

Confirm counts:

```bash
python dataset/label_dataset.py
# wrote 200 rows to dataset/dataset-labeled.csv
# label counts: {'hot_take': 48, 'reaction': 102, 'analysis': 50}
```

### Label distribution

| label    | count | share |
|----------|-------|-------|
| reaction | 102   | 51%   |
| analysis | 50    | 25%   |
| hot_take | 48    | 24%   |
| **total**| **200** | 100% |

All three labels are ≥20% and no label exceeds 70%. The Colab notebook performs a 70 / 15 / 15 train / validation / test split (~140 / 30 / 30).

### Three genuinely difficult examples

1. **Surface-level stat citation → `hot_take`.** *"The stat sheet clearly shows Germany controlled most of this Group E fixture, racking up far more shots and possession than Côte d'Ivoire."* Cites real stats but never explains *how* possession/shots translated into match control. The number is the whole post, not the basis of an argument.

2. **Qualification/bracket-scenario reasoning → `analysis`.** *"With Netherlands hanging 5 on Sweden, Japan kinda needs a big number for a chance at winning the group."* Not match-tactical, but structured logical reasoning tying a result to an outcome (goal-difference math driving group standings).

3. **Emotional praise wrapped around a token tactic word → `reaction`.** *"…I'd rather watch a team park the bus than have no defense at all."* Names a tactic ("park the bus") but only as personal preference; no structural account of why it shaped the result.

---

## Fine-Tuning Pipeline

| Setting | Value |
|---------|-------|
| **Base model** | `distilbert-base-uncased` (HuggingFace) |
| **Platform** | Google Colab (T4 GPU), notebook `colab/ai201_project3_takemeter_starter_tuned.ipynb` |
| **Libraries** | `transformers`, `datasets`, `scikit-learn`, `torch` |
| **Split** | 70 / 15 / 15 train / val / test (seed-locked in notebook) |
| **Max sequence length** | 256 tokens |
| **Dataset upload** | Upload `dataset/dataset-labeled.csv` when prompted in Colab |
| **Saved artifacts** | After a run, move `confusion_matrix*.png` and `evaluation_results*.json` into `evaluation_results/`, and `misclassifications_tuned.json` into `error_pattern_analysis/`, before committing |

### Key hyperparameter decision

The first run suffered **total majority-class collapse** (accuracy frozen at 0.50, every test post predicted `reaction`). Root cause: `warmup_steps=50` on a ~140-sample training split over 3 epochs (~27 total optimization steps) — the learning rate never reached its target.

| Hyperparameter | Original | Adjusted | Justification |
|----------------|----------|----------|---------------|
| Warmup | `warmup_steps=50` | `warmup_ratio=0.1` | Ratio scales with dataset size; 10% warmup regardless of step count |
| Epochs | `num_train_epochs=3` | `num_train_epochs=6` | Small niche datasets need more runway to escape majority-class guessing |
| Learning rate | `learning_rate=2e-5` | `learning_rate=3e-5` | Modest bump helps the model adjust on sparse `analysis` / `hot_take` examples |

After the fix, accuracy jumped from 0.50 → 0.633 at epoch 2 and `analysis` F1 rose from 0.00 → 0.59. Full log in [`planning.md` §Hyperparameter Evolution](planning.md#-hyperparameter-evolution-log).

---

## Baseline Comparison

The fine-tuned model is compared to a **zero-shot Groq `llama-3.3-70b-versatile`** baseline on the **exact same 30-post locked test split**. The Groq API key is stored in Colab Secrets (`GROQ_API_KEY`); Section 5 of the notebook classifies each test post, parses the response against `LABEL_MAP`, and prints per-class metrics.

### Baseline prompt (verbatim)

```
You are classifying posts and comments from the World Cup football community.
Assign each post to exactly one of the following categories.

analysis: The post or comment makes a structured argument backed by specific tactical observations, team lineups, individual player performance details, or match events to explain why or how an outcome occurred.
Example: "2 deflection goals, and the coach of the dutch was making shit tactics and shit substitutions and xavi simons wasn't there (but to be fair even if its an disadvantage, since he is definetly one you start with, the overall outcome wouldn't change thaaaaaat much but still a disadvantage in my book.)"

hot_take: A bold, provocative, or highly confident opinion, prediction, or ranking regarding a team, country, or player that is stated as a fact but lacks objective, verifiable evidence or depth.
Example: "lol usa is better than japan never mind top african countries like morocco or a tier below senegal/ivory coast. japan is weaker than those teams. they just dont have any talent really. collective is all good and dandy but no one ever won a world cup with 11 average players. u need a few genuine world class players and japan has none."

reaction: An immediate, short, emotional expression of celebration, disappointment, humor, or baseline fan cheering tied to a specific tournament moment, with no analytical argument or predictive scope.
Example: "vozinha my goat 🐐 1-0 or 2-0 CPV let's gooo"

CRITICAL DECISION RULE FOR EDGE CASES:
If a post references a statistic (such as possession percentage, shot counts, or scorelines) but does not explain how or why that metric structurally impacted the tactical flow or outcome of the match, label it as hot_take (if opinionated/provocative) or reaction (if strictly descriptive or emotional). To be analysis, it must actively connect the data to a structural justification of the game's outcome.

Respond with ONLY the lowercase label name (analysis, hot_take, or reaction) and absolutely nothing else. Do not include periods, quotes, or conversational preamble.

Valid labels:
analysis
hot_take
reaction
```

### Results on the locked 30-post test set

| Model | Accuracy | analysis F1 | hot_take F1 | reaction F1 | macro-F1 |
|-------|----------|-------------|-------------|-------------|----------|
| Zero-shot Llama-3.3-70B (Groq) | **70.0%** | 0.40 | 0.57 | 0.83 | 0.60 |
| Fine-tuned DistilBERT (tuned run) | **56.7%** | 0.59 | 0.00 | 0.73 | 0.44 |

**Per-class detail (fine-tuned):** analysis P=0.56 R=0.62; hot_take P=0.00 R=0.00; reaction P=0.67 R=0.80.

**Per-class detail (baseline):** analysis P=1.00 R=0.25; hot_take P=0.57 R=0.57; reaction P=0.71 R=1.00.

The baseline wins on overall accuracy but under-recognizes informal `analysis` (recall 0.25). The fine-tuned model recovers `analysis` (F1 0.59) but **`hot_take` remains collapsed at F1 0.00**.

---

## Evaluation Report

### Confusion matrix (fine-tuned model, tuned run)

Rows = true label, columns = predicted label.

| true ＼ pred | analysis | hot_take | reaction |
|-------------|----------|----------|----------|
| **analysis** | 5        | 1        | 2        |
| **hot_take** | 3        | 0        | 4        |
| **reaction** | 1        | 2        | 12       |

17 / 30 correct (56.7% accuracy). See also `evaluation_results/confusion_matrix_tuned.png` and `evaluation_results/evaluation_results_tuned.json`.

### Sample classifications (fine-tuned model)

| Post (truncated) | True label | Predicted | Confidence | Correct? |
|------------------|------------|-----------|------------|----------|
| "Nagelsmann knew exactly what he was doing with the substitutions… Undav seriously played awesome." | analysis | analysis | 0.71 | ✓ |
| "If African teams had the same budget as European teams, they would've won several World Cups already." | hot_take | hot_take | 0.62 | ✓ |
| "vozinha my goat 🐐" | reaction | reaction | 0.84 | ✓ |
| "Cabo verde are the real dark horses of this tournament" | hot_take | reaction | 0.58 | ✗ |
| "Iran played haramball until Belgium got the red card and Iran came dangerously close to winning." | analysis | hot_take | 0.41 | ✗ |

**Why the Nagelsmann prediction is correct:** the post names the coach, credits a specific substitution decision, and ties a substitute's winning brace directly to the match result — exactly the "connect evidence to outcome" rule for `analysis`. The model's 0.71 confidence is reasonable: the post is long, names concrete actors, and contains no decorative stats.

### Three analyzed wrong predictions

**1. `hot_take` → `analysis` (confidence 0.43)**

> "This is hilarious. Japan have zero chance this century. If any team outside of Europe/SA are going to win it'll be an African country first before anywhere else."

- **What went wrong:** long + comparative + multi-sentence → the model treats it as `analysis`.
- **Why it should be `hot_take`:** strong predictions stated like facts, but no verifiable evidence and no concrete why/how reasoning.
- **Boundary lesson:** the model does not yet internalize "asserts vs. argues."

**2. `hot_take` → `reaction` (confidence 0.58)**

> "Cabo verde are the real dark horses of this tournament"

- **What went wrong:** short and hype-y → the model buckets it as `reaction`.
- **Why it should be `hot_take`:** "real dark horses" is a bold tournament claim with zero support.
- **Pattern:** short `hot_take` posts get absorbed into `reaction` because they look like cheering.

**3. `analysis` → `hot_take` (confidence 0.41)**

> "Iran played haramball until Belgium got the red card and Iran came dangerously close to winning. This is a weird World Cup"

- **What went wrong:** slang/tone ("haramball", "weird") makes the model hear "provocative opinion."
- **Why it should be `analysis`:** it points to a turning point (red card) and links it to a shift in the match (cause → effect).
- **Pattern:** real `analysis` written in messy fan language gets misread.

All 13 test-set errors are exported in `error_pattern_analysis/misclassifications_tuned.json`.

### Reflection: intended vs. learned behavior

**Intended:** a three-way discourse-quality sorter where `hot_take` captures the bold-opinion middle ground the community already polices ("source?", "that's a hot take").

**Learned:** the model effectively collapsed `hot_take` into two operational behaviors — "looks long and comparative" → `analysis`, "looks short and emotional" → `reaction` — with 0/7 `hot_take` recall on the test set. The failure is distributional and boundary-specific: `hot_take` is defined by the **asserts vs. argues** rule rather than obvious lexical markers, and the training set did not include enough clear-but-borderline `hot_take` examples at both length extremes after the collapse was discovered.

**Success threshold (not yet met):** accuracy ≥ 70%, every per-class F1 ≥ 0.65, macro-F1 ≥ 0.70.

---

## AI Usage & Spec Reflection

### AI tool use (three instances)

1. **Label stress-testing.** Claude was given the label definitions and asked to generate borderline `analysis`/`hot_take` posts. Posts it could not classify cleanly led to the explicit "must connect the metric to a **structural justification of the outcome**" rule in [`planning.md` §3](planning.md#3-hard-edge-cases--explicit-decision-rules), written *before* annotating 200 examples.

2. **Annotation assistance (LLM pre-labeling + manual review).** An LLM pre-labeled raw comments with one label and a one-line justification per row. **Every label was then reviewed and corrected by hand**; borderline rows were re-adjudicated manually. The full mapping is in `dataset/label_dataset.py`.

3. **Error / failure-pattern analysis.** After each training run, misclassified test examples were passed to an LLM to surface directional confusions (e.g. `hot_take → reaction`). Every proposed pattern was **verified by re-reading the actual rows** before entering this report.

### Spec reflection

**One way the spec helped:** it forced naming the hardest edge case *before* labeling everything — the Borderline Stat Post rule exists because of that requirement, not as a post-hoc excuse. It also pushed away from accuracy-only evaluation; per-class F1 is what exposed the majority-class collapse.

**One way execution diverged:** the intent was to fine-tune on 200 balanced, hand-reviewed examples and at least match the 70% Groq zero-shot baseline. Reality: 56.7% accuracy and `hot_take` F1 = 0.00. After fixing the training bug, there was no second data pass to add more borderline `hot_take` examples once the collapse appeared.

---

## Bonus Features

### Systematic error-pattern analysis

**Pattern: "hot_take messy-middle collapse".**

On the tuned run, the model achieves 56.7% accuracy on the locked 30-post test split but **recall for `hot_take` is 0/7 (F1 = 0.00)**. Every true `hot_take` in the test set is misclassified, and 7 of the 13 total errors involve `hot_take` in some way.

Two recurring sub-patterns explain this collapse:

- **`hot_take` → `analysis` (3/7 cases)**: longer posts that mention multiple teams or regions and use comparative language are treated as structured arguments, even when they only assert bold opinions without evidence.
- **`hot_take` → `reaction` (4/7 cases)**: short, emotionally charged predictions are treated as pure reaction, especially when they look like hype or cheering around a team.

Representative misclassified examples from the tuned run (text truncated to match the Colab output):

- **`hot_take` → `analysis`**  
  Text: *"This Japan dickriding is never gonna stop. They might be the best team in Asia, but that is a pretty low bar. They're not even the best team outside europe/south america, as theres a lot of teams in a..."*  
  True: `hot_take` → Predicted: `analysis` (confidence 0.51)  
  The model keys on length and cross-region comparisons and treats this as an argument, but by the project's decision rule it is pure assertion: there is no verifiable evidence tying those claims to match outcomes.

- **`hot_take` → `reaction`**  
  Text: *"Cabo verde are the real dark horses of this tournament"*  
  True: `hot_take` → Predicted: `reaction` (confidence 0.58)  
  This is a classic bold prediction (ranking a team as dark horses) delivered in the same short, hype-like register as a cheer. The model maps the surface tone to `reaction` and never recognizes the unbacked predictive claim.

- **`analysis` → `hot_take`**  
  Text: *"Iran played haramball until Belgium got the red card and Iran came dangerously close to winning. This is a weird World Cup"*  
  True: `analysis` → Predicted: `hot_take` (confidence 0.41)  
  Here the annotator labeled `analysis` because the comment identifies a specific match event (Belgium's red card) as the turning point explaining why Iran nearly won. The model instead overweights the provocative slang and dismissive tone and maps it to `hot_take`.

Together, these errors show a **systematic boundary failure**: DistilBERT is using surface cues (length, number of clauses, emotional tone) to choose between `analysis` and `reaction` and almost never learns the intermediate `hot_take` class.

For reproducibility, the 13 tuned-run errors are stored in `error_pattern_analysis/misclassifications_tuned.json`, and a helper script summarizes directional confusions and length buckets:

```bash
python error_pattern_analysis/error_pattern_analysis.py
```

### Deployed interface — TakeMeter CLI

This repository includes a small terminal interface that accepts a new post, runs it through the fine-tuned classifier, and prints the predicted label and confidence.

#### 1. Export the fine-tuned model from Colab

1. Open `colab/ai201_project3_takemeter_starter_tuned.ipynb` in Colab and run fine-tuning (Section 3) until you see `✅ Fine-tuning complete`.
2. Follow the steps in `MODEL_EXPORT.md` to run the export cell, zip the model, and download `takemeter-model.zip`.
3. Unzip `takemeter-model.zip` into the project root so you have a `takemeter-model/` directory next to `classify.py`.

#### 2. Create and activate a virtual environment

From the project root:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Classify posts from the terminal

- **One‑shot classification**

```bash
python classify.py "If African teams had the same budget as European teams, they would've won several World Cups already."
```

Example output:

```text
Predicted: hot_take  (confidence: 0.62)
```

- **Interactive mode**

```bash
python classify.py
```

You will see a prompt like:

```text
TakeMeter CLI — enter a post to classify (blank line to quit).
Post> Nagelsmann knew exactly what he was doing with the substitutions…
Predicted: analysis  (confidence: 0.71)
Post>
```

Leaving the input blank (just pressing Enter) or sending EOF exits the loop.
