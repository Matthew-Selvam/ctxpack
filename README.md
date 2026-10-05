# ctxpack

**Pack any codebase into a token-budgeted context bundle.**

Zero dependencies. Knows what it measured. Never exceeds the budget.

```bash
pipx install ctxpack          # or: uv tool install ctxpack
```

```bash
ctxpack ~/code/myapp -b 60000 -o context.xml -f xml
```

---

## The problem

An LLM agent with a 200k context window does not need help *finding* files. It
needs help *not wasting* them. Point a tool at a real checkout and it hands the
model `node_modules`, four generated bundles, a `CHANGELOG` with 8,000 commits
in it, and 40 copies of the same locale file — then runs out of room before it
reaches the code.

`ctxpack` reads a repository and emits the most informative subset that fits a
budget you choose.

| | |
|---|---|
| **Dependencies** | none (`tiktoken` is optional, for exact counts) |
| **Budget accuracy** | fitted to tiktoken ground truth, **8.1%** mean error, holdout-validated |
| **Budget overrun** | structurally impossible |
| **Speed** | 88k files walked in 6.5s; 484MB repo packed to a full 80k budget in 10s |

---

## What it actually does

### 1. Drops the noise

Respects `.gitignore` (a real implementation of it, including negation and
`**`), plus a built-in list covering what shows up in essentially every
repository. Measured on this machine:

| Repository | Files on disk | Text files kept | Time |
|---|---:|---:|---:|
| `automaton-review` | 19,642 | 211 | 0.02s |
| `hermes` | 88,532 | 23,432 | 6.50s |

### 2. Ranks files by how much they're worth

Not by size, not alphabetically — by whether a new engineer would read them.
Every contribution is a named, inspectable signal:

```console
$ ctxpack explain . -n 2
  9.20  README.md
        +7.00 readme, +1.00 docs, +1.20 top-level
  6.80  src/ctxpack/__init__.py
        +2.40 anchor, +1.90 source-root, +1.50 source, +0.60 shallow, +0.40 small
```

### 3. Packs under diversity constraints

Naive packing fills a budget with whatever sorts first. `ctxpack` applies caps
on per-file, per-directory and per-extension share, collapses near-duplicates,
and cuts oversized files at a line boundary:

```
$ ctxpack pack ~/.hermes -b 80000 --stats
budget            80,000 tokens
index             23,432 entries, 7,747 tokens
files packed      97 of 23,432 discovered (924 ignored)
content           70,253 tokens
accounted         80,000 tokens (+0 vs budget)
truncated         2
note              index trimmed from 658,882 to 9,600 tokens
note              scanned the top 3,000 of 23,432 files by rank
```

Two things worth noting there: the index would have cost 658,882 tokens — more
than the entire budget — so it gets trimmed, and the report says so instead of
silently dropping 23,000 entries.

### 4. Tells you when it leaves budget unused

A bundle that uses 60% of its budget looks like a bug. It usually isn't, so
ctxpack names the reason:

```
note   left ~4,861 tokens unused; the main blocker was .py share cap (7 files)
```

---

## Token counting

This is the part most tools hand-wave, so it's the part with numbers attached.

With `tiktoken` installed, ctxpack counts exactly. Without it, it uses a bundled
estimator — no network, no 2MB ranks table, works in a sandboxed CI container.

The estimator is a per-class characters-per-token model fitted against
`tiktoken o200k_base` on 1,200 real files across nine languages:

```
train MAPE 7.62%   HOLDOUT MAPE 8.11%
```

Per-language, measured separately:

| | files | mean error | p90 | over-counts |
|---|---:|---:|---:|---:|
| Python | 215 | 5.7% | 10.6% | 54% |
| TypeScript | 220 | 5.2% | 10.4% | 30% |
| Markdown | 220 | 6.0% | 13.3% | 35% |
| JSON | 220 | 9.5% | 15.3% | 10% |
| YAML | 220 | 8.0% | 17.1% | 40% |

Over-counting is the safe direction: the bundle stays inside its budget.

Want better numbers on *your* code? Fit them:

```bash
ctxpack calibrate ~/code --write
```

```
raw estimator MAPE 6.37%
least-squares factor 0.9848
MAPE after scaling 6.66%
```

### Two things the fit taught me

Both of these are documented in `src/ctxpack/tokens.py`, because the reasoning is
the interesting part.

**Do not floor each piece at one token.** It's physically true that BPE can't
encode `"hello"` as zero tokens. It's also measurably wrong: the `cl100k`/`o200k`
split pattern *over-segments* relative to real BPE merges, because adjacent
pieces get merged back together. A small Python method splits into 22 pre-tokens
and 16 real ones. Flooring each piece made error go from **8% to 18%**, with
every fitted ratio jammed against its ceiling trying to compensate. So this is
not a piece-counting model at all — it's a character-mass estimator whose free
prefix absorbs short pieces, which is what the data actually supports.

**Some parameters aren't identifiable, and fitting them anyway is a lie.** The
digit and underscore classes are almost entirely 1–3 character pieces that fall
inside their own free prefix, so their ratios change nothing. Left unconstrained
the optimiser reported "1.2 chars per token" for `__` — meaningless, and
misleading to anyone reading the source. Only four of the seven classes are
fitted; the rest keep principled values.

`scripts/fit_params.py` reproduces the fit, including the random restarts
(coordinate descent on a lumpy objective landed in different basins across runs)
and the held-out split.

---

## Commands

```bash
ctxpack <path> -b 60000 -o context.xml -f xml   # pack (default subcommand)
ctxpack count . --limit 25                      # per-file tokens, density, rank
ctxpack explain . --filter 'src/*'              # why files rank where they do
ctxpack calibrate . --write                     # fit the estimator to your code
```

### Modes

| mode | behaviour |
|---|---|
| `balanced` *(default)* | rank order under all diversity caps |
| `coverage` | round-robin across directories — breadth first, for when you don't know what matters yet |
| `depth` | pure rank order, caps relaxed in spirit |

### Formats

`markdown` (default) · `xml` · `json` · `tree`

`tree` is the cheap one — just the index, no file bodies — which is often all
you want to paste first.

The renderers don't trust your source files: fences escalate past any ``` inside
content, CDATA sections split on `]]>`, and XML attributes are escaped with the
stdlib. There are tests for all three.

### Useful flags

```
-b, --budget TOKENS      total budget (default 32000)
-m, --mode               balanced | coverage | depth
-f, --format             markdown | xml | json | tree
-x, --exclude GLOB       skip paths (repeatable)
-i, --include GLOB       only these paths (repeatable)
    --per-dir-frac       cap one directory's share (default 0.40)
    --dedupe-threshold   collapse files this similar (default 0.85)
    --manifest           full | paths | none
    --stats              summary to stderr
```

---

## Honest limitations

- **Short strings are less accurate proportionally.** `"hello world"` scores 1
  against a true 2. The absolute error is about one token, which is irrelevant
  when pricing files of thousands — but don't use it to count a one-liner.
- **gitignore support is a subset, and one deviation is deliberate.** A `dir/`
  pattern also matches a *file* named `dir`, and — as in git — a `!` re-include
  cannot rescue something whose parent directory was already excluded, since
  ctxpack stops descending there. `**/` is handled properly. Both are noted in
  `walk.py`.
- **`--include` / `--exclude` use shell glob semantics**, not gitignore
  semantics: `*` crosses `/`. They're a different job and shouldn't surprise you.
- **Duplicate detection is lexical**, not semantic. It collapses the same file
  copied around; it will not collapse two files that merely *mean* the same thing.
- **The ranking weights are hand-tuned.** They're explainable and inspectable via
  `ctxpack explain`, which is the best you can say about any such heuristic.
- **It reads whole files into memory** for the top `--max-scan` (default 3000)
  files. On a repository with hundreds of thousands of text files, lower it.

---

## Development

```bash
git clone https://github.com/Matthew-Selvam/ctxpack
cd ctxpack
pip install -e ".[dev,exact]"
pytest -q
ruff check src tests scripts
```

The test suite runs both with and without `tiktoken`, because "zero dependencies"
is a claim that needs verifying: if the estimator tests start needing
`tiktoken`, the claim is wrong and CI says so.

## License

MIT