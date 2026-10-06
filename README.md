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
| **Budget accuracy** | fitted to tiktoken ground truth, **7.7%** token-weighted over 3,116 real files |
| **Budget overrun** | structurally impossible, verified across synthetic repo shapes in CI |
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
`tiktoken o200k_base`, then measured on 3,116 real files across 19 extensions:

```
train MAPE 7.62%   HOLDOUT MAPE 8.11%      (the fit, held out)
overall 7.48% mean error · 7.74% token-weighted · p90 15.9%   (independent)
```

Two error figures, because they answer different questions. *Mean error* treats
every file equally, which over-weights tiny ones — a 5-token `FUNDING.yml`
counted as 4 instead of 6 is "20% wrong" and costs nothing. *Token-weighted*
is total miscounted tokens over total tokens, which is what actually decides
whether a bundle fits.

| ext | files | median tokens | mean error | token-weighted | p90 |
|---|---:|---:|---:|---:|---:|
| `.js` | 220 | 1,442 | 4.7% | 6.6% | 9.4% |
| `.py` | 211 | 1,588 | 5.0% | 4.2% | 10.3% |
| `.rs` | 220 | 900 | 5.3% | 6.7% | 10.4% |
| `.toml` | 219 | 405 | 6.0% | 5.3% | 10.6% |
| `.ts` | 220 | 1,110 | 6.1% | 5.2% | 13.1% |
| `.md` | 220 | 880 | 6.2% | 5.9% | 14.0% |
| `.tsx` | 220 | 814 | 6.3% | 6.2% | 13.0% |
| `.yaml` | 220 | 57 | 7.5% | 10.3% | 16.9% |
| `.sh` | 220 | 650 | 8.1% | 9.4% | 14.7% |
| `.h` | 219 | 1,031 | 9.7% | 10.2% | 18.0% |
| `.json` | 220 | 116 | 10.0% | 14.0% | 16.0% |
| `.yml` | 220 | 350 | 14.7% | 11.9% | 24.2% |
| **all** | **3,116** | — | **7.48%** | **7.74%** | **15.9%** |

The two worst rows are the two with the smallest median file: short files are
where a character-mass model is weakest. Over-counting is the safe direction —
the bundle stays inside its budget.

Reproduce it, and catch regressions, with:

```bash
python3 scripts/bench.py --root ~/code --baseline bench.json --write
python3 scripts/bench.py --root ~/code --baseline bench.json --check
```

`bench.py` also builds adversarial synthetic repo shapes — 200 identical files,
one 20k-line file, 1500 tiny files, 40-deep nesting — because those are the
inputs that break greedy packers and none of them appear in a small sample of
real repos. CI asserts no budget is ever exceeded across them, and that every
rendered format still parses when fed deliberately hostile content.

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
ctxpack . --diff main...HEAD -b 40000           # only what the branch changed
ctxpack . --reach-weight 4                      # rank by import-graph distance
ctxpack . --outline                             # signatures, not bodies
```

### Config file

Stop retyping the same flags. ctxpack reads `ctxpack.toml` (or
`.ctxpack.toml`, or the `.json` equivalents) from the target directory or any
parent, stopping at a `.git` boundary:

```toml
# ctxpack.toml
budget = 60000
mode = "coverage"
format = "xml"
reach_weight = 4
exclude = ["**/__tests__/**", "*.snap"]
```

Precedence is **explicit flag > config file > built-in default**. The middle
case is the subtle one: argparse cannot tell a `-b 32000` a user typed from the
`32000` it filled in, so a config file would otherwise be un-overridable. ctxpack
recovers the set of flags actually typed by walking `argv` against the parser's
own option strings.

```console
$ ctxpack . --show-config
budget              60000  (config)
mode                coverage  (config)
format              markdown  (default)
```

A typo is reported rather than silently dropped -- and does not discard the rest
of the file:

```
ctxpack: ctxpack.toml: ignoring unknown key(s) 'budegt' (did you mean 'budget'?)
```

`exclude` and `include` **accumulate** (config first, then flags) rather than
replacing each other, since a shared exclude list plus a one-off addition is the
common case. Use `--no-config` to ignore the file entirely.

TOML needs `tomllib`, which is stdlib from 3.11. On 3.10 a `.toml` file says so
plainly and points at `.ctxpack.json`, which always works.

### Beyond the bundle

Five flags for the cases where "the most useful files" is not the same as "the
most important files".

**`--diff SPEC` — pack only what changed.** Reviewing a change is where the
whole repository is least useful. `--diff` takes any git range:

```bash
ctxpack . --diff main...HEAD          # merge-base diff, like git
ctxpack . --diff HEAD~3 --uncommitted # uncommitted work plus three commits
ctxpack . --diff HEAD --staged        # just the index
ctxpack . --diff HEAD --untracked     # including files git has never seen
```

`--untracked` exists because `git diff` cannot see the brand-new file you just
wrote, which is frequently the exact file an agent needs. Deleted files are
excluded; renames resolve to their new path. Every `git` invocation is
argv-based (never a shell), timeout-bounded via `CTXPACK_GIT_TIMEOUT`, and ref
names are validated before they reach git.

**`--reach-weight W` — rank by the import graph, not by filename.** The
heuristics in `rank.py` are guesses about importance. Reachability from a
detected entrypoint is evidence. Imports are parsed with `ast` for Python and by
regex for JS/TS, including TypeScript's `NodeNext` convention where
`from './x.js'` means `./x.ts`; files closer to an entrypoint get a boost that
decays with distance.

```console
$ ctxpack ~/repos/some-ts-app -b 8000 --reach-weight 5 --stats
ctxpack: import graph 117/211 files reachable from 3 entrypoint(s)
```

**`--outline` — signatures instead of implementations.** A structural summary of
a 4,000-line module costs a fraction of its body, so the same budget holds far
more files:

```console
note  outline mode: 6 file(s) reduced to structure
      (median 13.8x cheaper than the full body); 9 left as full text.
      This bundle shows signatures, not implementations.
```

Outlining happens *before* selection, not after, so the freed budget buys more
files rather than the same files more cheaply: 21 files in a 20k budget instead
of 15. Files where outlining would not pay for itself (a 3-line module, a config
file) are left alone automatically.

**`--factor-shared` — hoist repeated blocks.** A licence header repeated across
40 files is one fact paid for 40 times.

**Honest measurement: this one is not worth much on modern code.** Factoring
saved **0.4–0.6%** across every real corpus tried, including a vendored crates
registry that still carries repeated headers. Modern code consolidated those
into single-line SPDX identifiers, so the premise is largely historical. It
saves ~10% on a synthetic repo built to suit it. It is kept because it is free
when off, it does help older and template-generated repositories, and reporting
"we measured it and it saves 0.4%" beats implying otherwise.

**`--watch` — rebuild while you work.** The edit/repack loop is the tedious part
of keeping an agent's context current:

```console
$ ctxpack . --watch -b 20000 -o context.md
ctxpack: wrote context.md (19,028 tokens exact, budget 20,000)
ctxpack: watching 39 files, rebuilding on change (every 0.5s, ctrl-C to stop)
ctxpack: modified src/ctxpack/boiler.py
ctxpack: wrote context.md (19,028 tokens exact, budget 20,000)
```

The first bundle is written *before* the loop starts, so `--watch` never blocks
with nothing on disk. A burst of writes — a formatter plus a save, an editor
writing a temp file and renaming it — is coalesced into one rebuild, because
spending the CPU once per edit rather than once per write is the entire point. A
rebuild that fails is reported and the loop continues: a syntax error in a file
mid-edit is the normal state of a repository being worked on, and a watcher that
dies on the first one is useless exactly when it is needed. The previous bundle
stays valid on disk.

The debounce is bounded (`max_settle`, 5s). Without that bound a repository that
keeps changing never goes quiet, the settle deadline keeps being pushed out, and
the rebuild never happens at all — better to rebuild on a moving target than
never to rebuild.

It polls `(mtime, size)` per file rather than subscribing to filesystem events,
because every cross-platform watcher for Python is a third-party package and this
has none. `--watch-interval` overrides the poll period, and
`CTXPACK_WATCH_INTERVAL` sets the same thing for every invocation. Precedence
matches the rest of the CLI: typed flag > environment > default.

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
    --diff SPEC          only files changed in this git range
    --reach-weight W     boost files reachable from entrypoints
    --outline            replace bodies with structural outlines
    --factor-shared      hoist repeated blocks into one section
    --watch              repack on change until interrupted
    --per-dir-frac       cap one directory's share (default 0.40)
    --no-config          ignore ctxpack.toml
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
- **A .gitignore pattern with more than two wildcards in one path segment is
  ignored.** `*a*a*a*b` cannot be matched without an exponential search -- it is
  14ms with two wildcards, 1.1s with three, and never finishes with four, against
  a 200-character path. Real patterns use one or two, so ctxpack degrades the
  pathological ones to matching nothing rather than hanging. It fails *open*: a
  file that should have been filtered out stays in the bundle, rather than a file
  that belongs there being hidden.
- **Malformed bracket expressions in .gitignore are treated as a literal `[`**
  rather than guessed at, matching git's own behaviour for a bracket it cannot
  parse.
- **The import graph is partial by design.** No `sys.path` emulation, no
  `node_modules`, no dynamic `import()`, no conditional imports, no
  `__getattr__` re-export discovery. Each of those would make the graph bigger
  and less trustworthy as a ranking input. Unresolvable imports are dropped
  rather than guessed, so the graph under-claims rather than over-claims.
- **A repo with no recognisable entrypoint gets no graph.** `main`, `index`,
  `__init__`, `app`, `server`, `cli`, `entry`, `bootstrap` are the only
  evidence available; nothing reads a manifest or a `package.json` `main` field.
  `--reach-weight` then says so and carries on with the static ranking.
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

925 tests, run both with and without `tiktoken`, because "zero dependencies" is a
claim that needs verifying: if the estimator tests start needing `tiktoken`, the
claim is wrong and CI says so.

The fuzzer in `tests/test_fuzz.py` found three real defects that no fixture had:
an unescaped index fence in the markdown renderer, a `.gitignore` character class
that raised `re.PatternError` straight out of `ctxpack .`, and adjacent unbounded
quantifiers that turned one `.gitignore` line into a multi-minute hang. All three
have regression tests. `scripts/bench.py --check` runs in CI as a canary for the
budget invariant and for render integrity against hostile content.

Review found a fourth, and the worst kind: `--factor-shared` could rewrite valid
Python into Python that does not parse. The conservatism rules all read a
candidate run *in isolation*, so a run cut out of the middle of an unclosed
bracket passed every one of them — a comment inside the bracket satisfies the
anchor rule, a closing `}` satisfies the unit-boundary rule — and hoisting it left
the opener dangling. Nothing errored; the bundle just contained broken source.
The fix is a bracket-depth and triple-quote-region check that requires depth zero
at both ends of a run. There are now 32 grid shapes asserting the property that
actually matters: valid Python in, valid Python out, whatever the guards decide.

Every one of those was found by a review or a test, not by reading the code. That
is the argument for keeping both.

## License

MIT
