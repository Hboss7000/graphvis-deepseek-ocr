# Small OBQA retrieval comparison

These new wrappers compare the vendored MHGRN path-based retrieval in
`MHGRN/` with the project's current QA-GNN inputs. They do not modify either
vendor checkout or `scripts/generate_graphvis_datasets.py`, and every command
requires an explicit, small list of question indices.

## Environment

From the repository root:

```bash
python3 -m venv venv_mhgrn
venv_mhgrn/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
venv_mhgrn/bin/pip install -r mhgrn_code/requirements_mhgrn_cpu.txt
```

The two required TransE arrays must be supplied at:

```text
data/transe/glove.transe.sgd.ent.npy
data/transe/glove.transe.sgd.rel.npy
```

Do not regenerate missing OBQA grounding: that legacy step requires spaCy 2.x.
The wrappers contain a local `nx.read_gpickle` compatibility shim, but the
comparison environment intentionally pins `networkx<3` to match MHGRN.

## Small comparison

The following example processes eleven test questions, in the specified order:

```bash
venv_mhgrn/bin/python mhgrn_code/mhgrn_preflight.py \
  --split test --indices 60,0,1,2,3,4,5,6,7,8,9

venv_mhgrn/bin/python mhgrn_code/run_mhgrn_subset.py \
  --split test --indices 60,0,1,2,3,4,5,6,7,8,9 \
  --out-dir outputs/retrieval_compare --num-processes 1

tail -f outputs/retrieval_compare/test/mhgrn_run.log

venv_mhgrn/bin/python mhgrn_code/compare_retrieval.py \
  --split test --indices 60,0,1,2,3,4,5,6,7,8,9 \
  --out-dir outputs/retrieval_compare \
  --max-nodes 18 --max-edges 60 --max-degree 0 \
  --bridge-rule qa-bridge --engine dot --dpi 200
```

The comparison refuses to overwrite existing files. Use a fresh `--out-dir`
for another run. Montages are written as
`outputs/retrieval_compare/test/images/qNNNNN_compare.png`; numeric results are
in `comparison.jsonl` and `summary.md`.

Run the isolated synthetic tests with:

```bash
venv_mhgrn/bin/python -m pytest -q mhgrn_code/test_retrieval_compare.py
```
