# Symbolic Regression of Modular Forms via Transformers

This repository contains the code accompanying the paper *"Algorithmic and ML approaches to power series of
transcendental functions - I: A study of eta products
and holomorphic modular forms"*, arxiv number. The goal is to recover, from
the first Fourier coefficients of a modular form on some congruence group `Γ` of ` SL(2,Z)`, an explicit **eta product formula** that reproduces those coefficients — i.e. a
seq2seq problem mapping

```
a(1), a(2), …, a(N)   ⟼   Σ_i c_i · ∏_j η(m_{ij} τ)^{e_{ij}}
```

We use a transformer encoder–decoder trained on synthetically generated
formulas. Coefficients are predicted in **skeleton mode** (with a
`COEFF_PH` placeholder) and the integer coefficients `c_i` are recovered in a
post-hoc brute-force search over the candidate beams.

The broader research program (not all of it released here) explores related
modular-forms problems: predicting bases for Serre modular forms, and detecting
Swinnerton-Dyer congruences as anomaly-detection in eigenform coefficients.

## Cloning the repo 

Some files are quite heavy, so we had to use Large Storage File. You may need to pull them as follow 

```
git lfs install
git clone https://github.com/BastD/EtaProductsSR.git>
git lfs pull
```

## Repository layout

```
.
├── transformer_eta_product.py   # Model, vocabulary, syntax masking, embedding
├── dataset_generator.py         # Synthetic formula generation + q-series
├── train_tf_sr.py               # Training script (with checkpointing / resume)
├── analyze_prediction.py        # Beam-search evaluation + brute-force re-rank
├── plot_history.py              # Loss curves from training logs
└── dataset_stats.py             # Distribution analysis over a generated set
```

## `transformer_eta_product.py`

Core model code, shared by training and analysis. Provides:

- **Vocabulary and tokenization.** Tokens for the eta-product grammar: the
  `eta(m)` factors, integer exponents up to `±EXP_MAX`, the `COEFF_PH`
  placeholder for relative coefficients, and the structural tokens (`+`,`MUTL` ,`PAD`,...).
- **q-series embedding.** Sign + magnitude embedding of the input
  coefficients, with positional encoding indexed by the power of `q`.
- **Transformer encoder–decoder.** Pre-LN blocks, `d_model=256`, 4+4 layers,
  8 heads by default. Configurable from CLI in `train_tf_sr.py`.
- **Beam search.** Length-normalized beam search returning the top-`B`
  skeletons together with their log-probabilities, suitable for downstream
  brute-force coefficient recovery.

You should rarely call this module directly — it is imported by the three
scripts below.

## `dataset_generator.py`

Generates `(q-series → formula)` pairs.

Each formula is a sum of up to `--max_terms` eta products. Each term is a
product of up to `--max_factors` factors `η(m τ)^e`, with arguments
`m ∈ {1, …, max_arg}` and exponents `e ∈ {−EXP_MAX, …, EXP_MAX}\{0}`.
Relative coefficients between terms are integers in
`{−max_coeff, …, max_coeff}\{0}`, with the leading coefficient normalized to
`+1`. q-series are computed up to `q^{max_k}` using a memoized `φ`-function
expansion (Euler's pentagonal recursion).

### Example: generate a 1M-sample dataset of 3-term sums

```bash
python dataset_generator.py \
    --n_samples 1000000 \
    --max_terms 3 \
    --max_factors 4 \
    --max_arg 6 \
    --max_k 12 \
    --max_coeff 20 \
    --out_dir eta_product/data_3term_1M \
    --seed 0 \
    --n_workers 32
```

This writes a sharded dataset (`shard_XXXX.pt`) plus a `meta.json` with the
generation parameters. Duplicate formulas are filtered by canonical form
before being written.


## `train_tf_sr.py`

Trains the transformer in skeleton mode on a generated dataset.
The script handles checkpointing, auto-resume from the latest checkpoint in `--out_dir`.

### Minimal example

```bash
python train_tf_sr.py \
    --data_dir eta_product/data_3term_1M \
    --out_dir  eta_product/results_3term/run_1M \
    --d_model 256 \
    --n_enc 4 --n_dec 4 --n_heads 8 \
    --batch_size 512 \
    --lr 3e-4 \
    --warmup_steps 2000 \
    --max_steps 200000 \
    --val_every 2000 \
    --use_periodic
```

## `analyze_prediction.py`

Evaluates a trained checkpoint on a held-out set. Runs beam search with the
syntax mask, then performs **brute-force coefficient recovery**: for each
returned skeleton, the integer coefficients filling the `COEFF_PH` slots are
searched exhaustively in `{−max_coeff, …, max_coeff}\{0}` and the candidate
whose q-series matches the input is kept. The exact-match score reported in
the paper *already includes* this re-ranking step.

The script is parallelized with `multiprocessing.Pool` using the
`forkserver` start method, with per-worker model loading and
`OMP_NUM_THREADS=MKL_NUM_THREADS=1` set before any torch import to avoid
file-descriptor exhaustion.

### Example: evaluate a checkpoint at multiple beam widths

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python analyze_prediction.py \
    --ckpt /work/bduboeuf/eta_product/results_3term/run_1M/best.pt \
    --data_dir /work/bduboeuf/eta_product/data_3term_1M/val \
    --beam_widths 1 5 10 20 \
    --max_coeff 20 \
    --n_workers 16 \
    --out_json analysis_1M.json
```

### Output

The JSON export contains, per beam width:

- Exact-match accuracy (with and without coefficient re-rank)
- Per-token (skeleton) accuracy
- Confusion matrices for the eta arguments `m` and exponents `e`
- A breakdown of accuracy by formula complexity (number of factors, `max_k`,
  largest exponent)
- Beam scaling curves

It can be fed back into the plotting utilities to reproduce the paper's
figures.

## Reproducing the main result

The headline number in the paper (~88% exact match at beam 20 on the 2-term
benchmark) is reproduced by:

```bash
# 1. generate the dataset
python dataset_generator.py --n_samples 1000000 --max_terms 2 \
    --max_factors 4 --max_arg 6 --max_k 12 --max_coeff 20 \
    --out_dir data/2term_1M --n_workers 32

# 2. train
python train_tf_sr.py --data_dir data/2term_1M --out_dir runs/2term_1M \
    --d_model 256 --n_enc 4 --n_dec 4 --use_periodic --max_steps 200000

# 3. evaluate
python analyze_prediction.py --ckpt runs/2term_1M/best.pt \
    --data_dir data/2term_1M/val --beam_widths 1 5 10 20 \
    --out_json results/2term_1M.json
```

## Citation

```bibtex
@article{
}
```
