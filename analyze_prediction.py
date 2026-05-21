"""
Analysis Script for Eta Product Transformer (with Sums support)

Metrics:
- Greedy & beam: skeleton match (set-based, order-independent, ignoring coefficients)
- Brute-force coefficient search on every predicted skeleton
- Exact match after brute-force (skeleton + coefficients)
- Coefficient verification (recompute from predicted formula)
- Breakdown by number of sum terms, factors, eta weight, etc.
- Error examples & correct examples

Usage:
    python analyze_predictions.py \
        --checkpoint ./checkpoints/checkpoint_best.pt \
        --data_path ./data_eta_product/eta_product_dataset.pkl \
        --output_dir ./analysis \
        --beam_width 5
"""

import torch
import torch.nn.functional as F
import pickle
import argparse
import numpy as np
from fractions import Fraction
from pathlib import Path
from tqdm import tqdm
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple
from math import lcm
from functools import reduce
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer_eta_product import (
    EtaProductTransformer, EtaProductVocabulary, ModelConfig, SyntaxMasker,
)
from dataset_generator import (
    compute_formula_coefficients, compute_raw_product,
    compute_eta_weight, build_phi_cache,
)


# ============================================================================
# FORMULA COMPARISON
# ============================================================================

def normalize_formula(formula: Dict) -> Optional[set]:
    """Convert formula to set of (coeff, shift, frozenset(factors)) for exact comparison."""
    if not formula or 'sum_terms' not in formula:
        return None
    terms = set()
    for term in formula['sum_terms']:
        coeff = term.get('coeff', 1)
        shift = term.get('shift', 0)
        factors = frozenset(
            (Fraction(k) if isinstance(k, int) else k, exp)
            for k, exp in term['factors']
        )
        terms.add((coeff, shift, factors))
    return terms


def normalize_skeleton(formula: Dict) -> Optional[set]:
    """Convert formula to set of (shift, frozenset(factors)) — ignoring coefficients."""
    if not formula or 'sum_terms' not in formula:
        return None
    terms = set()
    for term in formula['sum_terms']:
        shift = term.get('shift', 0)
        factors = frozenset(
            (Fraction(k) if isinstance(k, int) else k, exp)
            for k, exp in term['factors']
        )
        terms.add((shift, factors))
    return terms


def compare_formulas(pred: Dict, true: Dict) -> Dict:
    """Compare predicted vs true formula (exact, with coefficients)."""
    pred_set = normalize_formula(pred)
    true_set = normalize_formula(true)

    if pred_set is None or true_set is None:
        n_true = len(true['sum_terms']) if true and 'sum_terms' in true else 0
        return {
            'valid': pred_set is not None,
            'exact': False,
            'matching_terms': 0,
            'pred_terms': 0,
            'true_terms': n_true,
        }

    return {
        'valid': True,
        'exact': pred_set == true_set,
        'matching_terms': len(pred_set & true_set),
        'pred_terms': len(pred_set),
        'true_terms': len(true_set),
    }


def compare_skeletons(pred: Dict, true: Dict) -> Dict:
    """Compare predicted vs true skeleton (ignoring coefficients)."""
    pred_set = normalize_skeleton(pred)
    true_set = normalize_skeleton(true)

    if pred_set is None or true_set is None:
        n_true = len(true['sum_terms']) if true and 'sum_terms' in true else 0
        return {
            'valid': pred_set is not None,
            'skeleton_exact': False,
            'matching_terms': 0,
            'pred_terms': 0,
            'true_terms': n_true,
        }

    return {
        'valid': True,
        'skeleton_exact': pred_set == true_set,
        'matching_terms': len(pred_set & true_set),
        'pred_terms': len(pred_set),
        'true_terms': len(true_set),
    }


# ============================================================================
# BRUTE-FORCE COEFFICIENT SEARCH
# ============================================================================

def bruteforce_coefficients(formula: Dict, true_coeffs: np.ndarray,
                            phi_cache: Dict, max_coeff: int = 20,
                            n_coeffs: int = 50) -> Optional[Dict]:
    """
    Take a predicted skeleton and brute-force search the best integer
    coefficient for each term.

    Term 0 is always fixed to coeff=1.
    Terms 1..N are searched in [1, max_coeff].
    Handles coeff=None (placeholder) transparently.

    Returns dict with corrected formula, match info, or None on failure.
    """
    if formula is None or 'sum_terms' not in formula:
        return None
    terms = formula['sum_terms']

    max_order = n_coeffs + 20
    true = true_coeffs[:n_coeffs].astype(np.float64)

    # Compute each term's raw series separately
    term_series = []
    for term in terms:
        raw = compute_raw_product(term['factors'], phi_cache, max_order)
        if raw is None:
            return None
        term_series.append((raw, term.get('shift', 0)))

    # Find common grid
    all_powers = set()
    for raw, shift in term_series:
        for p in raw:
            all_powers.add(p + shift)

    if not all_powers:
        return None

    min_p = min(all_powers)

    # Find the step (LCM denominator)
    denoms = set()
    for p in all_powers:
        if isinstance(p, Fraction):
            denoms.add(p.denominator)
        else:
            denoms.add(1)
    lcm_d = reduce(lcm, denoms, 1)

    def to_array(raw_poly, shift, length):
        arr = np.zeros(length, dtype=np.float64)
        for p, c in raw_poly.items():
            sp = p + shift
            idx_frac = (sp - min_p) * lcm_d
            idx = int(round(float(idx_frac)))
            if 0 <= idx < length:
                arr[idx] = c
        return arr

    n_grid = n_coeffs

    # Term 0 always has coeff=1
    base = to_array(term_series[0][0], term_series[0][1], n_grid)

    if len(terms) == 1:
        # Single term: coeff is always 1, just check match
        n_match = int(np.sum(np.abs(base[:len(true)] - true) < 0.5))
        return {
            'formula': {'sum_terms': [{'coeff': 1, 'shift': terms[0].get('shift', 0),
                                        'factors': terms[0]['factors']}]},
            'coeffs_found': [1],
            'n_match': n_match,
            'n_total': n_coeffs,
            'all_match': n_match == n_coeffs,
        }

    # Multi-term: search coefficients for terms 1..N
    other_arrays = []
    for i in range(1, len(terms)):
        arr = to_array(term_series[i][0], term_series[i][1], n_grid)
        other_arrays.append(arr)

    best_coeffs = [1]
    best_total = np.copy(base)

    for j, other in enumerate(other_arrays):
        best_c = 1
        best_matches = -1

        for c in range(1, max_coeff + 1):
            candidate = best_total + c * other
            n_match = int(np.sum(np.abs(candidate[:len(true)] - true) < 0.5))
            if n_match > best_matches:
                best_matches = n_match
                best_c = c

        best_coeffs.append(best_c)
        best_total = best_total + best_c * other

    # Final match count
    n_match = int(np.sum(np.abs(best_total[:len(true)] - true) < 0.5))
    all_match = n_match == n_coeffs

    # Build corrected formula
    corrected = {'sum_terms': []}
    for i, term in enumerate(terms):
        corrected['sum_terms'].append({
            'coeff': best_coeffs[i],
            'shift': term.get('shift', 0),
            'factors': term['factors'],
        })

    return {
        'formula': corrected,
        'coeffs_found': best_coeffs,
        'n_match': n_match,
        'n_total': n_coeffs,
        'all_match': all_match,
    }


# ============================================================================
# BEAM SEARCH
# ============================================================================

def beam_search(model, coeffs, vocab, beam_width=5, max_len=80, masker=None):
    """Beam search with optional syntax masking. Returns list of (tokens, log_prob)."""
    model.eval()
    device = coeffs.device

    with torch.no_grad():
        src = model.encode_coefficients(coeffs)
        memory = model.transformer.encoder(src)

        beams = [([vocab.start_idx], 0.0)]
        finished = []

        for _ in range(max_len):
            candidates = []
            for tokens, log_prob in beams:
                if tokens[-1] == vocab.end_idx:
                    finished.append((tokens, log_prob))
                    continue

                tgt_tokens = torch.tensor([tokens], device=device)
                tgt_len = tgt_tokens.size(1)
                positions = torch.arange(tgt_len, device=device)
                tgt = model.token_embedding(tgt_tokens)
                tgt = tgt + model.token_pos_encoding(positions).unsqueeze(0)
                tgt_mask = model.generate_square_subsequent_mask(tgt_len).to(device)
                output = model.transformer.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
                logits = model.output_projection(output[:, -1, :])

                if masker is not None:
                    syntax_mask = masker.create_mask(tokens, vocab.vocab_size).to(device)
                    logits = logits + syntax_mask.unsqueeze(0)

                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                topk_lp, topk_idx = torch.topk(log_probs, beam_width)

                for lp, tok in zip(topk_lp, topk_idx):
                    candidates.append((tokens + [tok.item()], log_prob + lp.item()))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

            if all(t[-1] == vocab.end_idx for t, _ in beams):
                finished.extend(beams)
                break

        finished.extend(beams)
        finished.sort(key=lambda x: x[1], reverse=True)
        # Deduplicate
        seen = set()
        unique = []
        for tokens, lp in finished:
            key = tuple(tokens)
            if key not in seen:
                seen.add(key)
                unique.append((tokens, lp))

    return unique[:beam_width]


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ETA PRODUCT TRANSFORMER — ANALYSIS")
    print("=" * 80)

    # --- Load data ---
    print(f"\nLoading dataset: {args.data_path}")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    dataset_cfg = data.get('config', {})

    split = 'test' if 'test' in data else 'val'
    val = data[split]
    n_total = len(val['coefficients'])
    n_samples = n_total if args.num_samples == 'all' else min(int(args.num_samples), n_total)
    print(f"  Split: {split}, total: {n_total}, analyzing: {n_samples}")

    # --- Reconstruct vocab ---
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt['config']
    ckpt_args = ckpt.get('args', {})

    denominators = ckpt.get('vocab_denominators',
                   ckpt_args.get('denominators',
                   dataset_cfg.get('denominators', [1])))
    max_exp = ckpt.get('vocab_max_exp', None)
    if max_exp is None:
        max_exp = max(abs(dataset_cfg.get('min_exp', -6)), abs(dataset_cfg.get('max_exp', 6)))
    max_k = ckpt_args.get('max_k', dataset_cfg.get('max_k', 12))

    max_num_coeff = ckpt.get('vocab_max_num_coeff',
                    ckpt_args.get('max_num_coeff',
                    dataset_cfg.get('max_num_coeff', 1)))
    max_shift = ckpt.get('vocab_max_shift',
                ckpt_args.get('max_shift',
                dataset_cfg.get('max_shift', 0)))

    vocab = EtaProductVocabulary(
        denominators=denominators, max_k=max_k, max_exp=max_exp,
        max_num_coeff=max_num_coeff, max_shift=max_shift,
    )

    # Auto-fix vocab size mismatch
    target_size = ckpt['model_state_dict']['token_embedding.weight'].shape[0]
    if vocab.vocab_size != target_size:
        print(f"  ⚠ Vocab mismatch: {vocab.vocab_size} vs {target_size}, searching...")
        found = False
        for try_mc in [1, 2, 3, 5, 10, max_num_coeff]:
            for try_ms in [0, 5, 10, max_shift]:
                for try_me in [max_exp, 6, 12, 24]:
                    v = EtaProductVocabulary(
                        denominators=denominators, max_k=max_k, max_exp=try_me,
                        max_num_coeff=try_mc, max_shift=try_ms,
                    )
                    if v.vocab_size == target_size:
                        vocab = v
                        max_exp, max_num_coeff, max_shift = try_me, try_mc, try_ms
                        found = True
                        break
                if found: break
            if found: break
        if not found:
            raise RuntimeError(f"Cannot match vocab size {target_size}")

    print(f"  Vocab: {vocab.vocab_size} tokens (max_exp={max_exp}, max_k={max_k}, "
          f"max_coeff={max_num_coeff}, max_shift={max_shift})")

    # --- Build model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = EtaProductTransformer(config, vocab)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    masker = SyntaxMasker(vocab) if args.syntax_masking else None

    # --- Phi cache for coefficient verification ---
    ds_denoms = dataset_cfg.get('denominators', denominators)
    ds_max_k = dataset_cfg.get('max_k', max_k)
    n_coeffs = dataset_cfg.get('n_coeffs', 50)
    print(f"  Building phi cache (denoms={ds_denoms}, max_k={ds_max_k})...")
    phi_cache = build_phi_cache(ds_denoms, max_k=ds_max_k, max_order=n_coeffs + 20)
    ds_max_num_coeff = dataset_cfg.get('max_num_coeff', max_num_coeff)

    # --- Run predictions ---
    print(f"\n{'='*80}")
    print(f"GENERATING PREDICTIONS (beam_width={args.beam_width})")
    print(f"{'='*80}")

    results = []

    for idx in tqdm(range(n_samples)):
        coeffs_np = val['coefficients'][idx].astype(np.float64)
        coeffs_t = torch.tensor(coeffs_np, dtype=torch.float32).unsqueeze(0).to(device)

        true_formula = val['formulas'][idx]
        # Normalize legacy format
        if 'sum_terms' not in true_formula:
            true_formula = {'sum_terms': [{'coeff': 1, 'shift': 0, 'factors': true_formula['factors']}]}

        # --- Greedy ---
        greedy_idx = model.generate_greedy(coeffs_t)
        greedy_tokens = vocab.decode(greedy_idx)
        greedy_formula = vocab.tokens_to_formula(greedy_tokens)
        greedy_skel = compare_skeletons(greedy_formula, true_formula)

        # Brute-force coefficients on greedy skeleton
        greedy_bf = bruteforce_coefficients(
            greedy_formula, coeffs_np, phi_cache,
            max_coeff=ds_max_num_coeff, n_coeffs=n_coeffs,
        )
        greedy_bf_exact = False
        if greedy_bf and greedy_bf['all_match']:
            bf_cmp = compare_formulas(greedy_bf['formula'], true_formula)
            greedy_bf_exact = bf_cmp['exact']

        r = {
            'idx': idx,
            'true_formula': true_formula,
            'true_coeffs': coeffs_np,
            'greedy_tokens': greedy_tokens,
            'greedy_formula': greedy_formula,
            'greedy_skel': greedy_skel,
            'greedy_bf': greedy_bf,
            'greedy_bf_exact': greedy_bf_exact,
            'beam_results': [],
            'beam_skel_rank': -1,
            'beam_bf_rank': -1,
        }

        # --- Beam ---
        if args.beam_width > 1:
            beams = beam_search(model, coeffs_t, vocab, args.beam_width, masker=masker)
            for rank, (b_idx, b_lp) in enumerate(beams):
                b_tokens = vocab.decode(b_idx)
                b_formula = vocab.tokens_to_formula(b_tokens)
                b_skel = compare_skeletons(b_formula, true_formula)

                # Brute-force on every beam skeleton
                b_bf = bruteforce_coefficients(
                    b_formula, coeffs_np, phi_cache,
                    max_coeff=ds_max_num_coeff, n_coeffs=n_coeffs,
                )
                b_bf_exact = False
                if b_bf and b_bf['all_match']:
                    bf_cmp = compare_formulas(b_bf['formula'], true_formula)
                    b_bf_exact = bf_cmp['exact']

                r['beam_results'].append({
                    'tokens': b_tokens, 'formula': b_formula, 'log_prob': b_lp,
                    'skel': b_skel, 'bf': b_bf, 'bf_exact': b_bf_exact,
                })

                if b_skel['skeleton_exact'] and r['beam_skel_rank'] < 0:
                    r['beam_skel_rank'] = rank
                if b_bf_exact and r['beam_bf_rank'] < 0:
                    r['beam_bf_rank'] = rank

        results.append(r)

    # --- Compute stats ---
    print(f"\n{'='*80}")
    print("COMPUTING STATISTICS")
    print(f"{'='*80}")

    # Global counters
    g_skel_exact, g_valid = 0, 0
    g_bf_coeff_match, g_bf_exact = 0, 0

    b_skel_top1, b_skel_top3, b_skel_top5, b_skel_any = 0, 0, 0, 0
    b_bf_top1, b_bf_top3, b_bf_top5, b_bf_any = 0, 0, 0, 0
    b_bf_coeff_any = 0

    g_partial_num, g_partial_den = 0, 0

    # Breakdown accumulators
    by_n_sum_terms = defaultdict(lambda: {'n': 0, 'g_skel': 0, 'g_bf': 0, 'b_skel': 0, 'b_bf': 0})
    by_n_factors = defaultdict(lambda: {'n': 0, 'g_skel': 0, 'g_bf': 0, 'b_skel': 0, 'b_bf': 0})
    by_max_exp = defaultdict(lambda: {'n': 0, 'g_skel': 0, 'g_bf': 0, 'b_skel': 0, 'b_bf': 0})

    # Component-level stats
    eta_k_correct = Counter()
    eta_k_total = Counter()
    eta_k_confused = Counter()
    exp_correct = Counter()
    exp_total = Counter()
    exp_confused = Counter()
    factor_correct, factor_total = 0, 0
    skeleton_correct = 0

    for r in results:
        gs = r['greedy_skel']
        if gs['valid']:
            g_valid += 1
        if gs['skeleton_exact']:
            g_skel_exact += 1
        g_partial_num += gs['matching_terms']
        g_partial_den += gs['true_terms']

        gbf = r['greedy_bf']
        if gbf and gbf['all_match']:
            g_bf_coeff_match += 1
        if r['greedy_bf_exact']:
            g_bf_exact += 1

        # Beam skeleton rank
        bsr = r['beam_skel_rank']
        if bsr >= 0:
            b_skel_any += 1
            if bsr < 1: b_skel_top1 += 1
            if bsr < 3: b_skel_top3 += 1
            if bsr < 5: b_skel_top5 += 1

        # Beam BF rank
        bbr = r['beam_bf_rank']
        if bbr >= 0:
            b_bf_any += 1
            if bbr < 1: b_bf_top1 += 1
            if bbr < 3: b_bf_top3 += 1
            if bbr < 5: b_bf_top5 += 1

        if any(br['bf'] and br['bf']['all_match'] for br in r['beam_results']):
            b_bf_coeff_any += 1

        # Breakdown keys
        tf = r['true_formula']
        n_st = len(tf['sum_terms'])
        n_fac = max(len(t['factors']) for t in tf['sum_terms'])
        max_ae = max(abs(e) for t in tf['sum_terms'] for _, e in t['factors'])
        g_skel_ok = gs['skeleton_exact']
        g_bf_ok = r['greedy_bf_exact']
        b_skel_ok = bsr >= 0
        b_bf_ok = bbr >= 0

        for bk, val_key in [(by_n_sum_terms, n_st), (by_n_factors, n_fac), (by_max_exp, max_ae)]:
            bk[val_key]['n'] += 1
            bk[val_key]['g_skel'] += g_skel_ok
            bk[val_key]['g_bf'] += g_bf_ok
            bk[val_key]['b_skel'] += b_skel_ok
            bk[val_key]['b_bf'] += b_bf_ok

        # --- Component-level analysis (on greedy) ---
        pf = r['greedy_formula']
        if pf is None or 'sum_terms' not in pf:
            for term in tf['sum_terms']:
                for k, exp in term['factors']:
                    k_str = str(Fraction(k) if isinstance(k, int) else k)
                    eta_k_total[k_str] += 1
                    exp_total[exp] += 1
                    factor_total += 1
            continue

        true_terms = tf['sum_terms']
        pred_terms = pf['sum_terms']

        # Align terms by factor overlap (ignoring coefficients)
        used_pred = set()
        alignments = []
        for ti, tt in enumerate(true_terms):
            true_factors = set((Fraction(k) if isinstance(k, int) else k, e) for k, e in tt['factors'])
            best_pi, best_overlap = -1, -1
            for pi, pt in enumerate(pred_terms):
                if pi in used_pred:
                    continue
                pred_factors = set((Fraction(k) if isinstance(k, int) else k, e) for k, e in pt['factors'])
                overlap = len(true_factors & pred_factors)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pi = pi
            if best_pi >= 0:
                used_pred.add(best_pi)
                alignments.append((ti, best_pi))
            else:
                alignments.append((ti, -1))

        all_factors_match = True
        for ti, pi in alignments:
            tt = true_terms[ti]

            if pi < 0:
                for k, exp in tt['factors']:
                    k_str = str(Fraction(k) if isinstance(k, int) else k)
                    eta_k_total[k_str] += 1
                    exp_total[exp] += 1
                    factor_total += 1
                all_factors_match = False
                continue

            pt = pred_terms[pi]

            true_facs = {(Fraction(k) if isinstance(k, int) else k): e for k, e in tt['factors']}
            pred_facs = {(Fraction(k) if isinstance(k, int) else k): e for k, e in pt['factors']}

            term_factors_ok = True
            for k_frac in true_facs:
                k_str = str(k_frac)
                eta_k_total[k_str] += 1
                true_exp = true_facs[k_frac]
                exp_total[true_exp] += 1
                factor_total += 1

                if k_frac in pred_facs:
                    eta_k_correct[k_str] += 1
                    pred_exp = pred_facs[k_frac]
                    if true_exp == pred_exp:
                        exp_correct[true_exp] += 1
                        factor_correct += 1
                    else:
                        exp_confused[(true_exp, pred_exp)] += 1
                        term_factors_ok = False
                else:
                    term_factors_ok = False
                    for pk in pred_facs:
                        if pk not in true_facs:
                            eta_k_confused[(k_str, str(pk))] += 1
                            break

            if not term_factors_ok:
                all_factors_match = False

        if all_factors_match:
            skeleton_correct += 1

    # --- Write report ---
    report_path = output_dir / 'report.txt'
    with open(report_path, 'w') as f:
        def w(s=''):
            f.write(s + '\n')
            print(s)

        w("=" * 80)
        w("RESULTS SUMMARY")
        w("=" * 80)
        w(f"Samples: {n_samples}  |  Beam width: {args.beam_width}  |  Syntax mask: {args.syntax_masking}")
        w(f"Epoch: {ckpt.get('epoch', '?')}")
        w(f"BF max_coeff: {ds_max_num_coeff}")
        w()

        pct = lambda a, b: f"{a/b*100:.1f}%" if b > 0 else "N/A"

        w("GREEDY:")
        w(f"  Valid syntax:           {g_valid:5d} / {n_samples}  ({pct(g_valid, n_samples)})")
        w(f"  Skeleton exact:         {g_skel_exact:5d} / {n_samples}  ({pct(g_skel_exact, n_samples)})")
        w(f"  BF coeff all match:     {g_bf_coeff_match:5d} / {n_samples}  ({pct(g_bf_coeff_match, n_samples)})")
        w(f"  BF exact (skel+coeff):  {g_bf_exact:5d} / {n_samples}  ({pct(g_bf_exact, n_samples)})")
        w(f"  Partial term match:     {g_partial_num:5d} / {g_partial_den}  ({pct(g_partial_num, g_partial_den)})")
        w()

        if args.beam_width > 1:
            w(f"BEAM SKELETON (width={args.beam_width}):")
            w(f"  Top-1 skeleton:  {b_skel_top1:5d} / {n_samples}  ({pct(b_skel_top1, n_samples)})")
            w(f"  Top-3 skeleton:  {b_skel_top3:5d} / {n_samples}  ({pct(b_skel_top3, n_samples)})")
            w(f"  Top-5 skeleton:  {b_skel_top5:5d} / {n_samples}  ({pct(b_skel_top5, n_samples)})")
            w(f"  Any skeleton:    {b_skel_any:5d} / {n_samples}  ({pct(b_skel_any, n_samples)})")
            w()

            w(f"BEAM + BRUTE-FORCE (width={args.beam_width}, max_coeff={ds_max_num_coeff}):")
            w(f"  Top-1 exact:     {b_bf_top1:5d} / {n_samples}  ({pct(b_bf_top1, n_samples)})")
            w(f"  Top-3 exact:     {b_bf_top3:5d} / {n_samples}  ({pct(b_bf_top3, n_samples)})")
            w(f"  Top-5 exact:     {b_bf_top5:5d} / {n_samples}  ({pct(b_bf_top5, n_samples)})")
            w(f"  Any exact:       {b_bf_any:5d} / {n_samples}  ({pct(b_bf_any, n_samples)})")
            w(f"  Any coeff match: {b_bf_coeff_any:5d} / {n_samples}  ({pct(b_bf_coeff_any, n_samples)})")
            w()

        # Breakdowns
        w("=" * 80)
        w("BREAKDOWN BY # SUM TERMS")
        w("-" * 80)
        for k in sorted(by_n_sum_terms):
            d = by_n_sum_terms[k]
            line = f"  {k} term(s): skel={d['g_skel']:4d}/{d['n']:4d} ({pct(d['g_skel'], d['n'])})"
            line += f"  bf={d['g_bf']:4d} ({pct(d['g_bf'], d['n'])})"
            if args.beam_width > 1:
                line += f"  |  beam_skel={d['b_skel']:4d} ({pct(d['b_skel'], d['n'])})"
                line += f"  beam_bf={d['b_bf']:4d} ({pct(d['b_bf'], d['n'])})"
            w(line)
        w()

        w("BREAKDOWN BY # FACTORS (max per term)")
        w("-" * 80)
        for k in sorted(by_n_factors):
            d = by_n_factors[k]
            line = f"  {k} factors: skel={d['g_skel']:4d}/{d['n']:4d} ({pct(d['g_skel'], d['n'])})"
            line += f"  bf={d['g_bf']:4d} ({pct(d['g_bf'], d['n'])})"
            if args.beam_width > 1:
                line += f"  |  beam_skel={d['b_skel']:4d} ({pct(d['b_skel'], d['n'])})"
                line += f"  beam_bf={d['b_bf']:4d} ({pct(d['b_bf'], d['n'])})"
            w(line)
        w()

        w("BREAKDOWN BY MAX |EXPONENT|")
        w("-" * 80)
        for k in sorted(by_max_exp):
            d = by_max_exp[k]
            line = f"  |e|≤{k:2d}: skel={d['g_skel']:4d}/{d['n']:4d} ({pct(d['g_skel'], d['n'])})"
            line += f"  bf={d['g_bf']:4d} ({pct(d['g_bf'], d['n'])})"
            if args.beam_width > 1:
                line += f"  |  beam_skel={d['b_skel']:4d} ({pct(d['b_skel'], d['n'])})"
                line += f"  beam_bf={d['b_bf']:4d} ({pct(d['b_bf'], d['n'])})"
            w(line)
        w()

        # Component-level accuracy
        w("=" * 80)
        w("COMPONENT-LEVEL ACCURACY (greedy)")
        w("=" * 80)
        w()

        w(f"  Skeleton correct (all factors match): {skeleton_correct:5d} / {n_samples}  ({pct(skeleton_correct, n_samples)})")
        w(f"  Factor (k,e) accuracy:  {factor_correct:5d} / {factor_total:5d}  ({pct(factor_correct, factor_total)})")
        w()

        # Accuracy per eta argument k
        w("ACCURACY BY η(q^k) ARGUMENT:")
        w("-" * 80)
        k_items = []
        for k_str in eta_k_total:
            tot = eta_k_total[k_str]
            cor = eta_k_correct.get(k_str, 0)
            k_items.append((k_str, cor, tot))
        def k_sort_key(x):
            try:
                return float(Fraction(x[0]))
            except:
                return 999
        k_items.sort(key=k_sort_key)
        for k_str, cor, tot in k_items:
            w(f"  η(q^{k_str:>3s}): {cor:5d} / {tot:5d}  ({pct(cor, tot)})")
        w()

        # Accuracy per exponent value
        w("ACCURACY BY EXPONENT:")
        w("-" * 80)
        for exp_val in sorted(exp_total.keys()):
            tot = exp_total[exp_val]
            cor = exp_correct.get(exp_val, 0)
            w(f"  e={exp_val:+2d}: {cor:5d} / {tot:5d}  ({pct(cor, tot)})")
        w()

        # Exponent confusion
        if exp_confused:
            w("EXPONENT CONFUSION (true → pred), top 15:")
            for (te, pe), cnt in exp_confused.most_common(15):
                w(f"  e={te:+2d} → e={pe:+2d}: {cnt}")
            w()

        # η(q^k) confusion
        if eta_k_confused:
            w("η(q^k) CONFUSION (true → pred), top 15:")
            for (tk, pk), cnt in eta_k_confused.most_common(15):
                w(f"  η(q^{tk}) → η(q^{pk}): {cnt}")
            w()

        # Examples: correct (skeleton + BF exact)
        correct = [r for r in results if r['greedy_bf_exact']]
        w("=" * 80)
        w(f"CORRECT EXAMPLES — greedy skeleton + BF (showing {min(10, len(correct))} of {len(correct)})")
        w("=" * 80)
        for r in correct[:10]:
            ts = vocab.formula_to_string(r['true_formula'])
            bf_str = vocab.formula_to_string(r['greedy_bf']['formula']) if r['greedy_bf'] else "?"
            w(f"  [{r['idx']:4d}] TRUE: {ts}")
            w(f"         BF:   {bf_str}")
            w(f"         coeffs found: {r['greedy_bf']['coeffs_found']}")
        w()

        # Examples: skeleton correct but BF coefficients wrong
        skel_ok_bf_wrong = [r for r in results
                            if r['greedy_skel']['skeleton_exact'] and not r['greedy_bf_exact']]
        if skel_ok_bf_wrong:
            w("=" * 80)
            w(f"SKELETON CORRECT BUT BF WRONG ({len(skel_ok_bf_wrong)} cases, showing {min(5, len(skel_ok_bf_wrong))})")
            w("=" * 80)
            for r in skel_ok_bf_wrong[:5]:
                ts = vocab.formula_to_string(r['true_formula'])
                w(f"  [{r['idx']:4d}] TRUE: {ts}")
                if r['greedy_bf']:
                    bf_str = vocab.formula_to_string(r['greedy_bf']['formula'])
                    w(f"         BF:   {bf_str}  (match={r['greedy_bf']['n_match']}/{r['greedy_bf']['n_total']})")
                if r['greedy_formula']:
                    ps = vocab.formula_to_string(r['greedy_formula'])
                    w(f"         PRED: {ps}")
            w()

        # Examples: errors
        errors = [r for r in results if not r['greedy_bf_exact']]
        n_show = min(args.max_errors, len(errors))
        w("=" * 80)
        w(f"ERROR EXAMPLES (showing {n_show} of {len(errors)})")
        w("=" * 80)
        for r in errors[:n_show]:
            ts = vocab.formula_to_string(r['true_formula'])
            w(f"\n  [{r['idx']:4d}] TRUE: {ts}")
            w(f"         coeffs[0:8] = {r['true_coeffs'][:8]}")

            gs = r['greedy_skel']
            if r['greedy_formula']:
                ps = vocab.formula_to_string(r['greedy_formula'])
                status = f"skel_match={gs['skeleton_exact']}, terms={gs['matching_terms']}/{gs['true_terms']}"
                if r['greedy_bf']:
                    status += f", bf_match={r['greedy_bf']['n_match']}/{r['greedy_bf']['n_total']}"
                    status += f", bf_coeffs={r['greedy_bf']['coeffs_found']}"
                w(f"         GREEDY: {ps}  ({status})")
                if r['greedy_bf'] and r['greedy_bf']['coeffs_found'] != [1] * len(r['greedy_bf']['coeffs_found']):
                    bf_str = vocab.formula_to_string(r['greedy_bf']['formula'])
                    w(f"         BF:     {bf_str}")
            else:
                w(f"         GREEDY: INVALID SYNTAX")
                w(f"         tokens: {' '.join(r['greedy_tokens'][:20])}...")

            if r['beam_results']:
                w(f"         BEAM (top {min(3, len(r['beam_results']))}):")
                for i, br in enumerate(r['beam_results'][:3]):
                    if br['formula']:
                        bs = vocab.formula_to_string(br['formula'])
                        skel_mark = "S" if br['skel']['skeleton_exact'] else " "
                        bf_mark = "✓" if br['bf_exact'] else " "
                        line = f"           {skel_mark}{bf_mark} [{i+1}] {bs}  (lp={br['log_prob']:.2f})"
                        if br['bf'] and br['bf']['all_match']:
                            bf_str = vocab.formula_to_string(br['bf']['formula'])
                            line += f"\n                  → BF: {bf_str} coeffs={br['bf']['coeffs_found']}"
                        w(line)
                    else:
                        w(f"             [{i+1}] INVALID")
        w()

    print(f"\nReport saved: {report_path}")

    # --- Plots ---
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Summary bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    labels = ['Valid\nsyntax', 'Skeleton\nexact', 'BF coeff\nmatch', 'BF exact\n(skel+coeff)']
    g_vals = [g_valid/n_samples*100, g_skel_exact/n_samples*100,
              g_bf_coeff_match/n_samples*100, g_bf_exact/n_samples*100]
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width/2, g_vals, width, label='Greedy', color='#3498db', edgecolor='black')
    if args.beam_width > 1:
        b_vals = [100, b_skel_any/n_samples*100, b_bf_coeff_any/n_samples*100, b_bf_any/n_samples*100]
        ax.bar(x + width/2, b_vals, width, label=f'Beam top-{args.beam_width}', color='#e74c3c', edgecolor='black')
    for i, v in enumerate(g_vals):
        ax.text(i - width/2, v + 1, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold')
    if args.beam_width > 1:
        for i, v in enumerate(b_vals):
            ax.text(i + width/2, v + 1, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Performance (Skeleton + Brute-Force)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'summary.png', dpi=150)
    plt.close()

    # Plot 2: By number of factors
    fig, ax = plt.subplots(figsize=(8, 4))
    ks = sorted(by_n_factors)
    vals_skel = [by_n_factors[k]['g_skel']/by_n_factors[k]['n']*100 for k in ks]
    vals_bf = [by_n_factors[k]['g_bf']/by_n_factors[k]['n']*100 for k in ks]
    ns = [by_n_factors[k]['n'] for k in ks]
    x = np.arange(len(ks))
    width = 0.35
    ax.bar(x - width/2, vals_skel, width, label='Skeleton', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, vals_bf, width, label='BF exact', color='#2ecc71', edgecolor='black')
    for i, (vs, vb, n) in enumerate(zip(vals_skel, vals_bf, ns)):
        ax.text(i, max(vs, vb)+2, f'n={n}', ha='center', fontsize=8)
    ax.set_xlabel('Max # factors per term')
    ax.set_ylabel('Greedy accuracy (%)')
    ax.set_title('Accuracy by # Factors')
    ax.set_xticks(x)
    ax.set_xticklabels(ks)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'by_factors.png', dpi=150)
    plt.close()

    # Plot 3: Beam improvement pipeline
    if args.beam_width > 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        labels_b = ['Greedy\nskeleton', 'Greedy\nBF exact',
                     f'Beam\nskeleton', f'Beam\nBF exact']
        vals_b = [g_skel_exact, g_bf_exact, b_skel_any, b_bf_any]
        vals_b = [v/n_samples*100 for v in vals_b]
        colors = ['#3498db', '#2980b9', '#e74c3c', '#c0392b']
        bars = ax.bar(labels_b, vals_b, color=colors, edgecolor='black')
        for bar, v in zip(bars, vals_b):
            ax.text(bar.get_x()+bar.get_width()/2, v+1, f'{v:.1f}%',
                    ha='center', fontsize=9, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Pipeline: Skeleton → Brute-Force Coefficients')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'pipeline.png', dpi=150)
        plt.close()

    print(f"Plots saved: {plots_dir}/")
    print(f"\n✅ Analysis complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./analysis')
    parser.add_argument('--num_samples', type=str, default='all')
    parser.add_argument('--beam_width', type=int, default=1)
    parser.add_argument('--max_errors', type=int, default=20)
    parser.add_argument('--syntax_masking', action='store_true')
    args = parser.parse_args()
    main(args)
