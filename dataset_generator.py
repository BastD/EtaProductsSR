"""
Eta Product Dataset Generator - with Sums, Coefficients, and Shifts

Generates formulas of the form:
  c_1 * q^{s_1} * PROD_1 + c_2 * q^{s_2} * PROD_2 + ...

where:
- c_i are positive integers from 1 to max_num_coeff
- s_i are non-negative integers from 0 to max_shift
- PROD_i = prod_j eta(q^{k_j})^{e_j}

Options:
- --product_only: all exponents positive (no quotients, no 1/eta)
- --modular: each product has integer eta weight (Σ(e_i*k_i) ≡ 0 mod 24)
- --same_weight: all sum terms have same modular weight k = Σ(e_i)/2
  (requires --modular)

The term with smallest eta weight always has coeff=1, guaranteeing
that the first coefficient of the q-series is 1.

Usage:
    # Products only (original behavior)
    python dataset_generator.py --denominators 1

    # Sums of 2 products with coefficients up to 5
    python dataset_generator.py --denominators 1 \
        --min_sum_terms 2 --max_sum_terms 2 --max_num_coeff 5

    # Modular eta products (integer weights), positive exponents only
    python dataset_generator.py --denominators 1 \
        --modular --product_only --max_exp 24

    # Modular sums with same weight
    python dataset_generator.py --denominators 1 \
        --min_sum_terms 2 --max_sum_terms 3 --max_num_coeff 10 \
        --modular --same_weight --product_only --max_exp 24
"""

import numpy as np
import pickle
import random
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from tqdm import tqdm
from pathlib import Path
from fractions import Fraction
import argparse
import math
import multiprocessing as mp


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def phi_function_q_power_int(n: int, max_order: int = 100) -> Dict[int, int]:
    result = {0: 1}
    k = 1
    while n * k <= max_order:
        nk = n * k
        new_result = defaultdict(int)
        for power, coeff in result.items():
            new_result[power] += coeff
            new_power = power + nk
            if new_power <= max_order:
                new_result[new_power] -= coeff
        result = dict(new_result)
        k += 1
    return result


def phi_function_q_power_inverse_int(n: int, max_order: int = 100) -> Dict[int, int]:
    result = {0: 1}
    k = 1
    while n * k <= max_order:
        nk = n * k
        new_result = defaultdict(int)
        for power, coeff in result.items():
            m = 0
            while power + m * nk <= max_order:
                new_result[power + m * nk] += coeff
                m += 1
        result = dict(new_result)
        k += 1
    return result


def phi_function_q_power_fraction(n: Fraction, max_order: int = 100) -> Dict[Fraction, int]:
    result = {Fraction(0): 1}
    k = 1
    while True:
        nk = n * k
        if nk > max_order:
            break
        new_result = defaultdict(int)
        for power, coeff in result.items():
            new_result[power] += coeff
            new_power = power + nk
            if new_power <= max_order:
                new_result[new_power] -= coeff
        result = dict(new_result)
        k += 1
    return result


def phi_function_q_power_inverse_fraction(n: Fraction, max_order: int = 100) -> Dict[Fraction, int]:
    result = {Fraction(0): 1}
    k = 1
    while True:
        nk = n * k
        if nk > max_order:
            break
        new_result = defaultdict(int)
        for power, coeff in result.items():
            m = 0
            while power + m * nk <= max_order:
                new_result[power + m * nk] += coeff
                m += 1
        result = dict(new_result)
        k += 1
    return result


# ============================================================================
# PHI CACHE
# ============================================================================

def build_phi_cache(denominators: List[int], max_k: int = 12, max_order: int = 100) -> Dict:
    cache = {'pos': {}, 'neg': {}}

    fractions_to_cache = []
    for k in range(1, max_k + 1):
        fractions_to_cache.append(Fraction(k, 1))
    for d in denominators:
        if d > 1:
            for p in range(1, d):
                if math.gcd(p, d) == 1:
                    fractions_to_cache.append(Fraction(p, d))

    fractions_to_cache = list(set(fractions_to_cache))

    for frac in tqdm(fractions_to_cache, desc="Pre-computing phi"):
        if frac.denominator == 1:
            pos = phi_function_q_power_int(int(frac), max_order)
            neg = phi_function_q_power_inverse_int(int(frac), max_order)
            cache['pos'][frac] = {Fraction(p): c for p, c in pos.items()}
            cache['neg'][frac] = {Fraction(p): c for p, c in neg.items()}
        else:
            cache['pos'][frac] = phi_function_q_power_fraction(frac, max_order)
            cache['neg'][frac] = phi_function_q_power_inverse_fraction(frac, max_order)

    return cache


def get_phi_from_cache(cache: Dict, frac: Fraction, positive: bool = True) -> Dict:
    key = 'pos' if positive else 'neg'
    return cache[key].get(frac, {Fraction(0): 1})


def compute_eta_weight(factors: List[Tuple[Fraction, int]]) -> Fraction:
    """
    Compute the eta weight w = sum(e_i * k_i) / 24.
    """
    w = Fraction(0)
    for k, e in factors:
        if not isinstance(k, Fraction):
            k = Fraction(k)
        w += e * k
    return w / 24


# ============================================================================
# FORMULA GENERATION
# ============================================================================

def build_k_pool(denominators: List[int], max_k: int = 12) -> List[Fraction]:
    pool = set()
    for k in range(1, max_k + 1):
        pool.add(Fraction(k, 1))
    for d in denominators:
        if d > 1:
            for p in range(1, d):
                if math.gcd(p, d) == 1:
                    pool.add(Fraction(p, d))
    return sorted(pool)


def generate_random_product(
    k_pool: List[Fraction],
    k_pool_frac: List[Fraction],
    k_pool_int: List[Fraction],
    min_terms: int = 2,
    max_terms: int = 6,
    min_exp: int = -6,
    max_exp: int = 6,
    product_only: bool = False,
    modular: bool = False,
    target_exp_sum: Optional[int] = None,
) -> Optional[List[Tuple[Fraction, int]]]:
    """
    Generate a single product: prod_i eta(q^{k_i})^{e_i}
    """
    pool = list(k_pool_int) if modular else list(k_pool)
    n_terms = random.randint(max(min_terms, 2), max_terms)

    if len(pool) < n_terms:
        return None

    chosen_ks = random.sample(pool, n_terms)
    adj_idx = random.randint(0, n_terms - 1)
    k_adj = chosen_ks[adj_idx]

    if product_only:
        exp_pool = list(range(1, max_exp + 1))
    else:
        exp_pool = list(range(min_exp, 0)) + list(range(1, max_exp + 1))

    if not exp_pool:
        return None

    factors = []
    partial_exp_sum = 0
    partial_weighted_sum = Fraction(0)

    for i, k in enumerate(chosen_ks):
        if i == adj_idx:
            continue
        e = random.choice(exp_pool)
        factors.append((k, e))
        partial_exp_sum += e
        partial_weighted_sum += k * e

    need_modular = modular
    need_target = (target_exp_sum is not None)

    if need_target and need_modular:
        e_adj = target_exp_sum - partial_exp_sum
        total_weighted = e_adj * k_adj + partial_weighted_sum
        if total_weighted % 24 != 0:
            return None
    elif need_target:
        e_adj = target_exp_sum - partial_exp_sum
    elif need_modular:
        remainder = (-partial_weighted_sum) % 24
        k_adj_int = int(k_adj)
        g = math.gcd(k_adj_int, 24)
        if int(remainder) % g != 0:
            return None
        reduced_k = k_adj_int // g
        reduced_rem = int(remainder) // g
        reduced_mod = 24 // g
        try:
            inv = pow(reduced_k, -1, reduced_mod)
        except ValueError:
            return None
        e_adj_mod = (reduced_rem * inv) % reduced_mod

        if product_only:
            lo, hi = 1, max_exp
        else:
            lo, hi = min_exp, max_exp

        candidates = []
        start = lo + ((e_adj_mod - lo) % reduced_mod)
        for v in range(start, hi + 1, reduced_mod):
            if v != 0:
                candidates.append(v)
        if not candidates:
            return None
        e_adj = random.choice(candidates)
    else:
        e_adj = random.choice(exp_pool)

    if product_only and e_adj < 1:
        return None
    if e_adj == 0 or e_adj < min_exp or e_adj > max_exp:
        return None

    factors.append((k_adj, e_adj))
    factors.sort(key=lambda x: float(x[0]))
    return factors


def generate_random_sum_formula(
    k_pool: List[Fraction],
    k_pool_frac: List[Fraction],
    k_pool_int: List[Fraction],
    min_terms: int = 2,
    max_terms: int = 6,
    min_exp: int = -6,
    max_exp: int = 6,
    min_sum_terms: int = 1,
    max_sum_terms: int = 1,
    max_shift: int = 0,
    max_num_coeff: int = 1,
    product_only: bool = False,
    modular: bool = False,
    same_weight: bool = False,
) -> Optional[Dict]:
    """
    Generate a formula: sum of eta products with coefficients and shifts.
    """
    n_sum_terms = random.randint(min_sum_terms, max_sum_terms)

    factors_1 = generate_random_product(
        k_pool, k_pool_frac, k_pool_int,
        min_terms, max_terms, min_exp, max_exp,
        product_only=product_only, modular=modular, target_exp_sum=None,
    )
    if factors_1 is None:
        return None

    target_exp_sum = None
    if same_weight:
        target_exp_sum = sum(e for _, e in factors_1)
        if target_exp_sum % 2 != 0:
            return None

    w_1 = compute_eta_weight(factors_1)

    sum_terms = [{
        'coeff': 1,
        'shift': 0,
        'factors': factors_1,
    }]

    max_effective_start = w_1

    for i in range(1, n_sum_terms):
        found = False
        for _attempt in range(100):
            factors_j = generate_random_product(
                k_pool, k_pool_frac, k_pool_int,
                min_terms, max_terms, min_exp, max_exp,
                product_only=product_only, modular=modular,
                target_exp_sum=target_exp_sum,
            )
            if factors_j is None:
                continue

            w_j = compute_eta_weight(factors_j)

            if max_shift == 0:
                shift_j = 0
            else:
                shift_j = random.randint(0, max_shift)

            effective_start_j = w_j + shift_j

            if effective_start_j > max_effective_start:
                coeff_j = random.randint(1, max_num_coeff)
                sum_terms.append({
                    'coeff': coeff_j,
                    'shift': shift_j,
                    'factors': factors_j,
                })
                max_effective_start = effective_start_j
                found = True
                break

        if not found:
            break

    if len(sum_terms) < min_sum_terms:
        return None

    return {'sum_terms': sum_terms}


# ============================================================================
# COEFFICIENT COMPUTATION
# ============================================================================

def compute_raw_product(
    factors: List[Tuple[Fraction, int]],
    phi_cache: Dict,
    max_order: int = 100,
    max_coeff: float = 1e15,
) -> Optional[Dict[Fraction, int]]:
    """
    Compute raw polynomial {power: coeff} for a single eta product.
    """
    prod_result = {Fraction(0): 1}

    for k, exp in factors:
        if isinstance(k, int):
            k = Fraction(k)

        n_iter = abs(exp)
        positive = (exp > 0)

        for _ in range(n_iter):
            phi_k = get_phi_from_cache(phi_cache, k, positive=positive)
            new_result = defaultdict(int)

            for p1, c1 in prod_result.items():
                for p2, c2 in phi_k.items():
                    new_power = p1 + p2
                    if new_power <= max_order:
                        new_result[new_power] += c1 * c2

            prod_result = dict(new_result)

            if any(abs(c) > max_coeff for c in prod_result.values()):
                return None

    eta_w = compute_eta_weight(factors)
    shifted_result = {}
    for power, coeff in prod_result.items():
        shifted_power = power + eta_w
        if shifted_power <= max_order:
            shifted_result[shifted_power] = coeff

    return shifted_result


def compute_formula_coefficients(
    formula: Dict,
    phi_cache: Dict,
    n_coeffs: int = 50,
    max_coeff: float = 1e15,
) -> Optional[Tuple[np.ndarray, Fraction, int]]:
    """
    Compute coefficients for a formula (possibly a sum of products).
    Returns (coeffs_array, min_power, lcm_denominator) or None.
    """
    try:
        if 'sum_terms' in formula:
            sum_terms = formula['sum_terms']
        else:
            sum_terms = [{'coeff': 1, 'shift': 0, 'factors': formula['factors']}]

        max_order = n_coeffs + 20

        total_poly = defaultdict(int)

        for term in sum_terms:
            coeff_mult = term.get('coeff', 1)
            shift = term.get('shift', 0)
            factors = term['factors']

            raw = compute_raw_product(factors, phi_cache, max_order, max_coeff)
            if raw is None:
                return None

            for power, c in raw.items():
                shifted_power = power + shift
                if shifted_power <= max_order:
                    total_poly[shifted_power] += coeff_mult * c

        if not total_poly:
            return None

        if any(abs(c) > max_coeff for c in total_poly.values()):
            return None

        all_powers = [p for p, c in total_poly.items() if c != 0]
        if not all_powers:
            return None

        min_power = min(all_powers)

        all_denoms = set()
        for p in all_powers:
            if isinstance(p, Fraction):
                all_denoms.add(p.denominator)
            else:
                all_denoms.add(1)
        lcm_d = 1
        for d in all_denoms:
            lcm_d = lcm_d * d // math.gcd(lcm_d, d)

        step = Fraction(1, lcm_d)
        coeffs = np.zeros(n_coeffs, dtype=np.int64)

        for i in range(n_coeffs):
            power = min_power + i * step
            if power in total_poly:
                coeffs[i] = total_poly[power]

        if np.count_nonzero(coeffs) < 20:
            return None
        if coeffs[0] == 0:
            return None

        return coeffs, min_power, lcm_d

    except (OverflowError, ValueError, MemoryError):
        return None


# ============================================================================
# SIGNATURES
# ============================================================================

def get_formula_signature(formula: Dict) -> str:
    """Canonical signature for deduplication."""
    if 'sum_terms' in formula:
        term_sigs = []
        for term in formula['sum_terms']:
            coeff = term.get('coeff', 1)
            shift = term.get('shift', 0)
            factors_sorted = sorted(term['factors'], key=lambda x: (x[0].numerator, x[0].denominator))
            factors_str = ','.join(f"{k.numerator}/{k.denominator}^{e}" for k, e in factors_sorted)
            term_sigs.append(f"{coeff}*q^{shift}*({factors_str})")
        return ' + '.join(term_sigs)
    else:
        factors_sorted = sorted(formula['factors'], key=lambda x: (x[0].numerator, x[0].denominator))
        return ','.join(f"{k.numerator}/{k.denominator}^{e}" for k, e in factors_sorted)


def formula_to_string(formula: Dict) -> str:
    """Human-readable string."""
    if 'sum_terms' in formula:
        sum_terms = formula['sum_terms']
    else:
        sum_terms = [{'coeff': 1, 'shift': 0, 'factors': formula['factors']}]

    term_strs = []
    for term in sum_terms:
        coeff = term.get('coeff', 1)
        shift = term.get('shift', 0)
        factors = term['factors']

        parts = []
        for k, exp in factors:
            if isinstance(k, Fraction) and k.denominator > 1:
                eta_str = f"η(q^({k}))"
            elif isinstance(k, Fraction):
                k_int = k.numerator
                eta_str = f"η(q^{k_int})" if k_int > 1 else "η(q)"
            else:
                eta_str = f"η(q^{k})" if k > 1 else "η(q)"

            if exp == 1:
                parts.append(eta_str)
            elif exp == -1:
                parts.append(f"1/{eta_str}")
            elif exp > 0:
                parts.append(f"{eta_str}^{exp}")
            else:
                parts.append(f"1/{eta_str}^{abs(exp)}")

        product_str = " · ".join(parts)

        prefix_parts = []
        if coeff != 1:
            prefix_parts.append(str(coeff))
        if shift > 0:
            prefix_parts.append(f"q^{shift}")

        if prefix_parts:
            prefix = " · ".join(prefix_parts)
            term_strs.append(f"{prefix} · ({product_str})")
        else:
            term_strs.append(product_str)

    return " + ".join(term_strs)


# ============================================================================
# WORKER
# ============================================================================

def worker_generate_formulas(args):
    (worker_id, n_formulas, k_pool, k_pool_frac, k_pool_int,
     min_terms, max_terms, min_exp, max_exp,
     min_sum_terms, max_sum_terms, max_shift, max_num_coeff,
     product_only, modular, same_weight,
     n_coeffs, phi_cache, base_seed,
     existing_signatures) = args

    random.seed(base_seed + worker_id * 1000000)

    formulas = []
    coefficients = []
    min_powers = []
    lcm_denoms = []
    # Start with the global existing signatures so we never produce dupes
    signatures = set(existing_signatures)
    initial_sig_count = len(signatures)

    attempts = 0
    max_attempts = n_formulas * 200

    pbar = tqdm(total=n_formulas, desc=f"Worker {worker_id}", position=worker_id, leave=False)

    while len(formulas) < n_formulas and attempts < max_attempts:
        attempts += 1

        formula = generate_random_sum_formula(
            k_pool=k_pool,
            k_pool_frac=k_pool_frac,
            k_pool_int=k_pool_int,
            min_terms=min_terms,
            max_terms=max_terms,
            min_exp=min_exp,
            max_exp=max_exp,
            min_sum_terms=min_sum_terms,
            max_sum_terms=max_sum_terms,
            max_shift=max_shift,
            max_num_coeff=max_num_coeff,
            product_only=product_only,
            modular=modular,
            same_weight=same_weight,
        )

        if formula is None:
            continue

        signature = get_formula_signature(formula)
        if signature in signatures:
            continue

        result = compute_formula_coefficients(formula, phi_cache, n_coeffs)

        if result is None:
            continue

        coeffs, min_power, lcm_d = result

        formulas.append(formula)
        coefficients.append(coeffs)
        min_powers.append(min_power)
        lcm_denoms.append(lcm_d)
        signatures.add(signature)

        pbar.update(1)

    pbar.close()

    # Return only the NEW signatures (not the ones we were given)
    new_signatures = signatures - set(existing_signatures)

    return formulas, coefficients, min_powers, lcm_denoms, new_signatures


# ============================================================================
# MAIN
# ============================================================================

def _generate_split(
    split_name: str,
    n_samples: int,
    k_pool: List[Fraction],
    k_pool_frac: List[Fraction],
    k_pool_int: List[Fraction],
    min_terms: int,
    max_terms: int,
    min_exp: int,
    max_exp: int,
    min_sum_terms: int,
    max_sum_terms: int,
    max_shift: int,
    max_num_coeff: int,
    product_only: bool,
    modular: bool,
    same_weight: bool,
    n_coeffs: int,
    phi_cache: Dict,
    seed: int,
    num_workers: int,
    global_signatures: Set[str],
    max_rounds: int = 20,
) -> Dict:
    """
    Generate exactly n_samples unique formulas for a split, retrying as needed.

    Uses a retry loop: after each round of parallel generation, if we're still
    short (due to cross-worker or cross-split duplicates), we launch another
    round requesting only the deficit. Stops after max_rounds to avoid infinite
    loops if the search space is exhausted.
    """
    all_formulas = []
    all_coefficients = []
    all_min_powers = []
    all_lcm_denoms = []
    # Track all signatures: global (cross-split) + this split's own
    known_signatures = set(global_signatures)

    round_num = 0
    round_seed = seed

    while len(all_formulas) < n_samples and round_num < max_rounds:
        remaining = n_samples - len(all_formulas)
        round_num += 1

        if round_num > 1:
            print(f"  Round {round_num}: need {remaining} more formulas...")

        # Distribute remaining across workers
        effective_workers = min(num_workers, remaining)
        samples_per_worker = remaining // effective_workers
        remainder = remaining % effective_workers

        worker_args = []
        for worker_id in range(effective_workers):
            n_formulas = samples_per_worker + (1 if worker_id < remainder else 0)
            if n_formulas == 0:
                continue
            worker_args.append((
                worker_id,
                n_formulas,
                k_pool, k_pool_frac, k_pool_int,
                min_terms, max_terms, min_exp, max_exp,
                min_sum_terms, max_sum_terms, max_shift, max_num_coeff,
                product_only, modular, same_weight,
                n_coeffs,
                phi_cache,
                round_seed + worker_id * 1000000,
                known_signatures,  # pass ALL known sigs to each worker
            ))

        with mp.Pool(effective_workers) as pool:
            results = pool.map(worker_generate_formulas, worker_args)

        # Merge results, deduplicating across workers within this round
        round_new = 0
        for formulas, coeffs, min_pows, lcm_ds, new_sigs in results:
            for i in range(len(formulas)):
                if len(all_formulas) >= n_samples:
                    break
                sig = get_formula_signature(formulas[i])
                if sig in known_signatures:
                    continue
                all_formulas.append(formulas[i])
                all_coefficients.append(coeffs[i])
                all_min_powers.append(min_pows[i])
                all_lcm_denoms.append(lcm_ds[i])
                known_signatures.add(sig)
                round_new += 1

        # Advance seed for next round so we don't repeat the same random draws
        round_seed += num_workers * 1000000 + round_num * 7777

        if round_new == 0:
            print(f"  ⚠ Round {round_num} produced 0 new formulas — search space may be exhausted.")
            break

    if len(all_formulas) < n_samples:
        print(f"  ⚠ WARNING: Could only generate {len(all_formulas)}/{n_samples} unique formulas "
              f"for {split_name} after {round_num} rounds.")

    # Update the global set so subsequent splits don't collide
    global_signatures.update(known_signatures)

    return {
        'formulas': all_formulas,
        'coefficients': np.array(all_coefficients, dtype=np.int64),
        'min_powers': all_min_powers,
        'lcm_denominators': all_lcm_denoms,
    }


def generate_dataset(
    denominators: List[int],
    n_train: int = 10000,
    n_val: int = 1000,
    n_test: int = 1000,
    n_coeffs: int = 50,
    output_dir: str = './data_eta_product',
    seed: int = 42,
    min_terms: int = 2,
    max_terms: int = 6,
    min_exp: int = -6,
    max_exp: int = 6,
    max_k: int = 12,
    min_sum_terms: int = 1,
    max_sum_terms: int = 1,
    max_shift: int = 0,
    max_num_coeff: int = 1,
    product_only: bool = False,
    modular: bool = False,
    same_weight: bool = False,
    num_workers: Optional[int] = None
):
    random.seed(seed)
    np.random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    print("=" * 80)
    print("ETA PRODUCT DATASET GENERATION (with Sums support)")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  Train samples:   {n_train}")
    print(f"  Val samples:     {n_val}")
    print(f"  Test samples:    {n_test}")
    print(f"  Num coeffs:      {n_coeffs}")
    print(f"  Product terms:   {min_terms}-{max_terms}")
    print(f"  Exponents:       [{min_exp}, {max_exp}]")
    print(f"  Max k:           {max_k}")
    print(f"  Denominators:    {denominators}")
    print(f"  Sum terms:       {min_sum_terms}-{max_sum_terms}")
    print(f"  Max shift:       {max_shift}")
    print(f"  Max num coeff:   {max_num_coeff}")
    print(f"  Product only:    {product_only}")
    print(f"  Modular:         {modular}")
    print(f"  Same weight:     {same_weight}")
    print(f"  Workers:         {num_workers}")
    print(f"  Output dir:      {output_dir}")

    # Phase 1: cache
    print(f"\n{'=' * 80}")
    print("PHASE 1: Pre-computing phi functions")
    print("=" * 80)
    phi_cache = build_phi_cache(denominators, max_k=max_k, max_order=n_coeffs + 20)
    print(f"✓ Cached {len(phi_cache['pos'])} phi functions")

    k_pool = build_k_pool(denominators, max_k=max_k)
    k_pool_frac = [k for k in k_pool if k.denominator > 1]
    k_pool_int = [k for k in k_pool if k.denominator == 1]
    print(f"✓ k pool: {len(k_pool)} values ({len(k_pool_frac)} fractional, {len(k_pool_int)} integer)")

    # Phase 2: generate all splits with global dedup
    datasets = {}
    global_signatures: Set[str] = set()

    for split_name, n_samples in [('train', n_train), ('val', n_val), ('test', n_test)]:
        print(f"\n{'=' * 80}")
        print(f"PHASE 2: Generating {split_name} set ({n_samples} samples)")
        print(f"{'=' * 80}")

        split_seed = seed + hash(split_name) % (2**31)

        datasets[split_name] = _generate_split(
            split_name=split_name,
            n_samples=n_samples,
            k_pool=k_pool,
            k_pool_frac=k_pool_frac,
            k_pool_int=k_pool_int,
            min_terms=min_terms,
            max_terms=max_terms,
            min_exp=min_exp,
            max_exp=max_exp,
            min_sum_terms=min_sum_terms,
            max_sum_terms=max_sum_terms,
            max_shift=max_shift,
            max_num_coeff=max_num_coeff,
            product_only=product_only,
            modular=modular,
            same_weight=same_weight,
            n_coeffs=n_coeffs,
            phi_cache=phi_cache,
            seed=split_seed,
            num_workers=num_workers,
            global_signatures=global_signatures,
        )

        actual = len(datasets[split_name]['formulas'])
        print(f"\n{split_name} complete: {actual}/{n_samples} formulas")

    # Statistics
    print(f"\n{'=' * 80}")
    print("STATISTICS")
    print(f"{'=' * 80}")

    sum_terms_dist = Counter()
    coeff_dist = Counter()
    shift_dist = Counter()
    eta_weight_diffs = []
    for split in datasets.values():
        for f in split['formulas']:
            n_terms = len(f['sum_terms'])
            sum_terms_dist[n_terms] += 1
            weights = []
            for term in f['sum_terms']:
                coeff_dist[term.get('coeff', 1)] += 1
                shift_dist[term.get('shift', 0)] += 1
                weights.append(compute_eta_weight(term['factors']) + term.get('shift', 0))
            if len(weights) > 1:
                eta_weight_diffs.append(float(weights[1] - weights[0]))

    print(f"\nSum terms distribution:")
    for n in sorted(sum_terms_dist.keys()):
        print(f"  {n} term(s): {sum_terms_dist[n]}")

    if max_num_coeff > 1:
        print(f"\nCoefficient distribution:")
        for c in sorted(coeff_dist.keys()):
            print(f"  c={c}: {coeff_dist[c]}")

    if max_shift > 0:
        print(f"\nShift distribution (top 10):")
        for s in sorted(shift_dist.keys())[:10]:
            print(f"  s={s}: {shift_dist[s]}")

    if eta_weight_diffs:
        print(f"\nEta weight gap (w2 - w1) for multi-term formulas:")
        print(f"  min: {min(eta_weight_diffs):.4f}, max: {max(eta_weight_diffs):.4f}, "
              f"mean: {np.mean(eta_weight_diffs):.4f}")

    if modular or same_weight:
        modular_weight_dist = Counter()
        for split in datasets.values():
            for f in split['formulas']:
                for term in f['sum_terms']:
                    mw = sum(e for _, e in term['factors']) // 2
                    modular_weight_dist[mw] += 1
        print(f"\nModular weight k = Σe/2 distribution (top 10):")
        for k in sorted(modular_weight_dist.keys())[:10]:
            print(f"  k={k}: {modular_weight_dist[k]}")

    # Save
    print(f"\n{'=' * 80}")
    print("SAVING DATASET")
    print(f"{'=' * 80}")

    dataset = {
        'train': datasets['train'],
        'val': datasets['val'],
        'test': datasets['test'],
        'config': {
            'n_coeffs': n_coeffs,
            'seed': seed,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'min_terms': min_terms,
            'max_terms': max_terms,
            'min_exp': min_exp,
            'max_exp': max_exp,
            'max_k': max_k,
            'denominators': denominators,
            'min_sum_terms': min_sum_terms,
            'max_sum_terms': max_sum_terms,
            'max_shift': max_shift,
            'max_num_coeff': max_num_coeff,
            'product_only': product_only,
            'modular': modular,
            'same_weight': same_weight,
        }
    }

    output_file = output_path / 'eta_product_dataset.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n✓ Dataset saved to {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Examples
    print(f"\n{'=' * 80}")
    print("EXAMPLE FORMULAS")
    print(f"{'=' * 80}")
    for i in range(min(10, len(datasets['train']['formulas']))):
        f = datasets['train']['formulas'][i]
        c = datasets['train']['coefficients'][i]
        print(f"\n{i+1}. {formula_to_string(f)}")
        weights = [compute_eta_weight(t['factors']) + t.get('shift', 0) for t in f['sum_terms']]
        mod_weights = [sum(e for _, e in t['factors']) / 2 for t in f['sum_terms']]
        print(f"   Terms: {len(f['sum_terms'])}, eta weights: {[str(w) for w in weights]}, "
              f"mod weights: {[str(w) for w in mod_weights]}, Coeffs[0:10]: {c[:10]}")

    print(f"\n{'=' * 80}")
    print("✅ Dataset generation complete!")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Eta Product Dataset (with Sums)')
    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_val', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_coeffs', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./data_eta_product')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_terms', type=int, default=2)
    parser.add_argument('--max_terms', type=int, default=6)
    parser.add_argument('--min_exp', type=int, default=-6)
    parser.add_argument('--max_exp', type=int, default=6)
    parser.add_argument('--max_k', type=int, default=12)
    parser.add_argument('--denominators', type=str, default='1')
    parser.add_argument('--min_sum_terms', type=int, default=1)
    parser.add_argument('--max_sum_terms', type=int, default=1)
    parser.add_argument('--max_shift', type=int, default=0)
    parser.add_argument('--max_num_coeff', type=int, default=1)
    parser.add_argument('--product_only', action='store_true',
                        help='Only positive exponents (no quotients)')
    parser.add_argument('--modular', action='store_true',
                        help='Σ(e_i * k_i) ≡ 0 mod 24 (integer eta weights)')
    parser.add_argument('--same_weight', action='store_true',
                        help='All sum terms have same modular weight k=Σe/2 (requires --modular)')
    parser.add_argument('--num_workers', type=int, default=None)

    args = parser.parse_args()
    denominators = [int(d.strip()) for d in args.denominators.split(',')]

    if args.same_weight and not args.modular:
        parser.error("--same_weight requires --modular")

    generate_dataset(
        denominators=denominators,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        n_coeffs=args.n_coeffs,
        output_dir=args.output_dir,
        seed=args.seed,
        min_terms=args.min_terms,
        max_terms=args.max_terms,
        min_exp=args.min_exp,
        max_exp=args.max_exp,
        max_k=args.max_k,
        min_sum_terms=args.min_sum_terms,
        max_sum_terms=args.max_sum_terms,
        max_shift=args.max_shift,
        max_num_coeff=args.max_num_coeff,
        product_only=args.product_only,
        modular=args.modular,
        same_weight=args.same_weight,
        num_workers=args.num_workers,
    )
