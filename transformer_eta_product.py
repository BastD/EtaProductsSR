"""
Transformer for Eta Product Symbolic Regression (with Sums support)

Supports:
- Products only (backward compatible)
- Sums of products: c_1 * q^{s_1} * PROD_1 + c_2 * q^{s_2} * PROD_2 + ...
- Fractional k: eta(q^(1/2)), eta(q^(1/3)), etc.
- Integer k from 1 to max_k
- Exponents from -max_exp to max_exp (excluding 0)
- Integer coefficients c_i from 1 to max_num_coeff
- Integer shifts s_i from 0 to max_shift
- COEFF_PH placeholder: model predicts skeleton, coefficients brute-forced later

Token grammar:
    sequence := START term (TERM_SEP term)* END
    term     := COEFF COEFF_PH SHIFT s_val product
              | COEFF c_val SHIFT s_val product      (full mode)
    product  := POW eta exp (MUL POW eta exp)*
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field
from fractions import Fraction


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 200
    max_formula_len: int = 80
    n_encoder_tokens: int = 16
    n_coeffs: int = 100
    use_periodic: bool = False
    vocab_size: int = 0


# ============================================================================
# VOCABULARY
# ============================================================================

class EtaProductVocabulary:
    """
    Vocabulary for eta product formulas with sums, coefficients, shifts, and fractional k.

    Tokens:
    - Special: <PAD>, <START>, <END>
    - Structural: MUL, POW, TERM_SEP, COEFF, SHIFT
    - COEFF_PH: placeholder for coefficient value (model predicts this instead of c1..cN)
    - Coefficient values: c1, c2, ..., c{max_num_coeff}  (kept for brute-force / analysis)
    - Shift values: s0, s1, ..., s{max_shift}
    - eta tokens: eta(q^k) for k=1..max_k (integer)
    - eta fractional: eta(q^(p/d)) for each valid p/d < 1
    - Exponents: -max_exp .. max_exp (excluding 0)
    """

    def __init__(self, denominators: List[int] = None, max_k: int = 12, max_exp: int = 6,
                 max_num_coeff: int = 1, max_shift: int = 0):
        if denominators is None:
            denominators = [1]

        self.denominators = denominators
        self.max_k = max_k
        self.max_exp = max_exp
        self.max_num_coeff = max_num_coeff
        self.max_shift = max_shift

        self.PAD = '<PAD>'
        self.START = '<START>'
        self.END = '<END>'
        self.MUL = 'MUL'
        self.POW = 'POW'
        self.TERM_SEP = 'TERM_SEP'
        self.COEFF = 'COEFF'
        self.SHIFT = 'SHIFT'
        self.COEFF_PH = 'COEFF_PH'

        # Build eta tokens
        self.eta_int_tokens = [f'eta(q^{k})' for k in range(1, max_k + 1)]

        self.eta_frac_tokens = []
        self._frac_to_token = {}
        self._token_to_frac = {}

        seen_fracs = set()
        for d in sorted(denominators):
            if d > 1:
                for p in range(1, d):
                    if math.gcd(p, d) == 1:
                        frac = Fraction(p, d)
                        if frac not in seen_fracs:
                            token = f'eta(q^({p}/{d}))'
                            self.eta_frac_tokens.append(token)
                            self._frac_to_token[frac] = token
                            self._token_to_frac[token] = frac
                            seen_fracs.add(frac)

        self.all_eta_tokens = self.eta_frac_tokens + self.eta_int_tokens

        # Exponent tokens (excluding 0)
        self.exponent_tokens = [str(i) for i in range(-max_exp, max_exp + 1) if i != 0]

        # Coefficient value tokens: c1, c2, ..., c{max_num_coeff}
        self.coeff_tokens = [f'c{i}' for i in range(1, max_num_coeff + 1)]

        # Shift value tokens: s0, s1, ..., s{max_shift}
        self.shift_tokens = [f's{i}' for i in range(max_shift + 1)]

        # Full token list — COEFF_PH added alongside coeff_tokens
        self.tokens = (
            [self.PAD, self.START, self.END] +
            [self.MUL, self.POW, self.TERM_SEP, self.COEFF, self.SHIFT] +
            [self.COEFF_PH] +
            self.coeff_tokens +
            self.shift_tokens +
            self.all_eta_tokens +
            self.exponent_tokens
        )

        self.token2idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}

        self.vocab_size = len(self.tokens)
        self.pad_idx = self.token2idx[self.PAD]
        self.start_idx = self.token2idx[self.START]
        self.end_idx = self.token2idx[self.END]
        self.coeff_ph_idx = self.token2idx[self.COEFF_PH]

        # Token category sets (for syntax masking)
        self.eta_token_indices = set(self.token2idx[t] for t in self.all_eta_tokens)
        self.exp_token_indices = set(self.token2idx[t] for t in self.exponent_tokens)
        self.coeff_value_indices = set(self.token2idx[t] for t in self.coeff_tokens)
        self.shift_value_indices = set(self.token2idx[t] for t in self.shift_tokens)

    def _k_to_eta_token(self, k) -> str:
        if isinstance(k, Fraction):
            if k.denominator == 1:
                return f'eta(q^{k.numerator})'
            else:
                if k in self._frac_to_token:
                    return self._frac_to_token[k]
                else:
                    return f'eta(q^({k.numerator}/{k.denominator}))'
        else:
            return f'eta(q^{int(k)})'

    def _eta_token_to_k(self, token: str):
        if token in self._token_to_frac:
            return self._token_to_frac[token]
        if 'eta(q^(' in token:
            inner = token.split('(')[2].rstrip(')')
            num, den = inner.split('/')
            return Fraction(int(num), int(den))
        elif 'eta(q^' in token:
            k_str = token.split('^')[1].rstrip(')')
            return int(k_str)
        return None

    def encode(self, formula_tokens: List[str]) -> List[int]:
        """Convert a list of tokens into the list of corresponding indices"""
        return [self.token2idx[token] for token in formula_tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert a list of indices into the list of corresponding tokens"""
        return [self.idx2token.get(idx, f'<UNK_{idx}>') for idx in indices]

    def formula_to_tokens(self, formula: Dict, use_placeholder: bool = True) -> List[str]:
        """
        Convert formula to token sequence.

        Args:
            formula: formula dict (sum_terms or legacy factors format)
            use_placeholder: if True, emit COEFF_PH instead of actual coefficient values.
                             The model learns to predict skeletons.
        """
        # Normalize to sum format
        if 'sum_terms' in formula:
            sum_terms = formula['sum_terms']
        else:
            sum_terms = [{'coeff': 1, 'shift': 0, 'factors': formula['factors']}]

        tokens = [self.START]

        for term_idx, term in enumerate(sum_terms):
            if term_idx > 0:
                tokens.append(self.TERM_SEP)

            coeff = term.get('coeff', 1)
            shift = term.get('shift', 0)

            tokens.append(self.COEFF)
            if use_placeholder:
                tokens.append(self.COEFF_PH)
            else:
                tokens.append(f'c{coeff}')
            tokens.append(self.SHIFT)
            tokens.append(f's{shift}')

            # Sort factors by k ascending
            sorted_factors = sorted(term['factors'], key=lambda x: float(x[0]))

            for i, (k, exp) in enumerate(sorted_factors):
                tokens.append(self.POW)
                tokens.append(self._k_to_eta_token(k))
                tokens.append(str(exp))
                if i < len(sorted_factors) - 1:
                    tokens.append(self.MUL)

        tokens.append(self.END)
        return tokens

    def tokens_to_formula(self, tokens: List[str]) -> Optional[Dict]:
        """Parse token sequence back to formula (sum format).

        Handles both COEFF_PH (placeholder) and concrete c{val} tokens.
        When COEFF_PH is encountered, coeff is set to None.
        """
        if not tokens or tokens[0] != self.START:
            return None

        sum_terms = []
        i = 1

        while i < len(tokens):
            if tokens[i] == self.END:
                break

            # Parse COEFF (c{val} | COEFF_PH)
            if tokens[i] != self.COEFF:
                return None
            i += 1
            if i >= len(tokens):
                return None

            if tokens[i] == self.COEFF_PH:
                coeff = None  # placeholder — to be brute-forced later
                i += 1
            elif tokens[i].startswith('c'):
                try:
                    coeff = int(tokens[i][1:])
                except ValueError:
                    return None
                i += 1
            else:
                return None

            # Parse SHIFT s{val}
            if i >= len(tokens) or tokens[i] != self.SHIFT:
                return None
            i += 1
            if i >= len(tokens) or not tokens[i].startswith('s'):
                return None
            try:
                shift = int(tokens[i][1:])
            except ValueError:
                return None
            i += 1

            # Parse product: (POW eta exp (MUL POW eta exp)*)
            factors = []
            while i < len(tokens):
                if tokens[i] != self.POW:
                    break
                i += 1

                if i >= len(tokens):
                    return None
                k = self._eta_token_to_k(tokens[i])
                if k is None:
                    return None
                i += 1

                if i >= len(tokens):
                    return None
                tok = tokens[i]
                if tok == self.END or tok == self.MUL or tok == self.TERM_SEP:
                    return None
                try:
                    exp = int(tok)
                except ValueError:
                    return None
                if exp == 0:
                    return None

                factors.append((k, exp))
                i += 1

                # Skip MUL if present
                if i < len(tokens) and tokens[i] == self.MUL:
                    i += 1

            if not factors:
                return None

            sorted_factors = sorted(factors, key=lambda x: float(x[0]))
            sum_terms.append({
                'coeff': coeff,
                'shift': shift,
                'factors': sorted_factors,
            })

            # If next token is TERM_SEP, skip it and continue
            if i < len(tokens) and tokens[i] == self.TERM_SEP:
                i += 1

        if not sum_terms:
            return None

        return {'sum_terms': sum_terms}

    def formula_to_string(self, formula: Dict) -> str:
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
            if coeff is None:
                prefix_parts.append("?")
            elif coeff != 1:
                prefix_parts.append(str(coeff))
            if shift > 0:
                prefix_parts.append(f"q^{shift}")

            if prefix_parts:
                prefix = " · ".join(prefix_parts)
                term_strs.append(f"{prefix} · ({product_str})")
            else:
                term_strs.append(product_str)

        return " + ".join(term_strs)

    def validate_formula_syntax(self, tokens: List[str]) -> bool:
        return self.tokens_to_formula(tokens) is not None


# ============================================================================
# SYNTAX MASKER
# ============================================================================

class SyntaxMasker:
    """
    Grammar:
        START     → COEFF
        COEFF     → c_val | COEFF_PH
        c_val     → SHIFT
        COEFF_PH  → SHIFT
        SHIFT     → s_val
        s_val     → POW
        POW       → eta
        eta       → exp
        exp       → MUL | TERM_SEP | END
        MUL       → POW
        TERM_SEP  → COEFF
        END       → (nothing)
    """

    def __init__(self, vocab: EtaProductVocabulary):
        self.vocab = vocab
        self.pow_token = vocab.token2idx[vocab.POW]
        self.mul_token = vocab.token2idx[vocab.MUL]
        self.end_token = vocab.token2idx[vocab.END]
        self.term_sep_token = vocab.token2idx[vocab.TERM_SEP]
        self.coeff_token = vocab.token2idx[vocab.COEFF]
        self.shift_token = vocab.token2idx[vocab.SHIFT]
        self.coeff_ph_token = vocab.coeff_ph_idx
        self.eta_tokens = vocab.eta_token_indices
        self.exp_tokens = vocab.exp_token_indices
        self.coeff_value_tokens = vocab.coeff_value_indices
        self.shift_value_tokens = vocab.shift_value_indices

    def get_valid_tokens(self, token_history: List[int]) -> Set[int]:
        if not token_history or token_history[-1] == self.vocab.start_idx:
            return {self.coeff_token}

        last = token_history[-1]

        # COEFF → c_val | COEFF_PH
        if last == self.coeff_token:
            return self.coeff_value_tokens | {self.coeff_ph_token}

        # c_val → SHIFT
        if last in self.coeff_value_tokens:
            return {self.shift_token}

        # COEFF_PH → SHIFT
        if last == self.coeff_ph_token:
            return {self.shift_token}

        # SHIFT → s_val
        if last == self.shift_token:
            return self.shift_value_tokens

        # s_val → POW
        if last in self.shift_value_tokens:
            return {self.pow_token}

        # POW → eta
        if last == self.pow_token:
            return self.eta_tokens

        # eta → exp
        if last in self.eta_tokens:
            return self.exp_tokens

        # exp → MUL | TERM_SEP | END
        if last in self.exp_tokens:
            return {self.mul_token, self.term_sep_token, self.end_token}

        # MUL → POW
        if last == self.mul_token:
            return {self.pow_token}

        # TERM_SEP → COEFF
        if last == self.term_sep_token:
            return {self.coeff_token}

        # END → nothing
        if last == self.end_token:
            return set()

        return {self.end_token}

    def create_mask(self, token_history: List[int], vocab_size: int) -> torch.Tensor:
        valid = self.get_valid_tokens(token_history)
        mask = torch.full((vocab_size,), float('-inf'))
        for idx in valid:
            mask[idx] = 0.0
        return mask


# ============================================================================
# EMBEDDING
# ============================================================================

class QSeriesEmbedding(nn.Module):
    """
    Embedding for eta product coefficients.
    """

    def __init__(self, d_model: int = 128, max_len: int = 101,
                 use_periodic: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.use_periodic = use_periodic

        self.n_asymp = 3
        self.n_residuals = max_len
        self.n_fft = max_len // 2
        self.n_second_diff = max_len - 2
        self.n_periodic = (max_len // 2 - 1) * 4
        self.n_stats = 4
        self.n_deligne = 1

        total = (
            self.n_asymp + self.n_residuals + self.n_fft +
            self.n_second_diff + self.n_stats + self.n_deligne
        )
        if self.use_periodic:
            total += self.n_periodic

        self.proj = nn.Sequential(
            nn.Linear(total, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )

        n_idx = torch.arange(max_len, dtype=torch.float32)
        self.register_buffer('sqrt_n', torch.sqrt(n_idx + 1))
        self.register_buffer('log_n', torch.log(n_idx + 1))
        self.register_buffer('n_idx', n_idx)

    def _fit_asymptotic(self, log_c: torch.Tensor):
        B, L = log_c.shape
        device = log_c.device

        sqrt_n = self.sqrt_n[:L].to(device)
        log_n = self.log_n[:L].to(device)
        ones = torch.ones(L, device=device)

        X = torch.stack([ones, sqrt_n, log_n], dim=1)
        XtX = X.T @ X
        XtX_inv = torch.linalg.inv(XtX + 1e-6 * torch.eye(3, device=device))
        params = (XtX_inv @ X.T @ log_c.T).T

        alpha, beta, gamma = params[:, 0:1], params[:, 1:2], params[:, 2:3]
        asymp = alpha + beta * sqrt_n.unsqueeze(0) + gamma * log_n.unsqueeze(0)
        return alpha, beta, gamma, asymp

    def _compute_fft_features(self, residuals: torch.Tensor) -> torch.Tensor:
        B, L = residuals.shape
        fft = torch.fft.rfft(residuals, dim=1)
        amplitudes = torch.abs(fft)
        features = amplitudes[:, 1:self.n_fft + 1]
        if features.shape[1] < self.n_fft:
            features = F.pad(features, (0, self.n_fft - features.shape[1]))
        return features

    def _compute_periodic_features(self, residuals: torch.Tensor) -> torch.Tensor:
        B, L = residuals.shape
        device = residuals.device
        max_period = L // 2

        features = []
        for b in range(2, max_period + 1):
            for k in range(4):
                n = k * b
                if n + b < L:
                    features.append((residuals[:, n + b] - residuals[:, n]).unsqueeze(1))
                else:
                    features.append(torch.zeros(B, 1, device=device))

        feat = torch.cat(features, dim=1)
        if feat.shape[1] < self.n_periodic:
            feat = F.pad(feat, (0, self.n_periodic - feat.shape[1]))
        else:
            feat = feat[:, :self.n_periodic]
        return feat

    def _compute_stats(self, residuals: torch.Tensor) -> torch.Tensor:
        mean = residuals.mean(dim=1, keepdim=True)
        std = residuals.std(dim=1, keepdim=True) + 1e-8
        centered = residuals - mean
        skew = (centered ** 3).mean(dim=1, keepdim=True) / (std ** 3)
        kurt = (centered ** 4).mean(dim=1, keepdim=True) / (std ** 4) - 3
        return torch.cat([mean, std, skew, kurt], dim=1)

    @staticmethod
    def _sieve_primes(n: int) -> List[int]:
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    def _estimate_deligne_bound(self, coeffs: torch.Tensor) -> torch.Tensor:
        B, L = coeffs.shape
        device = coeffs.device

        primes = self._sieve_primes(L - 1)
        if len(primes) < 3:
            return torch.zeros(B, device=device)

        prime_indices = torch.tensor(primes, device=device, dtype=torch.long)
        log_p = torch.log(prime_indices.float())
        a_p = coeffs[:, prime_indices].abs()
        log_a_p = torch.log(a_p + 1e-8)

        n_p = len(primes)
        ones = torch.ones(n_p, device=device)
        X = torch.stack([ones, log_p], dim=0)
        XtX = X @ X.T
        Xty = X @ log_a_p.T
        reg = 1e-6 * torch.eye(2, device=device)
        params = torch.linalg.solve(XtX + reg, Xty)
        return params[1]

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        B, L = coeffs.shape
        device = coeffs.device
        eps = 1e-8

        log_c = torch.log(coeffs.abs() + eps)
        alpha, beta, gamma, asymp = self._fit_asymptotic(log_c)
        residuals = log_c - asymp

        asymp_feat = torch.cat([alpha, beta, gamma], dim=1)

        if L < self.n_residuals:
            res_padded = F.pad(residuals, (0, self.n_residuals - L))
        else:
            res_padded = residuals[:, :self.n_residuals]

        fft_feat = self._compute_fft_features(residuals)

        sd = residuals[:, 2:] - 2 * residuals[:, 1:-1] + residuals[:, :-2]
        if sd.shape[1] < self.n_second_diff:
            sd = F.pad(sd, (0, self.n_second_diff - sd.shape[1]))
        else:
            sd = sd[:, :self.n_second_diff]

        stats_feat = self._compute_stats(residuals)
        deligne_feat = self._estimate_deligne_bound(coeffs).unsqueeze(1)

        parts = [asymp_feat, res_padded, fft_feat, sd, stats_feat, deligne_feat]

        if self.use_periodic:
            periodic_feat = self._compute_periodic_features(residuals)
            parts.append(periodic_feat)

        all_feat = torch.cat(parts, dim=1)
        return self.proj(all_feat)


# ============================================================================
# GLOBAL EMBEDDING EXPANDER
# ============================================================================

class GlobalEmbeddingExpander(nn.Module):
    def __init__(self, d_model: int, n_tokens: int = 16):
        super().__init__()
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_tokens)
        ])
        self.pos_encoding = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)

    def forward(self, global_emb: torch.Tensor) -> torch.Tensor:
        tokens = [proj(global_emb) for proj in self.projections]
        seq = torch.stack(tokens, dim=1)
        return seq + self.pos_encoding.unsqueeze(0)


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class EtaProductTransformer(nn.Module):
    def __init__(self, config: ModelConfig, vocab: EtaProductVocabulary):
        super().__init__()
        self.config = config
        self.vocab = vocab

        self.coeff_embedding = QSeriesEmbedding(
            d_model=config.d_model,
            max_len=config.n_coeffs + 1,
            use_periodic=config.use_periodic,
        )

        self.embedding_expander = GlobalEmbeddingExpander(
            d_model=config.d_model,
            n_tokens=config.n_encoder_tokens,
        )

        self.token_embedding = nn.Embedding(vocab.vocab_size, config.d_model)
        self.token_pos_encoding = nn.Embedding(config.max_formula_len, config.d_model)

        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )

        self.output_projection = nn.Linear(config.d_model, vocab.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def encode_coefficients(self, coeffs):
        global_emb = self.coeff_embedding(coeffs)
        return self.embedding_expander(global_emb)

    def forward(self, coeffs, target_tokens, tgt_mask=None, tgt_key_padding_mask=None):
        src = self.encode_coefficients(coeffs)

        tgt_len = target_tokens.size(1)
        positions = torch.arange(tgt_len, device=target_tokens.device)

        tgt = self.token_embedding(target_tokens)
        tgt = tgt + self.token_pos_encoding(positions).unsqueeze(0)
        tgt = self.dropout(tgt)

        output = self.transformer(
            src=src, tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.output_projection(output)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    @torch.no_grad()
    def generate_greedy(self, coeffs, max_len=60):
        self.eval()
        device = coeffs.device

        src = self.encode_coefficients(coeffs)
        memory = self.transformer.encoder(src)

        generated = [self.vocab.start_idx]

        for _ in range(max_len):
            tgt_tokens = torch.tensor([generated], device=device)
            tgt_len = tgt_tokens.size(1)
            positions = torch.arange(tgt_len, device=device)

            tgt = self.token_embedding(tgt_tokens)
            tgt = tgt + self.token_pos_encoding(positions).unsqueeze(0)

            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
            output = self.transformer.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            logits = self.output_projection(output[:, -1, :])

            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)

            if next_token == self.vocab.end_idx:
                break

        return generated

    @torch.no_grad()
    def generate_with_mask(self, coeffs, masker: SyntaxMasker, max_len=60, temperature=1.0):
        self.eval()
        device = coeffs.device

        src = self.encode_coefficients(coeffs)
        memory = self.transformer.encoder(src)

        generated = [self.vocab.start_idx]

        for _ in range(max_len):
            tgt_tokens = torch.tensor([generated], device=device)
            tgt_len = tgt_tokens.size(1)
            positions = torch.arange(tgt_len, device=device)

            tgt = self.token_embedding(tgt_tokens)
            tgt = tgt + self.token_pos_encoding(positions).unsqueeze(0)

            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
            output = self.transformer.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            logits = self.output_projection(output[:, -1, :])

            syntax_mask = masker.create_mask(generated, self.vocab.vocab_size).to(device)
            logits = logits + syntax_mask.unsqueeze(0)
            logits = logits / temperature

            probs = F.softmax(logits, dim=-1).squeeze(0)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            if next_token == self.vocab.end_idx:
                break

        return generated


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Test EtaProductTransformer (with COEFF_PH support)")
    print("=" * 60)

    vocab = EtaProductVocabulary(
        denominators=[1, 2, 3],
        max_num_coeff=5,
        max_shift=10,
    )
    print(f"\nVocab size: {vocab.vocab_size}")
    print(f"  COEFF_PH index: {vocab.coeff_ph_idx}")
    print(f"  Coeff tokens: {vocab.coeff_tokens}")
    print(f"  Shift tokens: {vocab.shift_tokens[:6]}...")

    # Test single product (legacy format) — with placeholder
    formula_legacy = {
        'factors': [
            (Fraction(1, 1), -2),
            (Fraction(1, 3), 3),
            (Fraction(7, 1), 4),
        ]
    }
    tokens_ph = vocab.formula_to_tokens(formula_legacy, use_placeholder=True)
    tokens_full = vocab.formula_to_tokens(formula_legacy, use_placeholder=False)
    print(f"\nLegacy formula: {vocab.formula_to_string(formula_legacy)}")
    print(f"Tokens (placeholder): {tokens_ph}")
    print(f"Tokens (full):        {tokens_full}")
    assert 'COEFF_PH' in tokens_ph
    assert 'COEFF_PH' not in tokens_full
    parsed_ph = vocab.tokens_to_formula(tokens_ph)
    parsed_full = vocab.tokens_to_formula(tokens_full)
    print(f"Round-trip (PH):   {vocab.formula_to_string(parsed_ph)}")
    print(f"Round-trip (full): {vocab.formula_to_string(parsed_full)}")
    assert parsed_ph['sum_terms'][0]['coeff'] is None
    assert parsed_full['sum_terms'][0]['coeff'] == 1

    # Test sum formula with placeholder
    formula_sum = {
        'sum_terms': [
            {'coeff': 1, 'shift': 0, 'factors': [(Fraction(1, 1), -3), (Fraction(2, 1), 1)]},
            {'coeff': 3, 'shift': 5, 'factors': [(Fraction(1, 1), -2)]},
        ]
    }
    tokens_sum_ph = vocab.formula_to_tokens(formula_sum, use_placeholder=True)
    tokens_sum_full = vocab.formula_to_tokens(formula_sum, use_placeholder=False)
    print(f"\nSum formula: {vocab.formula_to_string(formula_sum)}")
    print(f"Tokens (PH):   {tokens_sum_ph}")
    print(f"Tokens (full): {tokens_sum_full}")
    parsed_sum_ph = vocab.tokens_to_formula(tokens_sum_ph)
    assert parsed_sum_ph['sum_terms'][0]['coeff'] is None
    assert parsed_sum_ph['sum_terms'][1]['coeff'] is None
    print(f"Round-trip (PH): {vocab.formula_to_string(parsed_sum_ph)}")
    print("✓ Placeholder round-trip OK")

    # Test syntax masker with COEFF_PH
    masker = SyntaxMasker(vocab)
    valid = masker.get_valid_tokens([vocab.start_idx])
    assert valid == {vocab.token2idx[vocab.COEFF]}
    valid_after_coeff = masker.get_valid_tokens([vocab.start_idx, vocab.token2idx[vocab.COEFF]])
    assert vocab.coeff_ph_idx in valid_after_coeff
    valid_after_ph = masker.get_valid_tokens([vocab.start_idx, vocab.token2idx[vocab.COEFF], vocab.coeff_ph_idx])
    assert valid_after_ph == {vocab.token2idx[vocab.SHIFT]}
    print("✓ Syntax masker with COEFF_PH OK")

    # Model test
    config = ModelConfig(
        d_model=128, nhead=8,
        num_encoder_layers=4, num_decoder_layers=4,
        dim_feedforward=512, dropout=0.1,
        max_formula_len=80,
        n_encoder_tokens=16,
        n_coeffs=100,
        vocab_size=vocab.vocab_size,
    )

    model = EtaProductTransformer(config, vocab)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    B, L = 4, 100
    coeffs = torch.abs(torch.randn(B, L)) * 100 + 1
    target_len = 20
    target_tokens = torch.randint(0, vocab.vocab_size, (B, target_len))

    tgt_mask = model.generate_square_subsequent_mask(target_len)
    logits = model(coeffs, target_tokens, tgt_mask=tgt_mask)
    print(f"Forward OK: logits shape = {logits.shape}")

    gen_masked = model.generate_with_mask(coeffs[:1], masker, temperature=1.0)
    decoded_masked = vocab.decode(gen_masked)
    print(f"Masked gen: {decoded_masked}")
    print(f"Valid: {vocab.validate_formula_syntax(decoded_masked)}")

    print("\n✓ All tests passed!")
