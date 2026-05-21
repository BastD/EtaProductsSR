"""
Training script for Eta Product Symbolic Regression (with Sums support)

Supports COEFF_PH placeholder mode: the model predicts formula skeletons
(with COEFF_PH instead of actual coefficient values). Coefficients are
brute-forced in a separate analysis step.

Usage:
    python train_tf_sr.py \
        --data_path ./data_eta_product/eta_product_dataset.pkl \
        --epochs 100 --output_dir ./checkpoints

    # With syntax masking during validation
    python train_tf_sr.py \
        --data_path ./data_eta_product/eta_product_dataset.pkl \
        --epochs 100 --syntax_masking --output_dir ./checkpoints
"""

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import random
from pathlib import Path
import json
from typing import List, Dict, Optional
from fractions import Fraction

from transformer_eta_product import (
    EtaProductTransformer,
    ModelConfig,
    EtaProductVocabulary,
    SyntaxMasker,
)


# ============================================================================
# SET-BASED COMPARISON (SKELETON-AWARE)
# ============================================================================

def normalize_formula_skeleton(formula: Dict) -> Optional[set]:
    """
    Convert formula to a set of comparable terms for order-independent matching.
    Coefficients are IGNORED (treated as wildcard) — only structure matters.

    Each term becomes: (shift, frozenset((k, exp), ...))
    """
    if not formula or 'sum_terms' not in formula:
        return None

    terms = set()
    for term in formula['sum_terms']:
        shift = term.get('shift', 0)
        # Normalize k to Fraction for consistent comparison
        factors = []
        for k, exp in term['factors']:
            if isinstance(k, int):
                k = Fraction(k)
            factors.append((k, exp))
        # Coeff is NOT included — skeleton comparison
        terms.add((shift, frozenset(factors)))

    return terms


def compare_formulas_skeleton(pred_formula: Dict, true_formula: Dict) -> bool:
    """Compare formula skeletons (ignoring coefficients, order-independent)."""
    pred_set = normalize_formula_skeleton(pred_formula)
    true_set = normalize_formula_skeleton(true_formula)
    if pred_set is None or true_set is None:
        return False
    return pred_set == true_set


def count_matching_terms_skeleton(pred_formula: Dict, true_formula: Dict) -> int:
    """Count how many individual terms match in skeleton (ignoring coefficients)."""
    pred_set = normalize_formula_skeleton(pred_formula)
    true_set = normalize_formula_skeleton(true_formula)
    if pred_set is None or true_set is None:
        return 0
    return len(pred_set & true_set)


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_pickle_dataset(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def prepare_split(split_data: Dict) -> List[Dict]:
    """Convert a dataset split into list of sample dicts."""
    samples = []

    for i in range(len(split_data['formulas'])):
        formula = split_data['formulas'][i]
        coeffs = split_data['coefficients'][i]

        # Normalize to sum format
        if 'sum_terms' not in formula:
            formula = {
                'sum_terms': [{
                    'coeff': 1,
                    'shift': 0,
                    'factors': formula.get('factors', []),
                }],
                'lcm_denominator': formula.get('lcm_denominator', 1),
            }

        samples.append({
            'coeffs': coeffs.astype(np.float32),
            'formula': formula,
        })

    return samples


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class EtaProductDataset(Dataset):
    def __init__(self, samples: List[Dict], vocab: EtaProductVocabulary,
                 use_placeholder: bool = True):
        self.samples = samples
        self.vocab = vocab
        self.use_placeholder = use_placeholder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        coeffs = torch.tensor(s['coeffs'], dtype=torch.float32)

        formula = s['formula']
        tokens = self.vocab.formula_to_tokens(formula, use_placeholder=self.use_placeholder)
        token_ids = torch.tensor(self.vocab.encode(tokens), dtype=torch.long)

        return {
            'coeffs': coeffs,
            'token_ids': token_ids,
            'formula': formula,
        }


class EtaCollateFunction:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        return {
            'coeffs': torch.stack([item['coeffs'] for item in batch]),
            'token_ids': pad_sequence(
                [item['token_ids'] for item in batch],
                batch_first=True, padding_value=self.pad_idx
            ),
            'formulas': [item['formula'] for item in batch],
        }


# ============================================================================
# LR SCHEDULER
# ============================================================================

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            lr = self.base_lr * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, vocab, grad_clip=1.0):
    model.train()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        coeffs = batch['coeffs'].to(device)
        target = batch['token_ids'].to(device)

        optimizer.zero_grad()

        dec_in = target[:, :-1]
        dec_out = target[:, 1:]

        tgt_mask = model.generate_square_subsequent_mask(dec_in.size(1)).to(device)
        tgt_pad_mask = (dec_in == vocab.pad_idx)

        logits = model(
            coeffs, dec_in,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask
        )

        loss = criterion(logits.reshape(-1, logits.size(-1)), dec_out.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        mask = (dec_out != vocab.pad_idx)
        correct_tokens += ((preds == dec_out) & mask).sum().item()
        total_tokens += mask.sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct_tokens / max(total_tokens, 1):.3f}'
        })

    return {
        'loss': total_loss / len(dataloader),
        'token_accuracy': correct_tokens / max(total_tokens, 1),
    }


def validate(model, dataloader, criterion, device, vocab, masker, epoch,
             num_examples=10, syntax_masking=False):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    syntax_valid = 0
    skeleton_exact = 0
    terms_matched = 0
    terms_total = 0
    total_generated = 0
    examples = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

        for batch_idx, batch in enumerate(pbar):
            coeffs = batch['coeffs'].to(device)
            target = batch['token_ids'].to(device)

            dec_in = target[:, :-1]
            dec_out = target[:, 1:]
            tgt_mask = model.generate_square_subsequent_mask(dec_in.size(1)).to(device)
            tgt_pad_mask = (dec_in == vocab.pad_idx)

            logits = model(
                coeffs, dec_in,
                tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask
            )

            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_out.reshape(-1))
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            mask = (dec_out != vocab.pad_idx)
            correct_tokens += ((preds == dec_out) & mask).sum().item()
            total_tokens += mask.sum().item()

            # Generation check
            if batch_idx < 50:
                if syntax_masking:
                    gen = model.generate_with_mask(
                        coeffs[:1], masker, max_len=80, temperature=1.0)
                else:
                    gen = model.generate_greedy(coeffs[:1], max_len=80)

                pred_tokens = vocab.decode(gen)
                pred_formula = vocab.tokens_to_formula(pred_tokens)
                is_valid = pred_formula is not None
                if is_valid:
                    syntax_valid += 1
                total_generated += 1

                # True formula from the dataset (with real coefficients)
                true_formula = batch['formulas'][0]

                # Skeleton-based comparison (ignoring coefficients)
                skel_match = False
                if pred_formula and true_formula:
                    if compare_formulas_skeleton(pred_formula, true_formula):
                        skeleton_exact += 1
                        skel_match = True
                    matched = count_matching_terms_skeleton(pred_formula, true_formula)
                    terms_matched += matched
                    n_true_terms = len(true_formula.get('sum_terms',
                                      [{'factors': true_formula.get('factors', [])}]))
                    terms_total += n_true_terms

                if batch_idx < num_examples:
                    examples.append({
                        'pred_tokens': pred_tokens,
                        'pred_formula': pred_formula,
                        'true_formula': true_formula,
                        'valid': is_valid,
                        'skeleton_match': skel_match,
                    })

            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'syntax': f'{syntax_valid}/{total_generated}',
                'skel': f'{skeleton_exact}/{total_generated}',
            })

    return {
        'loss': total_loss / len(dataloader),
        'token_accuracy': correct_tokens / max(total_tokens, 1),
        'syntax_valid_rate': syntax_valid / max(total_generated, 1),
        'syntax_valid_count': syntax_valid,
        'syntax_total': total_generated,
        'skeleton_exact_rate': skeleton_exact / max(total_generated, 1),
        'terms_partial_rate': terms_matched / max(terms_total, 1),
        'examples': examples,
    }


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    raw = load_pickle_dataset(args.data_path)

    cfg = raw['config']
    denominators = cfg['denominators']
    max_num_coeff = cfg.get('max_num_coeff', 1)
    max_shift = cfg.get('max_shift', 0)
    print(f"Dataset config: denominators={denominators}, "
          f"max_num_coeff={max_num_coeff}, max_shift={max_shift}, "
          f"sum_terms={cfg.get('min_sum_terms',1)}-{cfg.get('max_sum_terms',1)}")

    train_samples = prepare_split(raw['train'])
    val_samples = prepare_split(raw['val'])
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Auto-detect exponent range
    all_exps = set()
    for split_key in ['train', 'val', 'test']:
        if split_key in raw:
            for f in raw[split_key]['formulas']:
                if 'sum_terms' in f:
                    for term in f['sum_terms']:
                        for k, e in term['factors']:
                            all_exps.add(e)
                elif 'factors' in f:
                    for k, e in f['factors']:
                        all_exps.add(e)

    actual_max_exp = max(abs(e) for e in all_exps)
    actual_max_k = cfg.get('max_k', 12)
    print(f"Exponent range: [{min(all_exps)}, {max(all_exps)}] → max_exp={actual_max_exp}")
    print(f"Max k: {actual_max_k}")

    # Vocabulary
    vocab = EtaProductVocabulary(
        denominators=denominators,
        max_k=actual_max_k,
        max_exp=actual_max_exp,
        max_num_coeff=max_num_coeff,
        max_shift=max_shift,
    )
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"  Integer eta: {len(vocab.eta_int_tokens)}, "
          f"Fractional eta: {len(vocab.eta_frac_tokens)}, "
          f"Exponents: {len(vocab.exponent_tokens)}, "
          f"Coefficients: {len(vocab.coeff_tokens)}, "
          f"Shifts: {len(vocab.shift_tokens)}, "
          f"COEFF_PH: idx={vocab.coeff_ph_idx}")

    # Verify tokenization (with placeholder)
    n_ok = 0
    for s in train_samples[:100]:
        try:
            vocab.encode(vocab.formula_to_tokens(s['formula'], use_placeholder=True))
            n_ok += 1
        except KeyError as e:
            print(f"  Missing token: {e}")
            break
    print(f"Tokenization check (placeholder): {n_ok}/{min(100, len(train_samples))} OK")

    masker = SyntaxMasker(vocab)

    # use_placeholder=True: targets have COEFF_PH instead of c1..cN
    train_ds = EtaProductDataset(train_samples, vocab, use_placeholder=True)
    val_ds = EtaProductDataset(val_samples, vocab, use_placeholder=True)

    # Model
    n_coeffs = cfg['n_coeffs']
    config = ModelConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_formula_len=args.max_formula_len,
        n_encoder_tokens=args.n_encoder_tokens,
        n_coeffs=n_coeffs,
        use_periodic=args.use_periodic,
        vocab_size=vocab.vocab_size,
    )

    model = EtaProductTransformer(config, vocab)

    start_epoch = 1
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(
        ignore_index=vocab.pad_idx,
        label_smoothing=args.label_smoothing,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay,
    )
    if args.checkpoint and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    total_steps = (len(train_ds) // args.batch_size) * args.epochs
    scheduler = WarmupScheduler(optimizer, args.warmup_steps, total_steps)

    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Mode: SKELETON (COEFF_PH placeholder)")
    print(f"Syntax masking (val): {'ENABLED' if args.syntax_masking else 'DISABLED'}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max formula len: {args.max_formula_len}")
    print(f"{'='*60}\n")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    history = {k: [] for k in [
        'train_loss', 'train_acc',
        'val_loss', 'val_acc', 'val_syntax_rate',
        'val_skeleton_exact_rate', 'val_terms_partial_rate',
        'lr',
    ]}

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers,
            collate_fn=EtaCollateFunction(vocab.pad_idx),
            pin_memory=(device.type == 'cuda'),
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            collate_fn=EtaCollateFunction(vocab.pad_idx),
            pin_memory=(device.type == 'cuda'),
        )

        train_m = train_epoch(model, train_loader, optimizer, criterion, device, epoch, vocab, args.grad_clip)
        lr = scheduler.step()

        val_m = validate(model, val_loader, criterion, device, vocab, masker, epoch,
                         syntax_masking=args.syntax_masking)

        # Print metrics
        print(f"\n📊 Train: loss={train_m['loss']:.4f} acc={train_m['token_accuracy']:.4f}")
        print(f"📊 Val:   loss={val_m['loss']:.4f} acc={val_m['token_accuracy']:.4f}")
        print(f"  🎯 Syntax valid: {val_m['syntax_valid_count']}/{val_m['syntax_total']} "
              f"({val_m['syntax_valid_rate']*100:.1f}%)")
        print(f"  ✅ Skeleton exact: {val_m['skeleton_exact_rate']*100:.1f}%")
        print(f"  ✅ Terms partial:  {val_m['terms_partial_rate']*100:.1f}%")

        for i, ex in enumerate(val_m['examples'][:3]):
            status = "✓" if ex['skeleton_match'] else ("~" if ex['valid'] else "✗")
            print(f"\n  {i+1}. [{status}]")
            if ex['pred_formula']:
                print(f"     Pred: {vocab.formula_to_string(ex['pred_formula'])}")
            else:
                pred_short = ' '.join(ex['pred_tokens'][:20])
                print(f"     Pred (raw): {pred_short}...")
            if ex['true_formula']:
                print(f"     True: {vocab.formula_to_string(ex['true_formula'])}")

        print(f"\n  LR: {lr:.6f}")

        # History
        history['train_loss'].append(train_m['loss'])
        history['train_acc'].append(train_m['token_accuracy'])
        history['val_loss'].append(val_m['loss'])
        history['val_acc'].append(val_m['token_accuracy'])
        history['val_syntax_rate'].append(val_m['syntax_valid_rate'])
        history['val_skeleton_exact_rate'].append(val_m['skeleton_exact_rate'])
        history['val_terms_partial_rate'].append(val_m['terms_partial_rate'])
        history['lr'].append(lr)

        # Checkpoints
        ckpt_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'vocab_denominators': denominators,
            'vocab_max_exp': actual_max_exp,
            'vocab_max_k': actual_max_k,
            'vocab_max_num_coeff': max_num_coeff,
            'vocab_max_shift': max_shift,
            'history': history,
            'args': vars(args),
        }
        torch.save(ckpt_data, output_dir / 'checkpoint_last.pt')

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            torch.save(ckpt_data, output_dir / 'checkpoint_best.pt')
            print(f"✓ New best val loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_encoder_tokens', type=int, default=16)
    parser.add_argument('--max_formula_len', type=int, default=80)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--early_stopping', type=int, default=10)

    parser.add_argument('--syntax_masking', action='store_true')
    parser.add_argument('--use_periodic', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    main(parser.parse_args())
