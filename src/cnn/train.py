"""Train response-only (RQ1) or prompt+response (RQ2) CNNs on fixed splits.

From the repository root:
    python -m src.cnn.train --mode response
    python -m src.cnn.train --mode prompt_response
"""

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import TextDataset, build_vocab, collate_batch, input_tokens, load_splits
from .model import TextCNN, initialize_glove

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def choose_device(requested='auto'):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, truth, predicted, probabilities = 0.0, [], [], []
    with torch.set_grad_enabled(training):
        for ids, lengths, labels in loader:
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(ids, lengths)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; check the inputs and learning rate.')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            truth.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(1).detach().cpu().tolist())
            probabilities.extend(logits.softmax(1).detach().cpu().tolist())
    metrics = {
        'loss': total_loss / len(truth),
        'accuracy': float(accuracy_score(truth, predicted)),
        'macro_f1': float(f1_score(truth, predicted, labels=list(range(model.classifier.out_features)),
                                   average='macro', zero_division=0)),
    }
    return metrics, truth, predicted, probabilities


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def save_plots(output_dir, history, matrix, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ('loss', 'macro_f1')):
        for split in ('train', 'validation'):
            ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] for r in history],
                    label=split, marker='o', markersize=3)
        ax.set(xlabel='Epoch', ylabel=metric.replace('_', ' ').title())
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'learning_curves.png', dpi=160)
    plt.close(fig)
    if matrix is not None:
        from sklearn.metrics import ConfusionMatrixDisplay
        display = ConfusionMatrixDisplay(matrix, display_labels=labels)
        display.plot(cmap='Blues', values_format='d', colorbar=False)
        display.ax_.set_title('CNN — held-out test set')
        display.figure_.tight_layout()
        display.figure_.savefig(output_dir / 'confusion_matrix.png', dpi=160)
        plt.close(display.figure_)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PROJECT_ROOT / 'data/splits')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--mode', choices=['response', 'prompt_response'], default='response')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--response-length', type=int, default=512)
    parser.add_argument('--prompt-length', type=int, default=128)
    parser.add_argument('--embedding-dim', type=int, default=100)
    parser.add_argument('--num-filters', type=int, default=100)
    parser.add_argument('--kernel-sizes', type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--max-vocab', type=int, default=30000)
    parser.add_argument('--min-frequency', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--glove', type=Path, help='Optional local GloVe text file; dimension must match.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--skip-test', action='store_true', help='Validate only, for debugging/tuning.')
    args = parser.parse_args(argv)
    positive = ['epochs', 'patience', 'batch_size', 'response_length', 'prompt_length',
                'embedding_dim', 'num_filters', 'min_frequency', 'threads']
    if any(getattr(args, key) < 1 for key in positive) or min(args.kernel_sizes) < 1:
        parser.error('Lengths, counts, kernel sizes and patience must be positive.')
    if args.max_vocab < 4 or not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0:
        parser.error('Invalid vocabulary size, dropout, learning rate or weight decay.')
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    output_dir = args.output_dir or PROJECT_ROOT / 'results/cnn' / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f'{output_dir} is not empty. Choose a new --output-dir for this run.')

    splits = load_splits(args.data_dir)
    labels = sorted(splits['train']['family'].unique().tolist())
    label_to_id = {label: i for i, label in enumerate(labels)}
    token_lists = {
        name: [input_tokens(row, args.mode, args.response_length, args.prompt_length)
               for row in frame.to_dict('records')]
        for name, frame in splits.items()
    }
    vocab = build_vocab(token_lists['train'], args.max_vocab, args.min_frequency)
    loaders = {}
    generator = torch.Generator().manual_seed(args.seed)
    for name, frame in splits.items():
        dataset = TextDataset(token_lists[name], frame['family'], vocab, label_to_id)
        loaders[name] = DataLoader(dataset, batch_size=args.batch_size, shuffle=name == 'train',
                                   collate_fn=collate_batch, generator=generator, num_workers=0)
    model_config = dict(vocab_size=len(vocab), num_classes=len(labels),
                        embedding_dim=args.embedding_dim, num_filters=args.num_filters,
                        kernel_sizes=args.kernel_sizes, dropout=args.dropout)
    model = TextCNN(**model_config)
    glove_matches = initialize_glove(model, vocab, args.glove) if args.glove else 0
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    preprocessing = dict(mode=args.mode, response_length=args.response_length, prompt_length=args.prompt_length)
    metadata = {
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'device': str(device), 'labels': labels, 'vocab_size': len(vocab), 'glove_matches': glove_matches,
        'split_counts': {name: {k: int(v) for k, v in frame.family.value_counts().items()}
                         for name, frame in splits.items()},
        'data_sha256': {name: hashlib.sha256((args.data_dir / f'{name}.parquet').read_bytes()).hexdigest()
                        for name in splits},
        'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                     'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
    }
    write_json(output_dir / 'run_config.json', metadata)
    print(f'Device: {device} | mode: {args.mode} | vocabulary: {len(vocab):,}', flush=True)
    print('Split sizes:', {name: len(frame) for name, frame in splits.items()}, flush=True)
    best_score, best_epoch, stale, history = -1.0, 0, 0, []
    checkpoint_path = output_dir / 'best_model.pt'
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        train_metrics, *_ = run_epoch(model, loaders['train'], device, optimizer)
        validation_metrics, *_ = run_epoch(model, loaders['validation'], device)
        history.append({'epoch': epoch, 'seconds': time.monotonic() - started,
                        **{f'train_{k}': v for k, v in train_metrics.items()},
                        **{f'validation_{k}': v for k, v in validation_metrics.items()}})
        pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)
        print(f'Epoch {epoch:02d} | loss {train_metrics["loss"]:.4f} | '
              f'val accuracy {validation_metrics["accuracy"]:.4f} | '
              f'val macro F1 {validation_metrics["macro_f1"]:.4f} | '
              f'{history[-1]["seconds"]:.1f}s', flush=True)
        if validation_metrics['macro_f1'] > best_score:
            best_score, best_epoch, stale = validation_metrics['macro_f1'], epoch, 0
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'model_config': model_config, 'vocab': vocab, 'labels': labels,
                        'preprocessing': preprocessing, 'best_epoch': best_epoch,
                        'validation_metrics': validation_metrics}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f'Early stopping; best epoch was {best_epoch}.', flush=True)
                break

    # Restore the best validation checkpoint BEFORE evaluating the test split.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    summary = {'mode': args.mode, 'best_epoch': best_epoch,
               'validation': checkpoint['validation_metrics'], 'test_evaluated': not args.skip_test}
    matrix = None
    if not args.skip_test:
        test_metrics, truth, predicted, probabilities = run_epoch(model, loaders['test'], device)
        class_ids = list(range(len(labels)))
        report = classification_report(truth, predicted, labels=class_ids, target_names=labels,
                                       output_dict=True, zero_division=0)
        matrix = confusion_matrix(truth, predicted, labels=class_ids)
        majority_id = label_to_id[splits['train']['family'].value_counts().idxmax()]
        majority_predictions = [majority_id] * len(truth)
        summary.update(test=test_metrics, classification_report=report,
                       majority_baseline={'accuracy': float(accuracy_score(truth, majority_predictions)),
                                          'macro_f1': float(f1_score(truth, majority_predictions,
                                                                   labels=class_ids, average='macro', zero_division=0))})
        pd.DataFrame(matrix, index=pd.Index(labels, name='true_family'), columns=labels).to_csv(
            output_dir / 'confusion_matrix.csv')
        predictions = splits['test'][['sample_id', 'family']].copy()
        predictions['predicted_family'] = [labels[i] for i in predicted]
        for i, label in enumerate(labels):
            predictions[f'probability_{label}'] = [row[i] for row in probabilities]
        predictions.to_csv(output_dir / 'test_predictions.csv', index=False)
        print(classification_report(truth, predicted, labels=class_ids, target_names=labels,
                                    digits=4, zero_division=0), flush=True)
    write_json(output_dir / 'metrics.json', summary)
    save_plots(output_dir, history, matrix, labels)
    print(f'Saved model and results to {output_dir}', flush=True)
    return summary


if __name__ == '__main__':
    main()
