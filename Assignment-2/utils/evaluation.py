# utils/evaluation.py
import numpy as np

def char_f1_score(pred: str, target: str):
    """Character-level F1 (micro-averaged)"""
    pred_chars = list(pred)
    target_chars = list(target)
    if not target_chars:
        return 1.0 if not pred_chars else 0.0

    tp = len([c for c in pred_chars if c in target_chars])
    fp = len(pred_chars) - tp
    fn = len(target_chars) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def word_accuracy(pred: str, target: str):
    return float(pred == target)

def evaluate_model(predict_fn, test_dataset):
    f1_scores = []
    accs = []
    for ex in test_dataset:
        pred = predict_fn(ex["source"])
        f1 = char_f1_score(pred, ex["target"])
        acc = word_accuracy(pred, ex["target"])
        f1_scores.append(f1)
        accs.append(acc)
    return {
        "char_f1": np.mean(f1_scores),
        "word_acc": np.mean(accs),
        "total": len(test_dataset)
    }