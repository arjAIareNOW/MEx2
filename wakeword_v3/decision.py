import torch
from wakeword.metrics import compute_metrics


def threshold_sweep(probabilities, labels, far_cap=0.0025):
    """Maximize validation F1 subject to a predeclared clip FAR <= 0.25%.

    Exact distinct observed thresholds, plus reject-all; ties prefer recall then
    fewer false accepts. FAR is FP/non-MARVIN examples, NOT activations/hour.
    """
    p, y = probabilities.float().cpu(), labels.long().cpu()
    if not len(p) or not torch.isfinite(p).all() or not 0 < int(y.sum()) < len(y):
        raise ValueError("Need finite scores and both validation classes")
    order = torch.argsort(p, descending=True)
    scores, yy = p[order], y[order]
    tp = yy.cumsum(0)
    fp = torch.arange(1, len(y)+1)-tp
    ends = torch.ones(len(y), dtype=torch.bool)
    ends[:-1] = scores[:-1] != scores[1:]
    rows = [compute_metrics(p, y, float(torch.nextafter(p.max(), torch.tensor(float('inf'))))).to_dict()]
    # Vectorized confusion counts keep the sweep O(N log N), not O(N^2).
    positives, negatives = int(y.sum()), int((y == 0).sum())
    for i in ends.nonzero().flatten().tolist():
        a, b = int(tp[i]), int(fp[i])
        precision, recall = a/(a+b), a/positives
        rows.append(dict(threshold=float(scores[i]), examples=len(y), positives=positives,
                         negatives=negatives, true_positives=a, true_negatives=negatives-b,
                         false_accepts=b, false_rejects=positives-a,
                         accuracy=(a+negatives-b)/len(y), precision=precision, recall=recall,
                         f1=2*a/(positives+a+b), false_accept_rate=b/negatives,
                         false_reject_rate=1-recall))
    eligible = [r for r in rows if r['false_accept_rate'] <= far_cap]
    best = max(eligible, key=lambda r: (r['f1'], r['recall'], -r['false_accepts']))
    return best, rows
