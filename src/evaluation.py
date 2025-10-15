import numpy as np

def to_bool(mask):
    m = (mask > 0)
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D (H,W), got shape {mask.shape}")
    return m

def compute_iou(mask1, mask2):
    m1 = to_bool(mask1)
    m2 = to_bool(mask2)
    if m1.shape != m2.shape:
        raise ValueError(f"Mask shapes differ: {m1.shape} vs {m2.shape}")
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return float(inter) / union if union > 0 else 0.0

def match_masks(pred_masks, gt_masks, iou_thr):
    # Handle empty cases explicitly
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 0, 0, 0
    if len(gt_masks) == 0:
        return 0, len(pred_masks), 0  # all preds are FP
    if len(pred_masks) == 0:
        return 0, 0, len(gt_masks)    # all GTs are FN

    tp, fp = 0, 0
    matched_gt = set()

    # Greedy one-to-one matching by best IoU per prediction
    for p in pred_masks:
        ious = [compute_iou(p, g) for g in gt_masks] 
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and j not in matched_gt:
            tp += 1
            matched_gt.add(j)
        else:
            fp += 1

    fn = len(gt_masks) - len(matched_gt)
    return tp, fp, fn

def f2_from_counts(tp, fp, fn, eps=1e-9):
    # F2 with beta=2: (5*TP)/(5*TP + 4*FN + FP)
    return (5*tp) / (5*tp + 4*fn + fp + eps)

def compute_f2_score(preds, gts, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    scores = []
    for thr in iou_thresholds:
        TP = FP = FN = 0
        for pred, gt in zip(preds, gts):
            t, f, n = match_masks(pred, gt, thr)
            TP += t; FP += f; FN += n
        scores.append(f2_from_counts(TP, FP, FN))
    return float(np.mean(scores))
