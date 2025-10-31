"""
Point Adjustment (PA) Evaluation Metrics for Time Series Anomaly Detection

Reference:
- "Revisiting Time Series Anomaly Detection" (TSB-UAD Benchmark)
- NAB (Numenta Anomaly Benchmark)
- Standard protocol for evaluating anomaly detection on time series
"""

import numpy as np
from typing import Tuple, List


def find_anomaly_segments(labels: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find continuous anomaly segments in labels
    
    Args:
        labels: Binary labels (0: normal, 1: anomaly)
    
    Returns:
        List of (start, end) tuples for each anomaly segment
    """
    segments = []
    in_segment = False
    start = 0
    
    for i, label in enumerate(labels):
        if label == 1 and not in_segment:
            # Start of new anomaly segment
            start = i
            in_segment = True
        elif label == 0 and in_segment:
            # End of anomaly segment
            segments.append((start, i))
            in_segment = False
    
    # Handle case where anomaly extends to end
    if in_segment:
        segments.append((start, len(labels)))
    
    return segments


def point_adjustment(
    predictions: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, dict]:

    gt = labels.copy().astype(int)
    pred = predictions.copy().astype(int)
    
    # Track anomaly segments for statistics
    anomaly_segments = find_anomaly_segments(gt)
    total_segments = len(anomaly_segments)
    detected_segments = 0
    
    # Point Adjustment algorithm
    anomaly_state = False
    
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            # Start of a detected anomaly segment
            anomaly_state = True
            detected_segments += 1
            
            # Expand backward to cover entire segment
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            
            # Expand forward to cover entire segment
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        
        elif gt[i] == 0:
            # Reset state when leaving anomaly region
            anomaly_state = False
        
        if anomaly_state:
            # Ensure all points in detected segment are marked
            pred[i] = 1
    
    info = {
        'total_segments': total_segments,
        'detected_segments': detected_segments,
        'segment_detection_rate': detected_segments / total_segments if total_segments > 0 else 0.0,
        'anomaly_segments': anomaly_segments
    }
    
    return pred, info


def compute_pa_metrics(
    predictions: np.ndarray,
    labels: np.ndarray
) -> dict:
    """
    Compute metrics with Point Adjustment (always applied).
    
    Args:
        predictions: Binary predictions (0 or 1)
        labels: Ground truth labels (0 or 1)
    
    Returns:
        Dictionary with metrics and PA info
    """
    predictions = predictions.astype(int)
    labels = labels.astype(int)
    
    # Always apply point adjustment
    adjusted_predictions, pa_info = point_adjustment(predictions, labels)
    
    # Compute confusion matrix
    TP = np.sum((adjusted_predictions == 1) & (labels == 1))
    TN = np.sum((adjusted_predictions == 0) & (labels == 0))
    FP = np.sum((adjusted_predictions == 1) & (labels == 0))
    FN = np.sum((adjusted_predictions == 0) & (labels == 1))
    
    # Compute metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': int(TP),
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        **pa_info
    }
    
    return metrics


def compute_best_f1_with_threshold_search(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200
) -> Tuple[float, float, dict]:
    """
    Search for best threshold that maximizes F1-score.
    Always uses Point Adjustment.
    
    Args:
        anomaly_scores: Anomaly scores per time step
        labels: Ground truth labels
        n_thresholds: Number of candidate thresholds to test
    
    Returns:
        (best_f1, best_threshold, best_metrics)
    """
    # Generate candidate thresholds
    min_score = np.min(anomaly_scores)
    max_score = np.max(anomaly_scores)
    thresholds = np.linspace(min_score, max_score, n_thresholds)
    
    best_f1 = 0.0
    best_threshold = thresholds[0]
    best_metrics = {}
    
    for threshold in thresholds:
        # Convert scores to predictions
        predictions = (anomaly_scores >= threshold).astype(int)
        
        # Compute metrics (always with PA)
        metrics = compute_pa_metrics(predictions, labels)
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
            best_metrics = metrics
    
    return best_f1, best_threshold, best_metrics


def evaluate_with_pa(
    anomaly_scores: np.ndarray,
    labels: np.ndarray
) -> dict:
    """
    Evaluate anomaly detection with Point Adjustment.
    Always searches for best threshold (maximize F1) and applies Point Adjustment.
    
    Args:
        anomaly_scores: Anomaly scores per time step
        labels: Ground truth labels
    
    Returns:
        Dictionary with best_threshold, best_f1, metrics, and predictions
    """
    # Always search for best threshold (with PA)
    best_f1, best_threshold, best_metrics = compute_best_f1_with_threshold_search(
        anomaly_scores, labels
    )
    
    results = {
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'threshold': best_threshold  # Use best threshold as the threshold
    }
    
    # Update with best metrics
    results.update({f'best_{k}': v for k, v in best_metrics.items()})
    
    # Convert to predictions at best threshold
    predictions = (anomaly_scores >= best_threshold).astype(int)
    results['predictions'] = predictions  # Include predictions for visualization
    
    # Compute final metrics (should be same as best_metrics, but keeping for consistency)
    results.update(best_metrics)
    
    return results


# Example usage
if __name__ == '__main__':
    # Example: Time series with anomaly segment
    np.random.seed(42)
    
    # Create synthetic labels with 2 anomaly segments
    labels = np.zeros(1000)
    labels[100:150] = 1  # First anomaly segment (50 points)
    labels[500:520] = 1  # Second anomaly segment (20 points)
    
    # Create synthetic predictions (detected 1st segment partially, missed 2nd)
    predictions = np.zeros(1000)
    predictions[120:130] = 1  # Detected part of 1st segment
    predictions[800:810] = 1  # False positive
    
    # With Point Adjustment (always applied)
    print("WITH Point Adjustment:")
    metrics = compute_pa_metrics(predictions, labels)
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Segments detected: {metrics['detected_segments']}/{metrics['total_segments']}")
    
    # With threshold search
    anomaly_scores = np.random.random(1000)
    anomaly_scores[100:150] += 0.5  # Higher scores for 1st segment
    
    print("\nWith Threshold Search:")
    best_f1, best_threshold, best_metrics = compute_best_f1_with_threshold_search(
        anomaly_scores, labels
    )
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Best threshold: {best_threshold:.4f}")

