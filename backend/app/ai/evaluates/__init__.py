"""
Evaluation System.

Object detection evaluation framework with:
- Per-class detailed metrics (mAP, mAP@0.5, mAP@0.75, mAP@0.5:0.95)
- Precision, Recall, F1 Score
- IoU distribution analysis
- Size-specific metrics
- Robustness analysis for adversarial evaluation
"""
from app.ai.evaluates.evaluator import Evaluator

__all__ = [
    "Evaluator",
]
