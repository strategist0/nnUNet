"""
Plotting utilities for reliability analysis.
"""
from typing import Dict, Tuple

import numpy as np


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, title: str, output_path: str) -> None:
	"""
	Plot ROC curve and save to file.
	"""
	from sklearn.metrics import roc_curve, auc
	import matplotlib.pyplot as plt

	fpr, tpr, _ = roc_curve(y_true, y_score)
	roc_auc = auc(fpr, tpr)

	plt.figure(figsize=(6, 5))
	plt.plot(fpr, tpr, label=f'AUROC = {roc_auc:.3f}')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.legend(loc='lower right')
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, title: str, output_path: str) -> None:
	"""
	Plot Precision-Recall curve and save to file.
	"""
	from sklearn.metrics import precision_recall_curve, average_precision_score
	import matplotlib.pyplot as plt

	precision, recall, _ = precision_recall_curve(y_true, y_score)
	ap = average_precision_score(y_true, y_score)

	plt.figure(figsize=(6, 5))
	plt.plot(recall, precision, label=f'AUPRC = {ap:.3f}')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title(title)
	plt.legend(loc='lower left')
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def plot_risk_coverage(coverage: np.ndarray, risk: np.ndarray, title: str, output_path: str) -> None:
	"""
	Plot risk-coverage curve and save to file.
	"""
	import matplotlib.pyplot as plt

	plt.figure(figsize=(6, 5))
	plt.plot(coverage, risk)
	plt.xlabel('Coverage')
	plt.ylabel('Risk')
	plt.title(title)
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def plot_correlation(x: np.ndarray, y: np.ndarray, title: str, output_path: str, xlabel: str, ylabel: str) -> None:
	"""
	Plot scatter with a simple linear fit.
	"""
	import matplotlib.pyplot as plt

	plt.figure(figsize=(6, 5))
	plt.scatter(x, y, s=12, alpha=0.7)

	if len(x) > 1:
		coeffs = np.polyfit(x, y, 1)
		line = coeffs[0] * x + coeffs[1]
		plt.plot(x, line, 'r--', linewidth=1)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def plot_bar_comparison(labels, values_a, values_b, title: str, output_path: str,
					label_a: str = 'Baseline', label_b: str = 'MC',
					yerr_a=None, yerr_b=None) -> None:
	"""
	Plot side-by-side bar chart for two models.
	"""
	import matplotlib.pyplot as plt

	indices = np.arange(len(labels))
	width = 0.35

	plt.figure(figsize=(7, 4))
	plt.bar(indices - width / 2, values_a, width, label=label_a, yerr=yerr_a, capsize=3)
	plt.bar(indices + width / 2, values_b, width, label=label_b, yerr=yerr_b, capsize=3)
	plt.xticks(indices, labels, rotation=0)
	plt.title(title)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()


def plot_risk_coverage_compare(coverage_a: np.ndarray, risk_a: np.ndarray,
							coverage_b: np.ndarray, risk_b: np.ndarray,
							title: str, output_path: str,
							label_a: str = 'Baseline', label_b: str = 'MC') -> None:
	"""
	Plot risk-coverage curves for two models.
	"""
	import matplotlib.pyplot as plt

	plt.figure(figsize=(6, 5))
	plt.plot(coverage_a, risk_a, label=label_a)
	plt.plot(coverage_b, risk_b, label=label_b)
	plt.xlabel('Coverage')
	plt.ylabel('Risk')
	plt.title(title)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=150)
	plt.close()
