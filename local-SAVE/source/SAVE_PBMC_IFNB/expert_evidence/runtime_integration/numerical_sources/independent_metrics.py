"""Trusted metric orientation; never use model-reported scores."""
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


def expression_mean_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if observed.ndim != 2 or predicted.ndim != 2:
        raise ValueError("Expression matrices must have cells x genes shape.")
    if not observed.shape[0] or not predicted.shape[0] or observed.shape[1] != predicted.shape[1]:
        raise ValueError("Expression matrix sizes are incompatible.")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Expression matrices contain nonfinite values.")
    true_mean = observed.mean(axis=0)
    pred_mean = predicted.mean(axis=0)
    correlation = float(pearsonr(true_mean, pred_mean)[0])
    return {
        "mse": float(mean_squared_error(true_mean, pred_mean)),
        "r2": float(r2_score(true_mean, pred_mean)),
        "pcc": correlation if np.isfinite(correlation) else None,
    }


if __name__ == "__main__":
    truth = np.array([[0., 2., 4.], [0., 2., 4.]])
    pred = np.array([[0., 1., 2.], [0., 1., 2.]])
    m = expression_mean_metrics(truth, pred)
    assert np.isclose(m["r2"], .375)
    assert np.isclose(m["mse"], 5 / 3)
    assert expression_mean_metrics(truth, truth)["mse"] == 0
    print("Metric orientation regression: passed")
