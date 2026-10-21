"""Public calibration origin-block comparisons; never reads development targets."""
import time
import numpy as np
from threadpoolctl import threadpool_limits
import tree_calibration
import spatial_calibration


def load():
    features = dict(np.load('/workspace/public_assets/train_features.npz'))
    targets = np.load('/workspace/public_assets/train_targets.npz')['target']
    return features, targets


def subset(features, index):
    return {key: value[index] for key, value in features.items()}


def mae(target, prediction):
    valid = np.isfinite(target) & (target > 0)
    return float(np.mean(np.abs(target[valid] - prediction[valid])))


def evaluate(fit, predict, name, splits=None):
    features, targets = load()
    if splits is None:
        splits = [(np.arange(50), np.arange(50, 74)),
                  (np.r_[0:25, 49:74], np.arange(25, 49))]
    start = time.monotonic()
    output = []
    for train, test in splits:
        state = fit(subset(features, train), targets[train], 17)
        pred = predict(state, subset(features, test))
        value = mae(targets[test], pred)
        output.append(value)
        print(name, 'test', test[0], test[-1], 'MAE', value,
              'elapsed', round(time.monotonic() - start, 2), flush=True)
    return output


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        evaluate(tree_calibration.fit, tree_calibration.predict, 'retained')
        for variant in ['neighbors', 'relative']:
            evaluate(lambda f, y, s: spatial_calibration.fit(f, y, s, variant=variant),
                     spatial_calibration.predict, variant)
