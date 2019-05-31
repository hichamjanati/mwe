import numpy as np
from smtr.estimators import MLL
from smtr.utils import build_dataset


def test_mll():
    n_features = 50
    x, y, coefs_true = build_dataset(n_samples=20, n_features=n_features,
                                     n_targets=3, positive=False)
    y += 0.6 * np.std(y) * np.random.RandomState(42).randn(*y.shape)
    n_samples, n_tasks = y.shape
    X = np.array(n_tasks * [x])
    y = y.T
    xty = np.array([x.T.dot(yi) for x, yi in zip(X, y)])
    alphamax = abs(xty).max()
    fracs = [0.02, 0.05, 0.15]
    tol = 1e-4
    for a in fracs:
        alpha = a * alphamax
        model = MLL(alpha=alpha, positive=False, tol=tol)
        model.fit(X, y, compute_obj=True)
        loss = model.loss_

        assert np.diff(loss).max() <= 1e-5
