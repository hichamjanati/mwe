"""Solvers for multitask group-stl with a simplex constraint."""
import numpy as np
from sklearn.linear_model import Lasso


def solver_stl(X, Y, alpha=None, callback=None, positive=False,
               maxiter=3000, tol=1e-4):
    """Perform CD to solve positive Lasso."""

    n_tasks, n_samples, n_features = X.shape
    theta = np.zeros((n_features, n_tasks))
    if callback:
        callback(theta)

    if alpha is None:
        alpha = np.ones(n_tasks)
    alpha = np.asarray(alpha).reshape(n_tasks)
    for k in range(n_tasks):
        lasso = Lasso(alpha=alpha[k], tol=tol, max_iter=maxiter,
                      positive=positive, fit_intercept=False)
        lasso.fit(X[k], Y[k])
        theta[:, k] = lasso.coef_
        if callback:
            callback(theta)

    return theta
