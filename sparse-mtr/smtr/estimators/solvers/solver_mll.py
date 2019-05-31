"""Solvers for multi Level Lasso."""
import numpy as np
from celer import Lasso
from scipy.sparse import block_diag


def objective(X, Y, C, S, alpha):
    n_tasks, n_samples, n_features = X.shape
    theta = C[:, None] * S
    R = np.array([x.dot(t) - y for x, t, y in zip(X, theta.T, Y)])
    obj = (0.5 * np.linalg.norm(R) ** 2) / n_samples
    obj += alpha * abs(C).sum()
    obj += alpha * abs(S).sum()
    return obj


def solver_mll(X, y, alpha=0.1, C=None, S=None, callback=None, positive=False,
               maxiter=1000, tol=1e-4, compute_obj=False):
    """Perform Lasso alternating to solve Multi-level lasso"""
    n_tasks, n_samples, n_features = X.shape
    lasso = Lasso(alpha=alpha, fit_intercept=False,
                  positive=positive)
    lasso_p = Lasso(alpha=alpha / n_tasks, fit_intercept=False,
                    positive=True)
    if S is None:
        S = np.zeros((n_features, n_tasks))
    if C is None:
        C = np.ones(n_features)
    else:
        if C.max() <= 0:
            C = np.ones(n_features)

    old_theta = C[:, None] * S
    objs = []
    if compute_obj or callback:
        ll = objective(X, y, C, S, alpha)
        objs.append(ll)
    for i in range(maxiter):
        # W = block_diag(X * C[None, None, :], "csc")
        # lasso.fit(W, y.flatten())
        # S = lasso.coef_.reshape(n_tasks, n_features).T
        W = X * C[None, None, :]
        for k in range(n_tasks):
            lasso.fit(W[k], y[k])
            S[:, k] = lasso.coef_
        Z = S.T[:, None, :] * X
        Z = Z.reshape(n_tasks * n_samples, n_features)
        lasso_p.fit(Z, y.flatten())
        C = lasso_p.coef_
        theta = C[:, None] * S
        dll = abs(theta - old_theta).max()
        dll /= max(theta.max(), old_theta.max(), 1.)
        old_theta = theta.copy()
        if compute_obj or callback:
            ll = objective(X, y, C, S, alpha)
            objs.append(ll)
        if callback:
            callback(theta, obj=ll)
        if dll < tol:
            break

    if i == maxiter - 1:
        print("**************************************\n"
              "******** WARNING: Stopped early. *****\n"
              "\n"
              "You may want to increase maxiter. Last err: %f" % dll)
    return C, S, objs
