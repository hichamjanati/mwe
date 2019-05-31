import numpy as np
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
from smtr import utils
from smtr.estimators import Dirty
from smtr.model_selection import cv_score_dirty as cross_val_score
from smtr.model_selection import best_score_dirty
from numpy.testing import assert_allclose
from smtr.utils import build_dataset


def test_dirty():
    n_features = 50
    x, y, coefs_true = build_dataset(n_samples=20, n_features=n_features,
                                     n_targets=3, positive=False)
    y += 0.6 * np.std(y) * np.random.RandomState(42).randn(*y.shape)
    n_samples, n_tasks = y.shape
    X = np.array(n_tasks * [x])
    y = y.T
    xty = np.array([x.T.dot(yi) for x, yi in zip(X, y)])
    alphamax = np.linalg.norm(xty, axis=0).max()
    betamax = abs(xty).max()
    fracs = [(0.15, 0.05), (0.15, 0.14), (0.15, 0.2)]
    tol = 1e-4
    tol_dg = tol * np.linalg.norm(y) ** 2
    for a, b in fracs:
        alpha = a * alphamax
        beta = b * betamax
        model = Dirty(alpha=alpha, beta=beta, positive=False, tol=tol)
        model.fit(X, y)
        R = model.residuals_
        coef1, coef2 = model.coefs_tuple_
        dualgap = np.linalg.norm(R) ** 2
        dualgap += alpha * np.linalg.norm(coef1, axis=1).sum()
        dualgap += beta * abs(coef2).sum()
        dualgap += - (R * y).sum()
        assert abs(dualgap) < tol_dg


def test_grouplasso_vs_sklearn():
    cv = 3
    X, y, coefs_true = build_dataset(n_samples=cv * 10, n_features=100,
                                     n_targets=3)

    y += 0.6 * np.std(y) * np.random.RandomState(42).randn(*y.shape)
    n_samples, n_tasks = y.shape
    YY = y.T
    XX = np.array([X] * n_tasks)
    model = Dirty(positive=False, tol=1e-8)

    # Test cv score
    cv_scores, model, params_grid = cross_val_score(model, XX, YY, cv=cv,
                                                    cv_size=10,
                                                    mtgl_only=True)

    cv_scores = np.mean(cv_scores, axis=-1)  # average across tasks
    alphas = np.array([p["alpha"] for p in params_grid])
    alphas = cv / (cv - 1) * alphas / n_samples
    mtl = MultiTaskLassoCV(fit_intercept=False, normalize=False, cv=cv,
                           alphas=alphas, tol=1e-6).fit(X, y)

    # assert cv path errors
    mtl_path = mtl.mse_path_.mean(axis=1)
    dirty_path = (-cv_scores.T).mean(axis=1)
    assert_allclose(mtl_path, dirty_path, rtol=1e-2)

    best_alpha_ = (cv - 1) / cv * mtl.alpha_
    mtl2 = MultiTaskLasso(alpha=best_alpha_,
                          fit_intercept=False, normalize=False,
                          tol=1e-10).fit(X, y)
    # assert best fit
    assert_allclose(model.coefs_, mtl2.coef_.T, rtol=1e-4)


def test_dirty_best_score():
    n_features = 50
    x, y, coefs_true = build_dataset(n_samples=20, n_features=n_features,
                                     n_targets=3, positive=False)
    y += 0.6 * np.std(y) * np.random.RandomState(42).randn(*y.shape)
    n_samples, n_tasks = y.shape
    X = np.array(n_tasks * [x])
    y = y.T
    xty = np.array([x.T.dot(yi) for x, yi in zip(X, y)])
    alphamax = np.linalg.norm(xty, axis=0).max()
    betamax = abs(xty).max()
    alpha = 0.1 * alphamax
    beta = 0.05 * betamax
    model = Dirty(alpha=alpha, beta=beta, positive=False, tol=1e-4)
    epsilon = 10. / n_features
    M = utils.groundmetric(n_features, normed=True)
    ot_params = {"M": M, "epsilon": epsilon, "log": True,
                 "gamma": 0., "tol": 1.}
    best_scores, scores, best_coefs, best_params, _, _ = \
        best_score_dirty(model, X, y, coefs_true, cv_size=5, **ot_params)
    for k, v in scores.items():
        assert best_coefs[k].shape == coefs_true.shape
        assert sorted(list(best_params[k].keys())) == ['alpha', 'beta']
