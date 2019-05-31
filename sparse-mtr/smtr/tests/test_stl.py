import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from sklearn.linear_model import LassoCV, Lasso
from smtr import utils
from smtr.estimators import STL
from smtr.model_selection import cv_score_stl as cross_val_score
from smtr.model_selection import best_score_stl
from smtr.utils import build_dataset


def test_stl_all_zero():
    X, y, _ = build_dataset(n_samples=50, n_features=200, n_targets=3)
    y += 0.6 * np.std(y) * np.random.RandomState(42).randn(*y.shape)
    n_samples, n_tasks = y.shape
    YY = y.T
    XX = np.array([X] * n_tasks)
    alphamax = np.array([X.T.dot(this_y) for this_y in YY]).max(axis=1)
    alpha = 1. * alphamax
    model = STL(alpha=alpha, positive=False)
    model.fit(XX, YY)

    assert_array_equal(model.coefs_, 0.)


def test_stl_vs_sklearn():
    cv = 3
    n_features = 100

    for positive in [True, False]:
        X, y, coefs_true = build_dataset(n_samples=cv * 10,
                                         n_features=n_features,
                                         n_targets=1, positive=positive)
        y += 0.6 * np.std(y) * np.random.RandomState(42).randn(*y.shape)
        n_samples, n_tasks = y.shape
        assert n_tasks == 1
        YY = y.T
        XX = X[None, :, :]
        scaling = XX.std(axis=1)
        XX /= scaling[:, None, :]
        model = STL(positive=positive, tol=1e-5)
        if positive:
            epsilon = 0. / n_features
            M = utils.groundmetric(n_features, normed=True)
            ot_params = {"M": M, "epsilon": epsilon, "maxiter": 2, "log": True,
                         "gamma": 0., "tol": 1}
            best_scores, scores, best_coefs, best_params, _, _ = \
                best_score_stl(model, XX, YY, coefs_true, cv_size=3,
                               scaling_vector=scaling.T,
                               **ot_params)
            for k, v in scores.items():
                assert best_coefs[k].shape == coefs_true.shape

        cv_scores, model, params_grid = cross_val_score(model, XX, YY, cv=cv,
                                                        cv_size=10)

        cv_scores = np.mean(cv_scores, axis=-1)  # average across tasks

        alphas = np.array([p["alpha"] for p in params_grid])
        lasso = LassoCV(fit_intercept=False, normalize=False, cv=cv,
                        alphas=alphas, tol=1e-5, positive=positive)
        lasso.fit(X, y.flatten())

        # assert cv path errors
        lasso_path = lasso.mse_path_.mean(axis=1)
        stl_path = (-cv_scores).mean(axis=0)
        assert_allclose(lasso_path, stl_path, rtol=1e-3, atol=1e-3)

        assert lasso.alpha_, model.alpha_[0]

        best_alpha_ = lasso.alpha_
        lasso = Lasso(alpha=best_alpha_,
                      fit_intercept=False, normalize=False,
                      tol=1e-5, positive=positive).fit(X, y)
        # assert best fit
        assert_allclose(model.coefs_[:, 0], lasso.coef_, rtol=1e-3, atol=1e-3)
