import numpy as np
from .solvers import solver_dirty, solver_mtgl

from ..base import BaseEstimator

try:
    import cupy as cp
    get_module = cp.get_array_module
except ImportError:
    def get_module(x):
        return np


class Dirty(BaseEstimator):
    """A class for MultiTask dirty model with L2,1 and L1 penalization.

    Attributes
    ----------
    n_features: int.
        number of features > 0.
    n_samples: int.
        number of samples > 0.
    n_tasks: int.
        length of the lists `X`and `Y` > 0.

    """

    def __init__(self, alpha=10., beta=10., positive=False,
                 callback=False, warmstart=False, tol=1e-4, **kwargs):
        """Creates instance of MTGL class.

        Parameters
        ----------
        alpha: float. (optional, default 1)
            Regression regularisation parameter. Weight of the L2,1
            penalty. Must be > 0
        positive: boolean. (optional, default True)
            If True, solve non-negative constrained problem.
        callback: boolean. (optional, default False)
            If True, sets a callback in the solver.

        """
        super().__init__(callback=False, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.positive = positive
        self.tol = tol
        self.warmstart = warmstart

    def _check_params(self):
        """prefit method, set proximal constraint."""
        pass

    def _get_solver(self):
        """Set solver to use: dirty or grouplasso depending on tuning
        parameters."""

        if self.alpha <= self.beta:
            return solver_mtgl
        else:
            return solver_dirty

    def fit(self, X, Y, coefs0=None, **kwargs):
        """Launch BCD gfb solver.

        Parameters
        ----------
        X: list of numpy arrays.
            list of design matrices of shape (n_samples, n_features).
        Y: list of numpy arrays.
            list of target çarrays of shape (n_samples,).
        **kwargs: supplementary parameters passed to the solver. See
            `multitask.mtgl.solvers.mtgl_solver`)

        Returns
        -------
        instance of self.

        """
        X, Y = self._pre_fit(X, Y)

        R = None
        if self.warmstart and hasattr(self, "coefs_"):
            coefs0 = self.coefs_tuple_0
            R = self.residuals_

        solver = self._get_solver()
        coefs1, coefs2, R, loss = solver(X, Y, coefs0=coefs0, R=R,
                                         alpha=self.alpha, beta=self.beta,
                                         positive=self.positive, tol=self.tol,
                                         **kwargs)
        self.coefs_tuple_ = coefs1, coefs2
        self.coefs_tuple_0 = coefs1.copy(), coefs2.copy()
        self.coefs_ = coefs1 + coefs2
        self.log_ = dict(loss=loss)
        self.residuals_ = R
        self._post_fit()
        return self

    def reset(self):
        if hasattr(self, 'coefs_'):
            del self.coefs_
            del self.residuals_
        if hasattr(self, 'scaling_'):
            del self.scaling_

    def get_params_grid(self, X, Y, cv_size, mtgl_only=False, eps=0.05,
                        do_mtgl=True):
        X, Y = self._pre_fit(X, Y)

        xty = np.array([x.T.dot(y) for (x, y) in zip(X, Y)])
        betamax = abs(xty).max()
        alphamax = np.linalg.norm(xty, axis=0).max()

        n_tasks = len(X)
        ratio = 1 / n_tasks ** 0.5
        params_grid = []
        cv_size_mtgl = cv_size
        mtgl_points = np.logspace(np.log10(eps), 0., cv_size_mtgl)
        mtgl_points *= betamax

        if do_mtgl:
            for this_alpha in mtgl_points[::-1]:
                params_grid += [(this_alpha, alphamax)]

        if not mtgl_only:
            hsize = alphamax - betamax
            vsize = betamax - alphamax * ratio
            hv_ratio = max(hsize / vsize, 0.)
            n_points_h = int(cv_size * hv_ratio / (1 + hv_ratio))
            n_points_v = cv_size - n_points_h
            if n_points_v < 1:
                n_points_h = 1
                n_points_v = 1
                cv_size = 2

            points = np.empty((cv_size, 2))
            points[:n_points_h, 1] = betamax
            points[:n_points_h, 0] = np.linspace(betamax,
                                                 alphamax,
                                                 n_points_h)

            points[n_points_h:, 0] = alphamax
            ratio = 1. / n_tasks ** 0.5
            points[n_points_h:, 1] = np.linspace(alphamax * ratio,
                                                 betamax,
                                                 n_points_v)

            logscale = np.logspace(0., -np.log10(eps), cv_size)

            for x, y in points:
                for s in logscale:
                    params_grid += [(x / s, y / s)]

        return [{"alpha": a, "beta": b} for a, b in params_grid]


if __name__ == '__main__':
    from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso
    from utils import build_dataset
    from model_selection import best_score_dirty, cv_score_dirty

    cv = 5
    X, y, coefs = build_dataset(n_samples=cv * 5, n_features=200, n_targets=3)
    y += 0.6 * np.std(y) * np.random.RandomState(42).randn(*y.shape)

    n_samples, n_tasks = y.shape
    YY = y.T
    XX = np.array([X] * n_tasks)
    model = Dirty(positive=False, tol=1e-8)

    cv_scores, model, params_grid = cv_score_dirty(model, XX, YY, cv=cv,
                                                   cv_size=5,
                                                   mtgl_only=True)

    alphas = cv / (cv - 1) * np.array(params_grid)[:, 0] / n_samples
    mtl = MultiTaskLassoCV(fit_intercept=False, normalize=False, cv=cv,
                           alphas=alphas, tol=1e-10).fit(X, y)
    best_alpha_ = (cv - 1) / cv * mtl.alpha_
    mtl2 = MultiTaskLasso(alpha=best_alpha_,
                          fit_intercept=False, normalize=False,
                          tol=1e-10).fit(X, y)
    print(abs(model.coefs_ - mtl2.coef_.T).max())
    b_scores, b_coefs, b_alpha, b_beta = best_score_dirty(model, XX, YY, coefs,
                                                          eps=1e-2, cv_size=5,
                                                          mtgl_only=True)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(121)
    plt.plot(np.log(alphas), (-cv_scores.T).mean(axis=1), label='dirty',
             lw=3)
    plt.plot(np.log(alphas), mtl.mse_path_.mean(axis=1), label='sklearn')
    plt.legend()
    plt.subplot(122)
    plt.plot(np.log(alphas), b_scores, label='MSE - best')
    plt.legend()
    plt.tight_layout()
    plt.show()
