import numpy as np
from ..base import BaseEstimator
from .solvers import solver_mll
from ..utils import inspector_mll


class MLL(BaseEstimator):
    """Class for reweighted independent stl estimators.

    Attributes
    ----------
    n_features: int.
        number of features > 0.
    n_samples: int.
        number of samples > 0.
    n_tasks: int.
        length of the lists `X`and `Y` > 0.

    """
    def __init__(self, alpha=None, positive=True, callback=False,
                 tol=1e-4, maxiter=1000, **kwargs):
        """Creates instance of independent STL class.

        Parameters
        ----------
        alpha: array, shape (n_tasks,)
            Regression regularisation parameters. Weight of the L1
            penalty. Must be > 0
        positive: boolean. (optional, default True)
            If True, solve non-negative constrained problem.
        callback: boolean. (optional, default False)
            If True, sets a callback in the solver.

        """
        super().__init__(callback=callback, **kwargs)
        self.alpha = alpha
        self.positive = positive
        self._inspector = inspector_mll
        self._solver = solver_mll
        self.tol = tol
        self.maxiter = maxiter

    def fit(self, X, y, **kwargs):
        """Launch FISTA solver.

        Parameters
        ----------
        X: list of numpy arrays.
            list of design matrices of shape (n_samples, n_features).
        Y: list of numpy arrays.
            list of target arrays of shape (n_samples,).
        **kwargs: supplementary parameters passed to the solver. See
            `multitask.mtgl.solvers.mtgl_solver`)

        Returns
        -------
        instance of self.

        """
        X, Y = self._pre_fit(X, y)
        C, S = None, None
        # Non-convex: no warmstart
        # if self.warmstart and hasattr(self, "C_"):
        #     C = self.C_
        #     S = self.S_
        self.C_, self.S_, self.loss_ = self._solver(X, y, alpha=self.alpha,
                                                    C=C,
                                                    S=S,
                                                    positive=self.positive,
                                                    callback=self.callback_f,
                                                    tol=self.tol,
                                                    maxiter=self.maxiter,
                                                    **kwargs)
        self.coefs_ = self.C_[:, None] * self.S_
        self._post_fit()

        return self

    def reset(self):
        if hasattr(self, 'C_'):
            del self.C_
        if hasattr(self, 'S_'):
            del self.S_
        if hasattr(self, 'coefs_'):
            del self.coefs_
        if hasattr(self, 'loss_'):
            del self.loss_

    def get_params_grid(self, X, y, cv_size=20, eps=0.02):
        X, Y = self._pre_fit(X, y)
        n_samples = X.shape[1]
        xty = np.array([x.T.dot(yy) for (x, yy) in zip(X, Y)])
        alphamax = abs(xty).max() / n_samples
        scale = np.logspace(np.log10(eps), -0.2, cv_size)
        params_grid = alphamax * scale

        d = []
        for a in params_grid:
            d.append({"alpha": a})
        return d


if __name__ == '__main__':
    pass
