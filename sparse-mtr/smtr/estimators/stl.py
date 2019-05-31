import numpy as np
from ..base import BaseEstimator
from .solvers import solver_stl
from ..utils import inspector_lasso


class STL(BaseEstimator):
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
                 tol=1e-5, **kwargs):
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
        self._inspector = inspector_lasso
        self._solver = solver_stl
        self.tol = tol

    def fit(self, X, Y, **kwargs):
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
        X, Y = self._pre_fit(X, Y)
        self.coefs_ = self._solver(X, Y, alpha=self.alpha,
                                   positive=self.positive,
                                   callback=self.callback_f,
                                   tol=self.tol,
                                   **kwargs)
        self._post_fit()

        return self

    def get_params_grid(self, X, Y, cv_size=50, eps=0.1):
        X, Y = self._pre_fit(X, Y)
        n_samples = X.shape[1]
        xty = np.array([x.T.dot(y) for (x, y) in zip(X, Y)])
        if not self.positive:
            xty = abs(xty)
        alphamax = xty.max(axis=1) / n_samples

        scale = np.logspace(np.log10(eps), 0., cv_size)
        params_grid = alphamax[:, None] * scale[None, :]
        if len(X) > 1:
            return [{"alpha": a} for a in params_grid.T[::-1]]
        else:
            return [{"alpha": float(a)} for a in params_grid.T[::-1]]


if __name__ == '__main__':
    pass
    # from utils import build_dataset
    # import matplotlib.pyplot as plt
    #
    # X, y, coefs = build_dataset(n_samples=50, n_features=200, n_targets=3,
    #                             positive=True)
    # y += 0.6 * np.std(y) * np.random.RandomState(42).randn(*y.shape)
    #
    # cv = 4
    # n_samples, n_tasks = y.shape
    # YY = y.T
    # XX = np.array([X] * n_tasks)
    # model = STL(positive=True)
    # cv_scores, model, params_grid = cv_score_stl(model, XX, YY, eps=1e-2,
    #                                              cv=cv, cv_size=30)
    # b_scores, b_coefs, b_alpha = best_score_stl(model, XX, YY, coefs,
    #                                             eps=1e-2, cv_size=30)
    # labels = ["Task " + str(i) for i in range(n_tasks)]
    #
    # f = plt.figure()
    # plt.subplot(121)
    # for params, scores, label in zip(params_grid, cv_scores, labels):
    #     plt.plot(np.log(params['alpha']), scores.mean(axis=1), label=label)
    # plt.title("Cross-validation")
    # plt.ylabel("CV error")
    # plt.xlabel(r"$\alpha$")
    #
    # plt.subplot(122)
    # for params, scores, label in zip(params_grid, b_scores, labels):
    #     plt.plot(np.log(params['alpha']), scores, label=label)
    # plt.ylabel("MSE")
    # plt.xlabel(r"$\alpha$")
    # plt.title("Best knowing the truth")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
