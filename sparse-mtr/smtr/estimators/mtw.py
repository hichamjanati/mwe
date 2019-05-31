import numpy as np
from .. import utils
from .solvers import solver_mtw
from ..utils import inspector_mtw
from ..base import BaseEstimator
try:
    import cupy as cp
    get_module = cp.get_array_module
except ImportError:
    def get_module(x):
        return np


class MTW(BaseEstimator):
    """A class for MultiTask Regression with Wasserstein penalization.

    Attributes
    ----------
    n_features : dimensionality of regression coefficients
    n_samples : number of samples per task
    n_tasks : length of the lists `X`and `Y`.

    """
    def __init__(self, M, alpha=1., beta=0., epsilon=None, gamma=None,
                 sigma0=0., callback=False, stable=True, maxiter_ot=1000,
                 tol_ot=1e-5, tol=1e-5, maxiter=4000, warmstart=False,
                 positive=False, n_jobs=1, cython=True, tol_cd=1e-4,
                 gpu=True, reweighting_steps=1, reweighting_tol=1e-2,
                 **kwargs):
        """Constructs instance of MTW.

        Parameters
        ----------
        M: array, shape (n_features, n_features)
            Ground metric matrix defining the Wasserstein distance.
        alpha: float.
            Optimal transport regularization parameter.
        beta : float.
            l1 norm regularization parameter.
        gamma: float.
            Kullback-Leibler marginal penalty regularization parameter.
        epsilon: float.
            Wasserstein regularization parameter > 0.
        stable: boolean.
            if True, use log-domain Sinhorn stabilization.

        """
        super().__init__(callback=callback, **kwargs)
        self.M = M
        self.xp = get_module(M)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.stable = stable
        self.maxiter_ot = maxiter_ot
        self.tol_ot = tol_ot
        self.tol = tol
        self._inspector = inspector_mtw
        self._solver = solver_mtw
        self.maxiter = maxiter
        self.warmstart = warmstart
        self.positive = positive
        self.n_jobs = n_jobs
        self.cython = cython
        self.tol_cd = tol_cd
        self.sigma0 = sigma0
        self.gpu = gpu
        self.reweighting_tol = reweighting_tol
        self.reweighting_steps = reweighting_steps
        self.t_ot = 0.
        self.t_cd = 0.

    def fit(self, X, Y, coefs01=None, coefs02=None, marginals1=None,
            marginals2=None, **kwargs):
        """Launch MTW solver.

        Parameters
        ----------
        X: list of numpy arrays (n_samples, n_features).
            Each object being the samples data of an independent
            regression task.
        Y: list of numpy arrays (n_samples,).
            Each object being the target vector of the correspondent
            regression task in X. Must have the same length as `X`.
        method: string. optional (default 'LBFGS'
        **kwargs: supplementary parameters passed to the solver. See
            `multitask.mtw.solvers.mtw_solver`)

        Returns
        -------
        instance of self.

        """

        X, Y = self._pre_fit(X, Y)
        R = None
        b1, b2 = None, None
        sigmas = np.ones(len(X))
        if self.stable:
            self.reset()
        if self.warmstart and hasattr(self, "coefs1_"):
            coefs01 = self.coefs1_
            coefs02 = self.coefs2_
            marginals1 = self.marginals1_
            marginals1[marginals1 < 1e-5] = 0.1
            marginals2 = self.marginals2_
            marginals2[marginals2 < 1e-5] = 0.1
            marginals1[~np.isfinite(marginals1)] = 0.1
            marginals2[~np.isfinite(marginals2)] = 0.1
            sigmas = self.sigmas_
            b1 = self.b1_
            b2 = self.b2_
            # sigmas = self.sigmas_
            # if self.stable:
            #     b1 = xp.exp(self.b1_)
            #     b2 = xp.exp(self.b2_)
            #     c = xp.isfinite(b1).all()
            #     d = xp.isfinite(b2).all()
            #     if c and d:
            #         self.stable = False
            #     else:
            #         b1 = self.b1_
            #         b2 = self.b2_
            # utils.free_gpu_memory(xp)
        coefs1, coefs2, R, bar1, bar2, log, sigmas, b1, b2, m1, m2 = \
            self._solver(X, Y,  coefs01=coefs01, coefs02=coefs02, R=R, b1=b1,
                         b2=b2, marginals1=marginals1, marginals2=marginals2,
                         alpha=self.alpha, beta=self.beta, sigma0=self.sigma0,
                         M=self.M, epsilon=self.epsilon, gamma=self.gamma,
                         stable=self.stable, tol=self.tol, sigmas=sigmas,
                         callback=self.callback_f, maxiter=self.maxiter,
                         tol_ot=self.tol_ot, maxiter_ot=self.maxiter_ot,
                         positive=self.positive, n_jobs=self.n_jobs,
                         cython=self.cython, tol_cd=self.tol_cd, gpu=self.gpu,
                         reweighting_tol=self.reweighting_tol,
                         reweighting_steps=self.reweighting_steps)
        self.coefs1_ = coefs1.copy()
        self.b1_ = b1
        self.marginals1_ = m1
        self.b2_ = b2
        self.marginals2_ = m2
        self.coefs2_ = coefs2.copy()
        self.coefs_ = coefs1 - coefs2
        self.sigmas_ = sigmas
        self.residuals_ = R
        self.barycenter1_ = bar1
        self.barycenter2_ = bar2
        self.barycenter_ = bar1 - bar2

        self.log_ = log
        self.stable = log['stable']
        self.t_ot += log["t_ot"]
        self.t_cd += log["t_cd"]

        self._post_fit()

        return self

    def get_params_grid(self, X, Y, cv_size=5, eps=0.01, alpha_range=(1., 50.),
                        alphas=None, betas=None):
        X, Y = self._pre_fit(X, Y)

        xty = np.array([x.T.dot(y) for (x, y) in zip(X, Y)])
        if not self.positive:
            xty = abs(xty)
        betamax = xty.max(axis=1).min() / Y.shape[-1]
        if self.sigma0:
            sigma00 = np.linalg.norm(Y, axis=1).min()
            sigma00 /= (Y.shape[-1] ** 0.5)
            betamax /= sigma00
        self.betamax = betamax
        alphamax, alphamin = alpha_range
        params_grid = []
        if betas is None:
            betas = np.logspace(np.log10(eps), 0., 2 * cv_size)
        else:
            betas = np.array(betas)
        betas *= betamax
        if alphas is None:
            alphas = np.r_[0., np.logspace(np.log10(alphamin),
                                           np.log10(alphamax),
                                           cv_size // 2)]
        else:
            alphas = np.array(alphas)
        alphas /= Y.shape[-1]

        params_grid_print = []
        for i, this_beta in enumerate(betas[::-1]):
            for this_alpha in alphas[::(- 1) ** i]:
                params_grid += [(this_alpha, this_beta)]
                b = this_beta / betamax
                a = this_alpha * Y.shape[-1]
                params_grid_print += [(a, b)]
        self.params_grid_print = params_grid_print

        return [{"alpha": a, "beta": b} for a, b in params_grid]

    def reset(self):
        if hasattr(self, 'coefs_'):
            del self.coefs_
        if hasattr(self, 'coefs1_'):
            del self.coefs1_
        if hasattr(self, 'coefs2_'):
            del self.coefs2_
        if hasattr(self, 'residuals_'):
            del self.residuals_
        if hasattr(self, 'b_'):
            del self.b_
        if hasattr(self, 'marginals_'):
            del self.marginals_
        if hasattr(self, 'barycenter2_'):
            del self.barycenter2_
        if hasattr(self, 'barycenter1_'):
            del self.barycenter1_
        if hasattr(self, 'barycenter_'):
            del self.barycenter_
        if hasattr(self, 'log_'):
            del self.log_
            self.stable = False
        self.t_ot = 0.
        self.t_cd = 0.


if __name__ == "__main__":
    from utils import groundmetric
    from time import time
    test_warmstart = False
    plot = True
    # Estimator params
    alpha = 50.
    beta_fr = 0.2

    seed = 1729
    width, n_tasks = 20, 3
    nnz = 2
    overlap = 0.
    denoising = False
    binary = False
    corr = 0.99
    toe = utils.toeplitz_2d(width, corr)

    # Gaussian Noise
    snr = 1.25
    # Deduce supplementary params
    n_features = width ** 2
    n_samples = n_features // 5

    # ot params
    threshold = 1e-2
    epsilon = 1. / n_features
    stable = False
    gamma = 10
    Mbig = utils.groundmetric2d(n_features, normed=False)
    m = np.median(Mbig)
    M = groundmetric(width, normed=False)
    M /= m
    # M = Mbig / m

    # Generate Coefs
    coefs = utils.generate_dirac_images(width, n_tasks, nnz=nnz,
                                        seed=seed, overlap=overlap,
                                        binary=binary)
    coefs_flat = coefs.reshape(-1, n_tasks)
    # # Generate X, Y data
    std = utils.get_std(n_samples, n_tasks, width, nnz, snr=snr, corr=corr,
                        seed=0, denoising=denoising, scaled=True,
                        binary=binary)
    X, Y = utils.gaussian_design(n_samples, coefs_flat,
                                 corr=corr,
                                 sigma=std,
                                 denoising=denoising,
                                 scaled=True,
                                 seed=seed)
    # validation set
    Xv, Yv = utils.gaussian_design(n_samples, coefs_flat,
                                   corr=corr,
                                   sigma=std,
                                   seed=seed + 1,
                                   denoising=denoising,
                                   scaled=True)
    betamax = np.array([x.T.dot(y) for x, y in zip(X, Y)]).max()
    beta = beta_fr * betamax

    callback_options = {'callback': True,
                        'x_real': coefs.reshape(- 1, n_tasks),
                        'verbose': True, 'rate': 20, 'prc_only': False}

    mtw_model = MTW(M=M, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma,
                    stable=stable, tol_ot=1e-6, tol=1e-5, warmstart=False,
                    maxiter_ot=1000, maxiter=20000,
                    **callback_options)
    t = time()
    mtw_model.fit(X, Y)
    log = mtw_model.log_
    loss = log['loss']
    dloss = np.diff(loss)
    t = time() - t
    print('First fit TIME = %.4f' % t)
    # Now change hyperparameter
    mtw_model.alpha = 5.
    if test_warmstart:
        # Fit using warmstart
        t = time()
        mtw_model.warmstart = True
        mtw_model.fit(X, Y)
        coefs_w = mtw_model.coefs_.reshape(width, width, -1)
        log_w = mtw_model.log_
        loss_w = log_w['loss']
        bar_w = mtw_model.barycenter_.reshape(width, width)[:, :, None]
        dloss_w = np.diff(loss_w)
        t = time() - t
        print('warmstart fit TIME = %.4f' % t)

        # Fit without using warmstart
        t = time()
        mtw_model.warmstart = False
        mtw_model.fit(X, Y)
        coefs_no_w = mtw_model.coefs_.reshape(width, width, -1)
        log_no_w = mtw_model.log_
        bar_no_w = mtw_model.barycenter_.reshape(width, width)[:, :, None]
        loss_no_w = log_no_w['loss']
        dloss_no_w = np.diff(loss_no_w)
        t = time() - t
        print('no warmstart fit TIME = %.4f' % t)

        print(abs(coefs_w - coefs_no_w).max())
    else:
        coefs_w = mtw_model.coefs_.reshape(width, width, -1)
        coefs_no_w = coefs_w
        bar_w = mtw_model.barycenter_.reshape(width, width)[:, :, None]
        bar_no_w = bar_w
    if plot:
        from matplotlib import pyplot as plt
        from matplotlib import cm

        colors = ["indianred", "cornflowerblue", "forestgreen", "purple",
                  "orange"]
        radiuses = [20, 70, 160, 270, 400, 600, 850]
        title = "%d Tasks |  %d-sparse | %.2f %% overlap" % (n_tasks, nnz,
                                                             overlap)

        f, axes = plt.subplots(2, 3, figsize=(9, 6))
        contours = [[coefs, coefs_w, bar_w], [coefs, coefs_no_w, bar_no_w]]
        titles = ["Warmstart", "No Warmstart"]
        fig_titles = ["True", "Recovered / Support", "Recovered / contour"]
        allmaps = 2 * [colors]
        allmaps.append(["grey"])
        cmaps = [cm.Reds, cm.Blues, cm.Greens, cm.Oranges, cm.Greys,
                 cm.Purples]
        for ax_row, t, to_plots in zip(axes, titles, contours):
            for ax, data, cmap, t in zip(ax_row.ravel(), to_plots, allmaps,
                                         fig_titles):
                data_ = data.copy()
                data_[data_ < threshold * data_.max()] = 0
                utils.scatter_coefs(data_, ax, colors, radiuses=radiuses)

        plt.tight_layout()
        # plt.suptitle('MTW - ' + title)
        plt.show()
