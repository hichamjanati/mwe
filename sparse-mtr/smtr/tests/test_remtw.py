from smtr import utils
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
from smtr.utils import groundmetric
from smtr.estimators import ReMTW, MTW

if 1:
#def test_reremtw():

    # Estimator params

    seed = 653
    width, n_tasks = 12, 2
    nnz = 1
    overlap = 0.
    denoising = False
    binary = False
    corr = 0.9

    # Gaussian Noise
    snr = 4
    # Deduce supplementary params
    n_features = width ** 2
    n_samples = n_features // 2

    # ot params
    epsilon = 0.05 / n_features
    stable = False
    gamma = 10.
    Mbig = utils.groundmetric2d(n_features, p=2, normed=False)
    m = np.median(Mbig)
    M = groundmetric(width, p=2, normed=False)
    M /= m
    M = Mbig / m

    # Generate Coefs
    coefs = utils.generate_dirac_images(width, n_tasks, nnz=nnz,
                                        seed=seed, overlap=overlap,
                                        binary=binary)
    coefs_flat = coefs.reshape(-1, n_tasks)
    # # Generate X, Y data
    std = 1 / snr
    X, Y = utils.gaussian_design(n_samples, coefs_flat,
                                 corr=corr,
                                 sigma=std,
                                 denoising=denoising,
                                 scaled=True,
                                 seed=seed)
    beta_fr = 0.2
    betamax = np.array([x.T.dot(y) for x, y in zip(X, Y)]).max()
    beta = beta_fr * betamax / n_samples
    alpha = 1. / n_samples
    remtw_model = ReMTW(M=M, alpha=alpha, beta=beta, epsilon=epsilon,
                        gamma=gamma, stable=stable, tol_ot=1e-6, tol=1e-6,
                        warmstart=False, maxiter_ot=10, maxiter=5000,
                        reweighting_steps=1)
    # mtw_model = MTW(M=M, alpha=alpha, beta=beta, epsilon=epsilon,
    #                 gamma=gamma, stable=stable, tol_ot=1e-6, tol=1e-6,
    #                 warmstart=False, maxiter_ot=10, maxiter=5000)
    # # first fit
    # mtw_model.fit(X, Y)

    # remtw_model.fit(X, Y)
    # theta_mtw = remtw_model.coefs_

    remtw_model.reweighting_steps = 5
    remtw_model.fit(X, Y)
    theta = remtw_model.coefs_

    assert remtw_model.log_['dloss'][-1] < 1e-5
