from smtr import utils
import numpy as np
from numpy.testing import assert_array_almost_equal
from smtr.utils import groundmetric
from smtr.estimators import MTW


def test_mtw_warmstart():

    # Estimator params
    alpha = 10.
    beta_fr = 0.3

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
    epsilon = 1. / n_features
    stable = False
    gamma = 10.
    Mbig = utils.groundmetric2d(n_features, p=2, normed=False)
    m = np.median(Mbig)
    M = groundmetric(width, p=2, normed=False)
    M /= m
    # M = Mbig / m

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

    betamax = np.array([x.T.dot(y) for x, y in zip(X, Y)]).max()
    beta = beta_fr * betamax

    mtw_model = MTW(M=M, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma,
                    stable=stable, tol_ot=1e-6, tol=1e-6, warmstart=False,
                    maxiter_ot=50, maxiter=5000)
    # first fit
    mtw_model.fit(X, Y)
    assert mtw_model.log_['dloss'][-1] < 1e-5

    # small change of hyperparamters
    # mtw_model.beta += 0.01
    mtw_model.alpha += 1

    mtw_model.warmstart = True
    mtw_model.fit(X, Y)
    assert mtw_model.log_['dloss'][-1] < 1e-5
    print(np.min(mtw_model.log_['loss']))
    coefs_warmstart = mtw_model.coefs_

    mtw_model.warmstart = False
    mtw_model.fit(X, Y)
    assert mtw_model.log_['dloss'][-1] < 1e-5
    print(np.min(mtw_model.log_['loss']))
    coefs_no_warmstart = mtw_model.coefs_

    assert_array_almost_equal(coefs_warmstart, coefs_no_warmstart, decimal=3)


def test_reweighting():
    # Estimator params
    alpha = 10.
    beta_fr = 0.1

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
    epsilon = 0.1 / n_features
    stable = False
    gamma = 10.
    Mbig = utils.groundmetric2d(n_features, p=2, normed=False)
    m = np.median(Mbig)
    M = groundmetric(width, p=2, normed=False)
    M /= m
    # M = Mbig / m

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

    betamax = np.array([abs(x.T.dot(y)) for x, y in zip(X, Y)]).max()
    beta = beta_fr * betamax / n_samples

    reweighting_steps = 10
    mtw_model = MTW(M=M, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma,
                    stable=stable, tol_ot=1e-5, tol=1e-5, warmstart=False,
                    maxiter_ot=50, maxiter=1000, reweighting_loss=1e-4,
                    reweighting_steps=reweighting_steps)
    # fit
    mtw_model.fit(X, Y)
    # coefs_mtw = mtw_model.coefs_.reshape(width, width, n_tasks)
    #
    # from matplotlib import pyplot as plt
    # from matplotlib import cm
    #
    # colors = ["indianred", "cornflowerblue", "forestgreen", "purple",
    #           "orange"]
    # bar = mtw_model.barycenter_.reshape(width, width, 1)
    # f, axes = plt.subplots(1, 3, figsize=(9, 3))
    # contours = [coefs, coefs_mtw, bar]
    # titles = ["Warmstart", "No Warmstart"]
    # fig_titles = ["True", "Recovered / Support", "Recovered / contour"]
    # cmaps = [cm.Reds, cm.Blues, cm.Greens, cm.Oranges, cm.Greys,
    #          cm.Purples]
    # threshold = 1e-3
    # radiuses = [20, 70, 160, 270, 400, 600, 850]
    # for ax, data, cmap in zip(axes.ravel(), contours, cmaps):
    #     data_ = abs(data).copy()
    #     data_[data_ < threshold * abs(data_).max()] = 0.
    #     utils.scatter_coefs(data_, ax, colors, radiuses=radiuses)

    assert np.diff(mtw_model.log_['reweighting_loss']).max() < 1e-3
