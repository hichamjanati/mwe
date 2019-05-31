import numpy as np

from sklearn.model_selection import check_cv
from time import time


def _cv_score(model, X, Y, cv, params_grid, mtgl_only=False,
              separate=False, random_rotations=False, warmstart=True):
    model.warmstart = warmstart
    model.reset()
    n_tasks = len(X)

    if not random_rotations:
        cv = check_cv(cv=cv)

        scores = np.empty((cv.n_splits, len(params_grid), n_tasks))

        for i_fold, (train, test) in enumerate(cv.split(X[0], Y[0])):
            print(" -> %d / %d" % (i_fold + 1, cv.n_splits))
            for i_param, params in enumerate(params_grid):
                model.set_params(params)
                model.fit(X[:, train], Y[:, train])
                scores[i_fold, i_param, :] = model.score(X[:, test],
                                                         Y[:, test])

            model.reset()

    else:
        pass
        # n_samples = Y.shape[-1]
        # rotations_train = special_ortho_group.rvs(n_samples, size=cv,
        #                                           random_state=42)
        # rotations_test = special_ortho_group.rvs(n_samples, size=cv,
        #                                     random_state=42)
        # for i_fold, rotation in enumerate(rotations):
        #     print(" -> rotation %d / %d" % (i_fold + 1, cv.n_splits))
        #     Xtrain, ytrain
        #     for i_param, params in enumerate(params_grid):
        #         model.set_params(params)
        #         model.fit(X[:, train], Y[:, train])
        #         scores[i_fold, i_param, :] = model.score(X[:, test],
        #                                                  Y[:, test])
        #
        #     model.reset()

    mean_scores = np.mean(scores, axis=0)  # n_params x n_tasks

    if separate:
        best_idx = np.argmax(mean_scores, axis=0)  # n_tasks
        if n_tasks > 1:
            alpha = [params_grid[k]['alpha'][i_task]
                     for i_task, k in enumerate(best_idx)]
        else:
            alpha = params_grid[best_idx[0]]['alpha']
        best_params = dict(alpha=np.array(alpha))
    else:
        tmp = np.mean(mean_scores, axis=1)  # average across tasks
        assert len(tmp) == len(params_grid)
        best_params = params_grid[np.argmax(tmp)]

    model.set_params(best_params)
    model.fit(X, Y)
    return scores, model, params_grid


def cv_score_dirty(model, X, Y, cv=3, cv_size=20, mtgl_only=False, eps=5e-2,
                   do_mtgl=True, warmstart=True):
    params_grid = model.get_params_grid(X, Y, cv_size=cv_size, eps=eps,
                                        mtgl_only=mtgl_only, do_mtgl=do_mtgl,
                                        )
    return _cv_score(model, X, Y, cv, params_grid, mtgl_only=mtgl_only,
                     warmstart=warmstart)


def cv_score_mll(model, X, Y, cv=3, cv_size=20, eps=5e-2, warmstart=True):
    params_grid = model.get_params_grid(X, Y, cv_size=cv_size, eps=eps)
    return _cv_score(model, X, Y, cv, params_grid, warmstart=warmstart)


def cv_score_mtw(model, X, Y, cv=3, cv_size=20, eps=5e-2,
                 alpha_range=(1., 50.), alphas=None, betas=None,
                 warmstart=True):
    params_grid = model.get_params_grid(X, Y, cv_size=cv_size, eps=eps,
                                        alpha_range=alpha_range, alphas=alphas,
                                        betas=None)
    return _cv_score(model, X, Y, cv, params_grid, warmstart=warmstart)


def cv_score_stl(model, X, Y, cv=3, cv_size=20, eps=5e-2, warmstart=True):
    params_grid = model.get_params_grid(X, Y, cv_size=cv_size, eps=eps)
    return _cv_score(model, X, Y, cv, params_grid, separate=True,
                     warmstart=warmstart)


def _best_score(model, X, Y, coefs_true, params_grid, scaling_vector=1.,
                warmstart=True, **kwargs):
    t0 = 0
    model.warmstart = warmstart
    model.reset()
    scores = dict(auc=[], mse=[], ot=[], aucabs=[], otabs=[])
    best_scores = dict(auc=-np.inf, mse=-np.inf, ot=-np.inf,
                       aucabs=-np.inf, otabs=-np.inf)
    best_coefs = dict(auc=None, mse=None, ot=None, aucabs=None, otabs=None)
    best_params = dict(auc=None, mse=None, ot=None, aucabs=None, otabs=None)
    barycenters = dict(auc=-np.inf, mse=-np.inf, ot=-np.inf,
                       aucabs=-np.inf, otabs=-np.inf)
    all_coefs = dict()
    try:
        params_print = model.params_grid_print
    except AttributeError:
        params_print = params_grid
    for i_param, (params, p_print) in enumerate(zip(params_grid,
                                                    params_print)):
        print(" -> %d / %d" % (i_param, len(params_grid)), p_print)
        t = time()
        model.set_params(params)
        model.fit(X, Y)
        model.coefs_ /= scaling_vector

        scores_dict = model.score_coefs(coefs_true,
                                        **kwargs)
        all_coefs[tuple(params.items())] = model.coefs_
        model.coefs_ *= scaling_vector
        for k, v in scores_dict.items():
            scores[k].append(v)
            if v > best_scores[k]:
                if hasattr(model, 'barycenter_'):
                    barycenters[k] = model.barycenter_
                best_coefs[k] = model.coefs_
                best_scores[k] = v
        t = time() - t
        t0 += t
        print(">>>>>> out of -> %d / %d in %.2f" %
              (i_param, len(params_grid), t))

    for k, v in scores.items():
        assert np.nanmax(v) == best_scores[k], \
            "key: %s, value: %.3f, best: %.3f" % (k, np.nanmax(v),
                                                  best_scores[k])
        best_params[k] = params_grid[np.argmax(v)]

    print(" >>> OUT OF WORKER - time = %.2f" % t0)
    return (best_scores, scores, best_coefs, best_params, barycenters,
            all_coefs)


def best_score_dirty(model, X, Y, coefs_true, cv_size=5, eps=1e-2,
                     mtgl_only=False, do_mtgl=True, **kwargs):
    params_grid = model.get_params_grid(X, Y, cv_size=cv_size, eps=eps,
                                        mtgl_only=mtgl_only,
                                        do_mtgl=do_mtgl)
    return _best_score(model, X, Y, coefs_true, params_grid, **kwargs)


def best_score_mll(model, X, Y, coefs_true, cv_size=5, eps=1e-2, **kwargs):
    params_grid = model.get_params_grid(X, Y, cv_size=cv_size, eps=eps)
    return _best_score(model, X, Y, coefs_true, params_grid, **kwargs)


def best_score_mtw(model, X, Y, coefs_true, cv_size=5, eps=1e-2, alphas=None,
                   alpha_range=(1., 50.), betas=None, **kwargs):
    params_grid = model.get_params_grid(X, Y, cv_size=cv_size, eps=eps,
                                        alpha_range=alpha_range, alphas=alphas,
                                        betas=betas)
    return _best_score(model, X, Y, coefs_true, params_grid, **kwargs)


def best_score_stl(model, X, Y, coefs_true, eps=1e-2, cv_size=20,
                   scaling_vector=None, **kwargs):
    if scaling_vector is None:
        scaling_vector = np.ones_like(coefs_true)
    scores = dict(auc=[], mse=[], ot=[], aucabs=[], otabs=[])
    best_scores = dict(auc=-np.inf, mse=-np.inf, ot=-np.inf,
                       aucabs=-np.inf, otabs=-np.inf)
    best_coefs = dict(auc=[], mse=[], ot=[], aucabs=[], otabs=[])
    best_params = dict(auc=[], mse=[], ot=[], aucabs=[], otabs=[])
    barycenters = dict(auc=-np.inf, mse=-np.inf, ot=-np.inf,
                       aucabs=-np.inf, otabs=-np.inf)
    all_coefs = dict()
    for i, (xx, yy, cc, sv) in enumerate(zip(X, Y, coefs_true.T,
                                             scaling_vector.T)):
        x, y, coef = xx[None, :, :], yy[None, :], cc[:, None]
        scaling = sv[:, None]
        params_grid = model.get_params_grid(x, y, eps=eps, cv_size=cv_size)
        bs, sc, btc, bp, _, ac = _best_score(model, x, y, coef, params_grid,
                                             scaling_vector=scaling, **kwargs)
        all_coefs[i] = ac
        for k, v in scores.items():
            v.append(sc[k])
            best_coefs[k].append(btc[k].flatten())
            best_params[k].append(bp[k])

    for d in [scores, best_coefs, best_params]:
        for k, v in d.items():
            d[k] = np.stack(d[k], axis=-1)

    for k, v in scores.items():
        best_scores[k] = scores[k].max(axis=0).mean()

    return best_scores, scores, best_coefs, best_params, barycenters, all_coefs


def plot_grid_params(params, scores, title, n_tasks=3):
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.DataFrame(params)
    df['score'] = scores

    # df.plot('alpha', 'beta', marker='o', legend=False)
    data = df.values
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.xlim([0, df['alpha'].max()])
    ratio = 1 / (n_tasks ** 0.5)
    xx = df.alpha.values
    ll = df.beta.values

    yy = xx * ratio
    plt.plot(xx, yy, color="indianred", lw=3,
             label=r"slope = $\frac{1}{\sqrt{T}}$")
    plt.plot(ll, ll, lw=3, color="orange", label=r"slope = $1$")
    plt.plot(xx, len(xx) * [data[:, 1].max()], color="forestgreen",
             ls="-", label="Group Lasso line")
    plt.legend()
    plt.xlabel(r"$\mu$")
    plt.ylabel(r"$\lambda$")
    plt.savefig("output/local/fig/grid.pdf")
    plt.show()
    plt.title(title)

    dd = df.pivot_table(index=['alpha'], columns=['beta'],
                        values=['score'])
    plt.figure()
    plt.plot(dd.index, dd.values, label=dd.columns)
    plt.ylabel("Score")
    plt.xlabel("alpha")
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
    # import utils
    # from mtw import MTW
    # from dirty import Dirty
    # from time import time
    #
    # seed = 653
    # width, n_tasks = 8, 2
    # nnz = 1
    # overlap = 1.
    # denoising = False
    # binary = False
    # corr = 0.9
    #
    # # Gaussian Noise
    # snr = 1.25
    # # Deduce supplementary params
    # n_features = width ** 2
    # n_samples = n_features // 2
    #
    # # OT params
    # Mbig = utils.groundmetric2d(n_features, normed=False)
    # m = np.median(Mbig)
    # M = utils.groundmetric(width, normed=False)
    # M /= m
    # # M = Mbig / m
    #
    # # Generate Coefs
    # coefs = utils.generate_dirac_images(width, n_tasks, nnz=nnz,
    #                                     seed=seed, overlap=overlap,
    #                                     binary=binary)
    # coefs_flat = coefs.reshape(-1, n_tasks)
    # # # Generate X, Y data
    # std = utils.get_std(n_samples, n_tasks, width, nnz, snr=snr, corr=corr,
    #                     seed=0, denoising=denoising, scaled=True,
    #                     binary=binary)
    # X, Y = utils.gaussian_design(n_samples, coefs_flat,
    #                              corr=corr,
    #                              sigma=std,
    #                              denoising=denoising,
    #                              scaled=True,
    #                              seed=seed)
    #
    # cv = 3
    # cv_size = 6
    # eps = 1e-2
    # alpha_range = (0.3, 20)
    #
    # epsilon = 1. / n_features
    # stable = False
    # gamma = 20.
    #
    # M = utils.groundmetric2d(n_features, normed=True)
    # mtw = MTW(M=M, epsilon=epsilon, gamma=gamma, stable=stable)
    # dirty = Dirty(positive=True)
    #
    # # scores_dirty, dirty, params_dirty = cv_score_dirty(dirty, X, Y,
    # #                                                    eps=eps,
    # #                                                    cv=cv,
    # #                                                    cv_size=cv_size)
    # # mean_cv_dirty = (-scores_dirty).mean(axis=(0, -1))
    # #
    # # coefs_scores_dirty, best_coefs_dirty, best_params_dirty = \
    # #     best_score_dirty(dirty, X, Y, coefs_flat, eps=eps, cv_size=cv_size)
    #
    # plt.close('all')
    #
    # # plot_grid_params(params_dirty, mean_cv_dirty, title='Dirty')
    # # plot_grid_params(params_dirty, coefs_scores_dirty,
    # #                  title='Dirty (best)')
    #
    # t = time()
    # # scores_mtw, mtw, params_mtw = cv_score_mtw(mtw, X, Y, eps=eps,
    # #                                            cv=cv, cv_size=cv_size,
    # #                                            alpha_range=alpha_range)
    # # mean_cv_mtw = (-scores_mtw).mean(axis=(0, -1))
    # ot_params = {"M": M, "epsilon": epsilon, "gamma": 0., "wyy0": None,
    #              "log": False}
    # coefs_scores_mtw, best_coefs_mtw, best_params_mtw = \
    #     best_score_mtw(mtw, X, Y, coefs_flat, eps=eps, cv_size=cv_size,
    #                    **ot_params)
    #
    # t = time() - t
    # print("Time with warmstart:", t)
    # plot_grid_params(params_mtw, mean_cv_mtw, title='MTW')
    # plot_grid_params(params_mtw, coefs_scores_mtw, title='MTW (best)')
