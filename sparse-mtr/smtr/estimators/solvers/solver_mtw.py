"""Solvers for multitask group-stl with a simplex constraint."""
import numpy as np
import warnings
from ... import utils
import numba as nb
from joblib import Parallel, delayed
from numba import jit, float64, int64, njit, boolean
from time import time
from scipy.optimize import nnls

from .solver_cd import cython_wrapper
try:
    import cupy as cp
    get_module = cp.get_array_module
except ImportError:
    def get_module(x):
        return np


def solver_mtw(X, Y, coefs01=None, coefs02=None, R=None, b1=None, b2=None,
               marginals1=None, marginals2=None, alpha=1., beta=0.,
               sigma0=0., M=None, sigmas=None,
               epsilon=0.01, gamma=1., stable=False, maxiter=2000,
               callback=None, tol=1e-5, maxiter_ot=10000, tol_ot=1e-4,
               positive=False, n_jobs=1, cython=True, tol_cd=1e-4,
               gpu=True, reweighting_steps=1, reweighting_tol=1e-2):
    """Perform Alternating Optimization of the MTW problem.

    Parameters
    ----------
    X : numpy array (n_tasks, n_samples, n_features)
    Y : numpy array (n_tasks, n_samples)

    0 : numpy array (n_features, n_tasks)
        initial guess.
    maxiter : int > 0. optional, default 100
        maximum number of AO iterations.
    callback : callable. optional, default None.
        printing function.
    ftol : float > 0. optional, default 1e-10
        Stopping criterion threshold on relative loss decrease.
    returnlog : boolean. optional, default False
        if True, returns convergence log.

    Returns
    -------
    theta : numpy array (n_features, n_tasks)
        optimal minimizer

    if `returnlog` == True:
        theta : numpy array
            optimal minimizer
        log : dict.
            objectives, errors.

    """
    log = {'loss': [], 'dloss': [], 'log_sinkhorn1': [], 'log_sinkhorn2': [],
           'stable': stable, 't_cd': 0., 't_ot': 0., "times": [], "objcd": [],
           "fot1": [], "fot2": [], 'reweighting_loss': []}
    n_tasks, n_samples, n_features = X.shape
    if n_jobs == -1:
        n_jobs = n_tasks
    if coefs01 is None:
        coefs01 = np.ones((n_features, n_tasks))
    if coefs02 is None:
        coefs02 = np.ones((n_features, n_tasks))
    if marginals1 is None:
        marginals1 = np.ones((n_tasks, n_features)) / n_features
    if marginals2 is None:
        marginals2 = np.ones((n_tasks, n_features)) / n_features

    Xf = np.asfortranarray(X)
    Yf = np.asfortranarray(Y)
    theta1 = coefs01.copy()
    theta2 = coefs02.copy()
    theta = theta1 - theta2

    thetaold = theta.copy()
    Ls = lipschitz_numba(Xf)
    Ls[Ls == 0.] = Ls[Ls != 0].min()

    ot_img = True
    if len(M) == n_features:
        ot_img = False

    update_ot_1 = set_ot_func(stable, ot_img)
    update_ot_2 = set_ot_func(stable, ot_img)

    t_cd = []
    t_ot = 0.
    xp = utils.set_module(gpu)
    M = xp.asarray(M)
    if b1 is not None:
        b1 = xp.asarray(b1)
    if b2 is not None:
        b2 = xp.asarray(b2)

    thetabar1 = np.ones_like(coefs01).mean(axis=-1)
    thetabar2 = np.ones_like(coefs02).mean(axis=-1)
    thetabar = thetabar1 - thetabar2

    if positive:
        theta2 *= 0.
        thetabar2 *= 0.
        theta = theta1.copy()
    n_tasks = len(X)
    a = n_samples * alpha * gamma / n_tasks
    b = n_samples * beta
    if alpha == 0.:
        theta1 *= 0.
        theta2 *= 0.
    if sigma0:
        sigma0 *= np.linalg.norm(Y, axis=1).min() / (n_samples ** 0.5)
    weights = np.ones((n_features, n_tasks))
    with Parallel(n_jobs=n_jobs, backend="threading") as pll:
        for rr in range(reweighting_steps):
            outer_theta = theta.copy()
            if alpha == 0.:
                t = time()
                R = Y.copy()
                theta1 = np.asfortranarray(theta1)
                theta, R, sigmas = update_coefs(pll, X, Y, Ls, marginals1,
                                                sigmas,
                                                a, b * weights,
                                                sigma0,
                                                coefs0=theta1,
                                                R=R,
                                                tol=tol_cd, maxiter=20000,
                                                positive=positive,
                                                cython=cython)
                obj = 0.5 * (R ** 2).sum(axis=1).dot(1 / sigmas) / n_samples
                obj += beta * abs(theta).sum() + 0.5 * sigmas.sum()
                theta1, theta2 = utils.get_unsigned(theta)
                t_cd.append(time() - t)
                log["t_cd"] = sum(t_cd)
                log['loss'].append(obj)
                if callback:
                    callback(theta, thetabar, v=obj)

                return (theta1, theta2, R, thetabar1, thetabar2, log, sigmas,
                        b1, b2, marginals1, marginals2)
            for i in range(maxiter):
                t0 = time()
                if not positive:
                    theta2f = np.asfortranarray(theta2)
                    Y1 = utils.residual(Xf, - theta2f, Yf)
                else:
                    Y1 = Yf
                theta1, R, sigmas = update_coefs(pll, Xf, Y1, Ls, marginals1,
                                                 sigmas,
                                                 a, b * weights,
                                                 sigma0,
                                                 coefs0=theta1,
                                                 R=R,
                                                 tol=tol_cd, maxiter=10000,
                                                 cython=cython)
                if not positive:
                    theta1f = np.asfortranarray(theta1)
                    Y2 = utils.residual(Xf, theta1f, Yf)
                    theta2, R, sigmas, = update_coefs(pll, - Xf, Y2, Ls,
                                                      marginals2,
                                                      sigmas,
                                                      a, b * weights,
                                                      sigma0,
                                                      coefs0=theta2,
                                                      R=R,
                                                      tol=tol_cd,
                                                      maxiter=10000,
                                                      cython=cython)
                    theta = theta1 - theta2
                else:
                    theta = theta1.copy()
                if not np.isfinite(theta).all():
                    warnings.warn("!!!!! NAN AFTER COORD DESCENT !!!!")
                obj = 0.5 * (R ** 2).sum(axis=1).dot(1 / sigmas) / n_samples
                obj += beta * abs(theta).sum() + 0.5 * sigmas.sum()
                log["objcd"].append(obj)
                tc = time() - t0
                t_cd.append(tc)
                dx = abs(theta - thetaold).max() / max(1, thetaold.max(),
                                                       theta.max())

                thetaold = theta.copy()

                t1 = time()
                if alpha:
                    if theta1.any(0).all():
                        fot1, log_ot1, marginals1, b1, q1 = \
                            update_ot_1(theta1 + 1e-100, M, epsilon, gamma,
                                        b=b1, tol=tol_ot, maxiter=maxiter_ot)
                        utils.free_gpu_memory(xp)
                        if fot1 is None or not theta1.max(0).all():
                            warnings.warn("""Nan found in positive, re-fit in
                                             log-domain.""")
                            b1 = xp.log(b1 + 1e-100, out=b1)
                            stable = True
                            update_ot_1 = set_ot_func(True, ot_img)
                            fot1, log_ot1, marginals1, b1, q1 = \
                                update_ot_1(theta1, M, epsilon, gamma, b=b1,
                                            tol=tol_ot, maxiter=maxiter_ot)
                            utils.free_gpu_memory(xp)

                        log["log_sinkhorn1"].append(log_ot1)
                        thetabar1 = q1
                        log["fot1"].append(fot1)
                        obj += alpha * fot1 / n_tasks
                    if not positive and theta2.any(0).all():
                        fot2, log_ot2, marginals2, b2, q2 = \
                            update_ot_2(theta2 + 1e-100, M, epsilon, gamma,
                                        b=b2, tol=tol_ot, maxiter=maxiter_ot)
                        utils.free_gpu_memory(xp)

                        if fot2 is None or not theta2.max(0).all():
                            warnings.warn("""Nan found in negative, re-fit in
                                             log-domain.""")
                            b2 = xp.log(b2 + 1e-100, out=b2)
                            stable = True
                            update_ot_2 = set_ot_func(True, ot_img)
                            fot2, log_ot2, marginals2, b2, q2 = \
                                update_ot_2(theta2, M, epsilon, gamma,
                                            b=b2, tol=tol_ot,
                                            maxiter=maxiter_ot)
                            utils.free_gpu_memory(xp)

                        log["log_sinkhorn2"].append(log_ot2)
                        thetabar2 = q2
                        log["fot2"].append(fot2)
                        obj += alpha * fot2 / n_tasks
                        thetabar = thetabar1 - thetabar2
                    else:
                        thetabar = thetabar1

                t_ot += time() - t1
                if callback:
                    callback(theta, thetabar, v=obj)

                log['loss'].append(obj)
                log['dloss'].append(dx)
                log['times'].append(time() - t0)

                # if df < ftol or dx < tol:
                if dx < tol:
                    break
                if i == maxiter - 1:
                    print("\n"
                          "******** WARNING: Stopped early. *****\n"
                          "\n"
                          "You may want to increase mtw.maxiter.")
            dx = abs(theta - outer_theta).max() / max(1, abs(theta).max(),
                                                      abs(outer_theta).max())
            obj -= beta * abs(theta).sum()
            obj += beta * (abs(theta) ** 0.5).sum()
            log["reweighting_loss"].append(obj)
            weights = beta * 0.5 / (abs(theta) ** 0.5 + 1e-10)
            if dx < reweighting_tol:
                break
            if rr == reweighting_steps - 1 and rr:
                print("\n"
                      "******** WARNING: Stopped while reweighting. *****\n"
                      "\n"
                      "You may want to increase mtw.reweighting_steps.")

    print("Time ot %.1f | Time cd %.1f" % (t_ot, sum(t_cd)))

    log['stable'] = stable
    log['t_cd'] = sum(t_cd)
    log['t_ot'] = t_ot

    if positive:
        theta2 *= 0.
        thetabar2 = xp.zeros_like(thetabar1)
        try:
            thetabar2 = thetabar2.get()
        except AttributeError:
            pass
    # sigmas = np.linalg.norm(R, axis=1) / (n_samples ** 0.5)
    return (theta1, theta2, R, thetabar1, thetabar2, log, sigmas, b1, b2,
            marginals1, marginals2)


def set_ot_func(stable, ot_img):
    """Set barycenter function."""
    if stable:
        update_ot = barycenterkl_log
    else:
        update_ot = barycenterkl

    if ot_img:
        if stable:
            update_ot = barycenterkl_img_log
        else:
            update_ot = barycenterkl_img
    else:
        if stable:
            update_ot = barycenterkl_log
        else:
            update_ot = barycenterkl
    return update_ot


@jit(float64(float64, float64, float64, float64),
     nopython=True, cache=True)
def proximal(theta, marginal, a, b):
    """Compute proximal operator of MTW penalty
       (b * Lasso + a * KL(.|marginal))"""
    z = theta - a - b
    delta = (z ** 2 + 4 * a * marginal) ** 0.5
    theta = (z + delta) / 2
    return theta


@njit(float64(float64))
def fsign(f):
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


@jit(float64[::1, :](float64[::1, :, :]), nopython=True, cache=True)
def lipschitz_numba(X):
    """Compute lipschitz constants."""
    T, n, p = X.shape
    L = np.zeros((T, p))
    L = np.asfortranarray(L)
    for k in range(T):
        for j in range(p):
            li = 0.
            for i in range(n):
                li = li + X[k, i, j] ** 2
            L[k, j] = li
    return L


def linreg(X, y):
    sol, _ = nnls(X, y)
    R = y - X.dot(sol)
    return sol, R


def update_coefs(pll, X, y, Ls, marginals, sigmas, a, b, sigma0,
                 coefs0=None, R=None, maxiter=20000, tol=1e-4, cython=False,
                 positive=False):
    """BCD in numba."""
    n_tasks, n_samples, n_features = X.shape

    if coefs0 is None:
        R = y.copy()
        coefs0 = np.zeros((n_features, n_tasks))
    elif R is None:
        coefs0 = np.asfortranarray(coefs0)
        R = utils.residual(X, coefs0, y)

    if cython:
        dell = delayed(cython_wrapper)
        it = (dell(n_samples, n_features, np.asfortranarray(X[k].copy()), y[k],
                   Ls[k], marginals[k], coefs0[:, k], R[k].copy(),
                   sigmas[k:k + 1], a, b[:, k], sigma0, tol, maxiter, positive)
              for k in range(n_tasks))
    else:
        dell = delayed(nb_solver_wrapper)
        it = (dell(X[k], y[k], Ls[k], marginals[k],
                   coefs0[:, k], R[k], sigmas[k], a, b[:, k], sigma0,
                   tol, maxiter, positive) for k in range(n_tasks))
    output = pll(it)

    thetas, R, sigmas = list(zip(*output))
    thetas = np.stack(thetas, axis=1)
    R = np.stack(R, axis=0)

    # output = [nb_solver_wrapper(X[k], y[k], Ls[k], marginals[k],
    #                             coefs0[:, k], R[k], alpha, gamma,
    #                             beta, tol, n_tasks, maxiter)
    #           for k in range(n_tasks)]
    sigmas = np.r_[sigmas]
    return thetas, R, sigmas


@jit(float64(float64[:], float64[:], float64),
     nopython=True, cache=True)
def mt_loss(theta, R, beta):
    """Compute objective function for multi task group lasso."""
    n_samples = len(R)
    obj = 0.
    for n in range(n_samples):
        obj += R[n] ** 2
    obj *= 0.5

    return obj

output_type = nb.types.Tuple((float64[:], float64[:], float64))


@njit(output_type(float64[::1, :], float64[:], float64[:],
                  float64[:], float64[:], float64[:], float64,
                  float64, float64[:], float64, float64, int64,
                  boolean),
      nogil=True, cache=True)
def nb_solver(X, y, Ls, marginals, theta, R, sigma, a, b, sigma0,
              tol=1e-6, maxiter=10000, positive=False):
    """CD solver for l2 reg kl interpolation."""
    n_samples, n_features = X.shape

    if sigma0:
        sigma = np.linalg.norm(R) / (n_samples ** 0.5)
    for i in range(maxiter):
        maxw = 0.
        for j in range(n_features):
            # tmp is the prox argument
            if theta[j] != 0.:
                R += X[:, j] * theta[j]

            tmp = X[:, j].dot(R)
            # l2 thresholding
            old_j = theta[j]
            if a or positive:
                theta[j] = proximal(tmp, marginals[j] * Ls[j], a, b[j] * sigma)
                theta[j] /= Ls[j]
            else:
                theta[j] = fsign(tmp) * max(abs(tmp) - b[j] * sigma, 0) / Ls[j]

            dw = abs(old_j - theta[j]) / max(1, abs(theta[j]),
                                             abs(old_j))
            if dw > maxw:
                maxw = dw

            if theta[j] != 0.:
                R += - theta[j] * X[:, j]
                if sigma0:
                    sigma = np.linalg.norm(R) / (n_samples ** 0.5)

        if maxw < tol:
            break
    else:
        print("\n"
              "******** WARNING: Stopped early in cd loop. *****\n"
              "\n"
              "You may want to increase maxiter in cd."
              "max_change = ", maxw)

    # obj = mt_loss(theta, R, beta)
    return theta, R, sigma


def nb_solver_wrapper(X, y, Ls, marginals, coefs0, R, sigma, a, b,
                      sigma0, tol=1e-6, maxiter=1000, positive=True):
    Xf = np.asfortranarray(X)
    theta, R, sigma = nb_solver(Xf, y, Ls, marginals, coefs0, R, sigma, a,
                                b, sigma0, tol, maxiter, positive)
    return theta, R, sigma


def barycenterkl_log(P, M, epsilon, gamma, b=None, tol=1e-4,
                     maxiter=1000, weights=None):
    """KL OT Barycenters for 2D images."""
    xp = get_module(M)
    frac = gamma / (gamma + epsilon)
    n_tasks = P.shape[-1]
    n_features = M.shape[-1]
    psum = P.sum()
    support = P.any(axis=1)
    logps = np.log(P[support] + 1e-100)
    logps = xp.asarray(logps)
    M = M[xp.asarray(support)]
    utils.free_gpu_memory(xp)

    if b is None:
        Kb = utils.logsumexp(M, axis=1)
        Kb = xp.tile(Kb, (n_tasks, 1)).T
        utils.free_gpu_memory(xp)

    else:
        Kb = xp.zeros((len(M), n_tasks))
        for k in range(n_tasks):
            Kb[:, k] = utils.logsumexp(b[:, k][None, :] + M, axis=1)
            utils.free_gpu_memory(xp)
    log = {'cstr': [], 'flag': 0, 'obj': []}
    if weights is None:
        weights = xp.ones(n_tasks) / n_tasks
    logweights = xp.log(weights)[None, :]
    qold = xp.ones(n_features)[:, None]
    Ka = xp.zeros((n_features, n_tasks))
    for i in range(maxiter):
        a = frac * (logps - Kb)  # it's actually a
        for k in range(n_tasks):
            Ka[:, k] = utils.logsumexp(a[:, k][:, None] + M, axis=0)
            utils.free_gpu_memory(xp)
        logq = logweights + Ka * (1 - frac)  # it's weighted ka
        logq = utils.logsumexp(logq, axis=1)  # this is q in log
        logq = (1 / (1 - frac)) * logq
        b = frac * (logq[:, None] - Ka)
        for k in range(n_tasks):
            Kb[:, k] = utils.logsumexp(b[:, k][None, :] + M, axis=1)
            utils.free_gpu_memory(xp)
        q = xp.exp(logq)
        cstr = float(abs(q - qold).max())
        cstr /= float(max(q.max(), qold.max(), 1e-20))
        qold = q.copy()
        log["cstr"].append(cstr)
        # marginals = np.exp(a + Kb).T
        # f = - (2 * gamma + epsilon) * marginals.sum()
        # f += gamma * (psum + q.sum() * n_tasks)
        # f += ((gamma + epsilon) * a + gamma * (Kb - logps))
        # f = (f * marginals.T).sum()
        # log["obj"].append(f)
        if cstr < tol and i > 5:
            break
        # f = utils.wklobjective_log(a, Kb, P, q, K0, epsilon, gamma)
        # log["obj"].append(f)

    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        log['flag'] = 3

    try:
        a = a.get()
        Kb = Kb.get()
        q = q.get()
        logps = logps.get()
        utils.free_gpu_memory(xp)

    except AttributeError:
        pass
    marginals = np.exp(a + Kb).T
    marginals[~np.isfinite(marginals)] = 1

    m = np.zeros((n_tasks, n_features))
    f = utils.wklobjective_converged(n_tasks * q.sum(), 0.,
                                     psum,
                                     epsilon, gamma)
    # f = - (2 * gamma + epsilon) * marginals.sum()
    # f += gamma * (psum + q.sum() * n_tasks)
    # f += (((gamma + epsilon) * a + gamma * (Kb - logps)) * marginals.T).sum()
    # log["obj"].append(f)
    m[:, support] = marginals
    marginals = m
    b[~np.isfinite(b)] = 0.

    return f, log, marginals, b, q


def barycenterkl(P, M, epsilon, gamma, b=None, tol=1e-4,
                 maxiter=1000, weights=None):
    """Compute Unblanced Wasserstein barycenter.
    """
    xp = get_module(M)
    frac = gamma / (gamma + epsilon)
    psum = P.sum()
    n_features, n_tasks = P.shape
    frac = gamma / (gamma + epsilon)
    support = (P > 1e-20).any(axis=1)
    if len(support) == 0:
        support = P.any(axis=1)
    P = P[support]
    P = xp.asarray(P)
    M = M[xp.asarray(support)]
    M = xp.exp(M)
    if b is None:
        b = xp.ones((n_features, n_tasks))
    Kb = M.dot(b)

    log = {'cstr': [], 'flag': 0, 'obj': []}
    if weights is None:
        weights = xp.ones(n_tasks) / n_tasks
    q = xp.ones(n_features)
    qold = q.copy()
    return_nan = False
    utils.free_gpu_memory(xp)

    for i in range(maxiter):
        a = (P / Kb) ** frac
        utils.free_gpu_memory(xp)

        Ka = M.T.dot(a)
        q = ((Ka ** (1 - frac)).dot(weights))
        q = q ** (1 / (1 - frac))
        Q = q[:, None]
        utils.free_gpu_memory(xp)

        cstr = float(abs(q - qold).max() / max(q.max(), qold.max(), 1e-10))
        qold = q.copy()
        b_old = b.copy()
        b = (Q / Ka) ** frac
        utils.free_gpu_memory(xp)

        if not xp.isfinite(b).all():
            return_nan = True
            break
        Kb = M.dot(b)
        log["cstr"].append(cstr)
        if abs(cstr) < tol and i > 5:
            break
        # marginals = (a * Kb).T
        utils.free_gpu_memory(xp)
        # f = - (2 * gamma + epsilon) * marginals.sum()
        # f += gamma * (psum + q.sum() * n_tasks)
        # f += (((gamma + epsilon) * np.log(a + 1e-100) + gamma *
        #       (np.log(Kb + 1e-100) - np.log(P + 1e-100))) *
        # marginals.T).sum()
        # log["obj"].append(f)
    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        log['flag'] = - 1
    marginals = (a * Kb).T

    try:
        marginals = marginals.get()
        q = q.get()
        utils.free_gpu_memory(xp)
        P = P.get()
        Kb = Kb.get()
        a = a.get()

    except AttributeError:
        pass

    f = utils.wklobjective_converged(n_tasks * q.sum(), 0.,
                                     psum,
                                     epsilon, gamma)
    m = np.zeros((n_tasks, n_features))
    marginals[~np.isfinite(marginals)] = 1
    m[:, support] = marginals
    marginals = m
    if return_nan or xp.isnan(f):
        f = None
        b = b_old
    # else:
    #     f = - (2 * gamma + epsilon) * marginals.sum()
    #     f += gamma * (psum + q.sum() * n_tasks)
    #     xx = ((Kb / P) ** gamma) * (a ** (epsilon + gamma))
    #     xx[xx == 0.] = 0.
    #     f += (np.log(xx) * marginals.T).sum()
    #     log["obj"].append(f)
    return f, log, marginals, b, q


def barycenterkl_img_log(P, M, epsilon, gamma, b=None, tol=1e-4,
                         maxiter=1000, xp=np):
    """KL OT Barycenters for 2D images."""
    xp = get_module(M)
    psum = P.sum()
    P = P.reshape(xp.r_[M.shape, -1])
    n_tasks = P.shape[-1]
    n_features = P.size // n_tasks
    frac = gamma / (gamma + epsilon)
    if b is None:
        b = xp.zeros_like(P)
    Kb = utils.kls(b, M)
    log = {'cstr': [], 'flag': 0, 'obj': []}
    weights = xp.ones(n_tasks) / n_tasks
    logweights = xp.log(weights)[None, None, :]
    logp = xp.log(P + 1e-10)
    b = xp.zeros_like(logp)
    qold = P.mean(axis=-1) + 1e-10
    for i in range(maxiter):
        a = frac * (logp - Kb)
        Ka = utils.kls(a, M.T)
        kaw = logweights + Ka * (1 - frac)
        logq = utils.logsumexp(kaw, axis=-1) - xp.log(weights.sum())
        logq = (1 / (1 - frac)) * logq
        logQ = logq[:, :, xp.newaxis]
        b = frac * (logQ - Ka)
        Kb = utils.kls(b, M)
        q = xp.exp(logq)

        if i % 10 == 0:
            cstr = float((abs(q - qold)).max())
            cstr /= float(max(q.max(), qold.max(), 1e-20))
        qold = q.copy()

        log["cstr"].append(cstr)
        if cstr < tol and i > 5:
            break

        # f = utils.wklobjective_log(a, Kb, P, q, K0, epsilon, gamma)
        # log["obj"].append(f)
        utils.free_gpu_memory(xp)

    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        log['flag'] = 3

    marginals = xp.exp(a + Kb).reshape(n_features, n_tasks).T
    try:
        marginals = marginals.get()
        q = q.get()
    except AttributeError:
        pass

    f = utils.wklobjective_converged(n_tasks * q.sum(), 0.,
                                     psum,
                                     epsilon, gamma)
    return f, log, marginals, b, q


def barycenterkl_img(P, M, epsilon, gamma, b=None, tol=1e-4,
                     maxiter=1000, xp=np):
    """KL OT Barycenters for 2D images."""
    xp = get_module(M)
    psum = P.sum()
    P = P.reshape(xp.r_[M.shape, -1])
    frac = gamma / (gamma + epsilon)
    n_tasks = P.shape[-1]
    if b is None:
        b = xp.ones_like(P)
    M = xp.exp(M)
    Kb = utils.klconv1d_list(b, M)

    log = {'cstr': [], 'flag': 0, 'obj': []}
    weights = xp.ones(n_tasks) / n_tasks
    # qold = P.mean(axis=-1)
    return_nan = False
    margs_old = P.copy()

    for i in range(maxiter):
        a = (P / Kb) ** frac
        Ka = utils.klconv1d_list(a, M.T)
        q = ((Ka) ** (1 - frac)).dot(weights)
        q = (q / (weights.sum())) ** (1 / (1 - frac))
        Q = q[:, :, xp.newaxis]
        b_old = b.copy()
        b = (Q / Ka) ** frac
        if not xp.isfinite(b).all():
            return_nan = True
            break

        Kb = utils.klconv1d_list(b, M)
        margs = a * Kb
        if i % 10 == 0:
            cstr = float((abs(margs - margs_old)).max())
            cstr /= float(max(margs.max(), margs_old.max(), 1e-20))
        margs_old = margs.copy()

        log["cstr"].append(cstr)
        if cstr < tol:
            break
        utils.free_gpu_memory(xp)
    if i == maxiter - 1:
        warnings.warn("Early stop, Maxiter too low !")
        log['flag'] = 3

    marginals = (a * Kb).reshape(- 1, n_tasks).T
    try:
        marginals = marginals.get()
        q = q.get()
    except AttributeError:
        pass
    f = utils.wklobjective_converged(n_tasks * q.sum(), 0.,
                                     psum,
                                     epsilon, gamma)
    if return_nan:
        f = None
        b = b_old
    return f, log, marginals, b, q
