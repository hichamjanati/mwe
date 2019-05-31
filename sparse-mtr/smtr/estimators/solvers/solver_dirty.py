"""Solvers for multitask group-stl with a simplex constraint."""
import numpy as np
import numba as nb
from numba import (jit, float64, int64, boolean)
from ... import utils


@jit(float64(float64[::1, :, :], float64[::1, :], float64[::1, :],
             float64[::1, :], float64[::1, :], float64, float64),
     nopython=True, cache=True)
def dualgap(X, theta1, theta2, R, y, alpha, beta):
    """Compute dual gap for multi task group lasso."""
    n_samples, n_tasks = R.T.shape
    n_features = X.shape[-1]
    xtr = np.zeros_like(theta1)
    dualnorm = 0.
    dualmax = 0.
    for k in range(n_tasks):
        xtr[:, k] = X[k].T.dot(R[k])
    for j in range(n_features):
        dn = np.linalg.norm(xtr[j])
        dm = np.abs(xtr[j]).max()
        if dn > dualnorm:
            dualnorm = dn
        if dm > dualmax:
            dualmax = dm
    R_norm = np.linalg.norm(R)
    const = 1.
    if dualnorm < alpha and dualmax < beta:
        dg = R_norm ** 2
    else:
        const = min(alpha / dualnorm, beta / dualmax)
        A_norm = R_norm * const
        dg = 0.5 * (R_norm ** 2 + A_norm ** 2)
    dg += - const * (R * y).sum()
    for j in range(n_features):
        dg += alpha * np.linalg.norm(theta1[j])
    dg += beta * np.abs(theta2).sum()
    return dg


@jit(float64(float64[::1, :], float64[::1, :], float64[::1, :],
             float64[::1, :], float64, float64),
     nopython=True, cache=True)
def dirtyobjective(theta1, theta2, R, y, alpha, beta):
    """Compute objective function for multi task group lasso."""
    n_samples, n_tasks = R.T.shape
    obj = 0.
    for t in range(n_tasks):
        for n in range(n_samples):
            obj += R[t, n] ** 2
    obj *= 0.5
    obj += alpha * utils.l21norm(theta1)
    obj += beta * utils.l1norm(theta2)
    return obj


def solver_dirty(X, y, coefs0=None, R=None, alpha=1., beta=1., maxiter=2000,
                 tol=1e-4, positive=False, verbose=False, computeobj=False):
    """BCD in numba."""
    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    if R is None:
        R = y.copy()
        n_tasks = len(X)
        n_features = X[0].shape[-1]
        coefs01 = np.zeros((n_features, n_tasks))
        coefs02 = np.zeros((n_features, n_tasks))
    else:
        coefs01, coefs02 = coefs0
    coefs01 = np.asfortranarray(coefs01)
    coefs02 = np.asfortranarray(coefs02)

    R = np.asfortranarray(R)

    if positive:
        theta1, theta2, R, loss = solver_numba_positive(X, y, coefs01, coefs02,
                                                        R, alpha, beta,
                                                        maxiter, tol,
                                                        verbose, computeobj)
    else:
        theta1, theta2, R, loss = solver_numba(X, y, coefs01, coefs02, R,
                                               alpha, beta, maxiter, tol,
                                               verbose, computeobj)
    theta1 = np.ascontiguousarray(theta1)
    theta2 = np.ascontiguousarray(theta2)
    return theta1, theta2, R, loss


output_type = nb.types.Tuple((float64[::1, :], float64[::1, :],
                              float64[::1, :], float64[:]))


@jit(output_type(float64[::1, :, :], float64[::1, :], float64[::1, :],
                 float64[::1, :], float64[::1, :], float64, float64, int64,
                 float64, boolean, boolean),
     nopython=True, cache=True)
def solver_numba_positive(X, y, theta1, theta2, R, alpha, beta, maxiter, tol,
                          verbose, computeobj):
    """Perform GFB with BCD to solve Multi-task Dirty group lasso."""
    n_tasks = len(X)
    n_samples, n_features = X[0].shape
    theta = theta1 + theta2
    Ls = utils.lipschitz(X)
    z1 = theta1.copy()
    z2 = theta2.copy()
    z1 = np.asfortranarray(z1)
    z2 = np.asfortranarray(z2)
    dg_tol = tol * np.linalg.norm(y) ** 2
    loss = []

    # dg = 1.
    for i in range(maxiter):
        w_max = 0.0
        d_w_max = 0.0
        for j in range(n_features):
            if Ls[j] == 0.:
                continue
            # compute residual
            grad = np.zeros(n_tasks)
            tmp1 = np.zeros(n_tasks)
            tmp2 = np.zeros(n_tasks)

            normtmp = 0.
            for t in range(n_tasks):
                for n in range(n_samples):
                    grad[t] += X[t, n, j] * R[t, n]
                grad[t] /= Ls[j]
                tmp1[t] = grad[t] + 2 * theta1[j, t] - z1[j, t]
                tmp2[t] = grad[t] + 2 * theta2[j, t] - z2[j, t]

                normtmp += tmp1[t] ** 2

            normtmp = np.sqrt(normtmp)

            # l2 thresholding
            if normtmp == 0.:
                thresholdl2 = - 1.
            else:
                thresholdl2 = 1. - alpha / (Ls[j] * normtmp)
            thresholdl1 = beta / Ls[j]

            for t in range(n_tasks):
                if thresholdl2 <= 0.:
                    tmp1[t] = 0.
                else:
                    tmp1[t] *= thresholdl2
                z1[j, t] += tmp1[t] - theta1[j, t]

                tmp2[t] = np.sign(tmp2[t]) * max(abs(tmp2[t]) - thresholdl1, 0)
                z2[j, t] += tmp2[t] - theta2[j, t]

                theta[j, t] = theta1[j, t] + theta2[j, t]
                if theta[j, t] != 0:
                    for n in range(n_samples):
                        R[t, n] += X[t, n, j] * theta[j, t]
                # positive constraint

                theta1[j, t] = max(z1[j, t], 0.)
                theta2[j, t] = max(z2[j, t], 0.)
                new_theta = theta1[j, t] + theta2[j, t]
                d_w_j = abs(theta[j, t] - new_theta)
                d_w_max = max(d_w_max, d_w_j)
                w_max = max(w_max, abs(new_theta))
                theta[j, t] = new_theta

                if theta[j, t] != 0:
                    for n in range(n_samples):
                        R[t, n] -= X[t, n, j] * theta[j, t]
        if computeobj:
            obj = dirtyobjective(theta1, theta2, R, y, alpha, beta)
            loss.append(obj)

        if (w_max == 0.0 or d_w_max / w_max < tol or
                i == maxiter - 1):
            dg = dualgap(X, theta1, theta2, R, y, alpha, beta)
            if verbose:
                print(dg)
            if dg < dg_tol:
                break
    if i == maxiter - 1:
        print("**************************************\n"
              "******** WARNING: Stopped early. *****\n"
              "\n"
              "You may want to increase maxiter.")
        print(dg)
    loss = np.array(loss)
    return theta1, theta2, R, loss


@jit(output_type(float64[::1, :, :], float64[::1, :], float64[::1, :],
                 float64[::1, :], float64[::1, :], float64, float64, int64,
                 float64, boolean, boolean),
     nopython=True, cache=True)
def solver_numba(X, y, theta1, theta2, R, alpha, beta, maxiter, tol,
                 verbose, computeobj):
    """Perform GFB with BCD to solve Multi-task Dirty group lasso."""
    n_tasks = len(X)
    n_samples, n_features = X[0].shape
    theta = theta1 + theta2
    Ls = utils.lipschitz(X)
    dg_tol = tol * np.linalg.norm(y) ** 2
    loss = []

    # dg = 1.
    for i in range(maxiter):
        w_max = 0.0
        d_w_max = 0.0
        for j in range(n_features):
            if Ls[j] == 0.:
                continue
            # compute residual
            grad = np.zeros(n_tasks)
            tmp1 = np.zeros(n_tasks)
            tmp2 = np.zeros(n_tasks)

            normtmp = 0.
            for t in range(n_tasks):
                for n in range(n_samples):
                    grad[t] += X[t, n, j] * R[t, n]
                grad[t] /= Ls[j]
                tmp1[t] = grad[t] + theta1[j, t]
                tmp2[t] = grad[t] + theta2[j, t]

                normtmp += tmp1[t] ** 2

            normtmp = np.sqrt(normtmp)

            # l2 thresholding

            thresholdl2 = 0.
            if normtmp:
                thresholdl2 = max(1. - alpha / (Ls[j] * normtmp), 0.)
            tmp1 *= thresholdl2
            thresholdl1 = beta / Ls[j]
            tmp2 = np.sign(tmp2) * np.maximum(np.abs(tmp2) - thresholdl1, 0.)
            new_theta = tmp1 + tmp2
            if theta[j].any():
                for t in range(n_tasks):
                    R[t] += X[t, :, j] * theta[j, t]

            d_w_j = np.abs(theta[j] - new_theta).max()
            d_w_max = max(d_w_max, d_w_j)
            w_max = max(w_max, np.abs(tmp1 + tmp2).max())
            theta1[j] = tmp1
            theta2[j] = tmp2
            theta[j] = new_theta

            if theta[j].any():
                for t in range(n_tasks):
                    R[t] -= X[t, :, j] * theta[j, t]
        if computeobj:
            obj = dirtyobjective(theta1, theta2, R, y, alpha, beta)
            loss.append(obj)

        if (w_max == 0.0 or d_w_max / w_max < tol or
                i == maxiter - 1):
            dg = dualgap(X, theta1, theta2, R, y, alpha, beta)
            if verbose:
                print(dg, dg_tol)
            if dg < dg_tol:
                break
    if i == maxiter - 1:
        print("**************************************\n"
              "******** WARNING: Stopped early. *****\n"
              "\n"
              "You may want to increase maxiter.")
        print(dg)
    loss = np.array(loss)
    return theta1, theta2, R, loss
