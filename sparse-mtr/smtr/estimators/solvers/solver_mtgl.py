"""Numba solver for multitask group lasso with a positive constraint."""
import numpy as np
import numba as nb

from numba import jit, float64, int64, boolean

from ... import utils


@jit(float64(float64[::1, :, :], float64[::1, :], float64[::1, :],
             float64[::1, :], float64),
     nopython=True, cache=True)
def dualgap(X, theta, R, y, alpha):
    """Compute dual gap for multi task group lasso."""
    n_samples, n_tasks = R.T.shape
    n_features = X.shape[-1]
    dualnorm = 0.
    xtr = np.zeros_like(theta)
    for k in range(n_tasks):
        xtr[:, k] = X[k].T.dot(R[k])
    for j in range(n_features):
        dn = np.linalg.norm(xtr[j])
        if dn > dualnorm:
            dualnorm = dn
    print(dualnorm)
    R_norm = np.linalg.norm(R)
    if dualnorm < alpha:
        dg = R_norm ** 2
        const = 1.
    else:
        const = alpha / dualnorm
        A_norm = R_norm * const
        dg = 0.5 * (R_norm ** 2 + A_norm ** 2)
    dg += - const * (R * y).sum()
    for j in range(n_features):
        dg += alpha * np.linalg.norm(theta[j])
    return dg


@jit(float64(float64[::1, :], float64[::1, :], float64[::1, :], float64),
     nopython=True, cache=True)
def mtlobjective(theta, R, y, alpha):
    """Compute objective function for multi task group lasso."""
    n_samples, n_tasks = R.T.shape
    n_features = theta.shape[0]
    obj = 0.
    for t in range(n_tasks):
        for n in range(n_samples):
            obj += R[t, n] ** 2
    obj *= 0.5
    for j in range(n_features):
        obj += alpha * np.linalg.norm(theta[j])
    return obj


def solver_mtgl(X, y, coefs0=None, R=None, alpha=1., beta=1.,
                maxiter=2000, tol=1e-3, positive=False, verbose=False,
                computeobj=False):
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

    R = np.asfortranarray(R)

    if positive:
        theta1, R, loss = solver_numba_positive(X, y, coefs01, R, alpha,
                                                maxiter, tol, verbose,
                                                computeobj)
    else:
        theta1, R, loss = solver_numba(X, y, coefs01, R, alpha,
                                       maxiter, tol, verbose, computeobj)
    theta1 = np.ascontiguousarray(theta1)
    theta2 = np.zeros_like(theta1)
    return theta1, theta2, R, loss


output_type = nb.types.Tuple((float64[::1, :], float64[::1, :],
                              float64[:]))


@jit(output_type(float64[::1, :, :], float64[::1, :],
                 float64[::1, :], float64[::1, :], float64, int64,
                 float64, boolean, boolean),
     nopython=True, cache=True)
def solver_numba_positive(X, y, theta, R, alpha, maxiter, tol,
                          verbose, computeobj):
    """Perform GFB with BCD to solve Multi-task group lasso."""
    n_tasks = len(X)
    n_samples, n_features = X[0].shape
    Ls = utils.lipschitz(X)
    dg_tol = tol * np.linalg.norm(y) ** 2
    z = theta.copy()
    z = np.asfortranarray(z)
    loss = []

    for i in range(maxiter):
        maxw = 0.
        dg = 1.
        w_max = 0.0
        d_w_max = 0.0
        for j in range(n_features):
            # compute residual
            if Ls[j] == 0.:
                continue
            tmp = np.zeros(n_tasks)
            normtmp = 0.
            for t in range(n_tasks):
                for n in range(n_samples):
                    tmp[t] += X[t, n, j] * R[t, n]
                tmp[t] /= Ls[j]
                tmp[t] += 2 * theta[j, t] - z[j, t]
                normtmp += tmp[t] ** 2

            # l2 thresholding
            if normtmp:
                normtmp = np.sqrt(normtmp)
                threshold = 1. - alpha / (Ls[j] * normtmp)

                for t in range(n_tasks):
                    if threshold <= 0.:
                        tmp[t] = 0.
                    else:
                        tmp[t] *= threshold
                    z[j, t] += tmp[t] - theta[j, t]

                    if theta[j, t] != 0:
                        for n in range(n_samples):
                            R[t, n] += X[t, n, j] * theta[j, t]
                    # positive constraint
                    new_theta = max(z[j, t], 0)
                    d_w_j = abs(theta[j, t] - new_theta)
                    d_w_max = max(d_w_max, d_w_j)
                    w_max = max(w_max, abs(new_theta))
                    theta[j, t] = new_theta

                    if theta[j, t] != 0:
                        for n in range(n_samples):
                            R[t, n] -= X[t, n, j] * theta[j, t]

        if computeobj:
            obj = mtlobjective(theta, R, y, alpha)
            loss.append(obj)
        if (w_max == 0.0 or d_w_max / w_max < tol or
                i == maxiter - 1):
            dg = dualgap(X, theta, R, y, alpha)
            if verbose:
                print(dg)
            if dg < dg_tol:
                break
        if maxw < tol:
            break
    if i == maxiter - 1:
        print("**************************************\n"
              "******** WARNING: Stopped early. *****\n"
              "\n"
              "You may want to increase maxiter.")
        dg = dualgap(X, theta, R, y, alpha)
        print(dg)
    loss = np.array(loss)
    return theta, R, loss


@jit(output_type(float64[::1, :, :], float64[::1, :],
                 float64[::1, :], float64[::1, :], float64, int64,
                 float64, boolean, boolean),
     nopython=True, cache=True)
def solver_numba(X, y, theta, R, alpha, maxiter, tol,
                 verbose, computeobj):
    """Perform GFB with BCD to solve Multi-task group lasso."""
    n_tasks = len(X)
    n_samples, n_features = X[0].shape
    Ls = utils.lipschitz(X)
    dg_tol = tol * np.linalg.norm(y) ** 2
    loss = []

    for i in range(maxiter):
        w_max = 0.0
        d_w_max = 0.0
        for j in range(n_features):
            # compute residual
            if Ls[j] == 0.:
                continue
            tmp = np.zeros(n_tasks)
            for t in range(n_tasks):
                for n in range(n_samples):
                    tmp[t] += X[t, n, j] * R[t, n]
            tmp /= Ls[j]
            tmp += theta[j]
            normtmp = np.linalg.norm(tmp)

            # l2 thresholding
            threshold = 0.
            if normtmp:
                threshold = max(1. - alpha / (Ls[j] * normtmp), 0.)
            tmp *= threshold
            if theta[j].any():
                for t in range(n_tasks):
                    R[t] += X[t, :, j] * theta[j, t]
            d_w_j = np.abs(theta[j] - tmp).max()
            d_w_max = max(d_w_max, d_w_j)
            w_max = max(w_max, np.abs(tmp).max())
            theta[j] = tmp

            if theta[j].any():
                for t in range(n_tasks):
                    R[t] -= X[t, :, j] * theta[j, t]

        if computeobj:
            obj = mtlobjective(theta, R, y, alpha)
            loss.append(obj)
        if (w_max == 0.0 or d_w_max / w_max < tol or
                i == maxiter - 1):
            dg = dualgap(X, theta, R, y, alpha)
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
    return theta, R, loss
