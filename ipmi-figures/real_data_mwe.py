import numpy as np
import os

from smtr import MTW, utils
from whiten_data import build_real_dataset
from joblib import Parallel, delayed
from time import time
from smtr.estimators.solvers.solver_mtw import barycenterkl, barycenterkl_log
import warnings
import pickle
from config import get_params, get_subjects_list


if os.path.exists("/home/parietal/"):
    gpu = True
    try:
        import cupy as cp
        device = 1
    except:
        pass

n_tasks = 32
dataset = "camcan"
params = get_params(dataset)
subjects = get_subjects_list(dataset, simu=False)[:n_tasks]
subjects_dir = params["subjects_dir"]
data_path = params["data_path"]
M_fname = data_path + "metrics/metric_fsaverage_lrh.npy"
M_ = np.load(M_fname)

depth = 0.9
X, y, subjects, times = build_real_dataset(dataset=dataset, subjects=subjects)
n_tasks, n_samples, n_features = X.shape
norms = np.linalg.norm(X, axis=1) ** depth
X_scaled = X / norms[:, None, :]
sigma00 = np.linalg.norm(y, axis=1).min() / (n_samples ** 0.5)
sigma0 = 0.01
betamax = min([abs(x.T.dot(yy)).max() for (x, yy) in zip(X_scaled, y)])
betamax /= n_samples
model = "mtw-S%s" % n_tasks
if sigma0:
    betamax /= sigma00
    model = "mwe-S%s" % n_tasks


epsilon = 10. / n_features
gamma = 1.
depth_ = int(100 * depth)
ss = "subjects_%s" % n_tasks
ss_dir = "brain_plots/%s/%s/" % (dataset, model)
coefs_path = ss_dir + "coefs/"
subjects_path = ss_dir + "subjects/"
bar_path = ss_dir + "barycenter/"
thetabar_path = ss_dir + "thetabar/"
log_path = ss_dir + "log/"

if not os.path.exists(ss_dir):
    os.makedirs(ss_dir)
if not os.path.exists(coefs_path):
    os.makedirs(coefs_path)
    os.makedirs(coefs_path + "img/")
subject_paths = []
for sub in subjects:
    subpath = subjects_path + sub + "/"
    subject_paths.append(subpath)
    if not os.path.exists(subpath):
        os.makedirs(subpath)
if not os.path.exists(bar_path):
    os.makedirs(bar_path)
    os.makedirs(bar_path + "img/")
if not os.path.exists(thetabar_path):
    os.makedirs(thetabar_path)
    os.makedirs(thetabar_path + "img/")
if not os.path.exists(log_path):
    os.makedirs(log_path)
np.save(ss_dir + "scaling.npy", norms)


def mtw_run(alpha, beta_fr, power, save=True, gpu=gpu, average=False):
    p = int(10 * power)
    b = int(100 * beta_fr)
    a = int(100 * alpha * 204)
    print(">> Entering worker alpha, beta, p", a / 100, b / 100, p / 10)
    beta = beta_fr * betamax
    M = M_.copy() ** power  # convert to mm
    M /= np.median(M)
    M = - M / epsilon
    if gpu:
        with cp.cuda.Device(device):
            mtw = MTW(M=M, alpha=alpha, beta=beta, epsilon=epsilon,
                      gamma=gamma, stable=False, maxiter=4000,
                      maxiter_ot=10, tol=1e-4, tol_ot=1e-4, positive=False,
                      n_jobs=4, cython=True, tol_cd=1e-4, sigma0=sigma0)
            mtw.fit(X_scaled, y)
            utils.free_gpu_memory(cp)

    else:
        mtw = MTW(M=M, alpha=alpha, beta=beta, epsilon=epsilon,
                  gamma=gamma, stable=False, maxiter=5000,
                  maxiter_ot=10, tol=1e-4, tol_ot=1e-4, positive=False,
                  n_jobs=4, cython=True, tol_cd=1e-4, sigma0=sigma0)
        mtw.fit(X_scaled, y)
    print("TIMES %f, %f : OT = %f s, CD = %f" %
          (a, b, mtw.log_["t_ot"], mtw.log_["t_cd"]))
    coefs_mtw = mtw.coefs_
    coefs_mtw = 1e9 * mtw.coefs_ / norms.T

    if save:
        fname = "p%d-b%d-a%d.npy" % (p, b, a)
        for coef, subpath in zip(coefs_mtw.T, subject_paths):
            np.save(subpath + fname, coef)
        np.save(coefs_path + fname, coefs_mtw)
        np.save(thetabar_path + fname, mtw.barycenter_[:, None])
        keys = ["log_", "sigmas_", "coefs_", "gamma", "epsilon", "barycenter_"]
        log = dict()
        for k in keys:
            log[k] = getattr(mtw, k)
        fname = "p%d-b%d-a%d-ave.pkl" % (p, b, a)
        with open(log_path + fname, "wb") as ff:
            pickle.dump(log, ff)

        print(">> LEAVING worker alpha, beta, p", a / 100, b / 100, p / 10)

    if average:
        print("Computing barycenter ...")
        compute_average(alpha, beta_fr, power, gpu, save)
    if gpu:
        utils.free_gpu_memory(cp)
    return 0.


def compute_average(alpha, beta_fr, power, gpu=True, save=True):
    log1, log2, logl1, logl2 = {"cstr": []}, {"cstr": []}, {"cstr": []}, \
        {"cstr": []}
    p = int(10 * power)
    b = int(100 * beta_fr)
    a = int(204 * alpha * 100)
    fname = "p%d-b%d-a%d.npy" % (p, b, a)
    coefs = np.load(coefs_path + fname)
    M = M_.copy() ** power  # convert to mm
    M /= np.median(M)
    M = - M / epsilon
    coefs1, coefs2 = utils.get_unsigned(coefs)
    mean = coefs1.mean(-1) - coefs2.mean(-1)
    mean = mean[:, None]
    f = bar_path + "euclidean-%s" % fname
    np.save(f, mean)
    with cp.cuda.Device(device):
        M = cp.asarray(M)
        fot1, log1, _, b1, bar1 = barycenterkl(coefs1 + 1e-100, M, epsilon,
                                               gamma, tol=1e-7,
                                               maxiter=3000)
        utils.free_gpu_memory(cp)

        if fot1 is None or not coefs1.max(0).all():
            warnings.warn("""Nan found when averagin, re-fit in
                             log-domain.""")
            b1 = cp.log(b1 + 1e-100, out=b1)
            fot1, logl1, m1, b1, bar1 = \
                barycenterkl_log(coefs1, M, epsilon, gamma,
                                 b=b1, tol=1e-7, maxiter=3000)
            utils.free_gpu_memory(cp)

        fot2, log2, _, b2, bar2 = barycenterkl(coefs2 + 1e-100, M, epsilon,
                                               gamma, tol=1e-7,
                                               maxiter=3000)
        utils.free_gpu_memory(cp)

        if fot2 is None or not coefs2.max(0).all():
            warnings.warn("""Nan found when averagin, re-fit in
                             log-domain.""")
            b2 = cp.log(b2 + 1e-100, out=b2)
            fot2, logl2, m2, b2, bar2 = \
                barycenterkl_log(coefs2, M, epsilon, gamma,
                                 b=b2, tol=1e-7, maxiter=3000)
            utils.free_gpu_memory(cp)

        bar = bar1 - bar2
    bar = bar[:, None]
    np.save(bar_path + fname, bar)
    fname = "p%d-b%d-a%d-bar.pkl" % (p, b, a)
    logs = [log1["cstr"], log2["cstr"], logl1["cstr"], logl2["cstr"]]
    if save:
        with open(log_path + fname, "wb") as ff:
            pickle.dump(logs, ff)
    print(">> LEAVING worker alpha, beta, p", a / 100, b / 100, p / 10)
    return 0.


if __name__ == "__main__":

    t = time()
    betafrs = np.array([0.5, 0.6, 0.7])
    n_betas = len(betafrs)
    alphas = np.array([5000., 10000., 15000.]) / 204
    n_alphas = len(alphas)
    powers = [2]
    n_jobs = 20
    average = False
    if gpu:
        n_jobs = 9
    pll = Parallel(n_jobs=n_jobs, backend="multiprocessing")
    dell = delayed(mtw_run)
    # dell = delayed(compute_average)
    it = (dell(a, b, p, gpu=gpu, save=True, average=True) for a in alphas for b in betafrs
          for p in powers)
    output = pll(it)

    t = time() - t
    print("=======> TIME: %.2f" % t)
