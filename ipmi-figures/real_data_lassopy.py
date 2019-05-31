import numpy as np
import os

from smtr import STL, utils, AdaSTL
from whiten_data import build_real_dataset
from joblib import delayed, Parallel
from smtr.estimators.solvers.solver_mtw import barycenterkl, barycenterkl_log
import warnings
import pickle
from config import get_params, get_subjects_list


if os.path.exists("/home/parietal/"):
    try:
        import cupy as cp
        device = 1
    except:
        pass

dataset = "camcan"
params = get_params(dataset)
subjects = get_subjects_list(dataset, simu=False)
subjects_dir = params["subjects_dir"]
data_path = params["data_path"]
M_fname = data_path + "metrics/metric_fsaverage_lrh.npy"
M_ = np.load(M_fname)

alpha_frs = [0.3, 0.4, 0.5, 0.6]
depth = 0.9
n_features = len(M_)
epsilon = 10. / n_features
gamma = 1.


def get_dirs(dataset, method):
    ss_dir = "brain_plots/%s/%s/" % (dataset, method)
    bar_path = ss_dir + "barycenter/"
    log_path = ss_dir + "log/"
    if not os.path.exists(bar_path):
        os.makedirs(bar_path)
        os.makedirs(bar_path + "img/")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return ss_dir, bar_path, log_path


def run_lasso(subject, alpha_fr, model="lasso"):
    ss_dir, bar_path, log_path = get_dirs(dataset, model)
    X, y, ss, ts = build_real_dataset(dataset, subjects=[subject])
    coefs_path = ss_dir + "%s/" % subject
    try:
        if not os.path.exists(coefs_path):
            os.makedirs(coefs_path)
    except FileExistsError:
        pass

    norms = np.linalg.norm(X, axis=1) ** depth
    X_scaled = X / norms[:, None, :]
    alphamax = max([abs(x.T.dot(yy)).max() for (x, yy) in zip(X_scaled, y)])
    alphamax /= y.shape[-1]
    alpha = alpha_fr * alphamax
    if model == "lasso":
        model = STL
    else:
        model = AdaSTL
    stl = model(alpha=[alpha], positive=False)
    stl.fit(X_scaled, y)
    coefs_stl = stl.coefs_ * 1e9 / norms.T
    alpha = int(1000 * alpha_fr)
    coefs_fname = coefs_path + "alpha-%s.npy" % alpha
    np.save(coefs_fname, coefs_stl)
    return 0.


def lasso_average(alpha_fr, power=1, model="lasso"):
    ss_dir, bar_path, log_path = get_dirs(dataset, model)
    alpha = int(1000 * alpha_fr)
    coefs = []
    for subj in subjects:
        f = ss_dir + "%s/alpha-%s.npy" % (subj, alpha)
        c = np.load(f)
        coefs.append(c)
    coefs = np.concatenate(coefs, axis=-1)
    coefs_mean = coefs.mean(axis=-1)[:, None]
    f = bar_path + "euclidean-alpha-%s.bpy" % alpha
    np.save(f, coefs_mean)

    log1, log2, logl1, logl2 = {"cstr": []}, {"cstr": []}, {"cstr": []}, \
        {"cstr": []}

    M = M_.copy() ** power  # convert to mm
    M /= np.median(M)
    M = - M / epsilon
    with cp.cuda.Device(device):
        M = cp.asarray(M)
        coefs1, coefs2 = utils.get_unsigned(coefs)
        fot1, log1, _, b1, bar1 = barycenterkl(coefs1 + 1e-100, M, epsilon,
                                               gamma, tol=1e-7,
                                               maxiter=5000)
        utils.free_gpu_memory(cp)

        if fot1 is None or not coefs1.max(0).all():
            warnings.warn("""Nan found when averagin, re-fit in
                             log-domain.""")
            b1 = cp.log(b1 + 1e-100, out=b1)
            fot1, logl1, m1, b1, bar1 = \
                barycenterkl_log(coefs1, M, epsilon, gamma,
                                 b=b1, tol=1e-5, maxiter=1000)
            utils.free_gpu_memory(cp)

        fot2, log2, _, b2, bar2 = barycenterkl(coefs2 + 1e-100, M, epsilon,
                                               gamma, tol=1e-7,
                                               maxiter=5000)
        utils.free_gpu_memory(cp)

        if fot2 is None or not coefs2.max(0).all():
            warnings.warn("""Nan found when averagin, re-fit in
                             log-domain.""")
            b2 = cp.log(b2 + 1e-100, out=b2)
            fot2, logl2, m2, b2, bar2 = \
                barycenterkl_log(coefs2, M, epsilon, gamma,
                                 b=b2, tol=1e-5, maxiter=1000)
            utils.free_gpu_memory(cp)

        bar = bar1 - bar2
    bar = bar[:, None]
    fname = "ot-alpha-%s.npy" % alpha
    np.save(bar_path + fname, bar)
    fname = "alpha-%s.pkl" % alpha
    logs = [log1["cstr"], log2["cstr"], logl1["cstr"], logl2["cstr"]]
    with open(log_path + fname, "wb") as ff:
        pickle.dump(logs, ff)
    print(">> LEAVING worker alpha", alpha)
    return 0.


args = [(s, a, m) for s in subjects for a in alpha_frs
        for m in ["lasso"]]
pll = Parallel(n_jobs=50)
dell = delayed(run_lasso)
it = (dell(i, a, m) for i, a, m in args)
out = pll(it)

# pll = Parallel(n_jobs=4)
# it = (delayed(lasso_average)(a, 1, m) for a in alpha_frs
#       for m in ["lasso", "adastl"])
# out = pll(it)
