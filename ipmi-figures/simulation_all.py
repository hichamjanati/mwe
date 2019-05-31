# brain data
import os
from joblib import Parallel, delayed
import pandas as pd
import pickle

from time import time
import numpy as np

from smtr import STL, Dirty, MLL, utils, MTW
from build_data import build_coefs, build_dataset
from smtr.model_selection import (best_score_dirty,
                                  best_score_stl,
                                  best_score_mll,
                                  best_score_mtw)
device = 1
dataset = "camcan"
savedir_name = "5sources_%s" % dataset
gpu = True
n_splits = 2
cv_size_dirty = 12
mtgl_only = False
cv_size_lasso = 30
cv_size_mtw = 10
cv_size_mll = 40
compute_ot = True
tol_ot = 0.1
positive = False

if os.path.exists("/home/parietal/"):
    if gpu:
        try:
            import cupy as cp
        except ImportError:
            gpu = False
    server = True
    results_path = "/home/parietal/hjanati/csvs/%s/" % dataset
    data_path = "/home/parietal/hjanati/data/"
    plot = False
else:
    server = False
    gpu = False
    data_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/"
    data_path = os.path.expanduser(data_path)
    results_path = data_path + "results/%s/" % dataset
M = np.load(data_path + "%s/metrics/metric_fsaverage_lh.npy" % dataset)
M_emd = np.ascontiguousarray(M.copy() * 100)  # Metric M in cm
n_features = len(M)
seed = 42
std = 0.25
resolution = 4
subject = 'fsaverage%d' % resolution


n_samples = 204
epsilon = 10. / n_features
epsilon_met = 0.
gamma = 1.
depth = 0.9
dirty = Dirty(positive=positive)
mll = MLL(positive=positive)
stl = STL(positive=positive)
sigma0 = 0.01
rw_steps = 5
mtw = MTW(M=M, epsilon=epsilon, gamma=gamma, sigma0=sigma0,
          stable=False, tol_ot=1e-6, maxiter_ot=15, tol=1e-4,
          maxiter=4000, positive=positive, cython=True, gpu=True,
          n_jobs=4
          )
models = [
    (stl, 'STL', dict(cv_size=cv_size_lasso, eps=2e-2),
     best_score_stl),
    (mll, 'MLL', dict(cv_size=cv_size_mll, eps=1e-2),
     best_score_mll),
    (dirty, 'Dirty', dict(cv_size=cv_size_dirty, mtgl_only=mtgl_only, eps=2e-2,
                          do_mtgl=False), best_score_dirty),
    (mtw, 'MTWold', dict(cv_size=cv_size_mtw, eps=0.1,
     alphas=np.array([0., 5., 10., 20., 50., 70., 100.]),
     betas=[0.1, 0.15, 0.2, 0.3, 0.35]), best_score_mtw)
    ]
savedir = results_path + "%s/" % savedir_name
coefsdir = results_path + "%s/coefs/" % savedir_name
cvpathdir = results_path + "%s/cvpath/" % savedir_name

if not os.path.exists(savedir):
    os.makedirs(savedir)
if not os.path.exists(coefsdir):
    os.makedirs(coefsdir)
if not os.path.exists(cvpathdir):
    os.makedirs(cvpathdir)


def one_run(seed, n_tasks, overlap, n_sources, same_design, grad_only,
            power, gamma, labels_type, warmstart=True, dataset=dataset):
    n_samples = 306
    if grad_only:
        n_samples = 204
    assert os.path.exists(savedir)
    M_ = M.copy() ** power
    M_ /= np.median(M_)
    M_ = - M_ / epsilon

    coefs = build_coefs(n_tasks=n_tasks, overlap=overlap,
                        n_sources=n_sources, seed=seed,
                        positive=positive, labels_type=labels_type,
                        overlap_on="tasks", dataset=dataset)
    assert abs(coefs).max(axis=0).all()
    ot_params = {"M": M_emd, "epsilon": epsilon_met, "compute_ot": compute_ot}
    X, y = build_dataset(coefs, std=std, same_design=same_design,
                         seed=seed, grad_only=grad_only,
                         randomize_subjects=False, dataset=dataset)
    Xs = X.reshape(n_tasks, n_samples, -1)
    Ys = y.reshape(n_tasks, n_samples)
    norms = np.linalg.norm(Xs, axis=1) ** depth
    scaling = norms.T * 1e4
    X_scaled = Xs / norms[:, None, :]
    auc, ot, mse = dict(), dict(), dict()
    aucabs, otabs = dict(), dict()
    coefs_dict = dict(truth=coefs, scaling=scaling)
    cvpath_dict = dict()
    t0 = time()
    for model, name, cv_params, best_score_model in models:
        print("Doing %s ..." % name)
        if "mtw" in name.lower():
            model.gamma = gamma
            model.M = M_
            cv_params["alphas"] *= n_tasks / 4
            try:
                cp.cuda.Device(device).use()
            except:
                pass
        t = time()
        bscores, scores, bc, bp, _, ac = \
            best_score_model(model, X_scaled, Ys,  coefs, warmstart=warmstart,
                             scaling_vector=scaling,  **cv_params, **ot_params)
        print(bp)
        cvpath_dict[name.lower()] = ac
        coefs_dict[name.lower()] = bc
        coefs_pred = bc['auc']
        model.coefs_ = coefs_pred.copy()
        auc[name.lower()] = bscores['auc']
        ot[name.lower()] = - bscores['ot']
        mse[name.lower()] = - bscores['mse']
        aucabs[name.lower()] = bscores['aucabs']
        otabs[name.lower()] = - bscores['otabs']

        t = time() - t
        print("Time %s : %f, n_tasks = %d" % (name, t, n_tasks))
        if "mtw" in name.lower():
            print("Best for MTW", bp)
    x_auc, x_ot, x_mse, names = [], [], [], []
    x_aucabs, x_otabs = [], []
    for name, v in auc.items():
        names.append(name)
        x_auc.append(v)
        x_ot.append(ot[name])
        x_mse.append(mse[name])
        x_aucabs.append(aucabs[name])
        x_otabs.append(otabs[name])
    t0 = time() - t0
    data = pd.DataFrame(x_auc, columns=["auc"])
    data["ot"] = x_ot
    data["mse"] = x_mse
    data["aucabs"] = x_aucabs
    data["otabs"] = x_otabs
    data["model"] = names
    data["computation_time"] = t0
    if "mtw" in name.lower():
        data["t_ot"] = model.t_ot
        data["t_cd"] = model.t_cd
        data["alpha_auc"] = bp["auc"]["alpha"] * n_samples
        data["beta_auc"] = bp["auc"]["beta"] / model.betamax
        data["alpha_ot"] = bp["ot"]["alpha"] * n_samples
        data["beta_ot"] = bp["ot"]["beta"] / model.betamax
        data["conco"] = sigma0 > 0
        data["steps"] = rw_steps

        coefs_dict["scores"] = scores

    t = int(1e5 * time())

    coefs_fname = coefsdir + "coefs_%s_%s.pkl" % (name.lower(), t)
    cvpath_fname = cvpathdir + "cvpath_%s_%s.pkl" % (name.lower(), t)

    settings = [("subject", subject), ("n_tasks", n_tasks),
                ("overlap", overlap), ("std", std), ("seed", seed),
                ("epsilon", epsilon * n_features), ("gamma", gamma),
                ("cv_size_mtw", cv_size_mtw), ("cv_size_stl", cv_size_lasso),
                ("cv_size_dirty", cv_size_dirty), ("same_design", same_design),
                ("n_features", coefs.shape[0]), ("grad_only", grad_only),
                ("power", power), ("n_sources", n_sources),
                ("label_type", labels_type), ("coefspath", coefs_fname),
                ("save_time", t), ("cvpath", cvpath_fname),
                ("warmstart", warmstart), ("depth", depth)]
    coefs_dict["settings"] = dict(settings)
    for var_name, var_value in settings:
        data[var_name] = var_value

    # with open(coefs_fname, "wb") as ff:
    #     pickle.dump(coefs_dict, ff)
    # with open(cvpath_fname, "wb") as ff:
    #     pickle.dump(cvpath_dict, ff)
    print("One worker out: \n", data)
    data_name = "results_%d" % t + ".csv"
    data.to_csv(savedir + data_name)
    return 0.


def wrapper(seed, n_tasks, overlap, n_sources, same_design, grad_only,
            power, gamma, labels_type, warmstart=True):
    x = one_run(seed, n_tasks, overlap, n_sources, same_design, grad_only,
                power, gamma, labels_type, warmstart=warmstart)
    try:
        utils.free_gpu_memory(cp)
    except:
        pass
    return x


if __name__ == "__main__":
    t0 = time()
    seed = 42
    rnd = np.random.RandomState(seed)
    n_repeats = 30
    seeds = rnd.randint(100000000, size=n_repeats)
    overlaps = [0.5]
    n_tasks = [2, 4, 8, 16, 24, 32]

    n_sources = [5]
    same_design = [False, True]
    grad_only = [True]
    powers = [1.]
    gammas = [1.]
    types = ["any"]
    ws = [False]
    seeds_points = [(s, k, o, n, d, g, p, ga, lt, w)
                    for n in n_sources for d in same_design
                    for g in grad_only for o in overlaps for lt in types
                    for p in powers for ga in gammas for s in seeds
                    for w in ws for k in n_tasks]
    parallel = Parallel(n_jobs=20, backend="multiprocessing")
    # parallel = Parallel(n_jobs=1)
    iterator = (delayed(wrapper)(s, k, o, n, d, g, p, ga, lt, w)
                for s, k, o, n, d, g, p, ga, lt, w in seeds_points)
    out = parallel(iterator)
    print('================================' +
          'FULL TIME = %d' % (time() - t0))
