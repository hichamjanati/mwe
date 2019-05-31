# brain data
import os
from joblib import Parallel, delayed
import pandas as pd

from time import time
import numpy as np
from smtr import STL
from build_data import build_coefs, build_dataset


dataset = "camcan"
savedir_name = "5sources_%s" % dataset
compute_ot = True
positive = False

if os.path.exists("/home/parietal/"):
    results_path = "/home/parietal/hjanati/csvs/%s/" % dataset
    data_path = "/home/parietal/hjanati/data/"
else:
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
model = STL(positive=positive)

savedir = results_path + "%s/" % savedir_name

if not os.path.exists(savedir):
    os.makedirs(savedir)


def sum_coefs(seed, n_tasks, overlap, n_sources, power,
              labels_type, dataset=dataset):
    t0 = time()
    assert os.path.exists(savedir)
    M_ = M.copy() ** power
    M_ /= np.median(M_)
    M_ = - M_ / epsilon
    auc, ot, mse = dict(), dict(), dict()
    aucabs, otabs = dict(), dict()
    coefs = build_coefs(n_tasks=n_tasks, overlap=overlap,
                        n_sources=n_sources, seed=seed,
                        positive=positive, labels_type=labels_type,
                        overlap_on="tasks", dataset=dataset)
    coefs_dict = dict(truth=coefs, scaling=None)

    coefs_mean = coefs.mean(axis=1)
    coefs_ = np.ones_like(coefs) * coefs_mean[:, None]
    assert abs(coefs).max(axis=0).all()
    ot_params = {"M": M_emd, "epsilon": epsilon_met, "compute_ot": compute_ot}
    model.coefs_ = coefs_
    name = "truth-mean"
    bscores = model.score_coefs(coefs, **ot_params)
    auc[name.lower()] = bscores['auc']
    ot[name.lower()] = - bscores['ot']
    mse[name.lower()] = - bscores['mse']
    aucabs[name.lower()] = bscores['aucabs']
    otabs[name.lower()] = - bscores['otabs']

    x_auc, x_ot, x_mse, names = [], [], [], []
    x_aucabs, x_otabs = [], []
    for name, v in auc.items():
        names.append(name)
        x_auc.append(v)
        x_ot.append(ot[name])
        x_mse.append(mse[name])
        x_aucabs.append(aucabs[name])
        x_otabs.append(otabs[name])
    data = pd.DataFrame(x_auc, columns=["auc"])
    data["ot"] = x_ot
    data["mse"] = x_mse
    data["aucabs"] = x_aucabs
    data["otabs"] = x_otabs
    data["model"] = names
    data["computation_time"] = t0

    t = int(1e5 * time())

    settings = [("subject", subject), ("n_tasks", n_tasks),
                ("overlap", overlap), ("std", std), ("seed", seed),
                ("epsilon", epsilon * n_features), ("gamma", gamma),
                ("n_features", coefs.shape[0]),
                ("power", power), ("n_sources", n_sources),
                ("label_type", labels_type)
                ("save_time", t)]
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


def wrapper(seed, n_tasks, overlap, n_sources, power, labels_type):
    x = sum_coefs(seed, n_tasks, overlap, n_sources, power, labels_type)
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
    powers = [1.]
    types = ["any"]
    seeds_points = [(s, k, o, n, p, lt)
                    for n in n_sources for o in overlaps for lt in types
                    for p in powers for s in seeds for k in n_tasks]
    parallel = Parallel(n_jobs=50, backend="multiprocessing")
    # parallel = Parallel(n_jobs=1)
    iterator = (delayed(wrapper)(s, k, o, n, p, lt)
                for s, k, o, n, p, lt in seeds_points)
    out = parallel(iterator)
    print('================================' +
          'FULL TIME = %d' % (time() - t0))
