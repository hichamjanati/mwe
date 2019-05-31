# brain data
import os
import numpy as np
import mne
from config import (get_subjects_list, get_ave_fname,
                    get_cov_fname, get_params, get_epo_fname)
from joblib import Memory


if os.path.exists("/home/parietal/"):
    location = '~/data/'
else:
    location = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/"
location = os.path.expanduser(location)
memory = Memory(location, verbose=0)

condition = None
hemi = "lrh"
spacing = "ico4"
time_point = "auto"
task = "passive"
ds117_times = np.array([(17, 162), (19, 179), (10, 151), (13, 184),
                        (2, 168), (8, 157), (6, 162), (1, 151),
                        (18, 157), (16, 168), (15, 157), (14, 168),
                        (11, 168), (9, 157), (7, 157), (5, 140),
                        (4, 168), (3, 168), (12, 168)])
ds117_times = dict(ds117_times)

ds117_times_fmri = np.array([(9, 157), (6, 162), (4, 168), (12, 168),
                             (3, 168), (17, 162), (18, 157), (8, 157),
                             (14, 168), (19, 179), (10, 151), (7, 157),
                             (13, 184), (2, 168), (15, 157), (11, 168),
                             (1, 151), (5, 140), (16, 168)])
ds117_fmri = dict(ds117_times_fmri)


def get_whitened_data(dataset, subject, time_point=time_point, task=task):
    params = get_params(dataset)
    data_path = params["data_path"]

    # read data
    evoked_fname = get_ave_fname(dataset, subject, task)
    # cov_fname = get_cov_fname(dataset, subject)
    gain_fname = data_path + "leadfields/X_%s_%s_ico4.npy" % (subject, hemi)
    gain = np.load(gain_fname)
    evoked = mne.read_evokeds(evoked_fname, condition=condition,
                              verbose=False)
    if dataset == "camcan":
        evoked = evoked[:3]
        evoked = mne.combine_evoked(evoked, "nave")
    else:
        evoked = evoked[0]
    epo_fname = get_epo_fname(dataset, subject)
    epochs = mne.read_epochs(epo_fname)
    noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0.05,
                                       method="auto", verbose=True, n_jobs=1,
                                       projs=None)
    # get grad indx
    grad_ind = mne.pick_types(evoked.info, meg="grad")
    gain = gain[grad_ind, :]  # keep grad only
    evoked = evoked.pick_types("grad", eeg=False)
    # ch_names = evoked.info['ch_names']
    # get whitener
    # whitener, _, rank = mne.cov._get_whitener(raw_cov, evoked.info,
    #                                           ch_names, rank=None,
    #                                           pca=False)
    picks = mne.pick_types(evoked.info, meg="grad", eeg=False, eog=False,
                           exclude='bads')
    whitener, _ = mne.cov.compute_whitener(noise_cov, evoked.info,
                                           picks, pca=False)
    if dataset == "camcan":
        evoked.shift_time(-0.05)
    y = evoked.data
    if time_point == "auto":
        _, time_index = evoked.get_peak("grad", tmin=0.05, tmax=0.15,
                                        merge_grads=True, time_as_index=True)
        y = y[:, time_index]
        y = y.flatten()
    elif isinstance(time_point, int):
        t = time_point / 1000  # get time in seconds
        time_index = evoked.time_as_index(t)  # get index of time point
        y = y[:, time_index]
        y = y.flatten()
    X = (evoked.nave) ** 0.5 * whitener.dot(gain)
    y = (evoked.nave) ** 0.5 * whitener.dot(y)
    return X, y, evoked.times[time_index]


def build_ds117_data(ids=None, time_point="auto"):
    Xs, ys, ts = [], [], []
    ss = []
    keys = ds117_times_fmri[ids, 0]
    t = None
    for key in keys:
        subject = "sub%03d" % key
        if time_point == "auto":
            t = int(ds117_fmri[key])
        X, y, t_ = get_whitened_data("ds117", subject, t, task=None)
        Xs.append(X)
        ys.append(y)
        ss.append(key)
        ts.append(t_)
    Xs = np.array(Xs)
    ys = np.array(ys)
    return Xs, ys, ss, ts


def build_real_dataset(dataset, n_tasks=1, subjects=None, time_point="auto",
                       task="passive"):
    if dataset == "ds117":
        if subjects is None:
            subjects = np.arange(n_tasks)
        return build_ds117_data(subjects, time_point)
    Xs, ys, ts = [], [], []
    ss = []
    if subjects is None:
        subjects = get_subjects_list(dataset)[:n_tasks]
    for s in subjects:
        X, y, t = get_whitened_data("camcan", s, time_point, task=task)
        Xs.append(X)
        ys.append(y)
        ss.append(s)
        ts.append(t)
    Xs = np.array(Xs)
    ys = np.array(ys)
    ts = np.array(ts)
    return Xs, ys, ss, ts


build_real_dataset = memory.cache(build_real_dataset)


if __name__ == "__main__":
    dataset = "camcan"
    subjects = get_subjects_list(dataset, simu=False)
    Xs, ys, ss, ts = build_real_dataset("camcan", subjects=subjects)
    # Xs2, ys2, ss2, ts2 = build_real_dataset("ds117", 3)
