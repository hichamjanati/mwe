# brain data
import os
import numpy as np
import mne
from config import get_ave_fname, get_subjects_list
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


dataset = "camcan"
plot_all = False
if os.path.exists("/home/parietal/"):
    server = True
else:
    server = False
    data_path = "~/Dropbox/neuro_transport/code/mtw_experiments"
    data_path = os.path.expanduser(data_path)

subjects = get_subjects_list(dataset)


def read_ave(s, plot=True, dataset=dataset):
    fname = get_ave_fname(dataset, s)
    y = mne.read_evokeds(fname, verbose=False)[0]
    ts = np.linspace(0.0, 0.5, 10)
    if plot:
        y.plot_topomap(ts, "grad", show=False)
    plt.show()
    return y


plot = True
subjects = ["CC120061"]
for s in subjects:
    fname = get_ave_fname(dataset, s)
    y = mne.read_evokeds(fname, verbose=False)[:3]
    y = mne.combine_evoked(y[:3], "nave")
    if plot:
        f2 = y.pick_types("grad").plot(show=False)
        f2.savefig("misc/plots/%s.png" % s)
    y.shift_time(-0.05)
    ts = np.linspace(0.08, 0.12, 10)
    ch, t, amp = y.get_peak("grad", tmin=0.0, tmax=0.15, merge_grads=True,
                            return_amplitude=True)
    print(s, t)
    if plot:
        f = y.plot_topomap(ts, "grad", show=False)
        f.savefig("misc/topomaps/%s.png" % s)
        f = y.plot_topomap(ts, "mag", show=False)
        f.savefig("misc/topomaps/%s-mag.png" % s)
