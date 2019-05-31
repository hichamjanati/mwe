import numpy as np
import os
from plot_brains import plot_source_estimate


# ss = "new9-6-4-12-3-17-18-8-14-19-10-7-13-2-15-11-depth-90"
path = "camcan/mwe-S32/barycenter/"
one_subject = False
perc = 0.1
hemis = ["lh", "rh"]
label_names = ["Brodmann.41", "Brodmann.42"]
label_names = []
one_subject = ""
subject = "CC110033"
lims = [2, 5, 8]
lims = None
plot_all = True
plot_this = "p10-b70-a600000.npy"
# subjects = list(map(int, ss[3:].split('-')[:-2]))
# X, y, ss_ = build_real_dataset(subject_ids=subjects)

# paths = ["fig/realmeg/lasso/sub003/"]


def plot(power, alpha, beta_fr, plot=False):
    p = int(10 * power)
    b = int(100 * beta_fr)
    a = int(100 * alpha)
    fname = "p%d-b%d-a%d.npy" % (p, b, a)
    coefs = np.load(path + fname)
    if plot:
        plot_source_estimate(coefs, hemi="both", blobs=False)
    return coefs


top_vertices = 0
do_all = True
if do_all:
    for r, h, dirs in list(sorted(os.walk(path))):
        n = len(dirs)
        for i, f in enumerate(dirs):
            # print(">> File %d / %d" % (i + 1, n))
            # print(f)
            if not plot_this == f:
                continue
            else:
                print(f)
            if f.split(".")[-1] == "npy":
                if not os.path.exists(r + "/img/"):
                    os.makedirs(r + "/img/")
                coefs = np.load(r + "/" + f)
                if "dspme" in path:
                    coefs = abs(coefs)
                fnamev = r + "/img/"
                if isinstance(one_subject, int):
                    coefs = coefs[:, one_subject: one_subject + 1]
                    fnamev += subject + "/"
                    if not os.path.exists(fnamev):
                        os.makedirs(fnamev)

                for hemi in hemis:
                    fname_h = fnamev + f"{hemi}-" + f[:-4]
                    #
                    if not os.path.exists(fname_h + ".png") or plot_all:
                        plot_source_estimate(coefs, hemi=hemi,
                                             views="lateral",
                                             save_fname=fname_h + ".pdf",
                                             label_names=label_names,
                                             perc=perc,
                                             lims=lims
                                             )
                    # coefss = [coefs[:2485], coefs[2485:]]
                    # for h, coefs in zip(["lh", "rh"], coefss):
                    #     fnamei = fnamem + "-%s.png" % h
                    #     plot_source_estimate(coefs, hemi=h,
                    #                          views="medial",
                    #                          save_fname=fnamei,
                    #                          blobs=blobs)

# signal = np.array([x.dot(t) for x, t in zip(X, coefs.T)])
# residuals = y - signal
# ev_r = mne.EvokedArray(residuals.mean(axis=0)[:, None],
#                        info=info_grad)
#
# ev_r.plot_topomap(show=False)
# plt.savefig(fname)
