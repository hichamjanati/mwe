import numpy as np
import os
import mne

from time import time
from surfer import Brain
from mayavi import mlab
from smtr.estimators.solvers.solver_mtw import barycenterkl_log
from matplotlib import pyplot as plt
from smtr.utils import mesh_all_distances
from matplotlib.lines import Line2D

params = {"legend.fontsize": 20,
          "axes.titlesize": 17,
          "axes.labelsize": 22,
          "xtick.labelsize": 20,
          "ytick.labelsize": 20,
          "pdf.fonttype": 42}
plt.rcParams.update(params)

data_path = "~/Dropbox/neuro_transport/code/mtw_experiments/ds117_old/"
data_path = os.path.expanduser(data_path)
subjects_dir = data_path + "subjects/"

compute_M = False
subject = "fsaverage"
hemi = "lh"
labels_type = "gyri"
spacing = 'ico4'
surface = "inflated"

labels_raw = mne.read_labels_from_annot("fsaverage", "aparc",
                                        subjects_dir=subjects_dir)
label = labels_raw[4]
f = mlab.figure(size=(600, 600))
brain = Brain(subject, 'lh', surface, subjects_dir=subjects_dir, figure=f,
              background="white", foreground='black')
surf = brain.geo['lh']
vertices = label.vertices
n_features = len(vertices)
points_ = np.stack((surf.x, surf.y, surf.z)).T
if compute_M:
    M = mesh_all_distances(points_, surf.faces, verts=vertices)
    np.save(data_path + "metrics/metric_%s.npy" % label.name, M)
else:
    M = np.load(data_path + "metrics/metric_%s.npy" % label.name)
M **= 2
M /= np.median(M)

points = points_[vertices]
coefs = np.zeros((n_features, 2))
verts1 = np.argsort(points[:, 2])[-40:]
verts0 = np.where((points[:, 2] < 25) * (points[:, 1] < 43))[0]

coefs[verts0, 0] = 10
coefs[verts1, 1] = 10
focal_point = points[121]
views = dict(azimuth=150, elevation=66, distance=440,
             focalpoint=np.array([30., 10., -10.]), roll=100.0)
gamma = 10.
epsilon = 1. / n_features
K = - M / epsilon
ws = np.arange(1, 5) / 5
bars = np.zeros((n_features, len(ws) + 2))
bars[:, 0] = coefs[:, 0]
bars[:, -1] = coefs[:, 1]

vs = [np.sort(vertices), np.array([])]
colors = plt.cm.RdBu(np.linspace(0, 1, 6))[:, :-1]

invisible = - np.ones(len(surf.x))
invisible[0] = 1.

for i, w in enumerate(ws):
    weights = np.array([1 - w, w])
    t = time()
    _, log, _, _, q = barycenterkl_log(coefs, K, epsilon, gamma,
                                       tol=0, maxiter=100,
                                       weights=weights)
    q[q < 1] = 0.
    bars[:, i + 1] = q

for b, col in zip(bars.T, colors):
    brain.add_data(b[b > 0.01],
                   vertices=np.sort(vertices[np.where(b > 0.01)[0]]),
                   transparent=True, colormap=[col], colorbar=False,
                   smoothing_steps=3, min=0, max=1)

brain.add_data(invisible[:, None], colormap="RdBu", min=0., max=1.,
               vertices=np.arange(len(surf.x)), alpha=1.,
               transparent=False, time_label=None, thresh=0., colorbar=False)
brain.show_view(views)
sv = "fig/interpol.pdf"
mlab.savefig(sv)

# plot legend
weights = np.linspace(0., 1., len(colors))
legend_weights = [Line2D([0], [0], color="w", marker="o",
                         markerfacecolor=col, markersize=15,
                         label="t = %.1f" % w)
                  for col, w in zip(colors, weights)]
f = plt.figure(figsize=(2.3 * len(colors), 1))
f.legend(handles=legend_weights, ncol=len(colors))
f.set_tight_layout(True)
f.savefig("fig/interpol_legend.pdf")
