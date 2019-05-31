import numpy as np
import os
import mne

from surfer import Brain
from mayavi import mlab
from smtr.estimators.solvers.solver_mtw import barycenterkl_log


data_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/camcan/"
data_path = os.path.expanduser(data_path)
subjects_dir = data_path + "subjects/"

subject = "fsaverage"
hemi = "lh"
labels_type = "gyri"
spacing = 'ico4'
surface = "inflated"

M = np.load(data_path + "metrics/metric_%s_%s.npy" % (subject, hemi))
M **= 2
M /= np.median(M)

labels_raw = mne.read_labels_from_annot("fsaverage", "aparc",
                                        subjects_dir=subjects_dir)
label = labels_raw[4].morph(subject_to=subject, grade=4)
label2 = labels_raw[62].morph(subject_to=subject, grade=4)

seed = 42
rnd = np.random.RandomState(seed)
f = mlab.figure(size=(600, 600))
brain = Brain(subject, 'lh', surface, subjects_dir=subjects_dir, figure=f,
              background="white", foreground='black')
surf = brain.geo['lh']
vertices1 = label.vertices
vertices2 = label2.vertices

n_features = len(M)
points_ = np.stack((surf.x, surf.y, surf.z)).T

points1 = points_[vertices1]
points2 = points_[vertices2]
coefs = np.zeros((n_features, 3))
verts0 = np.where((points1[:, 2] < 22) * (points1[:, 1] > 50))[0]
verts1 = np.where((points1[:, 2] > 42) & (points1[:, 1] > 43))[0]
verts2 = np.where((points2[:, 2] > 15) & (points2[:, 1] > -12))[0]

coefs[vertices1[verts0], 0] = abs(rnd.randn(len(verts0)) + 2)
coefs[vertices1[verts1], 1] = abs(rnd.randn(len(verts1)) + 2)
coefs[vertices2[verts2], 2] = abs(rnd.randn(len(verts2)) + 2)

gamma = 20.
epsilon = 5 / n_features
K = - M / epsilon

_, log, _, _, q = barycenterkl_log(coefs, K, epsilon, gamma,
                                   tol=0, maxiter=800)

q[q < 0.05 * q.max()] = 0.

for co in coefs.T:
    brain.add_data(co, vertices=np.arange(n_features),
                   transparent=True, colormap="Greens", colorbar=False,
                   smoothing_steps=15, min=0)

brain.add_data(q, vertices=np.arange(n_features),
               transparent=True, colormap="Reds", colorbar=False,
               smoothing_steps=10, min=0)
view = dict(azimuth=150, elevation=56, distance=440,
            focalpoint=np.array([0., 20., 0.]), roll=95.0)
brain.show_view(view)
sv = "fig/bar.pdf"
mlab.savefig(sv)
