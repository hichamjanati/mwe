import numpy as np
import os
import mne

from surfer import Brain
from mayavi import mlab
from ot import emd2


data_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/camcan/"
data_path = os.path.expanduser(data_path)
subjects_dir = data_path + "subjects/"

subject = "fsaverage"
hemi = "lh"
labels_type = "gyri"
spacing = 'ico4'
surface = "white"

M = np.load(data_path + "metrics/metric_%s_%s.npy" % (subject, hemi))
M *= 100
M = np.ascontiguousarray(M)
n_features = len(M)

labels_raw = mne.read_labels_from_annot("fsaverage", "aparc",
                                        subjects_dir=subjects_dir)
nv = 5
label = labels_raw[4]
sources = label.vertices[20: 20 + nv]
x = np.zeros(n_features)
x[sources] = 1 / nv
distances = np.zeros(n_features)
Y = np.eye(n_features)
for v in range(n_features):
    distances[v] = emd2(x, Y[v], M)

src_fname = data_path + "bem/%s-ico4-morphed-src.fif" % subject
src = mne.read_source_spaces(src_fname)[int(hemi == "rh")]
faces = src["use_tris"]
points = src["rr"][:n_features]
f = mlab.figure(size=(700, 600))
brain = Brain(subject, 'lh', surface, subjects_dir=subjects_dir, figure=f,
              background="white", foreground='black')
brain.add_data(distances, vertices=np.arange(n_features),
               hemi=hemi, transparent=True, smoothing_steps=2,
               colormap="RdBu", alpha=1, mid=12)
surf = brain.geo[hemi]
brain.add_data(x, vertices=np.arange(n_features),
               hemi=hemi, transparent=True, smoothing_steps=30,
               colormap="Greens", alpha=1, colorbar=False)
brain.add_text(0.9, 0.95, "cm", "unit", font_size=25)
engine = mlab.get_engine()
module_manager = engine.scenes[0].children[1].children[0].children[0]
sc_lut_manager = module_manager.scalar_lut_manager
sc_lut_manager.scalar_bar.number_of_labels = 3
sc_lut_manager.scalar_bar.label_format = '%1.0f'
sc_lut_manager.scalar_bar.unconstrained_font_size = True
sc_lut_manager.label_text_property.font_size = 34
sc_lut_manager.scalar_bar.orientation = "vertical"
sc_lut_manager.scalar_bar_representation.position = (0.9, 0.01)
sc_lut_manager.scalar_bar_representation.position2 = (0.05, 0.9)

lat = dict(azimuth=165.0, elevation=60.0, distance=334.75,
           focalpoint=np.array([-31.92, -30.81, 16.36]), roll=90.0)
med = dict(azimuth=0.0, elevation=90.0, distance=334.75,
           focalpoint=np.array([-31.92, -18.81, 16.36]), roll=-90.0)
vnames = ["lateral", "medial"]
for v, vname in zip([lat, med], vnames):
    brain.show_view(v)
    sv = "fig/ot-distance-%s.eps" % vname
    mlab.savefig(sv)
    sc_lut_manager.scalar_bar_representation.visibility = 0.
    brain.texts_dict["unit"]["text"].visible = False
# t = time() - t
# print("time = %f" % t)
# bars.append(q[:, None])
# sv = "fig/interpol/interpol_%d.pdf" % int(100 * w)
# b = plot_source_estimate(q[:, None], hemi="lh", views=views,
#                          colorbar=False, save_fname=sv)

# for i, c in enumerate(coefs.T):
#     sv = "fig/interpol/interpol_%d.pdf" % int(100 * (1 - i))
#     b = plot_source_estimate(c[:, None], hemi="lh", views=views,
#                              colorbar=False, save_fname=sv)
