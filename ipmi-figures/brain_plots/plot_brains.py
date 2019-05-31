import os
import numpy as np
from surfer import Brain
from mayavi import mlab
import mne
import warnings
from matplotlib import pyplot as plt

dataset = "camcan"
hemi = 'lh'
surface = "white"


def get_data_dir(dataset):
    data_path = "~/Dropbox/neuro_transport/code/"
    data_path += "mtw_experiments/meg/%s/" % dataset
    data_path = os.path.expanduser(data_path)
    subjects_dir = data_path + 'subjects/'
    os.environ['SUBJECTS_DIR'] = subjects_dir

    return data_path


def get_labels(label_names, annot_name="PALS_B12_Brodmann", hemi="lh",
               subjects_dir=None):
    labels_raw = mne.read_labels_from_annot("fsaverage", annot_name,
                                            subjects_dir=subjects_dir)
    labels = []
    for l in labels_raw:
        for name in label_names:
            if name + "-" + hemi in l.name:
                labels.append(l)
    sum_labels = labels[0]
    if len(label_names) > 1:
        for l in labels[1:]:
            sum_labels += l
    return sum_labels


def plot_source_estimate(coefs=None, subject='fsaverage', save_fname="",
                         title=None, surface="inflated", text="", hemi=hemi,
                         background="white", colorbar=True, perc=0.2,
                         views="ventral", dataset=dataset, label_names=[],
                         annot_name="PALS_B12_Brodmann", lims=None,
                         figsize=(800, 800), brain=None):
    data_path = get_data_dir(dataset)
    subjects_dir = data_path + "subjects/"
    foreground = "black"
    if background == "black":
        foreground = "white"
    if title is None:
        title = "Source estimates - %s" % subject

    vert_fname = data_path + "vertno/%s-ico4-filtered-%s-vrt.npy" %\
        (subject, hemi)
    vertices = np.load(vert_fname)
    order = np.argsort(vertices)
    vertices = np.sort(vertices)
    n_features = len(vertices)
    h_index = int(hemi == "rh")
    if h_index:
        verts = [np.array([]), vertices]
        coefs = coefs[-n_features:]
    else:
        verts = [vertices, np.array([])]
        coefs = coefs[:n_features]
    coefs_ = coefs[order].copy()
    m = np.abs(coefs_).max()
    if m <= 0:
        warnings.warn("SourceEstimate all zero for %s " % save_fname)
    if lims is None:
        lims = (0., perc * m, m)
    if coefs_.min() < 0:
        clim = dict(kind="value", pos_lims=lims)
    else:
        clim = dict(kind="value", lims=lims)
    surfer_kwargs = dict(subject=subject, surface=surface,
                         subjects_dir=subjects_dir,
                         views=views, time_unit='s', size=(800, 800),
                         smoothing_steps=6, transparent=True, alpha=0.8,
                         clim=clim, colorbar=colorbar, time_label=None,
                         background=background, foreground=foreground)

    stc = mne.SourceEstimate(data=coefs_.copy(), vertices=verts, tmin=0.17,
                             tstep=0.)
    # return stc, surfer_kwargs
    brain = stc.plot(hemi=hemi, **surfer_kwargs)
    engine = mlab.get_engine()
    module_manager = engine.scenes[0].children[1].children[0].children[0]
    sc_lut_manager = module_manager.scalar_lut_manager
    sc_lut_manager.scalar_bar.number_of_labels = 6
    sc_lut_manager.scalar_bar.label_format = '%.2f'
    if label_names:
        label = get_labels(label_names, annot_name=annot_name, hemi=hemi,
                           subjects_dir=subjects_dir)
        brain.add_label(label, color="green", alpha=0.5, borders=True)

    if save_fname:
        mlab.savefig(save_fname)
        f = mlab.gcf()
        mlab.close(f)

    return brain


def plot_blobs(coefs=None, subject='fsaverage', save_fname="",
               title=None, surface="inflated", text="", hemi=hemi,
               background="white",  views="ventral", top_vertices=None,
               datset=dataset, label_names=[], annot_name="PALS_B12_Brodmann",
               brain=None,
               figsize=(800, 800)):
    data_path = get_data_dir(dataset)
    subjects_dir = data_path + "subjects/"
    foreground = "black"
    if background == "black":
        foreground = "white"
    n_subjects = coefs.shape[-1]
    if top_vertices is None:
        top_vertices = int((abs(coefs) > 0).sum(0).mean())
    if title is None:
        title = "Source estimates - %s" % subject
    vert_fname = data_path + "vertno/%s-ico4-filtered-%s-vrt.npy" %\
        (subject, hemi)
    vertices = np.load(vert_fname)
    n_features = len(vertices)
    h_index = int(hemi == "rh")
    if h_index:
        coefs = abs(coefs[-n_features:])
    else:
        coefs = abs(coefs[:n_features])

    colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_subjects))[::-1, :-1]
    scales = np.linspace(4, 9, n_subjects)[::-1]
    threshs = np.sort(abs(coefs), axis=0)[-top_vertices]
    f = mlab.figure(size=figsize)
    if brain is None:
        brain = Brain(subject, hemi, surface, subjects_dir=subjects_dir,
                      views=views, offscreen=False, background=background,
                      foreground=foreground, figure=f)

    if label_names:
        label = get_labels(label_names, annot_name=annot_name, hemi=hemi,
                           subjects_dir=subjects_dir)
        brain.add_label(label, color="orange", alpha=0.4, borders=False)

    for data, col, s, t in zip(coefs.T, colors, scales, threshs):
        surf = brain.geo[hemi]
        support = np.where(abs(data) >= max(t, 1e-5))[0]
        sources = vertices[support]
        mlab.points3d(surf.x[sources], surf.y[sources],
                      surf.z[sources], color=tuple(col),
                      scale_factor=s, opacity=0.7, transparent=True)

    if save_fname:
        mlab.savefig(save_fname)
        f = mlab.gcf()
        mlab.close(f)

    return brain


if __name__ == "__main__":

    data_path = "~/Dropbox/neuro_transport/code/"
    data_path += "mtw_experiments/meg/%s/" % dataset
    data_path = os.path.expanduser(data_path)
    subject = "fsaverage"
    hemi = "rh"
    labels_type = "gyri"
    spacing = 'ico4'
    labels = np.load(data_path +
                     "label/labels-%s-%s.npy" % (labels_type, hemi))
    n_labels = int(sum(labels[2]))
    coefs = np.zeros((5124, 2))
    coefs[np.where(labels[2])[0]] = np.random.rand(n_labels, 2)
    coefs[np.where(labels[2])[0] + 2562] = np.random.rand(n_labels, 2)
    # coefs[3000:3100, 0] = np.random.randn(100)
    label_names = ["Brodmann.41", "Brodmann.42"]
    if 1:
        b = plot_blobs(coefs, subject=subject, hemi=hemi)
    else:
        b = plot_source_estimate(coefs[:, :1], subject=subject, hemi=hemi,
                                 label_names=label_names, views="lateral",
                                 )
    # b = plot_source_estimate(coefs, subject=subject, hemi="lh")
    #
    # subjects = ["sub%03d" % i for i in range(1, 20)]
    # subjects = ["sub001", "sub017"]
    #
    # for s in subjects:
    #     b = plot_source_estimate(coefs, subject=s,
    #                              hemi="both")
    #
    # coefs = np.zeros((3, 4991, 2))
    # coefs[0, :10] = 2.
    # coefs[1, 3900:4000] = 2.
    # coefs[2, 300:400] = 2.
    #
    # for i, c in enumerate(coefs):
    #     brain = plot_source_estimate(c, hemi="both",
    #                                  save_fname="fig/surfer-%d.png" % i)
