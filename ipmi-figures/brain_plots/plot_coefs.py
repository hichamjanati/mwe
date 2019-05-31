from brain_plots.plot_brains import plot_blobs
from build_data import build_coefs
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


params = {"legend.fontsize": 19,
          "axes.titlesize": 17,
          "axes.labelsize": 16,
          "xtick.labelsize": 14,
          "ytick.labelsize": 14}
plt.rcParams.update(params)

n_subjects = 6
names = [f"Subject {i}" for i in range(1, n_subjects + 1)]
colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_subjects))[::-1, :-1]
legend_models = [Line2D([0], [0], color="w",
                        markerfacecolor=color, markersize=13,
                        linewidth=3,
                        marker="o",
                        label=name)
                 for color, name in zip(colors, names)]
f, ax = plt.subplots(1, 1, figsize=(8, 2))
ax.legend(handles=legend_models, ncol=3, frameon=False,
          labelspacing=1.5)
ax.axis("off")
plt.savefig("brain_plots/coefs_legend.pdf")


dataset = "camcan"
seed = 42
fname = "brain_plots/coefs.pdf"
overlap = 0.5
n_sources = 5
labels = ["S_oc_sup_and_transversal",
          "G_front_inf-Opercular",
          "S_precentral-sup-part",
          "S_front_middle",
          "G_pariet_inf-Supramar"]

coefs = build_coefs(n_tasks=n_subjects, overlap=overlap,
                    n_sources=n_sources, seed=seed,
                    positive=True, labels_type="any",
                    overlap_on="tasks", dataset=dataset)

plot_blobs(coefs, subject='fsaverage', save_fname=fname,
           title=None, surface="inflated", text="", hemi="lh",
           background="white",  views="lateral", top_vertices=10,
           datset=dataset, label_names=labels, annot_name="aparc.a2009s",
           brain=None, figsize=(800, 800))
