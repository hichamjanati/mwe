import pandas as pd
from matplotlib import pyplot as plt
import seaborn.apionly as sns
from matplotlib.lines import Line2D
from matplotlib import ticker


dataset = "camcan"
name = "5sources_%s" % dataset
path = "data/"
df = pd.read_csv(path + "%s.csv" % name, index_col=0)
params = {"legend.fontsize": 16,
          "axes.titlesize": 16,
          "axes.labelsize": 16,
          "xtick.labelsize": 14,
          "ytick.labelsize": 14,
          "pdf.fonttype": 42}
plt.rcParams.update(params)
sns.set_context("paper", rc=params)

xlabel = "# subjects"
metrics = ["auc", "aucabs", "ot", "otabs", "mse"]
tick_bases = [0.1, 0.1, 1, 1, 0.5]
ylabels = ["AUC", "AUC(|.|)", "EMD in cm", "EMD(|.|) in cm", "MSE"]
titles = ["Same Leadfield", "Different Leadfields"]
models = ["mtw", "mtwold", "mll", "dirty", "grouplasso", "stl", "truth-mean", "adastl", "recmtw"]

model_names = ["MWE", "MTW", "MLL", "Dirty", "GL", "Lasso", "Truth-mean", "AdaSTL", "RecMTW"]
colors = ["forestgreen", "black", "indianred", "gold", "cornflowerblue",
          "purple", "cyan", "magenta", "yellow"]
ms = ["*", "o", "s", "^", "P", "D", "*", "o", "^"]
ls = ["-", "--", "-.", ":", "-", "-", "--", ":", "-."]
legend_models = [Line2D([0], [0], color=color, marker=m,
                        markerfacecolor=color, markersize=13,
                        linewidth=3,
                        linestyle=ll,
                        label=name)
                 for color, name, m, ll in zip(colors, model_names, ms, ls)]
ids = [0, 2, -1]
f, axes = plt.subplots(3, 2, sharey="row", sharex=True, figsize=(10, 7))

df.model = df.model.replace(to_replace=models, value=model_names)
df.loc[df['model'] == "Truth-mean", "same_design"] = True
tm = df[df.model == "Truth-mean"]
tm.same_design = False
df = df.append(tm)

remtw = df[df.model == "RecMTW"].copy()
remtw.same_design = True
df = df.append(remtw)
for i, (axrow, metric_id) in enumerate(zip(axes, ids)):
    metric = metrics[metric_id]
    met_name = ylabels[metric_id]
    for ax, b, ti in zip(axrow.ravel(), [True, False], titles):
        sns.pointplot(y=metric, x="n_tasks", hue="model",
                      hue_order=model_names[::1],
                      palette=colors[::1],
                      markers=ms[::1],
                      linestyles=ls[::1],
                      data=df[df.same_design == b], ax=ax)
        if b:
            ax.set_ylabel(met_name)
        else:
            ax.set_ylabel("")
        if i == 0:
            ax.set_title(ti)
        else:
            ax.set_title("")
        if i == 2:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(-3, -3))
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel("")
        ax.grid(True)
        if i == 0 and not b:
            ax.legend(handles=legend_models, frameon=False,
                      ncol=1, bbox_to_anchor=[1.05, 0.5],
                      labelspacing=1.7,
                      columnspacing=0.5)
        else:
            ax.get_legend().remove()

        if i < 2:
            formatter = ticker.MultipleLocator(tick_bases[metric_id])
            ax.yaxis.set_major_locator(formatter)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())


# f.set_tight_layout(True)
# plt.show()
plt.savefig("fig/%s-anatomies-truth-mean.pdf" % dataset, bbox_inches="tight")

# for metric, yl in zip(metrics, ylabels):
#     f, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
#     for ax, b, ti in zip(axes.ravel(), [True, False], titles):
#         sns.pointplot(y=metric, x="n_tasks", hue="model",
#                       hue_order=model_names,
#                       palette=colors,
#                       data=df[df.same_design == b], ax=ax)
#         if "auc" in metric:
#             ax.set_ylim([0.2, 1])
#         if "mse" in metric:
#             ax.set_title(ti)
#         if b:
#             ax.set_ylabel(yl)
#         else:
#             ax.set_ylabel("")
#         if "ot" in metric:
#             ax.set_xlabel(xlabel)
#         else:
#             ax.set_xlabel("")
#         ax.grid('on')
#         if not b and "mse" in metric:
#             ax.legend(ncol=1, bbox_to_anchor=[1., 1.05], labelspacing=1.,
#                       columnspacing=None)
#         else:
#             ax.get_legend().remove()
#     f.set_tight_layout(True)
#     plt.savefig("fig/ipmi/anatomies-%s.pdf" % metric, bbox_inches="tight")
