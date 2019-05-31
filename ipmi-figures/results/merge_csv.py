import pandas as pd
import os
from time import ctime


dataset = "camcan"
if os.path.exists("/home/parietal/"):
    results_path = "/home/parietal/hjanati/csvs/%s/" % dataset
else:
    data_path = "~/Dropbox/neuro_transport/code/mtw_experiments/meg/"
    data_path = os.path.expanduser(data_path)
    results_path = data_path + "results/%s/" % dataset


savedir_names = ["5sources_camcan"]
df = []
for savedir_name in savedir_names:
    datadir = results_path + savedir_name + "/"

    for root, dirs, files in os.walk(datadir):
        for f in files:
            if f.split('.')[-1] == "csv":
                d = pd.read_csv(root + f, header=0, index_col=0)
                df.append(d)

    if len(df):
        df = pd.concat(df, ignore_index=True)
        if savedir_name == "5sources_camcan":
            df["day"] = df["save_time"].apply(lambda x: int(ctime(x / 1e5)[7:10]))
            df = df[(df.model != "mtw") + (df.day != 2)]
            df.loc[df.model == "mtw", "model"] = "mtw"
            df.loc[df.model == "mtw2", "model"] = "mtw"

        df.to_csv("data/%s.csv" % savedir_name)
    else:
        print("No data for %s" % savedir_name)
