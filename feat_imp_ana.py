import os
import pickle
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

infile = open("log_perf.txt")

perf_dict = {}
best_perf = 0.0
best_perf_id = ""
main_class = ""

vis_dict = {}

rename_dict = [["16","6_pos_min"],
            ["15","5_pos_min"],
            ["14","4_pos_min"],
            ["13","3_pos_min"],
            ["12","2_pos_min"],
            ["11","1_pos_min"],
            ["10","6_pos"],
            ["9","5_pos"],
            ["8","4_pos"],
            ["7","3_pos"],
            ["6","2_pos"],
            ["5","1_pos"],
            ["4","1_to_min_20"],
            ["3","1_to_min_10"],
            ["2","1_to_10"],
            ["1","1_to_min_5"],
            ["0","1_to_5"]]
rename_dict = dict(rename_dict)

for line in infile:
	if line.startswith("="): continue
	if line.startswith("DONE!"):
		#Do analysis
		best_feat = list(perf_dict[str(int(best_perf_id)-1)].keys())
		for bf in best_feat:
			if main_class in bf:
				try: 
					bf = bf.split("|")[1]
					bf = rename_dict[bf]
				except: bf = "average"
			if main_class in vis_dict.keys():
				vis_dict[main_class][bf] = 1
			else:
				vis_dict[main_class] = {}
				vis_dict[main_class][bf] = 1
		print(pd.DataFrame(vis_dict).sum(axis=1).sort_values().index)
		
		#raw_input()
		perf_dict = {}
		best_perf_id = ""
		best_perf = 0.0
		continue
	split_line = line.rstrip().split(" ")
	if len(split_line) == 1:
		key_part = split_line[0]
		perf_dict[key_part] = {}
		if key_part == "28":
			main_class = "GET_MAIN_CLASS"
		continue
		
	if float(split_line[1]) > best_perf:
		best_perf = float(split_line[1])
		print(best_perf,key_part)
		
		best_perf_id = key_part
		
	if main_class == "GET_MAIN_CLASS" and "|" in split_line[0]:
		main_class = split_line[0].split("|")[0]
		
	perf_dict[key_part][split_line[0]] = float(split_line[1])

df_vis = pd.DataFrame(vis_dict)
df_vis.fillna(0,inplace=True)

cg = sns.clustermap(df_vis.loc[df_vis.sum(axis=1).sort_values().index],row_cluster=False,cmap="bwr")
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
#plt.tight_layout()
plt.show()
#for file in os.listdir("C:/Users/asus/Dropbox/secretome/kaggle/"):
#	if not file.endswith(".pickle"): continue
#	mod = pickle.load(open(file,"rb"))
#	print(mod.feats)