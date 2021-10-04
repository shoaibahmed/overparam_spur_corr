import os
import sys
import natsort
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <Results directory>")
    exit()
results_dir = sys.argv[1]
print("Evaluating directory:", results_dir)
result_files = glob(os.path.join(results_dir, "*/test.csv"), recursive=True)
print("Files found:", result_files)

results_dict = {"supcon": {}, "ce": {}}
widths = []
for file in result_files:
    print("Loading file:", file)
    df = pd.read_csv(file, sep=',')
    avg_acc_group = []
    for group in range(4):
        avg_acc_group.append(df[f"avg_acc_group:{group}"].to_numpy()[-1])
    avg_acc = df["avg_acc"].to_numpy()[-1]
    config_name = os.path.split(file)[-2]
    width = config_name.split("_")[-1]
    assert "w" in width
    width = int(width.replace("w", ""))
    if "supcon" in config_name:
        results_dict["supcon"][width] = [avg_acc_group, avg_acc]
    elif "ce" in config_name:
        results_dict["ce"][width] = [avg_acc_group, avg_acc]
        widths.append(width)
    else:
        raise NotImplementedError(f"Unknown config name: {config_name}")
print(results_dict)

sorted_widths = natsort.natsorted(widths)
print("Sorted widths:", sorted_widths)

# Plot the results
fig, ax = plt.subplots()
fig.set_size_inches(8, 5)

line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
cm = plt.get_cmap('rainbow')
NUM_COLORS = 2
# marker_colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
marker_colors = ["tab:red", "tab:green", "tab:blue"]

for idx in range(2):
    if idx == 0:
        relevant_item = "ce"
    elif idx == 1:
        relevant_item = "supcon"
    else:
        raise NotImplementedError
    
    avg_acc_list = [results_dict[relevant_item][width][1] for width in sorted_widths]
    worst_acc_list = [np.min(results_dict[relevant_item][width][0]) for width in sorted_widths]
    print(f"Loss: {relevant_item.upper()} / Average acc list: {avg_acc_list}")
    print(f"Loss: {relevant_item.upper()} / Worst acc list: {worst_acc_list}")
    
    line = plt.plot([int(x) for x in widths], avg_acc_list, linewidth=2., marker=marker_list[idx % len(marker_list)],
                    color=marker_colors[idx], alpha=0.75, markeredgecolor='k', label=f"Avg accuracy ({relevant_item.upper()})")
    line[0].set_color(marker_colors[idx])
    line[0].set_linestyle(line_styles[(idx*2) % len(line_styles)])
    
    line = plt.plot([int(x) for x in widths], worst_acc_list, linewidth=2., marker=marker_list[idx % len(marker_list)],
                    color=marker_colors[idx], alpha=0.75, markeredgecolor='k', label=f"Worst group accuracy ({relevant_item.upper()})")
    line[0].set_color(marker_colors[idx])
    line[0].set_linestyle(line_styles[(idx*2+1) % len(line_styles)])

plt.xlabel('Network Width')
plt.ylabel('Accuracy (%)')
plt.title("Results on ResNet-10 trained using CelebA dataset")
plt.legend()
# plt.ylim(0., 1.)
# plt.xticks(list(range(1, 5)))
plt.tight_layout()

output_file = "results.png"
if output_file is not None:
    plt.savefig(output_file, dpi=300)
plt.show()
plt.close('all')
