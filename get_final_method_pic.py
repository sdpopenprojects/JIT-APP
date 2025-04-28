import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_paths = [
    "D:/xyw/SZZM/result_method/shap_pic/RF_all_shap_plot.png",
    "D:/xyw/SZZM/result_method/shap_pic/DT_all_shap_plot.png",
    "D:/xyw/SZZM/result_method/shap_pic/LR_all_shap_plot.png",
    "D:/xyw/SZZM/result_method/shap_pic/NB_all_shap_plot.png",
    "D:/xyw/SZZM/result_method/shap_pic/GBM_all_shap_plot.png"
]

labels = ['RF', 'DT', 'LR', 'NB', 'GBM']

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))
axes = axes.flatten()

for i in range(3):
    img = mpimg.imread(image_paths[i])
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].annotate(
        labels[i],
        xy=(0.5, -0.08),
        xycoords='axes fraction',
        fontsize=20,
        ha='center',
        va='top'
    )

for i in range(2):
    img = mpimg.imread(image_paths[i + 3])
    ax = axes[3 + i]
    ax.imshow(img)
    ax.axis('off')
    ax.annotate(
        labels[i + 3],
        xy=(0.5, -0.08),
        xycoords='axes fraction',
        fontsize=20,
        ha='center',
        va='top'
    )

axes[5].axis('off')

fig.subplots_adjust(hspace=0.3, wspace=0.1)
plt.tight_layout()
plt.savefig("combined_method_shap_plots.png", dpi=600)
# plt.savefig("combined_method_shap_plots.pdf", bbox_inches='tight')
