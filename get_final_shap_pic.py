import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_paths = [
    "D:/xyw/SZZM/result_change/shap_pic/RF_all_shap_plot.png",
    "D:/xyw/SZZM/result_change/shap_pic/KNN_all_shap_plot.png",
    "D:/xyw/SZZM/result_change/shap_pic/DT_all_shap_plot.png",
    "D:/xyw/SZZM/result_change/shap_pic/LR_all_shap_plot.png",
    "D:/xyw/SZZM/result_change/shap_pic/NB_all_shap_plot.png",
    "D:/xyw/SZZM/result_change/shap_pic/GBM_all_shap_plot.png"
]

labels = ['RF', 'KNN', 'DT', 'LR', 'NB', 'GBM']

fig, axes = plt.subplots(nrows=3, ncols=2,figsize=(12, 18))
axes = axes.flatten()

for i, (path, label) in enumerate(zip(image_paths, labels)):
    img = mpimg.imread(path)
    axes[i].imshow(img)
    axes[i].axis('off')
    # axes[i].text(10, 30, label, fontsize=8,  color='black', backgroundcolor='white')
    axes[i].annotate(
        label,
        xy=(0.5, -0.08),
        xycoords='axes fraction',
        fontsize=20,
        ha='center',
        va='top'
    )

fig.subplots_adjust(hspace=0.3, wspace=0.1)
plt.tight_layout()
plt.savefig("combined_change_shap_plots.png", dpi=600)
# plt.savefig("combined_change_shap_plots_1.pdf", bbox_inches='tight')
