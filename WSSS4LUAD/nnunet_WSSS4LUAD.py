import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ROI_EVAL = [
    'Abnormal',
    'Normal',
    'Stroma',
    'Tumor',
]
METRICS = ['dice', 'hausdorff']
MODELS_FOLDER = os.path.join(
    '/Users', 'lfidon', 'data', 'nnUNet_data', 'nnUNet_trained_models',
    'nnUNet', '2d', 'Task121_WSSS4LUAD', 'nnUNetTrainerV2__nnUNetPlansv2.1'
)
# fold -> path to pkl
METRICS_PKL = {
    i: os.path.join(MODELS_FOLDER, 'fold_%d' % i, 'pred_test', 'metrics.pkl')
    for i in range(5)
}
METRICS_PKL['all'] = os.path.join(MODELS_FOLDER, 'pred_test', 'metrics.pkl')
HUE_ORDER = [0, 1, 2, 3, 4, 'all']
ORDER_ROI = ROI_EVAL


def load_metrics(metrics_path):
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

def create_df(metric):
    raw_data = []
    columns = ['Fold', 'ROI', metric]
    # Create the raw data
    for fold in list(METRICS_PKL.keys()):
        met = load_metrics(METRICS_PKL[fold])
        for roi in ROI_EVAL:
            metric_name = '%s_%s' % (metric, roi)
            for val in met[metric_name]:
                line = [fold, roi, val]
                raw_data.append(line)
    df = pd.DataFrame(raw_data, columns=columns)
    return df

def main(metric_name):
    df = create_df(metric_name)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 12))

    g = sns.boxplot(
        data=df,
        hue='Fold',
        hue_order=HUE_ORDER,
        y=metric_name,
        x='ROI',
        order=ORDER_ROI,
        ax=ax,
        palette='Set3',
        # palette='colorblind',
    )
    ax.set_xlabel('Region of interest', fontsize=18, fontweight='bold')

    if metric_name == 'hausdorff':
        ax.set_ylabel('Hausdorff dist. 95% (in pixels)', fontsize=18, fontweight='bold')
        ax.set_ylim((-0.5, 150.5))
        g.set(yticks=[5*i for i in range(33)])
    else:
        ax.set_ylabel('Dice score (in %)', fontsize=18, fontweight='bold')
        ax.set_ylim((-0.5, 100.5))
        g.set(yticks=[5*i for i in range(21)])

    fake_handles = []
    # Set the hatches for one every two of the boxplots
    # for i, patch in enumerate(ax.artists):
        # Create the legend patch for this category
        # legend_patch = mpatches.Patch(
        #     facecolor=patch.get_facecolor(),
        #     edgecolor=patch.get_edgecolor(),
        # )

        # if i % 6 == 5:  # exclude the case 100%
        #     patch.set_hatch('///')
        #     legend_patch.set_hatch('///')
        # fake_handles.append(legend_patch)
    # ax.legend(fake_handles, HUE_ORDER)
    # sns.move_legend(
    #     ax,
    #     "lower left",
    #     bbox_to_anchor=(1, 0.),
    # )
    save_name = 'nnunet_WSSS4LUAD_%s.pdf' % metric_name
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    main(METRICS[0])
    main(METRICS[1])
