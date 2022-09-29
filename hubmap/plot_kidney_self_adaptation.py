import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


METRICS = ['dice', 'hausdorff']
MODELS = ['U-Net', 'CoaT']
MODE = ['Baseline', 'Lower Thres.', 'Lower Thres. + TTAda']
DATA_FOLDER = os.path.join(
    '/Users', 'lfidon', 'workspace', 'he-seg-eval', 'hubmap', 'data',
)

PKL = {m: {} for m in MODELS}
PKL['U-Net']['Baseline'] = os.path.join(
    DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_kidney_metrics.pkl')
PKL['CoaT']['Baseline'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_images_3000_3000_kidney_metrics.pkl')
PKL['U-Net']['Lower Thres.'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trnnUNetTrainerV2_fold012_union_images_3000_3000_kidney_metrics.pkl')
PKL['CoaT']['Lower Thres.'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_union_images_3000_3000_kidney_metrics.pkl')
PKL['U-Net']['Lower Thres. + TTAda'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trnnUNetTrainerV2_fold012_union_self_adaptation_images_3000_3000_kidney_metrics.pkl')
PKL['CoaT']['Lower Thres. + TTAda'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_union_self_adaptation_images_3000_3000_kidney_metrics.pkl')


def load_metrics(metrics_path):
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics


def create_df(metric):
    raw_data = []
    columns = ['Model', 'Adaptation Method', 'Metric']
    # Create the raw data
    for m in MODELS:
        for ada in MODE:
            met = load_metrics(PKL[m][ada])
            for val in met[metric]:
                line = [m, ada, val]
                raw_data.append(line)
    df = pd.DataFrame(raw_data, columns=columns)
    return df


def main(metric_name):
    df = create_df(metric_name)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    g = sns.boxplot(
        data=df,
        y='Metric',
        x='Model',
        hue='Adaptation Method',
        ax=ax,
        palette='Set3',
        hue_order=MODE,
        # palette='colorblind',
    )
    ax.set_xlabel('Model Architecture', fontsize=18, fontweight='bold')

    if metric_name == 'hausdorff':
        ax.set_ylabel('Hausdorff dist. 95% (in micrometers)', fontsize=18, fontweight='bold')
        # ax.set_ylim((-0.5, 3000.5))
        # g.set(yticks=[5*i for i in range(33)])
    else:
        ax.set_ylabel('Dice score (in %) [higher is better]', fontsize=18, fontweight='bold')
        ax.set_ylim((-0.5, 100.5))
        g.set(yticks=[5*i for i in range(21)])
        ax.legend(bbox_to_anchor=(1.02, 1), title='Adaptation Method')

    save_name = 'kidney_DG_%s.png' % metric_name
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    for met in METRICS:
        main(met)
