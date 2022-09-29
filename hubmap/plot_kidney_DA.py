import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


METRICS = ['dice', 'hausdorff']
DATA_FOLDER = os.path.join(
    '/Users', 'lfidon', 'workspace', 'he-seg-eval', 'hubmap', 'data',
)
# TRAIN = ['IHC', 'Fake PAS GAN V1', 'IHC + Fake PAS GAN V1', 'Fake PAS GAN V2']
TRAIN = ['IHC', 'Fake PAS GAN']
# TEST = ['PAS', 'Fake IHC Vahadane', 'Fake IHC GAN V1', 'Fake IHC GAN V2', 'PAS + Fake IHC']
TEST = ['PAS', 'Fake IHC Vahadane', 'Fake IHC GAN']
PKL = {tr: {} for tr in TRAIN}
PKL['IHC']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_kidney_metrics.pkl')
PKL['IHC']['Fake IHC Vahadane'] = os.path.join(
    DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_shift_vahadane_V1_kidney_metrics.pkl')
# PKL['IHC']['Fake IHC GAN V1'] = os.path.join(
#     DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_shift_V1_kidney_metrics.pkl')
PKL['IHC']['Fake IHC GAN'] = os.path.join(
    DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_shift_V2_kidney_metrics.pkl')
# PKL['Fake PAS GAN V1']['PAS'] = os.path.join(
#     DATA_FOLDER, 'pred_task171_fold012_images_3000_3000_kidney_metrics.pkl')
PKL['Fake PAS GAN']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task173_fold012_images_3000_3000_kidney_metrics.pkl')
# PKL['IHC + Fake PAS']['PAS'] = os.path.join(
#     DATA_FOLDER, 'pred_task172_images_3000_3000_kidney_metrics.pkl')
# PKL['IHC + Fake PAS']['Fake IHC'] = os.path.join(
#     DATA_FOLDER, 'pred_task172_images_3000_3000_shift_V1_kidney_metrics.pkl')
# PKL['IHC + Fake PAS']['PAS + Fake IHC'] = os.path.join(
#     DATA_FOLDER, 'pred_task172_TTA_metrics.pkl')


PKL_COAT = {tr: {} for tr in TRAIN}
PKL_COAT['IHC']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_images_3000_3000_kidney_metrics.pkl')
PKL_COAT['IHC']['Fake IHC Vahadane'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_images_3000_3000_shift_vahadane_V1_kidney_metrics.pkl')
PKL_COAT['IHC']['Fake IHC GAN'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_images_3000_3000_shift_V2_kidney_metrics.pkl')
PKL_COAT['Fake PAS GAN']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task173_trcoatTrainer_fold012_images_3000_3000_kidney_metrics.pkl')


def load_metrics(metrics_path):
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics


def create_df(metric):
    raw_data = []
    columns = ['Config', 'Model', 'Metric']
    # Create the raw data
    for tr in TRAIN:
        for ts in TEST:
            config = 'Train: %s\nTest: %s' % (tr, ts)
            if ts not in PKL[tr].keys():
                continue
            met = load_metrics(PKL[tr][ts])
            for val in met[metric]:
                line = [config, 'U-Net', val]
                raw_data.append(line)
            if ts in PKL_COAT[tr].keys():
                met = load_metrics(PKL_COAT[tr][ts])
                for val in met[metric]:
                    line = [config, 'CoaT', val]
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
        x='Config',
        hue='Model',
        ax=ax,
        palette='Set3',
        # palette='colorblind',
    )
    ax.set_xlabel('Adaptation Method for Train and Test', fontsize=18, fontweight='bold')

    if metric_name == 'hausdorff':
        ax.set_ylabel('Hausdorff dist. 95% (in micrometers)', fontsize=18, fontweight='bold')
        # ax.set_ylim((-0.5, 3000.5))
        # g.set(yticks=[5*i for i in range(33)])
    else:
        ax.set_ylabel('Dice score (in %) [higher is better]', fontsize=18, fontweight='bold')
        ax.set_ylim((-0.5, 100.5))
        g.set(yticks=[5*i for i in range(21)])
        ax.legend(bbox_to_anchor=(1.02, 1), title='Model')

    save_name = 'kidney_DA_%s.png' % metric_name
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    main(METRICS[0])
    main(METRICS[1])
