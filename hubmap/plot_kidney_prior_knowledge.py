import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


METRICS = ['dice', 'hausdorff']
MODELS = ['U-Net', 'CoaT']
DATA_FOLDER = os.path.join(
    '/Users', 'lfidon', 'workspace', 'he-seg-eval', 'hubmap', 'data',
)
TRAIN = ['IHC', 'Fake PAS GAN']
TEST = ['PAS', 'Fake IHC Vahadane', 'Fake IHC GAN']

PKL = {tr: {} for tr in TRAIN}
PKL['IHC']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_kidney_metrics.pkl')
PKL['IHC']['Fake IHC Vahadane'] = os.path.join(
    DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_shift_vahadane_V1_kidney_metrics.pkl')
PKL['IHC']['Fake IHC GAN'] = os.path.join(
    DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_shift_V2_kidney_metrics.pkl')
PKL['Fake PAS GAN']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task173_fold012_images_3000_3000_kidney_metrics.pkl')

PKL_CONV = {tr: {} for tr in TRAIN}
PKL_CONV['IHC']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_kidney_postconvex_metrics.pkl')
PKL_CONV['IHC']['Fake IHC Vahadane'] = os.path.join(
    DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_shift_vahadane_V1_kidney_postconvex_metrics.pkl')
PKL_CONV['IHC']['Fake IHC GAN'] = os.path.join(
    DATA_FOLDER, 'pred_task162_fold012_images_3000_3000_shift_V2_kidney_postconvex_metrics.pkl')
PKL_CONV['Fake PAS GAN']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task173_fold012_images_3000_3000_kidney_postconvex_metrics.pkl')

PKL_COAT = {tr: {} for tr in TRAIN}
PKL_COAT['IHC']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_images_3000_3000_kidney_metrics.pkl')
PKL_COAT['IHC']['Fake IHC Vahadane'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_images_3000_3000_shift_vahadane_V1_kidney_metrics.pkl')
PKL_COAT['IHC']['Fake IHC GAN'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_images_3000_3000_shift_V2_kidney_metrics.pkl')
PKL_COAT['Fake PAS GAN']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task173_trcoatTrainer_fold012_images_3000_3000_kidney_metrics.pkl')

PKL_COAT_CONV = {tr: {} for tr in TRAIN}
PKL_COAT_CONV['IHC']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_images_3000_3000_kidney_postconvex_metrics.pkl')
PKL_COAT_CONV['IHC']['Fake IHC Vahadane'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_images_3000_3000_shift_vahadane_V1_kidney_postconvex_metrics.pkl')
PKL_COAT_CONV['IHC']['Fake IHC GAN'] = os.path.join(
    DATA_FOLDER, 'pred_task162_trcoatTrainer_fold012_images_3000_3000_shift_V2_kidney_postconvex_metrics.pkl')
PKL_COAT_CONV['Fake PAS GAN']['PAS'] = os.path.join(
    DATA_FOLDER, 'pred_task173_trcoatTrainer_fold012_images_3000_3000_kidney_postconvex_metrics.pkl')


def load_metrics(metrics_path):
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics


def create_df(metric, model):
    raw_data = []
    columns = ['Config', 'Prior knowledge', 'Metric']
    # Create the raw data
    for tr in TRAIN:
        for ts in TEST:
            config = 'Train: %s\nTest: %s' % (tr, ts)
            if model == 'U-Net':
                if ts in PKL[tr].keys():
                    met = load_metrics(PKL[tr][ts])
                    for val in met[metric]:
                        line = [config, 'No', val]
                        raw_data.append(line)
                if ts in PKL_CONV[tr].keys():
                    met = load_metrics(PKL_CONV[tr][ts])
                    for val in met[metric]:
                        line = [config, 'Yes', val]
                        raw_data.append(line)
            elif model == 'CoaT':
                if ts in PKL_COAT[tr].keys():
                    met = load_metrics(PKL_COAT[tr][ts])
                    for val in met[metric]:
                        line = [config, 'No', val]
                        raw_data.append(line)
                if ts in PKL_COAT_CONV[tr].keys():
                    met = load_metrics(PKL_COAT_CONV[tr][ts])
                    for val in met[metric]:
                        line = [config, 'Yes', val]
                        raw_data.append(line)
    df = pd.DataFrame(raw_data, columns=columns)
    return df


def main(metric_name, model_name):
    df = create_df(metric_name, model_name)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    g = sns.boxplot(
        data=df,
        y='Metric',
        x='Config',
        hue='Prior knowledge',
        ax=ax,
        palette='Set3',
        hue_order=['No', 'Yes']
        # palette='colorblind',
    )
    ax.set_xlabel('%s - Config for Train and Test' % model_name, fontsize=18, fontweight='bold')

    if metric_name == 'hausdorff':
        ax.set_ylabel('Hausdorff dist. 95% (in micrometers)', fontsize=18, fontweight='bold')
        # ax.set_ylim((-0.5, 3000.5))
        # g.set(yticks=[5*i for i in range(33)])
    else:
        ax.set_ylabel('Dice score (in %) [higher is better]', fontsize=18, fontweight='bold')
        ax.set_ylim((-0.5, 100.5))
        g.set(yticks=[5*i for i in range(21)])
        ax.legend(bbox_to_anchor=(1.02, 1), title='Prior knowledge')

    save_name = 'kidney_prior_knowledge_%s_%s.png' % (metric_name, model_name)
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    for model in MODELS:
        for met in METRICS:
            main(met, model)
