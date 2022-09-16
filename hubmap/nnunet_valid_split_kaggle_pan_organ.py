import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


TASK = 'Task160_HPA_HuBMAP'
FOLDS = [i for i in range(5)]
TRAINER = 'nnUNetTrainerV2__nnUNetPlansv2.1'
NNUNET_FOLDER = '/Users/lfidon/data/nnUNet_data/nnUNet_trained_models/nnUNet/2d'
METRIC_AVAILABLE = [
    'Accuracy', 'Dice', 'False Discovery Rate', 'False Negative Rate',
    'False Omission Rate', 'False Positive Rate',
    'Jaccard', 'Precision', 'Recall',
]
METRIC_EVAL = ['Dice']
KAGGLE_TRAIN_CSV = '/Users/lfidon/data/hubmap/train.csv'


def main():
    task_folder = os.path.join(NNUNET_FOLDER, TASK)
    model_folder = os.path.join(task_folder, TRAINER)
    df_data = pd.read_csv(KAGGLE_TRAIN_CSV)

    # Read json metric data
    json_dict = {}
    for fold in FOLDS:
        json_path = os.path.join(model_folder, 'fold_%d' % fold, 'validation_raw', 'summary.json')
        with open(json_path) as f:
            print('*** Fold %d' % fold)
            json_dict[fold] = json.load(f)['results']
            print('Mean results valid split: Dice=%.2f' % json_dict[fold]['mean']['1']['Dice'])

    # Create the data frame for plots
    columns = ['id', 'Organ', 'Fold', 'Metric', 'Value']
    rows = []
    for fold in FOLDS:
        data = json_dict[fold]['all']
        for met in data:
            id = met['reference'].split('/')[-1].replace('.nii.gz', '')
            met_fg = met['1']
            organ = df_data[df_data['id'] == int(id)]['organ'].values[0]
            for met_name in METRIC_EVAL:
                row = [id, organ, fold, met_name, met_fg[met_name]]
                rows.append(row)

    df = pd.DataFrame(data=rows, columns=columns)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 12))
    g = sns.boxplot(
        data=df,
        y='Value',
        x='Organ',
        hue='Fold',
        ax=ax,
        palette='colorblind',
    )
    ax.set_xlabel('Organ', fontsize=18, fontweight='bold')
    ax.set_ylabel('Dice score', fontsize=18, fontweight='bold')
    save_name = 'nnunet_kaggle_pan_organ'
    for fold in FOLDS:
        save_name += '_%d' % fold
    save_name += '.png'
    fig.savefig(save_name, bbox_inches='tight')
    print('Figure saved in', save_name)


if __name__ == '__main__':
    main()