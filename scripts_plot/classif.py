import os
import pickle
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import RocCurveDisplay

EVAL = [
    'Normal vs all',
    'Stroma vs all',
    'Tumor vs all',
    'Normal vs Stroma',
    # 'Normal/Stroma vs Tumor',
]

parser = ArgumentParser()
parser.add_argument('--predictions', required=True)


def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def main_auc(args):
    data = load_data(args.predictions)

    # Prepare data
    y_true = {n: [] for n in EVAL}
    y_pred = {n: [] for n in EVAL}

    for case_n in list(data.keys()):
        true_label = data[case_n]['true_label']
        proba = data[case_n]['pred']

        # Normal
        if true_label == 0:
            p_n = proba[0]
            p_ns = proba[0] + proba[1]

            y_true['Normal vs all'].append(1)
            y_true['Stroma vs all'].append(0)
            y_true['Tumor vs all'].append(0)
            y_true['Normal vs Stroma'].append(1)
            # y_true['Normal/Stroma vs Tumor'].append(1)

            y_pred['Normal vs all'].append(p_n)
            y_pred['Stroma vs all'].append(proba[1])
            y_pred['Tumor vs all'].append(proba[2])
            if p_ns == 0:
                y_pred['Normal vs Stroma'].append(0)
            else:
                # shall we normalize by p_ns here?
                y_pred['Normal vs Stroma'].append(p_n)
            # y_pred['Normal/Stroma vs Tumor'].append(p_ns)

        # Stroma
        elif true_label == 1:
            p_s = proba[1]
            p_ns = proba[0] + proba[1]

            y_true['Normal vs all'].append(0)
            y_true['Stroma vs all'].append(1)
            y_true['Tumor vs all'].append(0)
            y_true['Normal vs Stroma'].append(0)
            # y_true['Normal/Stroma vs Tumor'].append(1)

            y_pred['Normal vs all'].append(proba[0])
            y_pred['Stroma vs all'].append(p_s)
            y_pred['Tumor vs all'].append(proba[2])
            if p_ns == 0:
                y_pred['Normal vs Stroma'].append(0)
            else:
                # shall we normalize by p_ns here?
                y_pred['Normal vs Stroma'].append(proba[0])
            # y_pred['Normal/Stroma vs Tumor'].append(p_ns)

        # Tumor
        elif true_label == 2:
            p_ns = proba[0] + proba[1]

            y_true['Normal vs all'].append(0)
            y_true['Stroma vs all'].append(0)
            y_true['Tumor vs all'].append(1)
            # y_true['Normal/Stroma vs Tumor'].append(0)

            y_pred['Normal vs all'].append(proba[0])
            y_pred['Stroma vs all'].append(proba[1])
            y_pred['Tumor vs all'].append(proba[2])
            # y_pred['Normal/Stroma vs Tumor'].append(p_ns)

        # Stroma/Tumor
        else:
            y_true['Normal vs all'].append(0)

            y_pred['Normal vs all'].append(proba[0])

    ncols = len(EVAL)
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(4*ncols, 4))

    # Plot all the AUC curves
    for i, target in enumerate(EVAL):
        RocCurveDisplay.from_predictions(y_true[target], y_pred[target], ax=ax[i])
        ax[i].set_title(target, fontsize=18)

    save_name = 'ROC_WSSS4LUAD.pdf'
    # fig.savefig(save_name, bbox_inches='tight')
    fig.savefig(save_name)
    print('Figure saved in', save_name)


def main_confusion(args):
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    main_auc(args)
