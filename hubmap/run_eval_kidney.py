import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import pickle
from argparse import ArgumentParser
import pandas as pd
import nibabel as nib
from evaluation_metrics.segmentation_metrics import *
from definition import CSV_VAL_KIDNEY

METRIC_NAMES = ['dice', 'hausdorff']
MAX_HD = 3000

parser = ArgumentParser()
parser.add_argument('--pred_csv', required=True)
parser.add_argument('--valid_csv', default=CSV_VAL_KIDNEY)


def rle2mask(mask_rle, shape=(3000, 3000)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def print_results(metrics, save_path=None):
    print('\nGlobal statistics for the metrics')
    for metric in METRIC_NAMES:
        key = metric
        n_cases = len(metrics[key])
        if n_cases == 0:
            print('No data for %s' % key)
            continue
        mean = np.mean(metrics[key])
        std = np.std(metrics[key])
        q3 = np.percentile(metrics[key], 75)
        p95 = np.percentile(metrics[key], 95)
        q1 = np.percentile(metrics[key], 25)
        p5 = np.percentile(metrics[key], 5)
        print(key)
        print('%d cases' % n_cases)
        print('mean=%.1f, std=%.1f, p95=%.1f, q3=%.1f, q1=%.1f, p5=%.1f' % (mean, std, p95, q3, q1, p5))
        print('-----------')
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f, pickle.HIGHEST_PROTOCOL)


def main(args):
    metrics = {'%s' % metric: [] for metric in METRIC_NAMES}

    # keep_default_na=False allows to avoid reading any empty rle as NaN
    df = pd.read_csv(args.valid_csv, keep_default_na=False)
    df_pred = pd.read_csv(args.pred_csv, keep_default_na=False)
    for i, row in df.iterrows():
        idx = str(row.id)
        mpp = float(row.pixel_size)
        rle = str(row.rle)
        pred_rle = str(df_pred[df_pred['id'] == idx]['rle'].values[0])
        if pred_rle == '':
            assert len(rle) > 0
            # Save time as we know the true segmentation is never empty
            metrics['dice'].append(0.)
            metrics['hausdorff'].append(mpp * MAX_HD)
            continue

        true_seg = rle2mask(rle)
        pred_seg = rle2mask(pred_rle)
        dice = 100 * dice_score(pred_seg, true_seg, fg_class=1)
        metrics['dice'].append(dice)
        haus = mpp * min(
            MAX_HD,
            haussdorff_distance(
                pred_seg,
                true_seg,
                fg_class=1,
                percentile=95,
            )
        )
        metrics['hausdorff'].append(haus)
        print('\n\033[92mCase', idx, '\033[0m')
        print('Dice score:', dice)
        print('HD95:', haus)
        if dice == 0:
            print('non zeros pred:', np.sum(pred_seg))
            print(len(rle))

    # Print the results
    metrics_save_path = args.pred_csv.replace('.csv',  '_metrics.pkl')
    print_results(metrics, save_path=metrics_save_path)
    print('Metrics have been saved in %s' % metrics_save_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
