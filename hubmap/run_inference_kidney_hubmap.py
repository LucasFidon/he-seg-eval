import os
import cv2
import torch
import numpy as np
from skimage import io
import pandas as pd
from argparse import ArgumentParser
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.preprocessing.preprocessing import resample_patient
from nnunet.paths import network_training_output_dir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from self_adapt_norm import replace_batchnorm
from test_time_adaptation import tta_pred

torch.backends.cudnn.benchmark = True

DEFAULT_FOLDER_TO_INFER = '/storage/hpa_hubmap_kidney/validation_PAS/images_3000_3000_shift_V1_kidney'
DEFAULT_OUT_FOLDER = '/workspace'
DEFAULT_CSV = '/storage/hpa_hubmap_kidney/validation_PAS/val_kidney.csv'
DEFAULT_TASK = 162
DEFAULT_TRAINER = 'nnUNetTrainerV2'
PLAN = 'nnUNetPlansv2.1_HPA_HuBMAP'
PATCH_SIZE = 512
MODEL_MPP = 1.
SCALES_TTA = [0.75, 1.25]
ALPHA_SAN = 0.1

TRAINERS_AVAILABLE = ['nnUNetTrainerV2', 'coatTrainer']

parser = ArgumentParser()
parser.add_argument('--input', nargs='+', default=DEFAULT_FOLDER_TO_INFER)
parser.add_argument('--input_csv', default=DEFAULT_CSV, help='contains the mpp for all the tiles.')
parser.add_argument('--output_folder', default=DEFAULT_OUT_FOLDER)
parser.add_argument('--trainer', type=str, default=DEFAULT_TRAINER,
                    help='try values in %s' % str(TRAINERS_AVAILABLE))
parser.add_argument('--task', type=int, default=DEFAULT_TASK)
parser.add_argument('--folds', nargs='+', type=int, default=0)
parser.add_argument('--tta', action='store_true')
parser.add_argument('--san', action='store_true', help='Use this flag to use self-adaptive normalization (SaN)')
parser.add_argument('--alpha', type=float, default=ALPHA_SAN)
parser.add_argument('--adapt', action='store_true')
parser.add_argument('--patch_size', type=int, default=PATCH_SIZE)
parser.add_argument('--union', action='store_true',
                    help='Use union rather than average for aggregation in ensembling')

def load_image(tiff_path: str):
    img = io.imread(tiff_path)
    return img


def preprocess(img, mpp, model_mpp):
    ori_spacing = (999, mpp, mpp)
    target_spacing = (999, model_mpp, model_mpp)
    res = img.transpose((2, 0, 1))
    # Add z dimension
    res = res[:, None, :, :]
    res = res.astype(float)
    res, _ = resample_patient(
        data=res, seg=None, original_spacing=ori_spacing,
        target_spacing=target_spacing, order_data=1, force_separate_z=True,
    )
    mean = res.mean(axis=(1, 2, 3))
    std = res.std(axis=(1, 2, 3))
    res = (res - mean[:, None, None, None]) / std[:, None, None, None]
    return res


# https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
# with transposed mask
def rle_encode_less_memory(img):
    # the image should be transposed
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def main(args):
    from time import time
    t0 = time()
    df = pd.read_csv(args.input_csv)

    names = []
    preds = []

    # Prepare model
    task_name = convert_id_to_task_name(args.task)
    model_folder = os.path.join(
        network_training_output_dir,
        '2d',
        task_name,
        args.trainer + "__" + PLAN,
    )
    folds = args.folds
    assert os.path.exists(model_folder), 'Folder %s not found' % model_folder
    trainer, params = load_model_and_checkpoint_files(
        model_folder, folds, mixed_precision=False,
        checkpoint_name='model_final_checkpoint',
    )
    trainer.network.eval()

    # Inference
    for i, row in df.iterrows():
        idx = str(row.id)
        # if idx != '1e2425f28_53':
        #     continue
        print('\n\033[92m(%d/%d) Image' % (i + 1, len(df)), idx, '\033[0m')
        mpp = float(row.pixel_size)
        h = 3000
        w = 3000
        softmax_list = []
        # Allow TTA using more than one input folder
        for input_path in args.input:
            tiff_path = os.path.join(input_path, '%s.tiff' % idx)
            print('infer on', tiff_path)
            img0 = load_image(tiff_path)
            img = preprocess(img0, mpp, MODEL_MPP)
            aug_img = None
            if args.tta:
                print('Use TTA')
                aug_img = {}
                for s in SCALES_TTA:
                    aug_img[s] = preprocess(img0, mpp, s * MODEL_MPP)

            # Inference
            for p in params:
                if args.san:
                    print('\n*** Use SaN with alpha=', args.alpha)
                    trainer, _ = load_model_and_checkpoint_files(
                        model_folder, folds, mixed_precision=False,
                        checkpoint_name='model_final_checkpoint',
                    )
                    trainer.network.eval()
                    trainer.load_checkpoint_ram(p, False)
                    # We have to replace the BN layers after loading the network params
                    replace_batchnorm(trainer.network, alpha=args.alpha)
                else:
                    trainer.load_checkpoint_ram(p, False)
                softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(
                    img, do_mirroring=True, mirror_axes=trainer.data_aug_params['mirror_axes'],
                    use_sliding_window=True, step_size=0.5, use_gaussian=True, all_in_gpu=None,
                    mixed_precision=False)[1]
                softmax_list.append(softmax)
                if aug_img is not None:
                    for s in SCALES_TTA:
                        softmax_s = trainer.predict_preprocessed_data_return_seg_and_softmax(
                            aug_img[s], do_mirroring=True, mirror_axes=trainer.data_aug_params['mirror_axes'],
                            use_sliding_window=True, step_size=0.5, use_gaussian=True, all_in_gpu=None,
                            mixed_precision=False)[1]
                        softmax_model_mpp, _ = resample_patient(
                            data=softmax_s, seg=None,
                            original_spacing=(999, s * MODEL_MPP, s * MODEL_MPP),
                            target_spacing=(999, MODEL_MPP, MODEL_MPP),
                            order_data=1, force_separate_z=True,
                        )
                        softmax_list.append(softmax_model_mpp)
        if args.union:
            print('Use soft union rather than averaging of softmax predictions')
            # we sum foreground softmax proba rather than averaging them; like a soft union
            softmax = np.sum(softmax_list, axis=0)
            # still use mean for the background class
            softmax[0, ...] /= len(softmax_list)
            # normalize proba to sum to 1
            softmax[:, ...] /= np.sum(softmax, axis=0)
        else:
            # Classic ensembling by averaging
            softmax = np.mean(softmax_list, axis=0)

        # (optional) Test-time adaptation
        # NB: which img are we using if several args.input folders were given?
        if args.adapt:
            print('Apply test-time adaptation')
            softmax_target = np.copy(softmax)
            softmax = 0
            for p in params:
                softmax += tta_pred(
                    trainer=trainer,
                    param=p,
                    image=img,
                    softmax_target=softmax_target,
                    patch_size=args.patch_size,
                )
            softmax /= len(params)
            save_name = type(trainer).__name__ + '_ens.npz'
            print('Save final softmax in', save_name)
            np.savez(
                save_name,
                softmax=softmax,
            )

        pred_seg = np.argmax(softmax, axis=0)[0]
        seg = cv2.resize(pred_seg, (h, w), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        rle = rle_encode_less_memory(seg)
        with open('rle_%s_ens.txt' % type(trainer).__name__, 'w') as f:
            f.write(rle)
        names.append(idx)
        preds.append(rle)

    df = pd.DataFrame({'id': names, 'rle': preds})
    csv_n = 'pred_task%d_tr%s_fold' % (args.task, args.trainer)
    for f in folds:
        csv_n += str(f)
    if args.san:
        csv_n += '_SaN%f' % args.alpha
    if args.tta:
        csv_n += '_tta'
    if args.union:
        csv_n += '_union'
    if args.adapt:
        csv_n += '_self_adaptation'
    for p in args.input:
        csv_n += '_' + p.split('/')[-1]
    csv_n += '.csv'
    out_csv = os.path.join(args.output_folder, csv_n)
    df.to_csv(out_csv, index=False)

    print('Total time', time() - t0, 'seconds')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
