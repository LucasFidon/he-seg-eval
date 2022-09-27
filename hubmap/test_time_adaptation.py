import torch
import torch.nn.functional as F
import numpy as np
from loss_functions import CrossEntropyLoss, DiceCELoss
from transforms import WeightedRandomCrop

THRESHOLD = 0.7
N_ITER_TTA = 10
# LR = 0.05
LR = 0.025  # divide by 2 because we take the sum of two losses compared to one loss for the original paper
IGNORE = -1
EPS = 1e-3
BS = 4  # batch size was 1 in the original paper but here we have to adapt for patch-based approach


def create_pseudo_gt(softmax, threshold=THRESHOLD):
    # We assume the max proba is always 1
    # this is usually the case when the Dice loss is used for training
    softmax = np.squeeze(softmax)
    seg = np.argmax(softmax, axis=0)
    if np.sum(softmax[0, ...] <= threshold) > 0:
        seg[np.logical_and(seg == 0, softmax[0, ...] <= threshold)] = IGNORE
    if np.sum(softmax[1, ...] <= threshold) > 0:
        seg[np.logical_and(seg == 1, softmax[1, ...] <= threshold)] = IGNORE
    # print('Num uncertain', np.sum(seg == IGNORE))
    # print('Or ratio', np.sum(seg == IGNORE) / seg.size)
    return seg


def tta_pred(trainer, param, image, softmax_target, patch_size, threshold=THRESHOLD):
    # Set up image and pseudo gt
    # Compute numpy array pseudo ground-truth segmentation
    # pixels to ignore are set -1
    p_gt = create_pseudo_gt(softmax_target, threshold=threshold)
    pseudo_gt = torch.tensor(p_gt).cuda()
    # We assume the image has been preprocessed already
    img = torch.tensor(np.squeeze(image)).float().cuda()

    # Set up transformations
    tr = WeightedRandomCrop(
        size=patch_size,
        weight=[EPS, EPS, 1.],  # Three weights because of the ignore index
        labels=[IGNORE, 0, 1],
    )
    flip_cycle = ([], [-1], [-2], [-1, -2])

    # Set up model
    trainer.load_checkpoint_ram(param, False)
    trainer.network.train()
    trainer.network.do_ds = True

    # Set up optimizer and loss for self adaptation
    optimizer = torch.optim.SGD(
        trainer.network.parameters(),
        lr=LR,
        momentum=0.,
        weight_decay=0,
    )
    # loss_fct = CrossEntropyLoss(ignore_index=IGNORE)
    loss_fct = DiceCELoss(ignore_index=IGNORE)
    # scaler = torch.cuda.amp.GradScaler()

    # Perform the test-time adaptation by SGD
    for i in range(N_ITER_TTA):
        # Prepare inputs
        img_batch = []
        seg_batch = []
        for b in range(BS):
            sample = {
                'image': torch.clone(img),
                'segmentation': torch.clone(pseudo_gt),
            }
            # Crop to get a patch
            sample = tr(sample)
            # We flip following a cycle to limit randomness
            i_flip = (i + b) % len(flip_cycle)
            flip_axis = flip_cycle[i_flip]
            if len(flip_axis) > 0:
                sample['image'] = torch.flip(sample['image'], flip_axis)
                sample['segmentation'] = torch.flip(sample['segmentation'], flip_axis)
            # Add batch dimension
            sample['image'] = torch.unsqueeze(sample['image'], 0)
            sample['segmentation'] = torch.unsqueeze(sample['segmentation'], 0)
            # Add class dim for downsampling and loss
            sample['segmentation'] = torch.unsqueeze(sample['segmentation'], 1)
            # Downsample segmentation if needed
            if type(trainer).__name__ == 'coatTrainer':
                # need to add batch dim for interpolate
                sample['segmentation'] = F.interpolate(
                    sample['segmentation'].float(),
                    scale_factor=(1. / 8, 1. / 8),
                    mode='nearest-exact',
                ).long()
            img_batch.append(sample['image'])
            seg_batch.append(sample['segmentation'])

        # Create batch
        x = torch.cat(img_batch, dim=0)
        y = torch.cat(seg_batch, dim=0)
        y_pred = trainer.network(x)[0]
        loss = loss_fct(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # l_cpu = loss.cpu().detach().numpy()
        # print('Iter %d: loss=%f' % (i, l_cpu))

    softmax_pred = trainer.predict_preprocessed_data_return_seg_and_softmax(
        image, do_mirroring=True, mirror_axes=trainer.data_aug_params['mirror_axes'],
        use_sliding_window=True, step_size=0.5, use_gaussian=True, all_in_gpu=None,
        mixed_precision=False)[1]

    # Test
    save_name = type(trainer).__name__ + '.npz'
    print('Save data in', save_name)
    np.savez(
        save_name,
        img=image,
        pgt=p_gt,
        softmax1=softmax_target,
        softmax2=softmax_pred,
    )

    return softmax_pred
