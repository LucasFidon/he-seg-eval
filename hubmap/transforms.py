import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


class WeightedRandomCrop(transforms.RandomCrop):
    """
    Random crop that samples crops according to class probabilities.
    This can be used to help balancing the classes.
    from https://github.com/owkin/RL_benchmarks/blob/main/rl_benchmarks/data/transforms.py
    """

    def __init__(self, size, weight, labels, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)
        self.weight = torch.tensor(weight).float()
        w_sum = self.weight.sum()
        self.labels = labels
        if w_sum > 0:
            self.weight /= w_sum

    def get_parameters(self, sample):
        img = sample['image']
        seg = sample['segmentation']
        h, w = F.get_image_size(img)
        th, tw = self.size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        # Draw the class to sample for the center of the image
        class_present = torch.unique(seg, sorted=True)
        proba = torch.clone(self.weight)
        # We cannot sample classes that are not present
        for i, l in enumerate(self.labels):
            if not l in class_present:
                proba[i] = 0
        proba /= proba.sum()
        c = self.labels[torch.multinomial(proba, 1).item()]
        indices = torch.argwhere(seg == c)

        # Draw the pixels to use as lower left indices
        center = torch.randint(0, indices.size()[0], size=(1,)).item()
        i_c, j_c = indices[center]
        # translate the indices to have the low left corner
        i = i_c - (th // 2)
        j = j_c - (tw // 2)

        # Make sure the indices are valid for the crop
        i = max(0, min(h - th, i))
        j = max(0, min(w - tw, j))

        return i, j, th, tw

    def padding_if_needed(self, img, seg):

        height, width = F.get_image_size(img)
        # pad the width if needed
        if width < self.size[1]:
            margin = self.size[1] - width
            padding = [0, margin // 2, 0, margin - margin // 2]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            seg = F.pad(seg, padding, 0, self.padding_mode)
        # pad the height if needed
        if height < self.size[0]:
            margin = self.size[0] - height
            padding = [margin // 2, 0, margin - margin // 2, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            seg = F.pad(seg, padding, 0, self.padding_mode)

        return img, seg

    def forward(self, sample):
        img = sample['image']
        seg = sample['segmentation']
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        if self.pad_if_needed:
            img, seg = self.padding_if_needed(img, seg)

        i, j, h, w = self.get_parameters({'image': img, 'segmentation': seg})

        out_img = F.crop(img, i, j, h, w)
        out_seg = F.crop(seg, i, j, h, w)

        return {'image': out_img, 'segmentation': out_seg}
