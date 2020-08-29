import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import pickle
import numpy as np
import nltk
from build_vocab import Vocabulary, build_vocab
from pycocotools.coco import COCO

import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import matplotlib.pyplot as plt
import nltk
from skimage import io,transform
from collections import Counter

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        return image

class CocoDataset(data.Dataset):
# class CaptionsPreprocessing:
    """Preprocess the captions, generate vocabulary and convert words to tensor tokens

    Args:
        captions_file_path (string): captions tsv file path
    """
    def __init__(self, root, captions_file_path, img_transform=None):
        self.root = root

        self.img_transform = img_transform

        self.all_captions = []

        self.captions_file_path = captions_file_path
        self.raw_captions_dict = self.read_raw_captions()
        self.captions_dict = self.process_captions()
        
        self.ids = list(self.captions_dict.keys())
        self.vocab = self.generate_vocabulary()

    def read_raw_captions(self):
        """
        Returns:
            Dictionary with raw captions list keyed by image ids (integers)
        """

        onlyfiles = [f for f in listdir(self.root) if isfile(join(self.root, f))]

        captions_dict = {}
        with open(self.captions_file_path, 'r', encoding='utf-8') as f:
            for img_caption_line in f.readlines():
                img_captions = img_caption_line.strip().split('\t')
                if(str('image_'+img_captions[0]+'.jpg') not in onlyfiles):
                    continue
                captions_dict[int(img_captions[0])] = img_captions[1:]

        return captions_dict

    def process_captions(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """

        raw_captions_dict = self.raw_captions_dict

        # Do the preprocessing here
        for key in raw_captions_dict:
            raw_captions_dict[key] = [i[:-1] for i in raw_captions_dict[key]]
            self.all_captions += (raw_captions_dict[key])

        captions_dict = raw_captions_dict

        return captions_dict

    def generate_vocabulary(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """

        captions_dict = self.captions_dict

        vocab = build_vocab(self.all_captions, 10);

        return vocab

    def captions_transform(self, img_caption_list):
        """
        Use this function to generate tensor tokens for the text captions
        Args:
            img_caption_list: List of captions for a particular image
        """
        tokens_list = [nltk.tokenize.word_tokenize(str(caption).lower()) for caption in img_caption_list]
        vocab = self.vocab

        res = []
        # Generate tensors
        for tokens in tokens_list:
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            res.append(target)
        return (res)

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        img_name = os.path.join(self.root, 'image_{}.jpg'.format(self.ids[index]))
        image = (io.imread(img_name))
        captions = self.captions_dict[self.ids[index]]

        if self.img_transform:
            # print('transforming images')
            image = self.img_transform(image)

        if self.captions_transform:
            captions = self.captions_transform(captions)

        # returning only one cpation as of now
        return image, captions[0]

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_loader(root, captions_file_path, img_transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       captions_file_path=captions_file_path,
                       img_transform=img_transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

# root='../../../ml-a4-data/train_images/train_images'
# captions_file_path='../../../ml-a4-data/train_captions.tsv'

# IMAGE_RESIZE = (128, 128)
# # Sequentially compose the transforms
# img_transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor()])

# num_workers=2
# shuffle=True
# batch_size=128

# print('Loading CocoDataset')
# coco = CocoDataset(root=root,
#                        captions_file_path=captions_file_path,
#                        img_transform=img_transform)

# print('Creating data_loader')
# data_loader = torch.utils.data.DataLoader(dataset=coco,
#                                           batch_size=batch_size,
#                                           shuffle=shuffle,
#                                           num_workers=num_workers,
#                                           collate_fn=collate_fn)
# for i in data_loader:
#     print(i)
#     break