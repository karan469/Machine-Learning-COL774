import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import nltk
from build_vocab import Vocabulary, build_vocab
from pycocotools.coco import COCO

import os
from os import listdir
from os.path import isfile, join
import PIL
import matplotlib.pyplot as plt
import nltk
from collections import Counter

class CocoDataset(data.Dataset):
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
        raw_captions_dict = self.raw_captions_dict
        for key in raw_captions_dict:
            raw_captions_dict[key] = [i[:-1] for i in raw_captions_dict[key]]
            self.all_captions += (raw_captions_dict[key])

        captions_dict = raw_captions_dict

        return captions_dict

    def generate_vocabulary(self):
        captions_dict = self.captions_dict
        vocab = build_vocab(self.all_captions, 10);
        return vocab

    def captions_transform(self, img_caption_list):
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
        caption_ind = index // 29000
        index = index % 29000
        img_name = os.path.join(self.root, 'image_{}.jpg'.format(self.ids[index]))
        image = PIL.Image.open(img_name).convert('RGB')
        # image = (io.imread(img_name))
        captions = self.captions_dict[self.ids[index]]

        if self.img_transform:
            # print('transforming images')
            image = self.img_transform(image)

        if self.captions_transform:
            captions = self.captions_transform(captions)
        # rand_index = random.randint(0,4)
       # returning only one cpation as of now
        return image, captions[caption_ind]

    def __len__(self):
        return (5 * len(self.ids))

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

    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_loader(root, captions_file_path, img_transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    coco = CocoDataset(root=root,
                       captions_file_path=captions_file_path,
                       img_transform=img_transform)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
