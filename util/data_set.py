import os
import numpy as np
import torch.nn.functional as f
import torch.utils.data as data
import PIL.Image


class TripletDataSet(data.Dataset):
    def __init__(self, base_path, label_path, transforms=None):
        self.base_path = base_path
        self.transforms = transforms
        self.sample_dict, self.sample_arr = self.parse_label(label_path)

    def __getitem__(self, index):
        sample_path, pos_label = self.sample_arr[index]
        pos_path = np.random.choice(self.sample_dict[pos_label])

        keys = list(self.sample_dict.keys()).copy()
        keys.remove(pos_label)
        neg_label = np.random.choice(keys)
        neg_path = np.random.choice(self.sample_dict[neg_label])

        sample = PIL.Image.open(sample_path).convert('RGB')
        pos_sample = PIL.Image.open(pos_path).convert('RGB')
        neg_sample = PIL.Image.open(neg_path).convert('RGB')

        if self.transforms is not None:
            sample = self.transforms(sample)
            pos_sample = self.transforms(pos_sample)
            neg_sample = self.transforms(neg_sample)

        return sample, pos_sample, neg_sample, pos_label, neg_label

    def __len__(self):
        return len(self.sample_arr)

    def parse_label(self, label_path):
        sample_dict = {}
        sample_arr = []

        with open(label_path, 'r') as f:
            lines = f.read().splitlines()

        for line in lines:
            path, label = line.split(',')

            path = os.path.join(self.base_path, path)
            label = int(label)

            sample_arr.append((path, label))
            if label in sample_dict.keys():
                sample_dict[label].append(path)
            else:
                sample_dict[label] = [path]

        return sample_dict, sample_arr


class TestDataSet(data.Dataset):
    def __init__(self, base_path, image_name_path, transforms=None):
        self.base_path = base_path
        self.transforms = transforms

        with open(image_name_path, 'r') as f:
            self.image_name_list = f.read().splitlines()

    def __getitem__(self, index):
        image_path = os.path.join(self.base_path, self.image_name_list[index])
        img = PIL.Image.open(image_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        return img

    def __len__(self):
        return len(self.image_name_list)
