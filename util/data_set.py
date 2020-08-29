import os

import PIL.Image
import torch.utils.data as data


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
