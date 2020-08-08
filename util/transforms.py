import numpy as np
from torchvision.transforms import transforms


class ResizePad(transforms.Lambda):
    def __init__(self, size):
        self.size = size
        super(ResizePad, self).__init__(self.resize_pad)

    def resize(self, img):
        iw, ih = img.size

        scale = self.size / max(iw, ih)
        resize_trans = transforms.Resize(size=(int(ih * scale), int(iw * scale)))

        return resize_trans(img)

    def pad(self, img):
        iw, ih = img.size

        pwl = int(np.floor((self.size - iw) / 2))
        pwr = int(np.ceil((self.size - iw) / 2))
        pht = int(np.floor((self.size - ih) / 2))
        phb = int(np.ceil((self.size - ih) / 2))

        pad_trans = transforms.Pad(padding=(pwl, pht, pwr, phb))

        return pad_trans(img)

    def resize_pad(self, img):
        return self.pad(self.resize(img))
