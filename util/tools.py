import warnings
import PIL.ImageFile
import torch.backends.cudnn


def init():
    warnings.filterwarnings('ignore')
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
