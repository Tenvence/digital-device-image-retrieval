import os
import warnings

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as f
from PIL import ImageFile
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, ColorJitter
from tqdm import tqdm

from model.encoder import Encoder
from util.data_set import TripletDataSet, TestDataSet
from util.lr_scheduler import LinearCosineScheduler
from util.transforms import ResizePad

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

data_set_path = '../../DataSet/digital-device-dataset'
train_path = os.path.join(data_set_path, 'train')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 3097
contrast_dim = 16

model_name = './saved-output/triplet-contrast-16-cla-3097-model-r50-ex30.pkl'
param_name = './saved-output/triplet-contrast-16-cla-3097-param-r50-ex30.pth'
gallery_embedding_block_name = './saved-output/triplet-contrast-16-cla-3097-r50-gallery-embedding-block-ex30'
query_embedding_block_name = './saved-output/triplet-contrast-16-cla-3097-r50-query-embedding-block-ex30'


def compute_contrast_loss(sample_embedding, pos_embedding, neg_embedding, margin=0.7):
    pos_cos = 1 - f.cosine_similarity(sample_embedding, pos_embedding)
    neg_cos = 1 - f.cosine_similarity(sample_embedding, neg_embedding)

    contrast_loss = torch.mean(f.relu(pos_cos - neg_cos + margin))

    return contrast_loss


def compute_cla_loss(sample_pred, pos_pred, neg_pred, pos_label, neg_label):
    sample_cla_loss = torch.mean(torch.sum(f.binary_cross_entropy(sample_pred, pos_label, reduce=False), dim=-1))
    pos_cla_loss = torch.mean(torch.sum(f.binary_cross_entropy(pos_pred, pos_label, reduce=False), dim=-1))
    neg_cla_loss = torch.mean(torch.sum(f.binary_cross_entropy(neg_pred, neg_label, reduce=False), dim=-1))

    cla_loss = (sample_cla_loss + pos_cla_loss + neg_cla_loss) / 3.

    return cla_loss


def triplet_contrast_train():
    model = Encoder(num_classes, contrast_dim)
    torch.save(model.state_dict(), model_name)

    model.load_state_dict(torch.load('./saved-output/triplet-contrast-16-cla-3097-param-r50.pth'))

    if torch.cuda.is_available():
        model = nn.DataParallel(model).to(device=device)

    model.train()

    train_transforms = Compose([
        RandomCrop(300, pad_if_needed=True, fill=(128, 128, 128)),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ResizePad(300),
        ToTensor()
    ])

    triplet_data_set = TripletDataSet(train_path, os.path.join(train_path, 'label.txt'), transforms=train_transforms)
    data_loader = DataLoader(triplet_data_set, batch_size=64, shuffle=True, num_workers=16)

    epoch = 30
    warm_epoch = 3
    iter_per_epoch = len(data_loader)

    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = LinearCosineScheduler(optimizer, warm_steps=warm_epoch * iter_per_epoch, total_steps=epoch * iter_per_epoch, max_lr=0.01)

    for epoch_idx in range(epoch):
        loss_arr = []
        cla_loss_arr = []
        cont_loss_arr = []
        processor = tqdm(data_loader)
        for sample, pos_sample, neg_sample, pos_label, neg_label in processor:
            sample, pos_sample, neg_sample = sample.to(device=device), pos_sample.to(device=device), neg_sample.to(device=device)

            pos_label = f.one_hot(pos_label, num_classes=num_classes).float().to(device=device)
            neg_label = f.one_hot(neg_label, num_classes=num_classes).float().to(device=device)

            optimizer.zero_grad()

            triplet_sample = torch.cat([sample, pos_sample, neg_sample], dim=0)
            triplet_contrast_output, cla_output = model(triplet_sample)

            batch_size = sample.shape[0]

            sample_embedding = triplet_contrast_output[:batch_size, ...]
            pos_embedding = triplet_contrast_output[batch_size:2 * batch_size, ...]
            neg_embedding = triplet_contrast_output[2 * batch_size:, ...]

            contrast_loss = compute_contrast_loss(sample_embedding, pos_embedding, neg_embedding)

            sample_cla_pred = cla_output[:batch_size, ...]
            pos_cla_pred = cla_output[batch_size:2 * batch_size, ...]
            neg_cla_pred = cla_output[2 * batch_size:, ...]

            cla_loss = compute_cla_loss(sample_cla_pred, pos_cla_pred, neg_cla_pred, pos_label, neg_label)

            loss = contrast_loss + cla_loss

            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            loss_arr.append(float(loss))
            cla_loss_arr.append(float(cla_loss))
            cont_loss_arr.append(float(contrast_loss))

            mean_loss = sum(loss_arr) / len(loss_arr)
            mean_cla_loss = sum(cla_loss_arr) / len(cla_loss_arr)
            mean_cont_loss = sum(cont_loss_arr) / len(cont_loss_arr)

            processor.set_description('  Epoch=%d/%d; mLoss=%.4f; cont=%.4f; cla=%.4f' % (epoch_idx + 1, epoch, mean_loss, mean_cont_loss, mean_cla_loss))
        torch.save(model.module.state_dict(), param_name)


def encode_test_data(test_dataset, embedding_block_name):
    model = Encoder(num_classes=num_classes)
    model.load_state_dict(torch.load(param_name))
    if torch.cuda.is_available():
        model = nn.DataParallel(model).to(device=device)
    model.eval()

    data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=32)

    processor = tqdm(data_loader)
    embedding_block = []
    for img in processor:
        with torch.no_grad():
            output, _ = model(img)  # 只保留相似度量的嵌入表示, 丢掉分类表示
        embedding_block.append(output)
    embedding_block = torch.cat(embedding_block, dim=0).cpu().numpy()
    np.save(embedding_block_name, embedding_block)


if __name__ == '__main__':
    triplet_contrast_train()

    query_dataset = TestDataSet(os.path.join(data_set_path, 'test', 'query'), './saved-output/query_images.txt', transforms=Compose([ResizePad(300), ToTensor()]))
    gallery_dataset = TestDataSet(os.path.join(data_set_path, 'test', 'gallery'), './saved-output/gallery_images.txt', transforms=Compose([ResizePad(300), ToTensor()]))
    encode_test_data(query_dataset, query_embedding_block_name)
    encode_test_data(gallery_dataset, gallery_embedding_block_name)
