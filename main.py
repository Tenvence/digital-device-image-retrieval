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
from torchvision.transforms import Compose, ToTensor
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

model_name = './saved-output/triplet-contrast-cosine-model-r50.pkl'
param_name = './saved-output/triplet-contrast-cosine-param-r50.pth'
gallery_embedding_block_name = './saved-output/triplet-contrast-cosine-r50-gallery-embedding-block'
query_embedding_block_name = './saved-output/triplet-contrast-cosine-r50-query-embedding-block'


def triplet_contrast_train():
    model = Encoder(num_classes=num_classes)
    torch.save(model.state_dict(), model_name)

    if torch.cuda.is_available():
        model = nn.DataParallel(model).to(device=device)

    model.train()

    batch_size = 64

    triplet_data_set = TripletDataSet(train_path, os.path.join(train_path, 'label.txt'), transforms=Compose([ResizePad(300), ToTensor()]))
    data_loader = DataLoader(triplet_data_set, batch_size=batch_size, shuffle=True, num_workers=32)

    epoch = 30
    warm_epoch = 3
    iter_per_epoch = len(data_loader)

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = LinearCosineScheduler(optimizer, warm_steps=warm_epoch * iter_per_epoch, total_steps=epoch * iter_per_epoch, max_lr=0.1)

    for epoch_idx in range(epoch):
        loss_arr = []
        processor = tqdm(data_loader)
        for sample, pos_sample, neg_sample in processor:
            sample, pos_sample, neg_sample = sample.to(device=device), pos_sample.to(device=device), neg_sample.to(device=device)

            optimizer.zero_grad()

            triplet_sample = torch.cat([sample, pos_sample, neg_sample], dim=0)
            triplet_output = model(triplet_sample)

            sample_embedding = triplet_output[:batch_size, ...]
            pos_embedding = triplet_output[batch_size:2 * batch_size, ...]
            neg_embedding = triplet_output[2 * batch_size:, ...]

            pos_cos = 1 - f.cosine_similarity(sample_embedding, pos_embedding)
            neg_cos = 1 - f.cosine_similarity(sample_embedding, neg_embedding)

            margin = 1.0
            loss = pos_cos - neg_cos + margin
            loss = torch.max(loss, torch.zeros_like(loss))
            loss = torch.mean(loss)

            # sample_embedding = model(sample)
            # pos_embedding = model(pos_sample)
            # neg_embedding = model(neg_sample)
            #
            # loss = torch.mean(f.triplet_margin_loss(sample_embedding, pos_embedding, neg_embedding, margin=1.0, p=2))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            loss_arr.append(float(loss))
            processor.set_description('  Epoch=%d/%d; mean loss=%.4f; cur loss=%.4f' % (epoch_idx + 1, epoch, sum(loss_arr) / len(loss_arr), float(loss)))
        torch.save(model.module.state_dict(), param_name)


def encode_test_data():
    model = Encoder(num_classes=num_classes)
    model.load_state_dict(torch.load(param_name))
    if torch.cuda.is_available():
        model = nn.DataParallel(model).to(device=device)
    model.eval()

    query_dataset = TestDataSet(os.path.join(data_set_path, 'test', 'query'), './saved-output/query_images.txt', transforms=Compose([ResizePad(300), ToTensor()]))
    data_loader = DataLoader(query_dataset, batch_size=128, shuffle=False, num_workers=32)

    processor = tqdm(data_loader)
    query_embedding_block = []
    for img in processor:
        with torch.no_grad():
            output = model(img)
        query_embedding_block.append(output)
    query_embedding_block = torch.cat(query_embedding_block, dim=0).cpu().numpy()
    np.save('./saved-output/triplet-contrast-r50-query-embedding-block', query_embedding_block)

    gallery_dataset = TestDataSet(os.path.join(data_set_path, 'test', 'gallery'), './saved-output/gallery_images.txt', transforms=Compose([ResizePad(300), ToTensor()]))
    data_loader = DataLoader(gallery_dataset, batch_size=128, shuffle=False, num_workers=32)

    processor = tqdm(data_loader)
    gallery_embedding_block = []
    for img in processor:
        with torch.no_grad():
            output = model(img)
        gallery_embedding_block.append(output)
    gallery_embedding_block = torch.cat(gallery_embedding_block, dim=0).cpu().numpy()
    np.save('./saved-output/triplet-contrast-r50-gallery-embedding-block', gallery_embedding_block)


if __name__ == '__main__':
    triplet_contrast_train()
    # encode_test_data()
