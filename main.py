import csv
import os

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as fun
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model.encoder import Encoder
from util.data_set import TestDataSet
from util.lr_scheduler import LinearCosineScheduler
from util.tools import *

data_set_path = '../../DataSet/digital-device-dataset'
train_path = os.path.join(data_set_path, 'train')

device = torch.device('cuda:0')
num_classes = 3097
embedding_features = 3097

aug_norm_mean = [123.68, 116.779, 103.939]
aug_norm_std = [58.393, 57.12, 57.375]

name = 'r50-mlp-3097-amp'
output_base_path = os.path.join('./saved-output', name)

if not os.path.exists(output_base_path):
    os.mkdir(output_base_path)

model_name = os.path.join(output_base_path, 'model.pkl')
param_name = os.path.join(output_base_path, 'param.pth')
# gallery_embedding_block_name = os.path.join(output_base_path, 'gallery-embedding-block')
# query_embedding_block_name = os.path.join(output_base_path, 'query-embedding-block')


def get_test_model():
    model = Encoder(feature_num=embedding_features)
    model.load_state_dict(torch.load(param_name))

    if torch.cuda.is_available():
        model = nn.DataParallel(model).to(device=device)

    model.eval()

    return model


def encode(model, test_dataset):
    data_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=32)

    processor = tqdm(data_loader)
    embedding_block = []
    for img in processor:
        with torch.no_grad():
            embedded_feature = model(img)
        embedding_block.append(fun.normalize(embedded_feature, p=2, dim=-1))
    embedding_block = torch.cat(embedding_block, dim=0)

    return embedding_block.float()  # matmul 函数不支持fp16（HalfTensor），将其转换为fp32（FloatTensor）


def train():
    model = Encoder(feature_num=embedding_features)
    torch.save(model.state_dict(), model_name)
    model = nn.DataParallel(model)
    model.to(device=device).train()

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[m / 255. for m in aug_norm_mean], std=[s / 255. for s in aug_norm_std]),
    ])

    cla_dataset = ImageFolder(train_path, transform=train_transforms)
    data_loader = DataLoader(cla_dataset, batch_size=256, shuffle=True, num_workers=32)

    epoch = 200
    iter_per_epoch = len(data_loader)
    warm_epoch = 2

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = LinearCosineScheduler(optimizer, warm_epoch * iter_per_epoch, epoch * iter_per_epoch, max_lr=0.1)
    # arc_margin_product = ArcMarginProduct(in_features=embedding_feature, out_features=num_classes)

    scaler = amp.GradScaler()

    for epoch_idx in range(epoch):
        loss_arr = []
        processor = tqdm(data_loader)
        for data, label in processor:
            data = data.to(device=device)
            label = label.to(device=device)

            optimizer.zero_grad()

            with amp.autocast():
                cla_output = model(data)
                # loss = arc_margin_product(cla_output, label)
                loss = fun.cross_entropy(cla_output, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            loss_arr.append(float(loss))
            mean_loss = sum(loss_arr) / len(loss_arr)
            processor.set_description('  Epoch=%d/%d; mLoss=%.4f; loss=%.4f' % (epoch_idx + 1, epoch, mean_loss, float(loss)))
        torch.save(model.module.state_dict(), param_name)


def query(query_embedding_block, gallery_embedding_block, query_names_path, gallery_names_path):
    with open(query_names_path, 'r') as f:
        query_names = f.read().splitlines()

    with open(gallery_names_path, 'r') as f:
        gallery_names = f.read().splitlines()

    cosine_distance = torch.matmul(query_embedding_block, gallery_embedding_block.t())
    indices_top_10 = torch.topk(cosine_distance, k=10, dim=-1).indices.cpu().numpy()

    query_res = {}
    for idx, match_indices in enumerate(indices_top_10):
        query_name = query_names[idx]
        match_names = []

        for match_idx in match_indices:
            match_names.append(gallery_names[match_idx])
        query_res[query_name] = match_names

        query_res[query_name][0] = '{' + query_res[query_name][0]
        query_res[query_name][-1] += '}'

    with open(os.path.join(output_base_path, 'submission.csv'), 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        for query_key in query_res.keys():
            csv_writer.writerow([query_key] + query_res[query_key])


if __name__ == '__main__':
    init()

    train()

    test_model = get_test_model()

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[m / 255. for m in aug_norm_mean], std=[s / 255. for s in aug_norm_std]),
    ])

    query_dataset = TestDataSet(os.path.join(data_set_path, 'test', 'query'), './saved-output/query_images.txt', transforms=test_transforms)
    gallery_dataset = TestDataSet(os.path.join(data_set_path, 'test', 'gallery'), './saved-output/gallery_images.txt', transforms=test_transforms)

    query_embedding_block = encode(test_model, query_dataset)
    gallery_embedding_block = encode(test_model, gallery_dataset)

    query(query_embedding_block, gallery_embedding_block, './saved-output/query_images.txt', './saved-output/gallery_images.txt')
