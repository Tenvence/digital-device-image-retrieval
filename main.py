# python -m torch.distributed.launch --nproc_per_node=4 main.py

import os
import csv
import warnings

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
# import torch.distributed
import torch.nn.functional as fun
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torch.cuda.amp as amp
from PIL import ImageFile
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model.encoder import Encoder
from util.data_set import TestDataSet
from util.lr_scheduler import LinearCosineScheduler
from util.metrics import AdaCos, ArcMarginProduct

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# torch.distributed.init_process_group(backend='nccl')

data_set_path = '../../DataSet/digital-device-dataset'
train_path = os.path.join(data_set_path, 'train')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 3097
embedding_feature = 3097
aug_norm_mean = [123.68, 116.779, 103.939]
aug_norm_std = [58.393, 57.12, 57.375]

name = 'test'
output_base_path = os.path.join('./saved-output', name)

if not os.path.exists(output_base_path):
    os.mkdir(output_base_path)

model_name = os.path.join(output_base_path, 'model.pkl')
param_name = os.path.join(output_base_path, 'param.pkl')
gallery_embedding_block_name = os.path.join(output_base_path, 'gallery-embedding-block')
query_embedding_block_name = os.path.join(output_base_path, 'query-embedding-block')


def encode(test_dataset, embedding_block_name):
    model = Encoder(feature_num=embedding_feature)
    model.load_state_dict(torch.load(param_name))

    if torch.cuda.is_available():
        model = nn.DataParallel(model).to(device=device)
        # model = nn.parallel.DistributedDataParallel(model)

    model.eval()

    # sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

    processor = tqdm(data_loader)
    embedding_block = []
    for img in processor:
        with torch.no_grad():
            embedded_feature = model(img)
        embedding_block.append(fun.normalize(embedded_feature, p=2, dim=-1))
    embedding_block = torch.cat(embedding_block, dim=0)
    np.save(embedding_block_name, embedding_block.cpu().numpy())
    return embedding_block


def train():
    model = Encoder(feature_num=embedding_feature)
    torch.save(model.state_dict(), model_name)

    if torch.cuda.is_available():
        # model = model.cuda()
        model = nn.DataParallel(model).to(device=device)
        # model = nn.parallel.DistributedDataParallel(model.to(device=device))

    model.train()

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[m / 255. for m in aug_norm_mean], std=[s / 255. for s in aug_norm_std]),
    ])

    cla_dataset = ImageFolder(train_path, transform=train_transforms)
    # sampler = torch.utils.data.distributed.DistributedSampler(cla_dataset)
    data_loader = DataLoader(cla_dataset, batch_size=256, shuffle=True, num_workers=32)

    epoch = 200
    iter_per_epoch = len(data_loader)
    warm_epoch = 2

    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = LinearCosineScheduler(optimizer, warm_epoch * iter_per_epoch, epoch * iter_per_epoch, max_lr=0.1)
    arc_margin_product = ArcMarginProduct(in_features=embedding_feature, out_features=num_classes)

    # scaler = amp.GradScaler()

    for epoch_idx in range(epoch):
        loss_arr = []
        processor = tqdm(data_loader)
        for data, label in processor:
            data = data.to(device=device)
            label = label.to(device=device)

            optimizer.zero_grad()

            # with amp.autocast():
            cla_output = model(data)
            # loss = arc_margin_product(cla_output, label)
            loss = fun.cross_entropy(cla_output, label)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            loss.backward()

            optimizer.step()
            scheduler.step()

            loss_arr.append(float(loss))
            mean_loss = sum(loss_arr) / len(loss_arr)
            processor.set_description('  Epoch=%d/%d; mLoss=%.4f; loss=%.4f' % (epoch_idx + 1, epoch, mean_loss, float(loss)))
        torch.save(model.module.state_dict(), param_name)


def query(query_embedding_path, gallery_embedding_path, query_names_path, gallery_names_path):
    query_data = np.load(query_embedding_path)
    gallery_data = np.load(gallery_embedding_path)

    with open(query_names_path, 'r') as f:
        query_names = f.read().splitlines()

    with open(gallery_names_path, 'r') as f:
        gallery_names = f.read().splitlines()

    query_data = torch.tensor(query_data)
    gallery_data = torch.tensor(gallery_data)

    cosine_distance = torch.matmul(query_data, gallery_data.t())
    indices_top_10 = torch.topk(cosine_distance, k=10, dim=-1).indices.numpy()

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
    train()

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[m / 255. for m in aug_norm_mean], std=[s / 255. for s in aug_norm_std]),
    ])

    query_dataset = TestDataSet(os.path.join(data_set_path, 'test', 'query'), './saved-output/query_images.txt', transforms=test_transforms)
    gallery_dataset = TestDataSet(os.path.join(data_set_path, 'test', 'gallery'), './saved-output/gallery_images.txt', transforms=test_transforms)

    query_embedding_block = encode(query_dataset, query_embedding_block_name)
    gallery_embedding_block = encode(gallery_dataset, gallery_embedding_block_name)

    query(query_embedding_block_name + '.npy', gallery_embedding_block_name + '.npy', './saved-output/query_images.txt', './saved-output/gallery_images.txt')
