import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

from network import Net

class DuplicatedCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        img1 = img.copy()
        img2 = img.copy()
        for t in self.transforms:
            img1 = t(img1)
            img2 = t(img2)
        return img1, img2

def momentum_update(model_q, model_k, beta = 0.999):
    param_k = model_k.state_dict()
    param_q = model_q.named_parameters()
    for n, q in param_q:
        if n in param_k:
            param_k[n].data.copy_(beta*param_k[n].data + (1-beta)*q.data)
    model_k.load_state_dict(param_k)

def queue_data(data, k):
    return torch.cat([data, k], dim=0)

def dequeue_data(data, K=4096):
    if len(data) > K:
        return data[-K:]
    else:
        return data

def initialize_queue(model_k, device, train_loader):
    queue = torch.zeros((0, 128), dtype=torch.float) 
    queue = queue.to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
        x_k = data[1]
        x_k = x_k.to(device)
        k = model_k(x_k)
        k = k.detach()
        queue = queue_data(queue, k)
        queue = dequeue_data(queue, K = 10)
        break
    return queue

def train(model_q, model_k, device, train_loader, queue, optimizer, epoch, temp=0.07):
    model_q.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        x_q = data[0]
        x_k = data[1]

        x_q, x_k = x_q.to(device), x_k.to(device)
        q = model_q(x_q)
        k = model_k(x_k)
        k = k.detach()

        N = data[0].shape[0]
        K = queue.shape[0]
        l_pos = torch.bmm(q.view(N,1,-1), k.view(N,-1,1))
        l_neg = torch.mm(q.view(N,-1), queue.T.view(-1,K))

        logits = torch.cat([l_pos.view(N, 1), l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long)
        labels = labels.to(device)

        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits/temp, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        momentum_update(model_q, model_k)

        queue = queue_data(queue, k)
        queue = dequeue_data(queue)

    total_loss /= len(train_loader.dataset)

    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoCo example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    batchsize = args.batchsize
    epochs = args.epochs
    out_dir = args.out
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True}
    
    transform = DuplicatedCompose([
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(28, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    
    train_mnist = datasets.MNIST('./', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST('./', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_mnist, batch_size=batchsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_mnist, batch_size=batchsize, shuffle=True, **kwargs)
    
    model_q = Net().to(device)
    model_k = Net().to(device)
    optimizer = optim.SGD(model_q.parameters(), lr=0.01, weight_decay=0.0001)
    
    queue = initialize_queue(model_k, device, train_loader)
   
    for epoch in range(1, epochs + 1):
        train(model_q, model_k, device, train_loader, queue, optimizer, epoch)
    
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model_q.state_dict(), os.path.join(out_dir, 'model.pth'))
