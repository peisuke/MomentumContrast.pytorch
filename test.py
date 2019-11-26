import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from network import Net

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoCo example: MNIST')
    parser.add_argument('--model', '-m', default='result/model.pth',
                        help='Model file')
    args = parser.parse_args()
    model_path = args.model

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    mnist = datasets.MNIST('./', train=False, download=True, transform=transform)
    
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    data = []
    targets = []
    for m in tqdm.tqdm(mnist):
        target = m[1]
        targets.append(target)
        x = m[0]
        x = x.view(1, *x.shape)
        feat = model(x)
        data.append(feat.data.numpy()[0])
    
    ret = TSNE(n_components=2, random_state=0).fit_transform(data)
    
    plt.scatter(ret[:,0], ret[:,1], c=targets)
    plt.colorbar()
    plt.show()
