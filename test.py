import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from network import Net

def show(mnist, targets, ret):
    target_ids = range(len(set(targets)))
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'violet', 'orange', 'purple']
    
    plt.figure(figsize=(12, 10))
    
    ax = plt.subplot(aspect='equal')
    for label in set(targets):
        idx = np.where(np.array(targets) == label)[0]
        plt.scatter(ret[idx, 0], ret[idx, 1], c=colors[label], label=label)
    
    for i in range(0, len(targets), 250):
        img = (mnist[i][0] * 0.3081 + 0.1307).numpy()[0]
        img = OffsetImage(img, cmap=plt.cm.gray_r, zoom=0.5) 
        ax.add_artist(AnnotationBbox(img, ret[i]))
    
    plt.legend()
    plt.show()

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
   
    show(mnist, targets, ret)
