from torch.utils.data import DataLoader
from torch import FloatTensor
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import TensorDataset
from torch import Tensor

from sklearn.manifold import TSNE
import pandas as pd
from ggplot import ggplot, aes, geom_point, ggtitle

def dataloader(dataset, input_size, batch_size, missing_mixt,split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    remove_label=missing_mixt
    if dataset == 'mnist':
        data_loader = DataLoader(datasets.MNIST('data/mnist', train=True, download=True, transform=transform),batch_size=batch_size, shuffle=True)
        data_loader.dataset.data = data_loader.dataset.data[data_loader.dataset.targets != remove_label]
        data_loader.dataset.targets = data_loader.dataset.targets[data_loader.dataset.targets != remove_label]
        print(len(data_loader.dataset.data),len(data_loader.dataset.targets))
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('data/lsun', classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'bags':
        data_loader = DataLoader(datasets.ImageFolder('./data/bags/', transform=transform),batch_size=batch_size, shuffle=True)
        #data_loader.dataset = data_loader.dataset[data_loader.dataset.targets != remove_label]

        print('OK!!',len(data_loader))
    elif dataset=='8Gaussians':
        centers = [
            (0, 1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1, 0),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (0, -1),
            (-1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1, 0),
            (-1. / np.sqrt(2), 1. / np.sqrt(2))
        ]
        centers = [(2 * x, 2 * y) for x, y in centers]
        dataset = []
        one_hot_vectors = []
        removed_mix_samples = 0
        i = 0
        while i < 10000:
            missing = [missing_mixt]
            center_idx = np.random.randint(0, len(centers))
            one_hot_vector = np.zeros(8)
            point = np.random.randn(2) * .05
            if center_idx not in missing:
                point[0] += centers[center_idx][0]
                point[1] += centers[center_idx][1]
                one_hot_vector[center_idx] = 1
                one_hot_vectors.append(center_idx)
                dataset.append(point / 1.414)
                i += 1

        dataset = np.array(dataset, dtype='float32')
        dataset = TensorDataset(Tensor(dataset), Tensor(np.asarray(one_hot_vectors)))
        data_loader = DataLoader(dataset,batch_size=batch_size, shuffle=True)

    return data_loader


def t_sne_visualize(generated,n_sne,epoch):
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    #
    # mnist_ = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    # X=mnist_.data.numpy()/255
    # y=mnist_.targets.numpy()
    # X=np.reshape(np.ravel(X), (X.shape[0], 28*28))
    n_label=7
    X_sample=generated.data.numpy()/255
    y_sample=list(range(n_label))*n_label
    X_sample=np.reshape(np.ravel(X_sample), (X_sample.shape[0], 28*28*3))

    feat_cols = [ 'pixel'+str(i) for i in range(X_sample.shape[1]) ]
    df = pd.DataFrame(X_sample,columns=feat_cols)
    df['label'] = y_sample
    df['label'] = df['label'].apply(lambda i: str(i))
    n_sne=49
    rndperm = np.concatenate((list(range(df.shape[0],df.shape[0])),np.random.permutation(df.shape[0])))
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    print('INITIALIZED')
    tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)
    print('AFTER FITTING')
    df_tsne = df.loc[rndperm[:n_sne],:].copy()
    df_tsne['x-tsne'] = tsne_results[:,0]
    df_tsne['y-tsne'] = tsne_results[:,1]

    chart=ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) \
            + geom_point(size=70, alpha =0.7) \
            + ggtitle("tSNE dimensions colored by digit")
    chart.save("tsne"+str(epoch)+".png")

    return