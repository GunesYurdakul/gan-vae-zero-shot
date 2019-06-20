from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from ggplot import ggplot, aes, geom_point, ggtitle
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--remove-label', type=int, default=-1, metavar='N',
                    help='label to remove')
parser.add_argument('--few-shot', type=int, default=0,
                    help='label to remove')
parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'bags', 'fashion-mnist'],
                    help='The name of dataset')
parser.add_argument('--input-size', type=int, default=784, metavar='N',
                    help='input size 784 for mnist')
parser.add_argument('--encoding-vector-size', type=int, default=20, metavar='N',
                    help='size of encoding vector')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'mnist':
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
elif args.dataset == 'fashion-mnist':
    print('fashion')
    train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('../data', train=True, download=True,transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    if args.remove_label!=-1:
        print('remove label')
#remove samples for zero shot
zero_shot_data = train_loader.dataset.data[train_loader.dataset.targets != args.remove_label]
zero_shot_label = train_loader.dataset.targets[train_loader.dataset.targets != args.remove_label]

if args.few_shot>0:
    #get samples of removed class
    one_shot_data = train_loader.dataset.data[train_loader.dataset.targets == args.remove_label]
    one_shot_label = train_loader.dataset.targets[train_loader.dataset.targets == args.remove_label]
    save_image(one_shot_data[:args.few_shot].view(args.few_shot, 1, 28, 28),
               'results/mnist_1_samples_repeated_removed_label_' + str(args.remove_label) + '.png')
    for i in range(1000):
        #add the same first args.few_shot samples of removed label many times
        zero_shot_data = torch.cat((zero_shot_data, one_shot_data[:args.few_shot]), 0)
        zero_shot_label=torch.cat((zero_shot_label, one_shot_label[:args.few_shot]), 0)

    train_loader.dataset.data = zero_shot_data
    train_loader.dataset.targets = zero_shot_label
else:
    train_loader.dataset.data = zero_shot_data
    train_loader.dataset.targets = zero_shot_label

#VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.input_size  = args.input_size
        self.encoding_vector_size  = args.encoding_vector_size

        self.fc1 = nn.Linear(self.input_size, 400)
        self.fc21 = nn.Linear(400, self.encoding_vector_size)
        self.fc22 = nn.Linear(400, self.encoding_vector_size)
        self.fc3 = nn.Linear(self.encoding_vector_size, 400)
        self.fc4 = nn.Linear(400, self.input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    z_vector = torch.empty(20)
    label_vector = torch.empty(20)
    logvar_vector = torch.empty(20)
    mu_vector = torch.empty(20)
    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
        if batch_idx==0:
            z_vector = z
            mu_vector = mu
            logvar_vector = logvar
            label_vector = label
        else:
            z_vector = torch.cat((z_vector, z), 0)
            mu_vector = torch.cat((mu_vector, mu), 0)
            logvar_vector = torch.cat((logvar_vector, logvar), 0)
            label_vector = torch.cat((label_vector, label), 0)
    #scatter_2d(mu_vector, label_vector, epoch, "mu-fewshot")
    #scatter_2d(logvar_vector, label_vector, epoch, "var-fewshot")
    #scatter_2d(z_vector, label_vector, epoch, "z-fewshot")
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        z_vector = torch.empty(20)
        label_vector = torch.empty(20)
        logvar_vector = torch.empty(20)
        mu_vector = torch.empty(20)
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, z = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 10)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/fashion-mnist/mnist-reconstruction_' + str(epoch) + '.png', nrow=n)
                z_vector = z
                mu_vector = mu
                logvar_vector=logvar
                label_vector=label
            else:
                z_vector = torch.cat((z_vector, z), 0)
                mu_vector = torch.cat((mu_vector, mu), 0)
                logvar_vector = torch.cat((logvar_vector, logvar), 0)
                label_vector = torch.cat((label_vector, label), 0)
        #t_sne_visualize(z_vector, label_vector, epoch)
#        scatter_2d(mu_vector, label_vector, epoch,"mu")
 #       scatter_2d(logvar_vector, label_vector, epoch, "var")
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def t_sne_visualize(latent_vectors,labels,epoch):
    print(latent_vectors.shape)
    X_sample=latent_vectors.data.numpy()/255
    feat_cols = [ 'pixel'+str(i) for i in range(X_sample.shape[1]) ]
    nsne=1000
    df = pd.DataFrame(X_sample,columns=feat_cols)
    df['label'] = labels
    df['label'] = df['label'].apply(lambda i: str(i))
    rndperm = np.concatenate((list(range(df.shape[0],df.shape[0])),np.random.permutation(df.shape[0])))
    tsne = TSNE(n_components=2, verbose=1, perplexity=30)
    print('INITIALIZED')
    tsne_results = tsne.fit_transform(df.loc[rndperm[:nsne],feat_cols].values)
    print('AFTER FITTING')
    df_tsne = df.loc[rndperm[:nsne],:].copy()
    df_tsne['x-tsne'] = tsne_results[:,0]
    df_tsne['y-tsne'] = tsne_results[:,1]

    chart=ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) \
            + geom_point(size=70, alpha =0.7) \
            + ggtitle("tSNE dimensions colored by digit")
    chart.save(str(args.dataset)+"tsne-vae/2d-vec-miss"+ str(args.remove_label)+"/tsne"+str(epoch)+".png")

    return

def scatter_2d(latent_vectors,labels,epoch,name):
    print(latent_vectors.shape)

    colors=['b', 'g', 'r', 'c', 'm', 'y','k', 'darkorange', 'lime', 'magenta', 'pink', 'royalblue']
    plt.clf()
    for i in range(10):
        X_sample = latent_vectors[labels==i]
        X_sample=X_sample.data.numpy()
        if i==args.remove_label:
            print('NUMBER OF SAMPLES',len(X_sample))
        plt.scatter(X_sample[:,0],X_sample[:,1],color=colors[i], label=i, alpha=0.3)
    plt.legend()
#    plt.show()
    plt.savefig(str(args.dataset)+"tsne-vae/"+name+"/2d_vec_"+str(epoch)+'.png')

    return

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, args.encoding_vector_size).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/fashion-mnist/mnist-sample_'+str(args.remove_label)+ '_' + str(epoch) + '.png')
    print('will be saved')
    torch.save(model.state_dict(), 'mnist_5shot_repeated_missing_vector2_'+str(args.remove_label)+'.pt')
    print("model saved")
