import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
from tqdm import tqdm
from sklearn import linear_model
import scipy.stats as st
import os
import re
from PIL import Image
from model import ConvVAE
import numpy as np
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import math
from torch import nn
import argparse
from sklearn.model_selection import KFold

blacklist = ['insurance_hu_site_limit_point_granularity_55.png',
             'affect_02_MEQ_37.png']

def compute_corr(args):
    latents = args.latents
    cuda = not args.no_cuda
    img_size = args.img_size

    model = ConvVAE(latents)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    if cuda:
        model.cuda()
        FloatTensor = torch.cuda.FloatTensor
    else:
        FloatTensor = torch.FloatTensor

    perceived_dists = defaultdict(lambda: defaultdict(float))

    with open('study_chi16/ground_truth.csv') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)

        names = header[1:]

        for row in reader:
            name = row[0]

            for i, sim in enumerate(row[1:]):
                perceived_dists[name][names[i]] = float(sim)

    scags = pd.read_csv('study_chi16/scagnostics.csv')

    scag_vectors = {}
    scag_names = ['Outlying', 'Skewed', 'Clumpy', 'Sparse',
                  'Striated', 'Convex', 'Skinny', 'Stringy', 'Monotonic']

    for index, row in scags.iterrows():
        s = {}
        for t in scag_names:
            s[t] = row[t]

        name = row['file'].replace('___', '_')

        scag_vectors[name] = s

    def get_scag_vector(name):
        return np.asarray([scag_vectors[name][key] for key in scag_names])

    bvae_codes = {}

    names = [name for name in names if name not in blacklist]

    for name in names:
        path = os.path.join('study_chi16', 'images', name)
        if not os.path.exists(path):
            blacklist.append(name)
            continue

        seed = np.asarray(Image.open(path)).reshape(
            (1, 1, img_size, img_size)) / 255
        seed = FloatTensor(seed)

        seed_recon, seed_mu, _ = model(seed)
        bvae_codes[name] = seed_mu.detach().cpu().numpy().squeeze()

    perceived = []
    computed = []

    names = [name for name in names if name not in blacklist]

    X = []
    y = []

    bvae_features = args.bvae_features
    nb_scag_features = 0
    if args.use_scag:
        nb_scag_features = 9

    if bvae_features < latents:
        print(f'''WARNING: you are using only {bvae_features} latent dimensions out of {latents}. Features with large kld are discriminative and will be used. Be sure to update the kld in the source code to one you achieved during training (see bvae-log.txt)''')

    kld = [0.00, 0.00, 0.00, 0.00, 0.49, 2.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.27, 0.00, 0.36,
           1.30, 0.00, 0.30, 1.43, 0.44, 0.00, 0.49, 0.00, 0.00, 0.30, 1.05, 0.00, 0.00, 0.63, 1.90, 1.03]
    kld_sort = list(range(args.latents))
    kld_sort.sort(key=lambda x: kld[x], reverse=True)

    nb_features = bvae_features + nb_scag_features

    for name1 in tqdm(names):
        vec1 = bvae_codes[name1][kld_sort[:bvae_features]]

        if args.use_scag:
            vec1 = np.concatenate((vec1, get_scag_vector(name1)))

        for name2 in tqdm(names):
            if name1 == name2:
                continue

            vec2 = bvae_codes[name2][kld_sort[:bvae_features]]
            if args.use_scag:
                vec2 = np.concatenate((vec2, get_scag_vector(name2)))

            X.append(np.concatenate((vec1, vec2)))
            y.append(perceived_dists[name1][name2] / 100)

    mlp = torch.nn.Sequential(
        torch.nn.BatchNorm1d(nb_features * 2),
        torch.nn.Linear(nb_features * 2, args.hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, 1),
    )

    X = FloatTensor(X)
    y = FloatTensor(y)

    if cuda:
        mlp.cuda()

    kf = KFold(args.folds, shuffle=True)

    def weights_init(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        elif type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
            torch.nn.init.xavier_uniform_(m.weight)

    corr_sum = 0
    loss_sum = 0

    for train_index, test_index in tqdm(list(kf.split(X))):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
        loss_fn = torch.nn.MSELoss(reduction='sum')
        mlp.apply(weights_init)

        epochs_tqdm = tqdm(range(args.epochs))
        for t in epochs_tqdm:
            y_pred = mlp(X_train).squeeze()
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 100 == 0:
                y_pred = mlp(X_test).squeeze()
                corr_val = np.corrcoef(
                    y_pred.cpu().detach().numpy(), y_test.cpu().detach().numpy())

                epochs_tqdm.set_description('Training Loss: %.4f, Validation Corr: %.4f' %
                                            (loss.item() / len(X_train), corr_val[0][1]))

        y_pred = mlp(X_test).squeeze()
        corr_sum += np.corrcoef(y_pred.cpu().detach().numpy(),
                                y_test.cpu().detach().numpy())
        loss_sum += loss.item() / len(X_train)

    print(f'Mean Corr: {corr_sum[0,1] / args.folds}, Mean Loss: {loss_sum / args.folds}')


def parse():
    parser = argparse.ArgumentParser(
        description='Compute the correlation between predicted distances and human-perceived distances')

    parser.add_argument('model', type=str, metavar='model_path',
                        help='path to the trained model')        
    parser.add_argument('--latents', type=int, default=32, metavar='N',
                        help='number of latents (default: 32)')
    parser.add_argument('--bvae-features', type=int, default=32, metavar='N',
                        help='number of features to use (default: 32)')
    parser.add_argument('--img-size', type=int, default=64, metavar='N',
                        help='input img size (default: 64)')
    parser.add_argument('--hidden', type=int, default=32, metavar='N',
                        help='size of a hidden layer (default: 32)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--use-scag', action='store_true', default=False,
                        help='use scagnostics features?')
    parser.add_argument('--folds', type=int, default=5, metavar='N',
                        help='number of folds')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='L',
                        help='learning rate for Adam (default: 5e-3)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 10000)')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    compute_corr(args)
