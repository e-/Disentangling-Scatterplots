import os
import shutil
import time
import argparse
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from model import ConvVAE
from tqdm import tqdm

def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    elif type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight)

def make_results_dir(dirpath):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

    os.mkdir(dirpath)

def traverse_latents(args, model, datapoint, latents, epoch_nb, batch_idx, nb_partition=14):
    model.eval()

    datapoint = datapoint.unsqueeze(0).unsqueeze(1)
    mu, _ = model.encode(datapoint)

    recons = torch.zeros(
        (nb_partition, latents, args.img_size, args.img_size))
    for zi in range(latents):
        muc = mu.squeeze().clone()
        for i, val in enumerate(np.linspace(-3, 3, nb_partition)):
            muc[zi] = val
            recon = model.decode(muc).cpu()
            recons[i, zi] = recon.view(args.img_size, args.img_size)

    filename = os.path.join(args.log_path, 'traversal_' +
                            str(epoch_nb) + '_' + str(batch_idx) + '.png')
    save_image(recons.view(-1, 1, args.img_size, args.img_size),
               filename, nrow=latents, pad_value=1)


def recons_random(args, model, sample_imgs, latents, epoch_nb, batch_idx):
    model.eval()
    sample_imgs = sample_imgs[:100]

    original_path = os.path.join(
        args.log_path, '%d_%d_original.png' % (epoch_nb, batch_idx))
    recons_path = os.path.join(
        args.log_path, '%d_%d_recons.png' % (epoch_nb, batch_idx))

    save_image(sample_imgs, original_path, nrow=10, pad_value=1)
    recon_imgs, mu, logvar = model.forward(sample_imgs)
    save_image(recon_imgs, recons_path, nrow=10, pad_value=1)


def loss_function(recon_x, x, mu, logvar, beta):
    bce = F.binary_cross_entropy(
        recon_x.view(-1, args.img_size ** 2), x.view(-1, args.img_size ** 2), reduction='sum')
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kld.mean(dim=0), bce + beta*kld.sum()


def train(args):
    make_results_dir(args.log_path)
    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            root=args.data,
            transform=transforms.Compose([
                transforms.Grayscale(1),
                transforms.ToTensor(),
            ])
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    testpoint = torch.Tensor(dataloader.dataset[0][0]).view(
        args.img_size, args.img_size)
    if not args.no_cuda:
        testpoint = testpoint.cuda()

    model = ConvVAE(args.latents)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    else:
        model.apply(weights_init)

    model.train()
    if not args.no_cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.eta)

    runloss, runkld = None, np.array([])
    start_time = time.time()

    log = open(os.path.join(args.log_path, 'log.txt'), 'a+')

    print(args, file=log)
    print(model.__class__, file=log)

    for epoch_nb in range(1, args.epochs + 1):
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.view(-1, 1, 64, 64)
            if not args.no_cuda:
                data = data.cuda()

            recon_batch, mu, logvar = model(data)
            kld, loss = loss_function(
                recon_batch, data, mu, logvar, args.beta)

            # param update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss /= len(data)
            runloss = loss if not runloss else runloss*0.9 + loss*0.1
            runkld = np.zeros(args.latents) if not len(
                runkld) else runkld*0.9 + kld.data.cpu().numpy()*0.1

            if not batch_idx % args.log_interval:
                print('Epoch {}, batch: {}/{} ({:.2f} s), loss: {:.2f}, kl: [{}]'.format(
                    epoch_nb, batch_idx, len(
                        dataloader), time.time() - start_time, runloss,
                    ', '.join('{:.2f}'.format(kl) for kl in runkld)))
                print('Epoch {}, batch: {}/{} ({:.2f} s), loss: {:.2f}, kl: [{}]'.format(
                    epoch_nb, batch_idx, len(
                        dataloader), time.time() - start_time, runloss,
                    ', '.join('{:.2f}'.format(kl) for kl in runkld)), file=log)
                start_time = time.time()

            if not batch_idx % args.save_interval:
                traverse_latents(args, model, testpoint,
                                 args.latents, epoch_nb, batch_idx)
                recons_random(args, model, data, args.latents,
                              epoch_nb, batch_idx)

                torch.save(model.state_dict(), args.save_path)
                model.train()

    log.close()


def parse():
    parser = argparse.ArgumentParser(
        description='Train a beta-VAE using scatterplot images')
    parser.add_argument('--data', type=str, metavar='path', default='images',
                        help='path to the scatterplot dataset (default: images)')
    parser.add_argument('--eta', type=float, default=1e-3, metavar='L',
                        help='learning rate for Adam (default: 1e-3)')
    parser.add_argument('--beta', type=float, default=4.0, metavar='B',
                        help='the beta coefficient (default: 4)')
    parser.add_argument('--latents', type=int, default=32, metavar='N',
                        help='number of latents (default: 32)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--img-size', type=int, default=64, metavar='N',
                        help='input image size (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=300, metavar='T',
                        help='how many batches to wait before saving latent traversal')
    parser.add_argument('--log-path', type=str, default='training_logs', metavar='path',
                        help='output directory for training logs')
    parser.add_argument('--save-path', type=str, default='bvae.pt', metavar='path',
                        help='output path for the model')
    parser.add_argument('--resume', type=str, default=None,
                        metavar='path', help='continue learning (from path)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    train(args)
