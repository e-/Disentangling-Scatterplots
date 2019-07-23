from model import LargeConvVAE, ConvVAE
import numpy as np
import os
import csv
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from PIL import Image
import argparse
import glob
from tqdm import tqdm

def _make_results_dir(dirpath='results'):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

def init_model(model_path, latents):
    model = ConvVAE(latents)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    return model

def flipV(tensor):
    """vertically flip the result, since the generate images (computed by np.histogram2d) have an origin at the top left corner
    """
    inv_idx = torch.arange(tensor.size(0)-1, -1, -1).long()
    return tensor.index_select(0, inv_idx)


def traverse(args):
    model = init_model(args.model, args.latents)

    FloatTensor = torch.FloatTensor

    # TODO: update kld to the final KL divergence values from training logs. 
    # kld is used to horizontally order latent dimensions in the output images (largest to smallest)
    kld = [0.00, 0.00, 0.00, 0.00, 0.49, 2.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.27, 0.00, 0.36, 1.30, 0.00, 0.30, 1.43, 0.44, 0.00, 0.49, 0.00, 0.00, 0.30, 1.05, 0.00, 0.00, 0.63, 1.90, 1.03]

    kld_sort = list(range(args.latents))
    kld_sort.sort(key=lambda x: kld[x], reverse=True)
    print(list(sorted(kld)))

    top_n = args.features
    partitions = args.partitions
    img_size = args.img_size

    _make_results_dir(args.output)

    for test_index, path in enumerate(tqdm(glob.glob(args.input))):

        seed = np.asarray(Image.open(path)).reshape((1, 1, img_size, img_size)) / 255
        seed = FloatTensor(seed)
        
        seed_recon, seed_mu, _ = model(seed)
        seed_mu = seed_mu.squeeze()

        recons = torch.zeros((top_n, partitions, img_size, img_size))

        save_image(flipV(seed_recon.view(img_size, img_size)).view(-1, 1, img_size, img_size),
            os.path.join(args.output, 'seed_%d_recon.png' % test_index))

        save_image(flipV(seed.view(img_size, img_size)).view(-1, 1, img_size, img_size),
            os.path.join(args.output, 'seed_%d_ori.png' % test_index))
    

        for i, zi in enumerate(kld_sort[:top_n]): 
            for index, val in enumerate(np.linspace(-args.range, args.range, partitions)):
                muc = seed_mu.clone().detach()
                muc[zi] = val
                recon = model.decode(muc)
                
                recons[i, index] = flipV(recon.view(img_size, img_size))

        save_image(recons.view(-1, 1, img_size, img_size), 
            os.path.join(args.output, 'seed_%d.png' % test_index), nrow=partitions, normalize=True, scale_each=False,
            padding=1, pad_value=255)

def parse():
    parser = argparse.ArgumentParser(description='Traverse the latent space of beta-VAE using test images as an input')

    parser.add_argument('model', type=str, metavar='model_path',
                        help='path to the trained model')
    parser.add_argument('--input', type=str, metavar='path', default='test_images/*.png',
                        help='path to the input images')
    parser.add_argument('--latents', type=int, default=32, metavar='N',
                        help='number of latents (default: 32)')
    parser.add_argument('--features', type=int, default=8, metavar='N',
                        help='number of features to traverse (default: 8)')
    parser.add_argument('--img-size', type=int, default=64, metavar='N',
                        help='input img size (default: 64)')    
    parser.add_argument('--partitions', type=int, default=11, metavar='N',
                        help='number of latent space partitions (default: 11)')
    parser.add_argument('--range', type=float, default=2.5, metavar='N',
                        help='the maximum value for latent traversals (default: 2.5)')    
    parser.add_argument('--output', type=str, metavar='output_path', default='traverse_result',
                        help='path to the output directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    traverse(args)