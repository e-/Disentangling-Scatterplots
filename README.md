Disentangled Representation of Data Distributions in Scatterplots
===

This repository demonstrates source codes and the dataset used in the paper "Disentangled Representation of Data Distributions in Scatterplots (IEEE InfoVis 2019 Short, [PDF](https://github.com/e-/Disentangling-Scatterplots/blob/master/paper.pdf))".

## Dependency

We strongly recommend using [Anaconda Python](https://www.anaconda.com/) (>=3.6). Important dependencies are listed below.

```
torch==1.0.0
torchvision==0.2.2
```

## Download

1. Clone this repository.
2. Download the scatterplot images: [https://www.kaggle.com/jaeminjo/scatterplotimages](https://www.kaggle.com/jaeminjo/scatterplotimages)
3. Create a subdirectory named `all` under the `images` directory, unzip the image archive, and put the images in the subdirectory. The source codes will look up `images/all/*.png` for training images by default.

## Train the Network

To train a β-VAE on the training images, run `python train.py`. This will generate a Torch model in the root directory (`bvae.pt` by default) as well as training logs (in a subdirectory`training_logs/` by default). Note that the model and logs will be overwritten if you rerun the code. We also provide a pre-trained model, `bvae_pretrained.pt`.

```
usage: train.py [-h] [--data path] [--eta L] [--beta B] [--latents N]
                [--batch-size N] [--img-size N] [--epochs N] [--no-cuda]
                [--log-interval N] [--save-interval T] [--log-path path]
                [--save-path path] [--resume path]

Train a beta-VAE using scatterplot images

optional arguments:
  -h, --help         show this help message and exit
  --data path        path to the scatterplot dataset (default: images)
  --eta L            learning rate for Adam (default: 1e-3)
  --beta B           the beta coefficient (default: 4)
  --latents N        number of latents (default: 32)
  --batch-size N     input batch size for training (default: 128)
  --img-size N       input image size (default: 64)
  --epochs N         number of epochs to train (default: 100)
  --no-cuda          disable CUDA training
  --log-interval N   how many batches to wait before logging training status
  --save-interval T  how many batches to wait before saving latent traversal
  --log-path path    output directory for training logs
  --save-path path   output path for the model
  --resume path      continue learning (from path)
```

## Latent Traversals 

To traverse the latent space, run `python traverse.py <path_to_model>`, e.g., `python traverse.py bvae_pretrained.pt`. This will perform latent traversals using images in the directory `test_images` as an input. The results will be saved in a separate directory (`traverse_result` by default). To reproduce Figure 1 in the paper, open `figure.html` on a Web browser after running the code. 

**Important Notes**: Since the training procedure is stochastic, the order of latent dimensions from the largest KL divergence to the smallest will change if you rerun the training procedure. To include only the top `N` latent dimensions with the largest KL divergences in visualization, you need to provide the KL divergence of each latent dimension obtained from the training procedure. Open `training_logs/log.txt` and go to the last line. Copy the KL divergences (it is an array of as many floating values as the number of latent dimensions you used) and paste it into `traverse.py`, `corr.py`, and `figure.html` (search for 'TODO' in the codes to find the exact line). If you are using the pre-trained model, you can skip this.

```
usage: traverse.py [-h] [--input path] [--latents N] [--features N]
                   [--img-size N] [--partitions N] [--range N]
                   [--output output_path]
                   model_path

Traverse the latent space of beta-VAE using test images as an input

positional arguments:
  model_path            path to the trained model

optional arguments:
  -h, --help            show this help message and exit
  --input path          path to the input images
  --latents N           number of latents (default: 32)
  --features N          number of features to traverse (default: 8)
  --img-size N          input img size (default: 64)
  --partitions N        number of latent space partitions (default: 11)
  --range N             the maximum value for latent traversals (default: 2.5)
  --output output_path  path to the output directory
```

## Compute the Correlation between Predicted and Perceived Distances

Run `python corr.py <path_to_model>`, e.g., `python corr.py bvae_pretrained.pt` to measure Pearson's *ρ* between human-perceived distances and predicted distances. By default, the script will train a neural network with a single hidden layer (with 32 neurons) to approximate the distance. All the latent dimensions will be used, but if you want to use the top `N` latent dimensions with the largest KL divergences, use the ``--bvae-features N`` flag. If you built your own model, you must provide the KL divergences of the latent dimensions. See **Important Notes** above. If you want to include scagnostics measures in the input, use the ``--use-scag`` flag.

Here are some results:

|Name|MSE Loss (Validation)|Pearson's *ρ*|Flag|
|-|-|-|-|
|Scagnostics|0.124|0.644|`--use-scag --bvae-features 0`|
|β-VAE|0.087|0.710|Empty|
|β-VAE+Scagnostics|0.070|0.750|`--use-scag`|


```
usage: corr.py [-h] [--latents N] [--bvae-features N] [--img-size N]
               [--hidden N] [--no-cuda] [--use-scag] [--folds N] [--lr L]
               [--epochs N]
               model_path

Compute the correlation between predicted distances and human-perceived
distances

positional arguments:
  model_path         path to the trained model

optional arguments:
  -h, --help         show this help message and exit
  --latents N        number of latents (default: 32)
  --bvae-features N  number of features to use (default: 32)
  --img-size N       input img size (default: 64)
  --hidden N         size of a hidden layer (default: 32)
  --no-cuda          disables CUDA training
  --use-scag         use scagnostics features?
  --folds N          number of folds
  --lr L             learning rate for Adam (default: 5e-3)
  --epochs N         number of epochs to train (default: 10000)
```

## Files

- `study_chi16` contains the result, such as the human-perceived distances between scatterplots, from a previous study, ["Towards Understanding Human Similarity Perception in the Analysis of Large Sets of Scatterplots (ACM CHI 16)"](https://dl.acm.org/citation.cfm?id=2858155) by Pandey et al. The original repository is [https://github.com/nyuvis/scatter-plot-similarity](https://github.com/nyuvis/scatter-plot-similarity).
