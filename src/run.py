import torch

import dataset
import model
import trainer


def run(network_type, bottleneck_dim, lr, batch_size, epochs, device, log_dir):
    mnist_data = dataset.MyMNIST()

    if network_type == 'mlp':
        net = model.MLPVAE((1, 32, 32), bottleneck_dim)
    elif network_type == 'cnn':
        net = model.CNNVAE((1, 32, 32), bottleneck_dim)
    else:
        raise ValueError(f'Unsupported network type {network_type}. Chose between "mlp" and "cnn".')

    optim = torch.optim.Adam(net.parameters(), lr)
    vae_trainer = trainer.Trainer(net, mnist_data, optim, batch_size, device, log_dir)
    vae_trainer.train(epochs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the training for a VAE.')
    parser.add_argument('-t', '--network_type', required=True, choices=['mlp', 'cnn'], help='type of the VAE network')
    parser.add_argument('-n', '--bottleneck_dim', default=16, type=int, help='size of the VAE bottleneck')
    parser.add_argument('-r', '--lr', default=0.001, type=float, help='learning rate for training')
    parser.add_argument('-b', '--batch_size', required=True, type=int, help='batch size for training')
    parser.add_argument('-e', '--epochs', required=True, type=int, help='epochs to train')
    parser.add_argument('-d', '--device', default='cpu', help='device to train on, e.g. "cuda:0"')
    parser.add_argument('-l', '--logdir', default='./results', help='directory to log the models and event file to')
    opt = parser.parse_args()

    run(opt.network_type, opt.bottleneck_dim, opt.lr, opt.batch_size, opt.epochs, opt.device, opt.logdir)
