import os

import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, model, data, optimizer, batch_size, device, log_dir='./results', num_generated_images=10):
        self.model = model.to(device)
        self.data = data
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self._num_generated_images = num_generated_images

        self._epoch = 0
        self._step = 0
        self._train_data = DataLoader(self.data.train_data, batch_size, shuffle=True, num_workers=2)
        self._test_data = DataLoader(self.data.test_data, batch_size, shuffle=False, num_workers=2)
        self._progress_bar = None

        self.summary = SummaryWriter(log_dir)

    def train(self, epochs):
        self._epoch = 0
        self._step = 0
        for e in range(epochs):
            self._epoch = e
            self._train_epoch()
            self._eval_and_log()
            self._save_model()

    def _train_epoch(self):
        self._progress_bar = tqdm.tqdm(self._train_data)
        self._progress_bar.set_description(f'Epoch {self._epoch}')

        self.model.train()
        for batch in self._progress_bar:
            self._train_step(batch)

    def _train_step(self, batch):
        self.optimizer.zero_grad()
        kl_div_loss, recon_loss = self._calc_loss(batch)
        loss = recon_loss + kl_div_loss
        loss.backward()
        self.optimizer.step()

        self._progress_bar.set_postfix({'loss': loss.item()})
        self.summary.add_scalar('train/recon_loss', recon_loss.item(), self._step)
        self.summary.add_scalar('train/kl_div_loss', kl_div_loss.item(), self._step)
        self.summary.add_scalar('train/loss', loss.item(), self._step)
        self._step += 1

    def _calc_loss(self, batch):
        inputs, _ = batch
        mu, log_sigma = self.model.encode(inputs)
        latent_code = self.model.bottleneck(mu, log_sigma)
        outputs = self.model.decode(latent_code)
        recon_loss = F.mse_loss(outputs, inputs, reduction='sum')
        kl_div_loss = self._kl_divergence(log_sigma, mu)

        return kl_div_loss, recon_loss

    @staticmethod
    def _kl_divergence(log_sigma, mu):
        return 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma)

    @torch.no_grad()
    def eval(self):
        self.model.eval()

        eval_loss = 0.
        for batch in self._test_data:
            eval_loss += self._eval_step(batch)
        eval_loss /= len(self.data.test_data)

        return eval_loss

    def generate(self, n):
        self.model.eval()

        samples = torch.stack([self.data.test_data[i][0] for i in range(n)])
        images = self.model(samples)
        images = torch.cat([samples, images], dim=3)

        return images

    def _eval_step(self, batch):
        _, recon_loss = self._calc_loss(batch)

        return recon_loss.item()

    def _eval_and_log(self):
        print(f'Evaluate epoch {self._epoch}: ', end='')
        eval_loss = self.eval()
        print(eval_loss)
        self.summary.add_scalar('test/loss', eval_loss, self._epoch)

        images = self.generate(n=self._num_generated_images)
        for i, img in enumerate(images):
            self.summary.add_image(f'test/{i}', img, global_step=self._epoch)

    def _save_model(self):
        save_path = os.path.join(self.log_dir, f'model_{str(self._epoch).zfill(3)}.pth')
        torch.save(self.model.cpu(), save_path)
