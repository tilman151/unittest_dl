import torch


class ModelTestsMixin:
    @torch.no_grad()
    def test_shape(self):
        outputs = self.net(self.test_inputs)
        self.assertEqual(self.test_inputs.shape, outputs.shape)

    @torch.no_grad()
    def test_device_moving(self):
        net_on_gpu = self.net.to('cuda:0')

        torch.manual_seed(42)
        outputs_cpu = self.net(self.test_inputs)
        torch.manual_seed(42)
        outputs_gpu = net_on_gpu(self.test_inputs.to('cuda:0'))

        self.assertAlmostEqual(0., torch.sum(outputs_cpu - outputs_gpu.cpu()))

    def test_batch_independence(self):
        inputs = self.test_inputs.clone()
        inputs.requires_grad = True

        # Compute forward pass in eval mode to deactivate batch norm
        self.net.eval()
        outputs = self.net(inputs)
        self.net.train()

        # Mask loss for certain samples in batch
        batch_size = inputs[0].shape[0]
        mask_idx = torch.randint(0, batch_size, ())
        mask = torch.ones_like(outputs)
        mask[mask_idx] = 0
        outputs = outputs * mask

        # Compute backward pass
        loss = outputs.mean()
        loss.backward()

        # Check if gradient exists and is zero for masked samples
        for i, grad in enumerate(inputs.grad):
            if i == mask_idx:
                self.assertTrue(torch.all(grad == 0).item())
            else:
                self.assertTrue(not torch.all(grad == 0))

    def test_all_parameters_updated(self):
        optim = torch.optim.SGD(self.net.parameters(), lr=0.1)

        optim.zero_grad()
        outputs = self.net(self.test_inputs)
        loss = outputs.mean()
        loss.backward()

        for param in self.net.parameters():
            self.assertIsNotNone(param.grad)
            self.assertNotEqual(0., torch.sum(param.grad ** 2))

        # Clear gradients to make test independent
        optim.zero_grad()
