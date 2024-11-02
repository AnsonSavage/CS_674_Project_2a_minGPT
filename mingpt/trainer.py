"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, checkpoint_path=None):
        self.train_dataset = train_dataset
        if checkpoint_path is not None:
            print('Loading checkpoint from', checkpoint_path)
            print('Ignoring passed config and model')
            self.load_checkpoint(checkpoint_path)
        else:
            self.config = config
            self.model = model
            # setup the optimizer
            self.optimizer = model.configure_optimizers(config)
            self.callbacks = defaultdict(list)

            # determine the device we'll train on
            if config.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = config.device
            self.model = self.model.to(self.device)
            print("running on device", self.device)

            # variables that will be assigned to trainer class later for logging and etc
            self.iter_num = 0
            self.iter_time = 0.0
            self.iter_dt = 0.0
            self.loss_history = []

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)
    
    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
    
    def checkpoint(self, path):
        assert path.endswith('.pth'), "checkpoint path should be a .pth file"
        path.replace('.pth', f'_{self.iter_num}.pth')
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_num': self.iter_num,
            'iter_time': self.iter_time,
            'iter_dt': self.iter_dt,
            'config': self.config,
            'loss_history': self.loss_history,
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_num = checkpoint['iter_num']
        self.iter_time = checkpoint['iter_time']
        self.iter_dt = checkpoint['iter_dt']
        self.config = checkpoint['config']
        self.loss_history = checkpoint['loss_history']

    def run(self):
        model, config = self.model, self.config

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
                self.trigger_callbacks('on_epoch_end')
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            self.loss_history.append(self.loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end') # I'm assuming we'd just add our checkpointing code as a callback here
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
