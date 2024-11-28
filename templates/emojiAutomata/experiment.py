
#neural cellular automata, Growing-Neural-Cellular-Automata is original

#handles plotting as well, keep plot.py empty and just change where writeup looks

import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#from utils.CAModel import CAModel
#from utils.utils_vis import SamplePool, to_rgb, get_living_mask, make_seed, make_circle_masks

#stuff from utils so scientist can edit it
#utils vis
class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

def to_alpha(x):
    return np.clip(x[..., 3:4], 0, 0.9999)

def to_rgb(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb, 0, 0.9999)

def get_living_mask(x):
    return nn.MaxPool2d(3, stride=1, padding=1)(x[:, 3:4, :, :])>0.1

def make_seeds(shape, n_channels, n=1):
    x = np.zeros([n, shape[0], shape[1], n_channels], np.float32)
    x[:, shape[0]//2, shape[1]//2, 3:] = 1.0
    return x

def make_seed(shape, n_channels):
    seed = np.zeros([shape[0], shape[1], n_channels], np.float32)
    seed[shape[0]//2, shape[1]//2, 3:] = 1.0
    return seed

def make_circle_masks(n, h, w):
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.random.random([2, n, 1, 1])*1.0-0.5
    r = np.random.random([n, 1, 1])*0.3+0.1
    x, y = (x-center[0])/r, (y-center[1])/r
    mask = (x*x+y*y < 1.0).astype(np.float32)
    return mask

#CAModel
class CAModel(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128):
        super(CAModel, self).__init__()

        self.device = device
        self.channel_n = channel_n

        self.fc0 = nn.Linear(channel_n*3, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        with torch.no_grad():
            self.fc1.weight.zero_()

        self.fire_rate = fire_rate
        self.to(self.device)

    def alive(self, x):
        return F.max_pool2d(x[:, 3:4, :, :], kernel_size=3, stride=1, padding=1) > 0.1

    def perceive(self, x, angle):

        def _perceive_with(x, weight):
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1,1,3,3).repeat(self.channel_n, 1, 1, 1)
            return F.conv2d(x, conv_weights, padding=1, groups=self.channel_n)

        dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0  # Sobel filter
        dy = dx.T
        c = np.cos(angle*np.pi/180)
        s = np.sin(angle*np.pi/180)
        w1 = c*dx-s*dy
        w2 = s*dx+c*dy

        y1 = _perceive_with(x, w1)
        y2 = _perceive_with(x, w2)
        y = torch.cat((x,y1,y2),1)
        return y

    def update(self, x, fire_rate, angle):
        x = x.transpose(1,3)
        pre_life_mask = self.alive(x)

        dx = self.perceive(x, angle)
        dx = dx.transpose(1,3)
        dx = self.fc0(dx)
        dx = F.relu(dx)
        dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        x = x+dx.transpose(1,3)

        post_life_mask = self.alive(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        return x.transpose(1,3)

    def forward(self, x, steps=1, fire_rate=None, angle=0.0):
        for step in range(steps):
            x = self.update(x, fire_rate, angle)
        return x

def load_emoji(index, path="emojis.png"):
    import imageio
    im = imageio.imread(path)
    emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))
    emoji /= 255.0
    return emoji

def visualize_batch(x0, x, out_dir):
    vis0 = to_rgb(x0)
    vis1 = to_rgb(x)
    fig, axs = plt.subplots(2, x0.shape[0], figsize=(15, 5))
    for i in range(x0.shape[0]):
        axs[0, i].imshow(vis0[i])
        axs[0, i].axis('off')
    for i in range(x0.shape[0]):
        axs[1, i].imshow(vis1[i])
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'batch_visualization.png'))
    plt.close()

def plot_loss(loss_log, out_dir):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log), '.', alpha=0.1)
    plt.savefig(os.path.join(out_dir, 'loss_history.png'))
    plt.close()

def train(ca, x, target, steps, optimizer, scheduler):
    x = ca(x, steps=steps)
    loss = F.mse_loss(x[:, :, :, :4], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return x, loss

def loss_f(x, target):
    return torch.mean(torch.pow(x[..., :4] - target, 2), [-2, -3, -1])

def main(args):
    # Set up directories and device
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load target emoji
    target_img = load_emoji(args.target_emoji)
    p = args.target_padding
    pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
    h, w = pad_target.shape[:2]
    pad_target = np.expand_dims(pad_target, axis=0)
    pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(device)

    # Initialize CA model and training components
    seed = make_seed((h, w), args.channel_n)
    pool = SamplePool(x=np.repeat(seed[None, ...], args.pool_size, 0))
    batch = pool.sample(args.batch_size).x
    ca = CAModel(args.channel_n, args.cell_fire_rate, device).to(device)
    if os.path.exists(args.model_path):
        ca.load_state_dict(torch.load(args.model_path))

    optimizer = optim.Adam(ca.parameters(), lr=args.lr, betas=args.betas)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
    loss_log = []

    #Have baseline metrics... compute the training time
    start_time = time.time()

    for i in range(args.n_epoch + 1):
        if args.use_pattern_pool:
            batch = pool.sample(args.batch_size)
            x0 = torch.from_numpy(batch.x.astype(np.float32)).to(device)
            loss_rank = loss_f(x0, pad_target).detach().cpu().numpy().argsort()[::-1]
            x0 = batch.x[loss_rank]
            x0[:1] = seed
            if args.damage_n:
                damage = 1.0 - make_circle_masks(args.damage_n, h, w)[..., None]
                x0[-args.damage_n:] *= damage
        else:
            x0 = np.repeat(seed[None, ...], args.batch_size, 0)
        x0 = torch.from_numpy(x0.astype(np.float32)).to(device)

        x, loss = train(ca, x0, pad_target, np.random.randint(64, 96), optimizer, scheduler)

        if args.use_pattern_pool:
            batch.x[:] = x.detach().cpu().numpy()
            batch.commit()

        step_i = len(loss_log)
        loss_log.append(loss.item())

        if step_i % 100 == 0:
            print(f"Step {step_i}, loss = {loss.item()}")
            visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy(), out_dir)
            plot_loss(loss_log, out_dir)
            torch.save(ca.state_dict(), args.model_path)

    final_metrics = {
        "final_train_loss": loss_log[-1],
        "total_train_time": time.time() - start_time,
        "learning_rate_final": scheduler.get_last_lr()[0],
        "training_epochs": args.n_epoch,
    }

    # Store results in the desired JSON format
    results = {
        "experiment": {
            "means": {
                "final_train_loss_mean": final_metrics["final_train_loss"],
                "total_train_time_mean": final_metrics["total_train_time"],
                "training_epochs": final_metrics["training_epochs"]
            },
            "final_info_dict": final_metrics,
            "losses": loss_log
        }
    }

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CA Model Training")
    parser.add_argument("--channel_n", type=int, default=16, help="Number of CA state channels")
    parser.add_argument("--target_padding", type=int, default=16, help="Target image border padding")
    parser.add_argument("--target_size", type=int, default=40, help="Target image size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--lr_gamma", type=float, default=0.9999, help="Learning rate decay")
    parser.add_argument("--betas", type=tuple, default=(0.5, 0.5), help="Optimizer betas")
    parser.add_argument("--n_epoch", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--pool_size", type=int, default=1024, help="Pattern pool size")
    parser.add_argument("--cell_fire_rate", type=float, default=0.5, help="Cell fire rate")
    parser.add_argument("--target_emoji", type=int, default=0, help="Index of target emoji")
    parser.add_argument("--experiment_type", type=str, default="Regenerating", choices=["Growing", "Persistent", "Regenerating"], help="Type of experiment")
    parser.add_argument("--use_pattern_pool", type=int, default=1, help="Use pattern pool")
    parser.add_argument("--damage_n", type=int, default=3, help="Number of patterns to damage in a batch")
    parser.add_argument("--model_path", type=str, default="mymodel.pth", help="Path to save/load model")
    parser.add_argument("--out_dir", type=str, default="output", help="Directory for output images and logs")
    args = parser.parse_args()
    print(f"Running for {args.n_epoch} training epochs...")
    main(args)
