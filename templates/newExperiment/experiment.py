
#neural cellular automata, Growing-Neural-Cellular-Automata is original

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


from utils.CAModel import CAModel

from utils.utils_vis import SamplePool, to_rgb, get_living_mask, make_seed, make_circle_masks

def load_emoji(index, path="emojis.png"):
    import imageio
    im = imageio.imread(path)
    emoji = np.array(im[:, index*40:(index+1)*40].astype(np.float32))
    emoji /= 255.0
    return emoji

def visualize_batch(x0, x, output_dir):
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
    plt.savefig(os.path.join(output_dir, f'batch_visualization.png'))
    plt.close()

def plot_loss(loss_log, output_dir):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log), '.', alpha=0.1)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
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
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
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
            visualize_batch(x0.detach().cpu().numpy(), x.detach().cpu().numpy(), output_dir)
            plot_loss(loss_log, output_dir)
            torch.save(ca.state_dict(), args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CA Model Training")
    parser.add_argument("--channel_n", type=int, default=16, help="Number of CA state channels")
    parser.add_argument("--target_padding", type=int, default=16, help="Target image border padding")
    parser.add_argument("--target_size", type=int, default=40, help="Target image size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--lr_gamma", type=float, default=0.9999, help="Learning rate decay")
    parser.add_argument("--betas", type=tuple, default=(0.5, 0.5), help="Optimizer betas")
    parser.add_argument("--n_epoch", type=int, default=80000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--pool_size", type=int, default=1024, help="Pattern pool size")
    parser.add_argument("--cell_fire_rate", type=float, default=0.5, help="Cell fire rate")
    parser.add_argument("--target_emoji", type=int, default=0, help="Index of target emoji")
    parser.add_argument("--experiment_type", type=str, default="Regenerating", choices=["Growing", "Persistent", "Regenerating"], help="Type of experiment")
    parser.add_argument("--use_pattern_pool", type=int, default=1, help="Use pattern pool")
    parser.add_argument("--damage_n", type=int, default=3, help="Number of patterns to damage in a batch")
    parser.add_argument("--model_path", type=str, default="mymodel.pth", help="Path to save/load model")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for output images and logs")
    args = parser.parse_args()

    main(args)
