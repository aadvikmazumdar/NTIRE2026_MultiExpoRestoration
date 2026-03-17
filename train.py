import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os, time, random
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm
to_pil_image = transforms.ToPILImage()

from DataLoader.rawfusion_dataset import RawFusionDataset
from cidautai_va_v2 import DarkIRFusion, DarkIRFusionLoss
from utils.utils import *
from utils.checkpoint import *


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate(model, cuda):
    val_set = RawFusionDataset(
        input_dir='datasets/val_input/',
        gt_dir='datasets/val_gt/',
        crop_size=None,
        augment=False
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    total_psnr = 0
    with torch.no_grad():
        for burst, gt in val_loader:
            if cuda:
                burst, gt = burst.cuda(), gt.cuda()
            pred, _ = model(burst)
            pred = torch.clamp(pred, 0, 1)
            total_psnr += calculate_psnr(pred, gt)
    model.train()
    return total_psnr / len(val_set)


def train(num_threads=1, cuda=True, restart_train=False, mGPU=False):
    seed = 86395
    set_seed(seed)
    print(f"*** FIXED SEED: {seed} ***")
    torch.set_num_threads(num_threads)

    batch_size = 1
    lr = 1e-3
    lr_min = 1e-7
    weight_decay = 1e-3
    n_epoch = 800
    crop_size = (384, 768)

    checkpoint_dir = 'checkpoint_dir_v2_800'
    output_dir = 'output_v2_800'
    logs_dir = 'logs_dir_v2_800'
    for d in [checkpoint_dir, output_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    data_set = RawFusionDataset(
        input_dir='datasets/trn/',
        gt_dir='datasets/trn/',
        crop_size=crop_size,
        augment=True
    )
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
    print(f"Train: {len(data_set)} scenes, {len(data_loader)} batches/epoch")

    model = DarkIRFusion(width=32, align_nf=8, align_r=4)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {num_params/1e6:.3f}M | limit: 5M | {'PASS' if num_params < 5e6 else 'FAIL'}")

    if cuda:
        model = model.cuda()
    if mGPU:
        model = nn.DataParallel(model)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=lr_min)
    criterion = DarkIRFusionLoss(edge_weight=50.0, guide_weight=1.0)

    start_epoch = 0
    global_step = 0
    best_val_psnr = 0.0
    best_epoch = 0

    if not restart_train:
        try:
            ckpt = load_checkpoint('checkpoint_dir_v2_800', 'best')
            start_epoch = 0
            global_step = ckpt['global_iter']
            best_val_psnr = ckpt.get('best_val_psnr', 0.0)
            best_epoch = ckpt.get('best_epoch', 0)
            model.load_state_dict(ckpt['state_dict'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer'])
            for pg in optimizer.param_groups: pg['lr'] = lr
            print(f'Resumed from epoch {start_epoch}, best_val={best_val_psnr:.2f} @ epoch {best_epoch}')
        except Exception as e:
            print(f'No checkpoint: {e}, training from scratch.')

    print('\n--- cidautai_va3 Training started ---\n')

    for epoch in range(start_epoch, n_epoch):
        model.train()
        epoch_t0 = time.time()
        lr_cur = optimizer.param_groups[0]['lr']
        epochs_done = epoch - start_epoch
        epochs_total = n_epoch - start_epoch
        epoch_pct = 100.0 * epochs_done / epochs_total

        print(f'\nEpoch [{epoch}/{n_epoch}] ({epoch_pct:.1f}% of run) | lr={lr_cur:.2e} | best_val={best_val_psnr:.2f}dB @ epoch {best_epoch}')

        avg_loss, avg_psnr, avg_ssim, steps = 0, 0, 0, 0

        pbar = tqdm(data_loader,
                    desc=f'  E{epoch}',
                    ncols=100,
                    unit='batch',
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')

        for step, (burst, gt) in enumerate(pbar):
            if cuda:
                burst, gt = burst.cuda(), gt.cuda()

            pred, guide = model(burst)
            pred = torch.clamp(pred, 0, 1)
            loss, _ = criterion(pred, guide, gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            psnr = calculate_psnr(pred, gt)
            ssim = calculate_ssim(pred, gt)
            avg_loss += loss.item()
            avg_psnr += psnr
            avg_ssim += ssim
            steps += 1

            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             psnr=f'{psnr:.2f}',
                             ssim=f'{ssim:.4f}')
            global_step += 1

        scheduler.step()
        epoch_time = time.time() - epoch_t0
        print(f'  Done in {epoch_time:.0f}s | '
              f'avg loss={avg_loss/steps:.4f} PSNR={avg_psnr/steps:.2f} SSIM={avg_ssim/steps:.4f}')

        if epoch % 50 == 0:
            with torch.no_grad():
                to_pil_image(gt[0].cpu()).save(f'{output_dir}/E{epoch}_gt.png')
                to_pil_image(pred[0].cpu()).save(f'{output_dir}/E{epoch}_pred.png')

        if epoch % 10 == 0:
            val_psnr = validate(model, cuda)
            is_best = val_psnr > best_val_psnr
            if is_best:
                best_val_psnr = val_psnr
                best_epoch = epoch
            print(f'  *** Val PSNR: {val_psnr:.2f} dB (best val: {best_val_psnr:.2f} dB @ epoch {best_epoch}) ***')

            save_dict = {
                'epoch': epoch,
                'best_epoch': best_epoch,
                'global_iter': global_step,
                'state_dict': model.state_dict(),
                'best_val_psnr': best_val_psnr,
                'best_loss': avg_loss/steps,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'seed': seed
            }
            save_checkpoint(save_dict, is_best, checkpoint_dir, global_step, max_keep=5)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--mgpu', action='store_true')
    args = parser.parse_args()
    train(num_threads=args.threads, cuda=True, restart_train=args.restart, mGPU=args.mgpu)
