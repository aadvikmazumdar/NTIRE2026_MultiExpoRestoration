import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os, time
import tifffile
from fvcore.nn import FlopCountAnalysis
from DataLoader.rawfusion_dataset import RawFusionDataset, RawFusionTestDataset
from cidautai_va_v2_mid2 import DarkIRFusion, self_ensemble
from utils.utils import *
from utils.checkpoint import *

def eval(cuda=True, mGPU=False, test_only=False):
    checkpoint_dir = './checkpoint_dir_v2_mid2_800'
    eval_dir = './res'
    os.makedirs(eval_dir, exist_ok=True)

    if test_only:
        data_set = RawFusionTestDataset(input_dir='datasets/test_input/')
    else:
        data_set = RawFusionDataset(
            input_dir='datasets/val_input/',
            gt_dir='datasets/val_gt/',
            crop_size=None, augment=False
        )
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=2)
    print(f"Eval scenes: {len(data_set)}")

    model = DarkIRFusion(width=32, align_nf=8, align_r=4)
    if cuda:
        model = model.cuda()
    if mGPU:
        model = nn.DataParallel(model)

    ckpt = load_checkpoint(checkpoint_dir, 'best')
    model.load_state_dict(ckpt['state_dict'])
    print(f'Loaded checkpoint (epoch {ckpt["epoch"]})')

    device = torch.device('cuda:0' if cuda else 'cpu')
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {num_params/1e6:.3f}M | Limit: 5.000M | {'PASS' if num_params < 5e6 else 'FAIL'}")

    try:
        dummy = torch.ones(1, 9, 768, 1536).to(device)
        flops = FlopCountAnalysis(model, (dummy,))
        flops.unsupported_ops_warnings(False)
        total_g = flops.total() / 1e9
        print(f"FLOPs: {total_g:.1f}G | Limit: 100G | {'PASS' if total_g < 100 else 'FAIL'}")
        del dummy
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"FLOPs measurement failed: {e}")

    model.eval()
    start_time = time.time()
    total_psnr, total_ssim = 0.0, 0.0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if test_only:
                burst, scene_name = batch
                gt = None
            else:
                burst, gt = batch

            t0 = time.time()
            if cuda:
                burst = burst.cuda()
                if gt is not None:
                    gt = gt.cuda()

            pred, _ = model(burst)
            pred = torch.clamp(pred, 0.0, 1.0)

            if gt is not None:
                psnr_t = calculate_psnr(pred.unsqueeze(1), gt.unsqueeze(1))
                ssim_t = calculate_ssim(pred.unsqueeze(1), gt.unsqueeze(1))
                total_psnr += psnr_t
                total_ssim += ssim_t
            else:
                psnr_t, ssim_t = 0.0, 0.0

            out_img = pred[0].cpu().permute(1, 2, 0).numpy().astype(np.float32)
            if test_only:
                out_name = f'{scene_name[0]}-out.tif'
            else:
                sid = data_set.scene_ids[i]
                out_name = f'Scene-{sid}-out.tif'
            tifffile.imwrite(os.path.join(eval_dir, out_name), out_img)

            t1 = time.time()
            if gt is not None:
                print(f'Scene {i}: PSNR={psnr_t:.2f}dB SSIM={ssim_t:.4f} ({t1-t0:.2f}s)')
            else:
                print(f'Scene {i}: saved ({t1-t0:.2f}s)')

    elapsed = time.time() - start_time
    if not test_only:
        n = len(data_set)
        print(f'\nAverage PSNR: {total_psnr/n:.2f}dB, SSIM: {total_ssim/n:.4f}')
    print(f'Total time: {elapsed:.1f}s ({elapsed/len(data_set):.2f}s/scene)')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--mgpu', action='store_true')
    args = parser.parse_args()
    eval(cuda=True, mGPU=args.mgpu, test_only=args.test_only)
