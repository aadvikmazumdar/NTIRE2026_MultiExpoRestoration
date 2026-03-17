import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
import tifffile
from fvcore.nn import FlopCountAnalysis, flop_count_table
import os, time
from torchvision.transforms import transforms
to_pil_image = transforms.ToPILImage()

from models.Model_08_MultiExpoRestoration import DarkIRFusion as My_model
from DataLoader.rawfusion_dataset import RawFusionTestDataset


def test(cuda=True, mGPU=False):
    print('Results on the test set ......')

    test_dir = './test_img_results'
    os.makedirs(test_dir, exist_ok=True)

    data_set = RawFusionTestDataset(input_dir='./testset/')
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=2)
    print(f"Length of the data_loader: {len(data_loader)}")

    model = My_model(width=32, align_nf=8, align_r=4)

    if cuda:
        model = model.cuda()
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if mGPU:
        model = nn.DataParallel(model)

    ckpt = torch.load('./model_zoo/Ckpt_08_MultiExpoRestoration.pth',
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    print('Model loaded successfully.')

    # params and FLOPs
    flops = FlopCountAnalysis(model, torch.ones(1, 9, 768, 1536).to(device))
    flops.unsupported_ops_warnings(False)
    print(flop_count_table(flops))

    num_params = sum(p.numel() for p in model.parameters())
    print("\n" + "="*20 + " Model params and FLOPs " + "="*20)
    print(f"\tTotal # of model parameters : {num_params / (1000**2):.3f} M")
    print(f"\tTotal FLOPs of the model : {flops.total() / (1000**3):.3f} G")
    print("=" * 64)
    print('\n------- Fusion started -------\n')

    model.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for i, (burst, scene_name) in enumerate(data_loader):
            if cuda:
                burst = burst.cuda()

            start.record()
            pred, _ = model(burst)
            end.record()
            torch.cuda.synchronize()
            timei = start.elapsed_time(end)
            timings.append(timei)

            pred = torch.clamp(pred, 0.0, 1.0)

            out_img = pred[0].cpu().permute(1, 2, 0).numpy().astype(np.float32)
            out_file_name = os.path.join(test_dir, f'{scene_name[0]}-out.tif')
            tifffile.imwrite(out_file_name, out_img)
            print(f'{i+1}-th image completed. | {out_file_name} | time: {timei:.2f} ms.')

    mean_time = np.mean(timings)
    print(f'Total average time: {mean_time:.2f} ms.')


if __name__ == '__main__':
    test()