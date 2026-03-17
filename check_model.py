import torch
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
from cidautai_va_v2_mid2 import DarkIRFusion

H, W = 768, 1536
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_flops(model, *inputs):
    model.eval()
    f = FlopCountAnalysis(model, inputs)
    f.unsupported_ops_warnings(False)
    f.uncalled_modules_warnings(False)
    try:
        return f.total() / 1e9
    except:
        return -1.0

def manual_dcn_flops(H, W, in_ch, nf, groups, K=9):
    macs = H * W * (nf // groups) * nf * K
    return 2 * macs / 1e9

model = DarkIRFusion(width=32, align_nf=8, align_r=4).to(device)

print(f"\n{'='*60}")
print(f"  Model: cidautai_va_v2_mid2 | Resolution: {H}x{W}")
print(f"{'='*60}")

burst = torch.randn(1, 9, H, W).to(device)
class FullModelWrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x)

fvcore_total = get_flops(FullModelWrap(model), burst)
print(f"\nfvcore total (misses DCN): {fvcore_total:.3f}G")

fused_1ch = torch.randn(1, 1, H, W).to(device)
class DarkIRWrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x, side_loss=True)

darkir_g = get_flops(DarkIRWrap(model.darkir), fused_1ch)
print(f"DarkIR (fvcore):           {darkir_g:.3f}G")

nf   = 8
r    = 4
Hlr  = H // r
Wlr  = W // r
align = model.alignment.align

feat_g    = get_flops(align.feat_extract, torch.randn(1, 1,        H,   W  ).to(device)) * 2
proj_g    = get_flops(align.proj_in,      torch.randn(1, nf*2*r*r, Hlr, Wlr).to(device))
offset_g  = get_flops(align.offset_conv,  torch.randn(1, nf,       Hlr, Wlr).to(device))
mask_g    = get_flops(align.mask_conv,    torch.randn(1, nf,       Hlr, Wlr).to(device))
outproj_g = get_flops(align.out_proj,     torch.randn(1, nf,       H,   W  ).to(device))
dcn_manual = manual_dcn_flops(H, W, nf, nf, groups=4)

single_dcn_total = feat_g + proj_g + offset_g + mask_g + outproj_g + dcn_manual
print(f"\nAlignment (per DCN call):")
print(f"  feat_extract x2:        {feat_g:.4f}G")
print(f"  proj_in:                {proj_g:.4f}G")
print(f"  offset_conv:            {offset_g:.4f}G")
print(f"  mask_conv:              {mask_g:.4f}G")
print(f"  out_proj:               {outproj_g:.4f}G")
print(f"  deform_conv2d (manual): {dcn_manual:.4f}G")
print(f"  Single DCN call total:  {single_dcn_total:.4f}G")
print(f"  x6 calls:               {single_dcn_total*6:.3f}G")

fuse_in = torch.randn(1, 3, H, W).to(device)
fuse_g  = get_flops(model.alignment.fuse_short, fuse_in)
print(f"\n  fuse_short/med/long ea: {fuse_g:.4f}G  x3 = {fuse_g*3:.4f}G")

class CrossAttnWrap(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return self.m(x, x, x)

cross_in = torch.randn(1, 1, H, W).to(device)
cross_g = get_flops(CrossAttnWrap(model.alignment.cross_attn), cross_in)
print(f"  CrossExposureAttention: {cross_g:.4f}G")

align_total = single_dcn_total*6 + fuse_g*3 + cross_g
print(f"  Alignment total:        {align_total:.3f}G")

dem_g = get_flops(model.fusion_demosaic, fused_1ch)
print(f"\nfusion_demosaic:          {dem_g:.4f}G")

total_manual = darkir_g + align_total + dem_g
print(f"\n{'='*60}")
print(f"fvcore total (no DCN):     {fvcore_total:.3f}G")
print(f"Manual total (with DCN):   {total_manual:.3f}G")
print(f"DCN contribution:          {dcn_manual*6:.3f}G")
print(f"Limit:                     100G | {'PASS' if total_manual < 100 else 'FAIL'}")
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Params:                    {params:.3f}M | limit: 5M | {'PASS' if params < 5 else 'FAIL'}")
print(f"{'='*60}")
