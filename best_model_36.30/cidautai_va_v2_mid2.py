import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), \
               grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CustomSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, use_adapter=False):
        for module in self.modules_list:
            x = module(x)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FreMLP(nn.Module):
    def __init__(self, nc, expand=2):
        super().__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        return torch.fft.irfft2(x_out, s=(H, W), norm='backward')


class Branch(nn.Module):
    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.branch = nn.Sequential(
            nn.Conv2d(self.dw_channel, self.dw_channel, 3, 1, dilation,
                      groups=self.dw_channel, bias=True, dilation=dilation))

    def forward(self, x):
        return self.branch(x)


class EBlockNoFreq(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True) if extra_depth_wise else nn.Identity()
        self.conv1 = nn.Conv2d(c, self.dw_channel, 1, 1, 0, bias=True)
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=d))
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, 1, bias=True))
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(self.dw_channel // 2, c, 1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        return inp + self.beta * x


class EBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=True) if extra_depth_wise else nn.Identity()
        self.conv1 = nn.Conv2d(c, self.dw_channel, 1, 1, 0, bias=True)
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=d))
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, 1, bias=True))
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(self.dw_channel // 2, c, 1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc=c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        y = inp
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        x_step2 = self.norm2(y)
        x_freq = self.freq(x_step2)
        x = y * x_freq
        x = y + x * self.gamma
        return x


class DBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        self.dw_channel = DW_Expand * c
        self.conv1 = nn.Conv2d(c, self.dw_channel, 1, bias=True)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, 3, 1, 1,
                                    groups=c, bias=True) if extra_depth_wise else nn.Identity()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand=1, dilation=d))
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dw_channel // 2, self.dw_channel // 2, 1, bias=True))
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(self.dw_channel // 2, c, 1, bias=True)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, adapter=None):
        y = inp
        x = self.norm1(inp)
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)
        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)
        x = y + x * self.gamma
        return x


class DarkIR(nn.Module):
    def __init__(self, img_channel=1, width=32,
                 middle_blk_num_enc=2, middle_blk_num_dec=2,
                 enc_blk_nums=[1, 1, 2], dec_blk_nums=[1, 1, 1],
                 dilations=[1, 4], extra_depth_wise=True):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, 3, 3, 1, 1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            if i == 0:
                self.encoders.append(
                    CustomSequential(*[EBlockNoFreq(chan, extra_depth_wise=False) for _ in range(num)]))
            else:
                self.encoders.append(
                    CustomSequential(*[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks_enc = CustomSequential(
            *[EBlock(chan, extra_depth_wise=False) for _ in range(middle_blk_num_enc)])
        self.middle_blks_dec = CustomSequential(
            *[DBlock(chan, dilations=dilations, extra_depth_wise=False) for _ in range(middle_blk_num_dec)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(
                CustomSequential(*[DBlock(chan, dilations=dilations,
                                         extra_depth_wise=False) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)
        self.side_out = nn.Conv2d(width * 2 ** len(self.encoders), 3, 3, 1, 1)

    def forward(self, input, side_loss=False):
        _, _, H, W = input.shape
        input = self.check_image_size(input)
        x = self.intro(input)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x_light = self.middle_blks_enc(x)
        if side_loss:
            out_side = self.side_out(x_light)
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        out = x[:, :, :H, :W]
        if side_loss:
            return out_side, out
        return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)
        return x


class EfficientDCNAlignment(nn.Module):
    def __init__(self, in_ch=1, nf=8, groups=4, r=4):
        super().__init__()
        self.r = r
        self.groups = groups
        self.K = 9
        self.feat_extract = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.unshuffle = nn.PixelUnshuffle(r)
        pair_unsh_ch = nf * 2 * (r * r)
        self.proj_in = nn.Conv2d(pair_unsh_ch, nf, 1, 1, 0)
        offset_ch = 2 * groups * self.K
        mask_ch   = groups * self.K
        self.offset_conv = nn.Conv2d(nf, offset_ch, 3, 1, 1)
        self.mask_conv   = nn.Conv2d(nf, mask_ch,   3, 1, 1)
        self.dcn_weight = nn.Parameter(torch.randn(nf, nf // groups, 3, 3) * 0.01)
        self.dcn_bias   = nn.Parameter(torch.zeros(nf))
        self.out_proj   = nn.Conv2d(nf, in_ch, 1)
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.constant_(self.mask_conv.bias, 0.5)

    def forward(self, ref, target):
        B, C, H, W = ref.shape
        ref_feat    = self.feat_extract(ref)
        target_feat = self.feat_extract(target)
        pair = torch.cat([ref_feat, target_feat], dim=1)
        pair_lr = self.unshuffle(pair)
        pair_lr = self.proj_in(pair_lr)
        offset_lr = self.offset_conv(pair_lr)
        mask_lr   = self.mask_conv(pair_lr)
        offset = F.interpolate(offset_lr, size=(H, W), mode='bilinear', align_corners=False)
        mask   = F.interpolate(mask_lr,   size=(H, W), mode='bilinear', align_corners=False)
        offset = offset * float(self.r)
        mask   = torch.sigmoid(mask)
        aligned_feat = deform_conv2d(
            input=target_feat,
            offset=offset,
            weight=self.dcn_weight,
            bias=self.dcn_bias,
            mask=mask,
            padding=1,
        )
        return self.out_proj(aligned_feat)


class CrossExposureAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.short_to_med  = nn.Conv2d(2, 1, 3, 1, 1)
        self.long_to_med   = nn.Conv2d(2, 1, 3, 1, 1)
        self.final         = nn.Conv2d(3, 1, 3, 1, 1)
        self.scale         = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        self.final.weight.data[0, 0, 1, 1] = 0.34
        self.final.weight.data[0, 1, 1, 1] = 0.33
        self.final.weight.data[0, 2, 1, 1] = 0.33

    def forward(self, short_map, medium_map, long_map):
        # medium as anchor, attends to short and long
        attn_s = torch.sigmoid(self.short_to_med(torch.cat([medium_map, short_map], dim=1)))
        attn_l = torch.sigmoid(self.long_to_med(torch.cat([medium_map, long_map],  dim=1)))
        enriched_short  = short_map  * attn_s
        enriched_long   = long_map   * attn_l
        return medium_map + self.scale * self.final(torch.cat([enriched_short, medium_map, enriched_long], dim=1))


class BurstAlignment(nn.Module):
    def __init__(self, in_ch=1, nf=8, r=4):
        super().__init__()
        self.align = EfficientDCNAlignment(in_ch=in_ch, nf=nf, groups=4, r=r)
        self.fuse_short  = nn.Conv2d(3, 1, 3, 1, 1)
        self.fuse_medium = nn.Conv2d(3, 1, 3, 1, 1)
        self.fuse_long   = nn.Conv2d(3, 1, 3, 1, 1)
        # exposure embeddings — 3 learnable scalars
        self.exposure_embed = nn.Parameter(torch.zeros(3, 1, 1, 1))
        self.cross_attn = CrossExposureAttention()
        for fuse in [self.fuse_short, self.fuse_medium, self.fuse_long]:
            nn.init.zeros_(fuse.weight)
            nn.init.zeros_(fuse.bias)
            fuse.weight.data[0, 0, 1, 1] = 0.34
            fuse.weight.data[0, 1, 1, 1] = 0.33
            fuse.weight.data[0, 2, 1, 1] = 0.33

    def forward(self, burst):
        # within-group references
        ref_short  = burst[:, 0:1]  # frame 0
        ref_medium = burst[:, 3:4]  # frame 3
        ref_long   = burst[:, 6:7]  # frame 6

        # within-group alignment — 6 DCN calls total
        short_aligned  = [ref_short,
                          self.align(ref_short,  burst[:, 1:2]),
                          self.align(ref_short,  burst[:, 2:3])]
        medium_aligned = [ref_medium,
                          self.align(ref_medium, burst[:, 4:5]),
                          self.align(ref_medium, burst[:, 5:6])]
        long_aligned   = [ref_long,
                          self.align(ref_long,   burst[:, 7:8]),
                          self.align(ref_long,   burst[:, 8:9])]

        # intra-group fusion
        short_map  = self.fuse_short( torch.cat(short_aligned,  dim=1))
        medium_map = self.fuse_medium(torch.cat(medium_aligned, dim=1))
        long_map   = self.fuse_long(  torch.cat(long_aligned,   dim=1))

        # exposure embeddings
        short_map  = short_map  + self.exposure_embed[0]
        medium_map = medium_map + self.exposure_embed[1]
        long_map   = long_map   + self.exposure_embed[2]

        # cross-exposure attention
        return self.cross_attn(short_map, medium_map, long_map)


class DarkIRFusion(nn.Module):
    def __init__(self, width=32, align_nf=8, align_r=4):
        super().__init__()
        self.alignment = BurstAlignment(in_ch=1, nf=align_nf, r=align_r)
        self.darkir = DarkIR(
            img_channel=1, width=width,
            middle_blk_num_enc=2, middle_blk_num_dec=2,
            enc_blk_nums=[1, 1, 2], dec_blk_nums=[1, 1, 1],
            dilations=[1, 4], extra_depth_wise=True)
        self.fusion_demosaic = nn.Conv2d(1, 3, 3, 1, 1)

    def forward(self, burst):
        fused    = self.alignment(burst)
        residual = self.fusion_demosaic(fused)
        out_side, darkir_out = self.darkir(fused, side_loss=True)
        return darkir_out + residual, out_side


def self_ensemble(model, burst):
    model.eval()
    with torch.no_grad():
        preds = []
        out, _ = model(burst);                       preds.append(out)
        out, _ = model(torch.flip(burst, [-1]));     preds.append(torch.flip(out, [-1]))
        out, _ = model(torch.flip(burst, [-2]));     preds.append(torch.flip(out, [-2]))
        out, _ = model(torch.flip(burst, [-1, -2])); preds.append(torch.flip(out, [-1, -2]))
    return torch.stack(preds).mean(0)


def ssim_loss(x, y, window_size=11, eps=1e-8):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    B, C, H, W = x.shape
    x = x.view(B * C, 1, H, W)
    y = y.view(B * C, 1, H, W)
    coords = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * (window_size / 6) ** 2))
    g = g / g.sum()
    kernel = (g.unsqueeze(0) * g.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    pad = window_size // 2
    mu_x  = F.conv2d(x, kernel, padding=pad)
    mu_y  = F.conv2d(y, kernel, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad) - mu_y2
    sigma_xy  = F.conv2d(x * y, kernel, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps)
    return 1.0 - ssim_map.mean()


class DarkIRFusionLoss(nn.Module):
    def __init__(self, edge_weight=50.0, guide_weight=1.0):
        super().__init__()
        self.edge_weight  = edge_weight
        self.guide_weight = guide_weight
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.register_buffer("kernel", torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1))

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        kernel = self.kernel.to(img.device)
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(img, kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4
        filtered = self.conv_gauss(new_filter)
        return current - filtered

    def forward(self, output, guide, gt):
        l2         = F.mse_loss(output, gt)
        edge       = F.mse_loss(self.laplacian_kernel(output), self.laplacian_kernel(gt))
        gt_down    = F.interpolate(gt, size=guide.shape[2:], mode="bilinear", align_corners=False)
        guide_loss = F.l1_loss(guide, gt_down)
        total      = l2 + self.edge_weight * edge + self.guide_weight * guide_loss
        return total, l2
