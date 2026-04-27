import math
import torch
import torch.nn.functional as F
from torch import nn
from networks.backbone import Backbone, Backbone_dinov3
from networks.position_encoding import PositionEmbeddingSine, BVPositionalEncoding2
from networks.transformer import build_transformer
from networks.attention_block import TempSpatioAttention, dAFF, SEfusion, AttentionBlocks
from networks.FFN import FFN
from networks.fusion import Gussan_sampler, Dirichletmodel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.train_backbone = args.lr > 0
        self.d_model = args.d_model
        self.t_model = args.times_num + 1
        self.backbone_img = Backbone(args.backbone, self.train_backbone)
        # self.backbone_img = Backbone_dinov3(args.backbone, args.weight_path, hidden_dim=args.d_model,finetune=False)
        self.img_gap = nn.AdaptiveAvgPool2d(1)
        self.pos_encoding_img = PositionEmbeddingSine(args.d_model, temperature=10000, normalize=True)
        self.pos_encoding_bv = BVPositionalEncoding2(args)
        self.to_hidden_dim_i = nn.Conv2d(self.backbone_img.num_channels, args.d_model, 1, padding=0)

        self.to_hidden_dim_v = nn.Linear(args.vel_input, args.d_model)
        self.to_hidden_dim_b = nn.Linear(args.bbox_input, args.d_model)
        self.encoder = build_transformer(args)
        # self.temp_spatio_atten_b = TempSpatioAttention(args)
        # self.temp_spatio_atten_v = TempSpatioAttention(args)
        self.daff = dAFF(args.d_model, r=4, d=3)
        # self.sefusion = SEfusion(args.d_model, 4)
        self.daffconv1 = nn.Conv1d(args.times_num, args.times_num, 3, 1, 1)
        self.daffconv2 = nn.Conv1d(args.times_num, args.times_num, 5, 1, 2)
        self.lstm = nn.LSTM(args.d_model, args.d_model, num_layers=6, batch_first=True)
        # self.ffn_b = FFN(args.d_model, args.d_model * 2)
        # self.ffn_v = FFN(args.d_model, args.d_model * 2)
        self.ffn_fusion = FFN(args.d_model, args.d_model * 2)
        self.SA_b = AttentionBlocks(args.d_model, args.num_head)
        self.SA_v = AttentionBlocks(args.d_model, args.num_head)
        self.cross_atten = AttentionBlocks(args.d_model, args.num_head)
        # self.bv_mlp = MLP(args.d_model * 2, args.d_model * 4, args.d_model, 3)
        self.log_var = nn.Linear(args.d_model, args.d_model)
        self.compute_u = Dirichletmodel(3, args.d_model, 2)
        self.fused_atten = AttentionBlocks(args.d_model, args.num_head)
        self.norm = nn.LayerNorm(args.times_num * args.d_model)

        # self.end_P = nn.Linear(args.d_model * args.times_num, 4)
        self.class_i = nn.Linear(3 * args.d_model, 2)
        self.class_bv = nn.Linear(args.times_num * args.d_model, 2)
        self.class_embed = nn.Linear(args.d_model * args.times_num, 1)


    def forward(self, image, bbox, vel, is_train=True):
        img, mask = image.decompose()
        del image
        torch.cuda.empty_cache()

        feat = self.backbone_img(img, mask)
        pos_img = self.pos_encoding_img(feat)
        src, mask = feat.decompose()
        feature = self.to_hidden_dim_i(src)
        mask = mask.flatten(1)
        # feature: [b, h*w, c]
        # feature = feature.flatten(2).transpose(1, 2)
        enc_out = self.encoder(feature, src_key_padding_mask=mask, pos=pos_img)

        e2, e3, e4 = enc_out[5], enc_out[8], enc_out[11]
        c2 = self.img_gap(e2).flatten(2).transpose(1, 2)
        c3 = self.img_gap(e3).flatten(2).transpose(1, 2)
        c4 = self.img_gap(e4).flatten(2).transpose(1, 2)

        # linear+pos embedding
        # vel = self.backbone_v(vel)
        # B,T,C
        vel = self.to_hidden_dim_v(vel)
        vel = self.pos_encoding_bv(vel)
        bbox = self.to_hidden_dim_b(bbox)
        bbox = self.pos_encoding_bv(bbox)
        # bv = self.bv_mlp(torch.cat((bbox, vel), dim=2))

        # # # temporal_encoder
        # enc_out = self.encoder(feature, src_key_padding_mask=mask, pos=pos_img)
        # # b,c,h,w--b,1,c
        # enc_out = self.img_gap(enc_out).flatten(2).transpose(1, 2)

        # vel = self.temp_spatio_atten(vel)
        # bbox = self.ffn_b(self.SA_b(bbox))
        # vel = self.ffn_v(self.SA_v(vel))
        bbox = self.SA_b(bbox)
        vel = self.SA_v(vel)
        # B,T,C
        f_v = self.ffn_fusion(self.cross_atten(bbox, vel))

        #B,T,C
        f_v, (h_n, c_n) = self.lstm(f_v)
        mu = f_v
        log_var = self.log_var(f_v)
        sigma_dynamic = torch.exp(log_var)
        h_bv = Gussan_sampler(mu, log_var, is_train=is_train)
        w_dynamic_raw = torch.norm(sigma_dynamic, p=2, dim=-1, keepdim=True)
        u_img = self.compute_u(torch.cat((c2, c3, c4), dim=-1)).unsqueeze(-1)
        w_static_raw = 1 / (u_img + 1e-6)
        #weight_fusion dynamic:{bs,T,1};static:{bs,1}
        w_dynamic = w_dynamic_raw / (w_dynamic_raw + w_static_raw)
        w_static = 1 - w_dynamic
        # fused = w_dynamic * h_bv + w_static * c4
        fused = self.fused_atten(w_dynamic * h_bv, w_static * c4)

        out = self.daff(self.daffconv1(fused), self.daffconv2(fused))
        pred_tag = self.class_embed(self.norm(out.flatten(1)))
        tag_i = self.class_i(torch.cat((c2, c3, c4), dim=-1).flatten(1))
        tag_bv = self.class_bv(h_bv.flatten(1))

        return pred_tag, tag_i, tag_bv, u_img, sigma_dynamic, mu


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)  # [1]*3 = [1,1,1]    [1,1,1] + [2] = [1,1,1,2]
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x





