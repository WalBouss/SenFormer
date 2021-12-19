_base_ = [
    './senformer_fpnt_swin_base_640x640_160k_ade20k.py'
]
model = dict(
    pretrained='pretrain/swin_large_patch4_window12_384_22k.pth',
    backbone=dict(embed_dims=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
    neck=dict(in_channels=[192, 384, 768, 1536], out_channels=512),
    decode_head=dict(in_channels=[512, 512, 512, 512], num_classes=150),
    auxiliary_head=dict(in_channels=768, num_classes=150))

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)