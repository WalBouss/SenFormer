_base_ = [
    './senformerNWS_fpnt_swin_tiny_512x512_160k_ade20k.py'
]
model = dict(
    pretrained='pretrain/swin_small_patch4_window7_224.pth',
    backbone=dict(depths=[2, 2, 18, 2]),
    neck=dict(in_channels=[96, 192, 384, 768], out_channels=512),
    decode_head=dict(in_channels=[512, 512, 512, 512], num_classes=150),
    auxiliary_head=dict(in_channels=384, num_classes=150))

# By default, models are trained on 8 GPUs with 2 images per GPU or 4 GPUs with 4 images per GPU
data=dict(samples_per_gpu=4)