_base_ = [
    '../../_base_/models/senformer_swin.py', '../../_base_/datasets/pascal_context.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained='pretrain/swin_large_patch4_window12_384_22k.pth',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        with_cp=False
        ),
    neck=dict(
        type='FPNT',
        in_channels=[192, 384, 768, 1536],
        out_channels=512,
        num_outs=4,
        depth_swin=1,
        num_heads=8),
    decode_head=dict(
        type='SenFormer',
        num_heads=8,
        branch_depth=6,
        in_channels=[512, 512, 512, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        num_classes=59,
        align_corners=False),
    auxiliary_head=dict(in_channels=768, num_classes=59))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'queries': dict(decay_mult=0.),
                                                 }))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1., norm_type=2))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)