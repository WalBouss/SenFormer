_base_ = [
    '../../_base_/models/senformer_r50.py', '../../_base_/datasets/coco-stuff10k.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(depth=50,),
    neck=dict(
        type='FPNT',
        in_channels=[256, 512, 1024, 2048],
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
        channels=128,  # not used
        dropout_ratio=0.1,
        num_classes=171,
        align_corners=False),
    auxiliary_head=dict(in_channels=1024, num_classes=171))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'queries': dict(decay_mult=0.),
                                                 'pos_embed': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1),
                                                 }))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1., norm_type=2))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4)