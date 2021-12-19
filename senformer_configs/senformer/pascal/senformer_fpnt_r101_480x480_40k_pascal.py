_base_ = [
    './senformer_fpnt_r50_480x480_40k_pascal.py'
]

model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))

# By default, models are trained on 8 GPUs with 2 images per GPU or 4 GPUs with 4 images per GPU
data=dict(samples_per_gpu=4)