# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette, get_classes

import mmcv
from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0', classes=None, palette=None):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = get_classes(classes)
        model.PALETTE = get_palette(palette)
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--classes',
        default='cityscapes',
        type=str,
        help='Classes used for segmentation map')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device, classes=args.classes, palette=args.palette)
    # test a single image
    result = inference_segmentor(model, args.img)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity)


if __name__ == '__main__':
    main()
