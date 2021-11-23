# import os
# import sys
# sys.path.append(os.path.abspath('.'))

import structlog
from pathlib import Path
from training.pyrenderer import PyrenderDataset


logger = structlog.get_logger(__name__)



synsets_cat = {
    '02691156': 'airplane', '02773838': 'bag', '02808440': 'bathtub', '02818832': 'bed', '02843684': 'birdhouse',
    '02871439': 'bookshelf', '02924116': 'bus', '02933112': 'cabinet', '02942699': 'camera', '02958343': 'car',
    '03001627': 'chair', '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'display', '03325088': 'faucet',
    '03636649': 'lamp', '03642806': 'laptop', '03691459': 'loudspeaker', '03710193': 'mailbox', '03761084': 'microwaves',
    '03790512': 'motorbike', '03928116': 'piano', '03948459': 'pistol', '04004475': 'printer', '04090263': 'rifle',
    '04256520': 'sofa', '04379243': 'table', '04468005': 'train', '04530566': 'watercraft', '04554684': 'washer'
}


def get_shape_paths(dataset_dir, whitelist_synsets=None, blacklist_synsets=None):
    """
    Returns shape paths for ShapeNet.

    Args:
        dataset_dir: the directory containing the dataset
        blacklist_synsets: a list of synsets to exclude

    Returns:

    """
    shape_index_path = (dataset_dir / 'paths.txt')
    if shape_index_path.exists():
        with open(shape_index_path, 'r') as f:
            paths = [Path(dataset_dir, p.strip()) for p in f.readlines()]
    else:
        paths = list(dataset_dir.glob('**/*.obj'))

    logger.info("total models", num_shape=len(paths))
    
    if whitelist_synsets is not None:
        num_filtered = sum(1 for p in paths if p.parent.parent.parent.name in whitelist_synsets)
        paths = [p for p in paths if p.parent.parent.parent.name in whitelist_synsets]
        logger.info("selected shapes from whitelist", num_filtered=num_filtered)

    if blacklist_synsets is not None:
        num_filtered = sum(1 for p in paths if p.parent.parent.parent.name in blacklist_synsets)
        paths = [p for p in paths if p.parent.parent.parent.name not in blacklist_synsets]
        logger.info("selected shapes byond blacklist", num_filtered=num_filtered)

    return paths


class ShapeNetV2(PyrenderDataset):
    def __init__(self, *args, data_dir, 
                whitelist_synsets=None, 
                blacklist_synsets=None, 
                scale_jitter=(0.05, 0.5),
                **kwargs):
        shape_paths = get_shape_paths(data_dir, 
                                      whitelist_synsets=whitelist_synsets,
                                      blacklist_synsets=blacklist_synsets,
                                     )

        super().__init__(shape_paths, scale_jitter=scale_jitter, *args, **kwargs)

