import torch

BASE_LR = 1e-3         # starting learning rate 
MAX_EPOCHS = 50        # maximum training epochs
NUM_VIEWS = 16         # the sampling number of viewpoint for each object
WARMUP_EPOCHS = 0      # warmup epochs during training
RANKING_MARGIN = 0.1   # the triplet margin for ranking
USE_DATA_AUG = True    # whether apply data augmentation during training process
HIST_BIN_NUMS = 100    # the number of histogram bins
MIN_DEPTH_PIXELS = 200 # the minimum number of valid depth values for a valid training depth image
VISIB_FRAC = 0.1       # the minimum visible surface ratio

RENDER_WIDTH = 720       # the width of rendered images
RENDER_HEIGHT = 540      # the height of rendered images
MIN_HIST_STAT = 50       # the histogram threshold for filtering out ambiguous inter-viewpoint training pairs
RENDER_DIST = 5          # the radius distance factor of uniform sampling relative to object diameter.
ZOOM_MODE = 'bilinear'   # the target zooming mode (bilinear or nearest)
ZOOM_SIZE = 128          # the target zooming size
ZOOM_DIST_FACTOR = 0.01    # the distance factor of zooming (relative to object diameter)


INTRINSIC = torch.tensor([
        [1.0757e+03, 0.0000e+00, 3.6607e+02],
        [0.0000e+00, 1.0739e+03, 2.8972e+02],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]
    ], dtype=torch.float32)


# RENDER_WIDTH = 640       # the width of rendered images
# RENDER_HEIGHT = 480      # the height of rendered images
# MIN_HIST_STAT = 30       # the histogram threshold for filtering out ambiguous inter-viewpoint training pairs
# RENDER_DIST = 5          # the radius distance factor of uniform sampling relative to object diameter.
# ZOOM_MODE = 'bilinear'   # the target zooming mode (bilinear or nearest)
# ZOOM_SIZE = 128          # the target zooming size
# ZOOM_DIST_FACTOR = 8     # the distance factor of zooming (relative to object diameter)

# INTRINSIC = torch.tensor([
#         [615.1436, 0.000000, 315.3623],
#         [0.0000,   615.4991, 251.5415],
#         [0.0000,   0.000000, 1.000000],
#     ], dtype=torch.float32)


