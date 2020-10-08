from yacs.config import CfgNode as CN

_C = CN()

# Input
_C.INPUT = CN()
_C.INPUT.TRAIN_VAL_SPLIT_DIR = '/data/spacenet7/split'
_C.INPUT.TRAIN_VAL_SPLIT_ID = 0
_C.INPUT.CLASSES = [
    'building_footprint', 'building_boundary', 'building_contact'
]
_C.INPUT.TEST_DIR = '/data/spacenet7/spacenet7/test_public'
_C.INPUT.CONCAT_PREV_FRAME = False
_C.INPUT.CONCAT_NEXT_FRAME = False

# Transforms
_C.TRANSFORM = CN()
_C.TRANSFORM.TRAIN_RANDOM_CROP_SIZE = (192, 192)
_C.TRANSFORM.TRAIN_RANDOM_ROTATE_DEG = (0, 0)
_C.TRANSFORM.TRAIN_RANDOM_ROTATE_PROB = 1.0
_C.TRANSFORM.TRAIN_HORIZONTAL_FLIP_PROB = 0.0
_C.TRANSFORM.TRAIN_VERTICAL_FLIP_PROB = 0.0
_C.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_STD = 0.0
_C.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_PROB = 1.0
_C.TRANSFORM.TEST_SIZE = (1024, 1024)
_C.TRANSFORM.SIZE_SCALE = 1.0

# Test time augmentations
_C.TTA = CN()
_C.TTA.RESIZE = []  # e.g., [[928, 928], [1152, 1152]]
_C.TTA.RESIZE_WEIGHTS = []  # e.g., [0.125, 0.25]

# Data loader
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH_SIZE = 16
_C.DATALOADER.VAL_BATCH_SIZE = 16
_C.DATALOADER.TEST_BATCH_SIZE = 16
_C.DATALOADER.TRAIN_NUM_WORKERS = 8
_C.DATALOADER.VAL_NUM_WORKERS = 8
_C.DATALOADER.TEST_NUM_WORKERS = 8
_C.DATALOADER.TRAIN_SHUFFLE = True

# Model
_C.MODEL = CN()
_C.MODEL.ARCHITECTURE = 'unet'  # ['unet', 'fpn', 'pan', 'pspnet', 'deeplabv3', 'linknet']
_C.MODEL.BACKBONE = 'timm-efficientnet-b3'
_C.MODEL.ENCODER_PRETRAINED_FROM = 'imagenet'
_C.MODEL.ACTIVATION = 'sigmoid'
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.UNET_DECODER_CHANNELS = (256, 128, 64, 32, 16)
_C.MODEL.UNET_ENABLE_DECODER_SCSE = False
_C.MODEL.FPN_DECODER_DROPOUT = 0.2
_C.MODEL.PSPNET_DROPOUT = 0.2
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.WEIGHT = 'none'

# Solver
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 300
_C.SOLVER.OPTIMIZER = 'adam'  # ['adam', 'adamw']
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.LR = 1e-4
_C.SOLVER.LR_SCHEDULER = 'multistep'  # ['multistep', 'annealing']
_C.SOLVER.LR_MULTISTEP_MILESTONES = [
    270,
]
_C.SOLVER.LR_MULTISTEP_GAMMA = 0.1
_C.SOLVER.LR_ANNEALING_T_MAX = 300
_C.SOLVER.LR_ANNEALING_ETA_MIN = 0.0
_C.SOLVER.LOSSES = ['dice', 'bce']  # ['dice', 'bce', 'focal']
_C.SOLVER.LOSS_WEIGHTS = [1.0, 1.0]
_C.SOLVER.FOCAL_LOSS_GAMMA = 2.0

# Eval
_C.EVAL = CN()
_C.EVAL.METRICS = [
    'iou',
]  # ['iou']
_C.EVAL.MAIN_METRIC = 'iou/building_footprint'
_C.EVAL.EPOCH_TO_START_VAL = 0

# Misc
_C.LOG_ROOT = '/logs'
_C.WEIGHT_ROOT = '/weights'
_C.CHECKPOINT_ROOT = '/checkpoints'
_C.PREDICTION_ROOT = '/predictions'
_C.ENSEMBLED_PREDICTION_ROOT = '/ensembled_predictions'
_C.POLY_ROOT = '/polygons'
_C.TRACKED_POLY_ROOT = '/tracked_polygons'
_C.SOLUTION_OUTPUT_PATH = 'none'
_C.SAVE_CHECKPOINTS = True
_C.DUMP_GIT_INFO = True
_C.TEST_TO_VAL = False
_C.BOUNDARY_SUBTRACT_COEFF = 0.50
_C.CONTACT_SUBTRACT_COEFF = 1.00
_C.METHOD_TO_MAKE_POLYGONS = 'watershed'  # ['contours', 'watershed']
_C.BUILDING_SCORE_THRESH = 0.5  # for 'contours'
_C.BUILDING_MIM_AREA_PIXEL = 8.0  # for 'contours'
_C.WATERSHED_MAIN_THRESH = 0.3
_C.WATERSHED_SEED_THRESH = 0.7
_C.WATERSHED_MIN_AREA_PIXEL = 6.0  # for 'watershed'  # XXX: smaller may be better
_C.WATERSHED_SEED_MIN_AREA_PIXEL = 0.0  # for 'watershed'
_C.TRACKING_MIN_IOU = 0.1  # XXX: smaller may be better
_C.EXP_ID = 9999  # 0~9999
_C.ENSEMBLE_EXP_IDS = []


def get_default_config():
    """[summary]

    Returns:
        [type]: [description]
    """
    return _C.clone()
