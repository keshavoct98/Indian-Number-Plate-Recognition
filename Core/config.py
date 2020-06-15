from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Set the class name
__C.YOLO.CLASSES              = "./data/classes/obj.names"
__C.YOLO.ANCHORS              = "./data/anchors/yolov4_anchors.txt"
# __C.YOLO.ANCHORS_V3           = "./data/anchors/yolov3_anchors.txt"
# __C.YOLO.ANCHORS_TINY         = "./data/anchors/basline_tiny_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
# __C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


