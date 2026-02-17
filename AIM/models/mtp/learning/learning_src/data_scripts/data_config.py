# params in normalized coordinate system
# ---------------------------
VEHICLE_ACCEL = 0.025
VEHICLE_DECEL = 0.045
VEHICLE_SIGMA = 0.5
VEHICLE_LENGTH = 0.05
VEHICLE_MAX_SPEED = 0.4
VEHICLE_MIN_GAP = 0.025
COLLECT_DATA_RADIUS = 0.5
COOL_DATA_RADIUS1 = 0.2
COOL_DATA_RADIUS2 = 0.3
# ---------------------------

ALLIGN_INITIAL_DIRECTION_TO_X = True  # allign initial direction of motion to +X
NUM_AUGMENTATION = 0
NORMALIZE_DATA = True
ZSCORE_NORMALIZE = True

INPUT_VECTOR_SIZE = 6
PREDICT_VECTOR_SIZE = 2

IMG_SHAPE = 128
BG_COLOR = 0  # black (in numpy will be 0)
ROAD_COLOR = 255  # white (in numpy will be 1)
ROAD_WIDTH_PX = 1
MAP_IMG_NAME = "map"
MAP_PARTS_NAME = "map_parts_info"
MAP_LANE_NAME = "map_lane_info"

OBS_LEN = 20
PRED_LEN = 1  # len of seq to be predicted by model
NUM_PREDICT_ON_PREDICT = 29  # number of predictions based on prediteced by model before
NUM_PREDICT = PRED_LEN + NUM_PREDICT_ON_PREDICT  # number of data needed to be generated

SAMPLE_RATE = 10
DT = float(1 / SAMPLE_RATE)  # delta_time
