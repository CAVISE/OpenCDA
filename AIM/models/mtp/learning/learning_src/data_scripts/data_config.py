# vehicle dynamics parameters in normalized coordinate system while sumo simulation
# ---------------------------
VEHICLE_ACCEL = 0.025  # vehicle acceleration
VEHICLE_DECEL = 0.045  # vehicle deceleration
VEHICLE_SIGMA = 0.5  # driver imperfection
VEHICLE_LENGTH = 0.05  # vehicle length
VEHICLE_MAX_SPEED = 0.4  # maximum vehicle speed
VEHICLE_MIN_GAP = 0.025  # minimum gap between vehicles
COLLECT_DATA_RADIUS = 0.5  # radius for data collection area
COOL_DATA_RADIUS1 = 0.2  # first importance radius (weights of vehicles in this area are higher)
COOL_DATA_RADIUS2 = 0.3  # second importance radius (weights of vehicles in this area are higher)
# ---------------------------

ALLIGN_INITIAL_DIRECTION_TO_X = True  # align initial direction of motion to +x axis
NUM_AUGMENTATION = 0  # number of data augmentation operations
NORMALIZE_DATA = True  # enable data normalization
ZSCORE_NORMALIZE = True  # use z-score normalization after min_max

# model input/output dimensions
INPUT_VECTOR_SIZE = 6  # size of input feature vector
PREDICT_VECTOR_SIZE = 2  # size of prediction output vector

# image and map configuration
IMG_SHAPE = 128  # image dimensions
BG_COLOR = 0  # background color value (black in numpy)
ROAD_COLOR = 255  # road color value (white in numpy)
ROAD_WIDTH_PX = 1  # road width in pixels
MAP_IMG_NAME = "map"  # map image file name
MAP_PARTS_NAME = "map_parts_info"  # map parts information file name
MAP_LANE_NAME = "map_lane_info"  # map lane information file name

# sequence length parameters
OBS_LEN = 20  # observation sequence length (number of past timesteps)
PRED_LEN = 1  # length of sequence to be predicted by model
NUM_PREDICT_ON_PREDICT = 29  # number of predictions based on previous predictions
NUM_PREDICT = PRED_LEN + NUM_PREDICT_ON_PREDICT  # total number of predictions to generate

# temporal parameters
SAMPLE_RATE = 10  # data sampling rate (samples per second)
DT = float(1 / SAMPLE_RATE)  # time delta between samples
