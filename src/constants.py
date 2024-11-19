"""
This file will contain constants used throughout the project
"""

from pathlib import Path


# Directory paths
PARENT_PATH = Path(__file__).parent
DATA_PATH = PARENT_PATH / '../data'
RAW24H_PATH = DATA_PATH / 'raw-seismic'
DETECTIONS_PATH = DATA_PATH / 'detections'
PICKLISTS_PATH = DATA_PATH / 'pick-lists'
STALOCS_PATH = DATA_PATH / 'station-locations'
VELMODEL_PATH = DATA_PATH / 'velocity-models'
EQLOCS_PATH = DATA_PATH / 'eq-locations'

# Contants 
NUM_SECS_DAY = 86400  # 60*60*24 
NUM_SECS_HOUR = 3600  # 60*60 