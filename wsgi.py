import sys
import os

path = '/home/alpman/rater_calibration'
if path not in sys.path:
    sys.path.insert(0, path)

os.environ['DB_PATH'] = '/home/alpman/rater_calibration/calibration.db'
os.environ['UPLOAD_DIR'] = '/home/alpman/rater_calibration/uploads'

from app import app as application
