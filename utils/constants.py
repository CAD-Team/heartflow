from os.path import dirname as get_dir, join, abspath

#
# Directories
WORKING_DIR = get_dir(get_dir(abspath(__file__)))
DATA_DIR = join(WORKING_DIR, "data")

#
# Logging status
INFO = 0
ERROR = 1
