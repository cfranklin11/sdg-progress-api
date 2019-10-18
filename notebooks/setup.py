import os
import sys

BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
