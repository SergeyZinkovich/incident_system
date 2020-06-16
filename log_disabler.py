import os
import sys


class NullWriter:
    def write(self, s):
        pass


sys.stderr = NullWriter()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
