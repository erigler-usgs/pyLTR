from . import ffmpeg
from . import mencoder

import unittest

class TestEncoder(unittest.TestCase):
    """
    This class tests the CX form library wrapped via Python.    
    """

    def test_mencoder_files_not_found(self):
        e = mencoder.mencoder()
        self.assertRaises(IOError, e.encode, '/dev/null', 'GarbageFiles','%03d', 'png', '/dev/null')

    def test_ffmpeg_files_not_found(self):
        e = ffmpeg.ffmpeg()
        self.assertRaises(IOError, e.encode, '/dev/null', 'GarbageFiles','%03d', 'png', '/dev/null')                

if __name__ == "__main__":
    unittest.main()
