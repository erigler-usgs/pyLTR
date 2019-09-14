import glob

class Encoder(object):
    """
    Abstract base class for movie encoder.  See ffmpeg.py and mencoder.py for sample implementations
    """

    def __init__(self):
        """
        Raises EnvironmentError if encoder is not installed.
        """
        raise NotImplementedError
    
    def _isInstalled(self):
        """
        Returns True if encoder is properly installed.        
        """
        raise NotImplementedError

    def _hasFiles(self, path_with_wildcard):
        """ Returns true if we find files matching the pattern """
        if glob.glob(path_with_wildcard):
            return True
        else:
            return False        
        

    def encode(self, directory, imagePrefix, printfSpecifier, imageExtension, outputFilename):
        """
        stitch images together using to create a movie.  Each image
        will become a single frame in the movie.

        Implementations may vary, but this will likely make a system
        call to a particular command-line encoder (eg. ffmpeg or mencoder).

        parameters:
          directory: path to directory containing images
          imagePrefix: common prefix to images
          printfSpecifer: c-style printf specifier of integer-based filenames (ie. %03d' for files named 001, 002, ...)
          imageExtension: file extension (png, tiff, etc)
          outputFilename: path to file to write.
        """
        raise NotImplementedError
