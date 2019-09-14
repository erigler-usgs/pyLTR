from . import Encoder

import os
import subprocess

class mencoder(Encoder.Encoder):
    def __init__(self):
        if not self._isInstalled():
            not_found_msg = """
            The mencoder command was not found;
            mencoder is used by this script to make an avi file from a set of pngs.
            It is typically not installed by default on linux distros because of
            legal restrictions, but it is widely available.
            http://www.mplayerhq.hu/

            We recommend ffmpeg with the x264 codec for the best quality movies. Try that instead.
            """
            
            raise EnvironmentError(not_found_msg)
        
    def _isInstalled(self):        
        """ Returns True if mencoder is properly installed"""
        try:
            p=subprocess.Popen(['mencoder'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            hasEncoder = True
        except OSError:
            hasEncoder = False
            
        return hasEncoder

    def encode(self, directory, imagePrefix, printfSpecifer, imageExtension, outputFilename):
        """
        stitch images together using Mencoder to create a movie.  Each
        image will become a single frame in the movie.
        
        this function makes a system call to mencode which looks like:
        
        mencoder mf://*.png -mf type=png:fps=25 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o output.avi
        
        See the MPlayer and Mencoder documentation for details.        

        parameters:
          directory: path to directory containing images
          imagePrefix: common prefix to images
          printfSpecifer: c-style printf specifier of integer-based filenames (ie. %03d' for files named 001, 002, ...)
          imageExtension: file extension (png, tiff, etc)
          outputFilename: path to file to write.
        """
        path_with_wildcard = os.path.join(directory,imagePrefix)+"*."+imageExtension
        if not self._hasFiles( path_with_wildcard ):
            raise IOError('Could not find image files to encode at "%s"' % path_with_wildcard)
        
        command = ('mencoder',
                   'mf://' + path_with_wildcard,
                   '-mf',
                   'type=' + imageExtension+':fps=15',
                   '-ovc',
                   'lavc',
                   '-lavcopts',
                   'vcodec=mpeg4',
                   '-oac',
                   'copy',
                   '-o',
                   outputFilename)

        #os.spawnvp(os.P_WAIT, 'mencoder', command)
    
        print(("\n\nabout to execute:\n%s\n\n" % ' '.join(command)))
        subprocess.call(command)
        
        print(("\n\n The movie was written to '%s'" % outputFilename))
        
        print(("\n\n You may want to delete the frames at "+os.path.join(directory, '*.'+imageExtension)+" now.\n\n"))
        
