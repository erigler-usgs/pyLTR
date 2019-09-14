from . import Encoder

import os
import sys
import subprocess
import tempfile

class ffmpeg(Encoder.Encoder):
    def __init__(self):
        if not self._isInstalled():
            not_found_msg = """            
            Encoder 'ffmpeg' was not found; ffmpeg with the x264 codec
            is used to make an mp4 file from a set of images.  ffmpeg
            and the x264 codec are freely available on the Internet:
            http://ffmpeg.org/
            http://ffmpeg.org/trac/ffmpeg/wiki/x264EncodingGuide

            ffmpeg with the x264 codec produces the best quality, but mencoder works in a pinch.
            """            
            raise EnvironmentError(not_found_msg)
                
    def _isInstalled(self):        
        """ Returns True if ffmpeg is properly installed"""
        try:
            p=subprocess.Popen(['ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            hasEncoder = True
        except OSError:
            hasEncoder = False
            
        return hasEncoder

    def encode(self, directory, imagePrefix, printfSpecifier, imageExtension, outputFilename):
        """
        stitch images together using Ffmpeg to create a movie.  Each
        image will become a single frame in the movie.

        this function makes a system call to ffmpeg which looks like:

        ffmpeg -f image2 -i "frame_00%03d.png" output_south.mp4

        parameters:
          directory: path to directory containing images
          imagePrefix: common prefix to images
          printfSpecifier: c-style printf specifier of integer-based filenames (ie. %03d' for files named 001, 002, ...)
          imageExtension: file extension (png, tiff, etc)
          outputFilename: path to file to write.
        """
        path_with_wildcard = os.path.join(directory,imagePrefix+"*."+imageExtension)
        if not self._hasFiles( path_with_wildcard ):
            raise IOError('Could not find image files to encode at "%s"' % path_with_wildcard)
        
        command = ['ffmpeg',
                   '-f', 'image2',
                   '-i', os.path.join(directory,imagePrefix) + printfSpecifier + '.' + imageExtension, 
                   '-pix_fmt', 'yuv420p', # required by many non-opensource video players
                   '-vf', '"pad=ceil(iw/2)*2:ceil(ih/2)*2"',
                   outputFilename]

        print('\n\nabout to execute:\n%s\n\n' % ' '.join(command))
        subprocess.call(" ".join(command),shell=True)
        #os.sys(command)

        print("\n\n The movie was written to '%s'" % outputFilename)
        
        print("\n\n You may want to delete the frames at "+os.path.join(directory, '*.'+imageExtension)+" now.\n\n")
    
    
    def encode_list(self, frameList, outputFilename, frameRate=15):
        """
        !!! NEEDS TESTING !!!
        
        Stitch images together using ffmpeg to create a movie.  Each image
        will become a single frame in the movie, and may have an arbitrary
        filename...the frame ordering is dictated by the list order.

        ...effectively makes a system call to ffmpeg that looks something like:
        
        ffmpeg -f concat -i frameList.txt -c copy output.mp4 
        
        ...where frameList.txt contains a list of the filenames to be concat'ed,
        and those filenames are of files that were created via something like:
           
        ffmpeg -f image2 -i frame.png frame.mp4
        
        parameters:
          frameList: Python list of filenames of frames to include in movie
          outputFilename: path to file to write.
          
          NOTE: theoretically, the file types in frameList may be anything that 
                ffmpeg recognizes, but things have only been tested using .png; 
                similarly, the movie file type may be any extension recognized 
                by ffmpeg, but things have only been tested using .mp4.

        """
        # make sure all files actually exist
        if subprocess.call(['ls']+frameList, stdout=open(os.devnull, 'wb'),
                                             stderr=open(os.devnull, 'wb')) != 0:
           raise IOError('Could not find all images to encode from list: '+frameList)
        
        
        #
        # a method that allows a list of files to be converted into a movie 
        # regardless of platform, and that doesn't use ffmpeg's buggy image2pipe
        #
        
        # break up outputFilename and determine output file type
        outputFilename, outputExtension = os.path.splitext(outputFilename)
        if len(outputFilename) == 0:
           outputFilename = 'output' # default to outputFilename of 'out'
           
        if len(outputExtension) == 0:
           outputExtension = '.mp4' # default to mp4 if not specified in outputFilename
        
        
        # create a unique temporary file that holds a two-element list of files to 
        # merge/concat; these being the movie so far, and the next frame.
        # NOTE: we need unique temporary filenames so that this can be called in
        #       parallel within the same directory; there are almost certainly 
        #       cleaner ways to do this, but we just want the random string
        #       returned by tempfile, and to open/close/etc. everything ourselves.
        tf=tempfile.NamedTemporaryFile(dir='./',prefix='Tmp_')
        tf.close()
        tmpFileString = os.path.basename(tf.name)
        
        f=open('movieMerge'+tmpFileString+'.txt','w')
        
        f.write('file'+' \''+outputFilename+tmpFileString+outputExtension+'\''+'\n')
        f.write('file'+' \''+'frame'+tmpFileString+outputExtension+'\'')
        f.close()
        
        
        # generate first frame in movie; important to specify same framerate
        # here as all subsequent frames
        subprocess.call(['ffmpeg', '-loglevel', 'error',
                         '-f', 'image2',
                         '-r', str(frameRate),
                         '-i', frameList[0],
                         '-pix_fmt', 'yuv420p',
                         outputFilename+tmpFileString+outputExtension])
        
        print('\nFFMPEG concat\'ing frame '+frameList[0], end=' ')
        sys.stdout.flush()
        
        # append/concat subsequent frames in a way that doesn't use too much
        # disk space
        for i in range(1,len(frameList)):
           
           # generate next frame in movie; important to specify same framerate
           # here as all subsequent frames
           subprocess.call(['ffmpeg', '-y',  '-loglevel', 'error',
                            '-f', 'image2',
                            '-r', str(frameRate),
                            '-i', frameList[i],
                            '-pix_fmt', 'yuv420p',
                            'frame'+tmpFileString+outputExtension])
           
           
           # append/concat the next frame to the current movie; specifying a
           # framerate along with '-f concat' does nothing, so don't bother
           subprocess.call(['ffmpeg', '-y',  '-loglevel', 'error',
                            '-f', 'concat',
                            '-i', 'movieMerge'+tmpFileString+'.txt',
                            '-c', 'copy',
                            outputFilename+tmpFileString+tmpFileString+outputExtension])
           
           print('\rFFMPEG concat\'ing frame '+frameList[i], end=' ')
           sys.stdout.flush()
           
           # concat'ing to the same output file as the input file is problemmatic
           os.rename(outputFilename+tmpFileString+tmpFileString+outputExtension, outputFilename+tmpFileString+outputExtension)
           
           # remove temporary single frame
           os.remove('frame'+tmpFileString+outputExtension)
        
        # remove merge config file
        os.remove('movieMerge'+tmpFileString+'.txt')
        
        # rename temp output file to final output file
        os.rename(outputFilename+tmpFileString+outputExtension, outputFilename+outputExtension)
        
        print("\n")
        print("The movie was written to '%s'" % outputFilename+outputExtension)
                
        
