# Abstract Base class
from .SolarWind import SolarWind
# Implementation classes
from .CCMC import CCMC
from .ENLIL import ENLIL
from .LFM import LFM
from .OMNI import OMNI
from .OMNI2 import OMNI2

# Helper functions to write data
from . import Writer

import re

def fileType(filename):
    """
    Returns a string corresponding to the class/reder required to read 'filename'
    
    >>> fileType('examples/data/solarWind/CCMC_wsa-enlil.dat')
    'CCMC'

    >>> fileType('examples/data/solarWind/ENLIL-cr2068-a3b2.Earth.dat')
    'ENLIL'

    >>> fileType('examples/data/solarWind/LFM_SW-SM-DAT')
    'LFM'

    >>> fileType('examples/data/solarWind/OMNI_HRO_1MIN_14180.txt1')
    'OMNI'
    """

    def isCCMC(filename):
        fh = open(filename)
        # Data printout from ...
        if not re.match(r'Data printout from[\w\W]*', fh.readline().strip('#').strip() ):
            return False
        # Data type: ...
        if not re.match(r'Data type\:[\w\W]*', fh.readline().strip('#').strip() ):
            return False
        # Run name: ...
        if not re.match(r'Run name\:[\w\W]*', fh.readline().strip('#').strip() ):
            return False        

        return True

    def isENLIL(filename):
        fh = open(filename)

        # Path to filename
        fh.readline()
        # location of solar wind data
        if not re.match(r'^Temporal profiles at EARTH', fh.readline().strip('#').strip() ):            
            return False
        # Blank line
        fh.readline()
        # program name
        if not re.match(r'^program[\s]*=[\s]*[\w]+$', fh.readline().strip('#').strip() ):
            return False
        # version number
        if not re.match(r'^version[\s]*=[\s]*[\d\.]+$', fh.readline().strip('#').strip() ):
            return False
        # observatory
        if not re.match(r'^observatory[\s]*=[\s]*[\w]+$', fh.readline().strip('#').strip() ):
            return False

        return True
        

    def isLFM(filename):
        fh = open(filename)

        # Should be 'YEAR DOY HOUR MINUTE'
        if not re.match(r'^\d{4}\s*\d{1,3}\s*\d{1,2}\s*\d{1,2}$', fh.readline().strip() ):
            return False
        # Should be 'nROWS nCOLS'
        if not re.match(r'^\d*\s*(10|11)$', fh.readline().strip() ):
            return False
        # Should contain the text 'DATA:'
        if not re.match(r'^DATA:$', fh.readline().strip() ):
            return False

        return True
        

    def isOMNI(filename):
        fh = open(filename)
        # Example line: '************************************'
        if not re.match(r'^#?[\s\*]+$', fh.readline().strip() ):
            print('a')
            return False
        # Example line: '*****    GLOBAL ATTRIBUTES    ******'
        if not re.match(r'^#?[\s\*]+GLOBAL ATTRIBUTES[\s\*]+$', fh.readline().strip() ):
            print('b')
            return False
        # Example line: '************************************'
        if not re.match(r'^#?[\s\*]+$', fh.readline().strip() ):
            print('c')
            return False
        # Blank line
        fh.readline()
        # PROJECT                  NSSDC
        if not re.match(r'^#?[\s]*PROJECT[\s\w]*$', fh.readline().strip() ):        
            return False
        # DISCIPLINE               Space Physics>Interplanetary Studies
        if not re.match(r'^#?[\s]*DISCIPLINE[\s]*[\>\w\s]*$', fh.readline().strip() ):
            return False

        return True
                    
    def isOMNI2(filename):
        fh = open(filename)
        
        if not re.match(r'^\d{4}\s+\d+\s+\d+\s+\d+',fh.readline()):
            return False
        print('Is OMNI2')
        return True
        
    if isCCMC(filename):
        return 'CCMC'
    elif isENLIL(filename):
        return 'ENLIL'
    elif isLFM(filename):
        return 'LFM'
    elif isOMNI(filename):
        return 'OMNI'
    elif isOMNI2(filename):
        return 'OMNI2'
    else:
        raise Exception('Did not understand filetype for file "%s" ' % filename)
