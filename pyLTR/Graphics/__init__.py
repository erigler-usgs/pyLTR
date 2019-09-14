"""
The pyLTR.Graphics module contains a variety of routines for post-processing:

Three basic types of plots:
* TimeSeries (aka line plots)
* Polar color 'maps' (i.e. standard MIX plots)
* Polar color Basemap maps (hopefully new standard MIX plot)
* Cartesian cut plane color maps (i.e. standard LFM 2d cutplanes)
Encode multiple images (png, tiff, etc). to a movie:
* ffmpeg with x264 codec (recommended)
* mencoder AVI (not recommended: movie shows artifacts, AVIs are large)

FIXME: Implement color map (polar & cartesian) code.  See #102.
"""
from . import TimeSeries
from . import PolarPlot
from . import MapPlot
from . import CutPlane

# Video encoders
from .ffmpeg import ffmpeg
from .mencoder import mencoder
