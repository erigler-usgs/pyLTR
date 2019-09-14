""" Release data for the pyLTR package
$Id$
"""
#*****************************************************************************
# Copyright (C) 2009-2012 Peter Schmitt <schmitt@ucar.edu>
#
# Distributed under the terms of a to-be-determined license
#*****************************************************************************

name = "pyLTR"

revision = "2145"

version = "2.2.0.svn.r" + revision.rstrip("M")

description = "Tools for LTR modeling."

long_description = \
"""
pyLTR is a collection of tools that support LFM-TIEGCM-RCM coupled
geospace models.

Main features:

* Solar Wind processing (pyLTR.SolarWind)

* Data structures for time series (pyLTR.TimeSeries)

* Standard 1d and 2d plots for ionosphere models (i.e. MIX) and
  magnetosphere models (i.e. LFM).
"""

license = "TBD"

authors = {"Peter" : ("Peter Schmitt", "schmitt@ucar.edu")
           }

url = "http://www.hao.ucar.edu"

download_url = "https://wiki.ucar.edu/display/LTR/Home"

platforms = ["Linux","Mac OSX"]
