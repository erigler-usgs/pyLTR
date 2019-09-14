#!/bin/sh

# This script uses Nose to find & execute all tests in the pyCISM package.
# http://somethingaboutorange.com/mrl/projects/nose/

nosetests --with-doctest
#argument to print stdout: --nocapture
