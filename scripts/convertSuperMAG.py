#!/usr/bin/env python
import numpy as n
import os,argparse,pickle
import datetime

parser = argparse.ArgumentParser("Converts SuperMAG ASCII to python Dictionary")
parser.add_argument("i",metavar='in-file',type=argparse.FileType('r'))
parser.add_argument("o",metavar='out-file',type=argparse.FileType('wb'))

try:
    results = parser.parse_args()
except IOError as msg:
    parser.error(str(msg))
    

#Read in the SuperMAG File
x = n.genfromtxt(results.i,delimiter=',',
                 dtype=('|S19','|S3',float,float,float,float,float,float),
                 names=True)

#Create an array of datetime objects from the date strings
a = []

for i in range(len(x['Date_UTC'])):
    a.append(datetime.datetime.strptime(x['Date_UTC'][i],'%Y-%m-%d %H:%M:%S'))
times = n.array(a)

superMAG = {}
locs = n.where(x['IAGA'] == x['IAGA'][0])
for i in range(locs[0][1]-1):
    locs = n.where(x['IAGA'] == x['IAGA'][i])
    stat = {'name':x['IAGA'][i],'mlt':x['MLT'][locs],'mlat':x['MLAT'][locs],
            'sza':x['SZA'][locs],'time':times[locs],
            'n':x['N'][locs],'e':x['E'][locs],'z':x['Z'][locs]}
    superMAG[x['IAGA'][i]] = stat
      
pickle.dump(superMAG,results.o,protocol=2)
results.o.close()
