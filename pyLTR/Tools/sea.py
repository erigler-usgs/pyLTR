import numpy as np
import pylab as p

class sea(object):
    """
    Class to conduct a superposed epoch analysis on dataset returning the
    mean,median, and quartile information
    """
    def __init__(self,data):
        """
        User provides a (n,m) data array with n being the number of epochs and 
        m being the length of the analysis window
        
        Returns the mean,median, lower quartile, and upper quartile as numpy
        arrays of size m
        """
        if (data.ndim != 2):
            print("Data must be a 2d array")
            return None
        (n,m)=data.shape 
        self.num = n
        self.semean=[np.mean(data[:,i]) for i in range(m)]
        self.semedian=[np.median(data[:,i]) for i in range(m)]
        self.lowerquart=np.zeros((m))
        self.upperquart=np.zeros((m))
        for i in range(m):
            dum=np.sort(data[:,i])
            qul=p.mlab.prctile(dum,p=(25,75))
            self.lowerquart[i],self.upperquart[i]=qul[0],qul[1]
            
        return 
        
    def plot(self,x,mediancolor='k-',meancolor='r--',quartcolor='#7F7FFF'):
        """
        Makes a simple plot of the SEA analysis
        """
        sefig=p.figure()
        ax0=sefig.add_subplot(111)
        ax0.fill_between(x,self.lowerquart,self.upperquart,facecolor=quartcolor,interpolate=True,alpha=0.25)
        ax0.plot(x,self.semedian,mediancolor,lw=2.0)
        ax0.plot(x,self.semean,meancolor,lw=1.25)
        p.show()
        
        return
        
        
        
             
