from scipy import signal
import numpy as np

def slidingwindow(data,W,T,winType):
    
    """    
    This function computes the matrix for the sliding window analysis for dynamic functional connectivity.

    A temporal window (W) is chosen, and within the temporal interval (t=1 to t=w_len), 
    Pearson's correlation coefficient is computed between each pair of time courses.
    Then the window is shifted by a step T, and the same calculations are repeated over the time interval [1+ T, W + T]. \
    This process is iterated until the window spans the end part of the timecourses, to eventually obtain a connectivity timecourse.
    
    INPUTS:
    data: input timeseries of size m x n, where m is the number of timeseries
    and n is the length of each timeseries in samples
    W: length of each sliding window in samples
    T: overlap between successive sliding windows in samples
    winType: if specified as 'gauss', will implement a gaussian window, else
    will implement a rectangular window.
    
    OUTPUT:
    FCsliding: matrix of size m x m x p where m is the number of timeseries and p is the number of sliding windows.
    
    """

    if winType == 'gauss':
        win = signal.gaussian(W,(W/5)) # gaussian window
    else:
        win = np.ones(W) # rectangular window

    # Get number of time sries
    numParcels = np.shape(data)[0]
    
    # Calculate number of sliding windows
    numWindows = int(np.floor((np.shape(data)[1]-W)/(W-T) + 1))
    
    # Creating empty 3D matrix
    FCsliding = np.zeros((numParcels,numParcels,numWindows))
    
    for iter in range(numWindows):
        # Get first W time series
        ts = data[:,((iter-1)*(W-T)+1):((iter-1)*(W-T)+W+1)]
        
        # Get window matrix
        window_matrix = ts*np.tile(win, (numParcels, 1))
        
        # Compute correlation for Wth time series 
        FCsliding[:,:,iter] = np.corrcoef(window_matrix) 
           
    return FCsliding
        