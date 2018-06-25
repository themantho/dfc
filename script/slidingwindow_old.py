from scipy import signal
import numpy as np

def slidingwindow(data,W,overlap,winType = 'rect'):
    
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
    overlap: overlap between successive sliding windows in samples
    winType: if specified as 'gauss', will implement a gaussian window, else
    will implement a rectangular window.
    
    OUTPUT:
    FCsliding: matrix of size m x m x p where m is the number of timeseries and p is the number of sliding windows.
    
    """

    if winType == 'gauss':
        win = signal.gaussian(W,(W/5)) # gaussian window
    else:
        win = np.ones(W) # rectangular window

    # Get number of nodes
    numParcels = np.shape(data)[1]
    
    # Calculate number of sliding windows [CORRECT]
    numWindows = int(np.floor((np.shape(data)[0]-W)/(W-overlap) + 1))
    
    # Initialize correlation heatmap
    FCsliding = np.zeros((numWindows,numParcels,numParcels))
    
    # Add empty first window
    data1 = data
    #vals = np.empty((1, 630))
    data1 = np.insert(data, [0], 0, axis=0)
    
    # Loop Correlation
    for iter in range(1, numWindows+1):
        # Get first W time series
        #ts = np.transpose(data[((iter-1)*(W-overlap)+1):((iter-1)*(W-overlap)+W+1),:])
        ts = np.transpose(data1[((iter-1)*(W-overlap)+1):((iter-1)*(W-overlap)+W+1),:])
        
        # Rep matrix
        rep_matr = np.tile(win, (numParcels, 1))
        
        # Get window matrix
        window_matrix = ts*rep_matr
        
        # Compute correlation for Wth time series 
        FCsliding[iter-1,:,:] = np.corrcoef(window_matrix)
        np.fill_diagonal(FCsliding[iter-1,:,:], 0)
           
    return FCsliding
        