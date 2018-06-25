from scipy import signal
import numpy as np

def coevolution(data_all, W, overlap, fc1_index, fc2_index, winType = 'rect'):
     
    """
    This function computes the co-evolution of two networks (coupling between two networks across time series)
    
    (1) compute all node-to-node correlations; 
    (2) Fisher r to z transform the correlations; 
    (3) extract the relevant connections (e.g., all unique pairwise DAN-DN connections) in a given window,
        and compute the average strength of fc across these connections
    (4) compute the correlation between sets of values across time 
        (e.g., mean DAN-DN core values for each time window were correlated with mean FPCN-DAN values for each time window)
        
    INPUT:
    data_all: input timeseries of list size n containing a m x m matrix, where n is the number of subjects, and m is the number of nodes.
    W: length of each sliding window in samples
    overlap: overlap between successive sliding windows in samples
    fc_index1: index of the first network in the matrix
    fc_index2: index of the second network in the matrix
    winType: if specified as 'gauss', will implement a gaussian window, else will implement a rectangular window.    
        
    OUTPUT:
    co_evolve_matr: co-evolution matrix size n, with each containing w values, where w is the number of windows.
    """

    if winType == 'gauss':
        win = signal.gaussian(W,(W/5)) # gaussian window
    else:
        win = np.ones(W) # rectangular window
   
    # Initialize matrix     
    co_evolve_matr = [None]*len(data_all)
    
    # Loop over subject
    for qq in range(len(data_all)): 
        
        # Get single subject
        data = data_all[qq]
        
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
        
        # For each window:
        co_evol = np.zeros(numWindows)
        
        """
        (1) compute all node-to-node correlations
        (2) Fisher r to z transform the correlations
        (3) extract the relevant connections (e.g., all unique pairwise DAN-DN connections)
        (4) compute the average strength of fc across these connections
        (5) compute the correlation between sets of values across time 
            (e.g., mean DAN-DN core values for each time window were correlated with mean FPCN-DAN values for each time window)
        """
        
        for iter in range(1, numWindows+1):
            # Get first W time series
            #ts = np.transpose(data[((iter-1)*(W-overlap)+1):((iter-1)*(W-overlap)+W+1),:])
            ts = np.transpose(data1[((iter-1)*(W-overlap)+1):((iter-1)*(W-overlap)+W+1),:])
            
            # Rep matrix
            rep_matr = np.tile(win, (numParcels, 1))
            
            # Get window matrix
            window_matrix = ts*rep_matr
            
            # (1) Compute correlation for Wth time series 
            FCsliding = np.corrcoef(window_matrix)
            np.fill_diagonal(FCsliding, 0)
               
            # (2) Fisher r to z transformation
            z_score_matrix = 0.5 * np.log((1+FCsliding)/(1-FCsliding))
            
            # (3) extract the relevant connections (e.g., all unique pairwise DAN-DN connections)
            int_matr = z_score_matrix[min(fc1_index):max(fc1_index)+1,min(fc2_index):max(fc2_index)+1]
            
            # (4) compute the average strength of fc across these connections
            co_evol[iter-1] = int_matr.mean()
            
            # (5) compute the correlation between sets of values across time 
        
        # return co-evolution matrix for subject
        co_evolve_matr[qq] = co_evol
        
    return co_evolve_matr