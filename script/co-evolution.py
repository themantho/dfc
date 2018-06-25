from scipy import signal
import numpy as np

def coevolution(data_all, W, overlap, fc1_index, fc2_index, winType = 'rect'):
     
    """
    (1) compute all node-to-node correlations; 
    (2) Fisher r to z transform the correlations; 
    (3) extract the relevant connections (e.g., all unique pairwise DAN-DN connections) in a given window,
        and compute the average strength of fc across these connections
    (4) compute the correlation between sets of values across time 
        (e.g., mean DAN-DN core values for each time window were correlated with mean FPCN-DAN values for each time window)
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