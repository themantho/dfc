##########################################
### USEFUL FUNCTIONS FOR AGING PROJECT ###
##########################################


#### Subset time series 

def subset_ts(ts_matr,network_indx):
    """
    INPUT:
        ts_matr = list of N time series, where N is the number of participants
        network_indx = index of ROIs to subset
    
    OUTPUT:
        ts_matr_filt = list of N time series which is a subset of ts_matr, where N is the number of participants
    
    """
    
    # Initialize list
    ts_matr_filt = []
    
    # Loop over subjects
    for j in range(len(ts_matr)):
        ts_subj = ts_matr[j]
        ts_filt = ts_subj[:,network_indx]
        ts_matr_filt.append(ts_filt)
    
    return ts_matr_filt
    
    


#### Extract functional connectivity matrix
    
def extract_fc(ts_matr_filt,kind='correlation'):

    """
    INPUT:
        ts_matr_filt = list of N subset time series, where N is the number of participants
        kind = The kind of matrix to compute (optional).
               It could be “correlation”, “partial correlation”, “tangent”, “covariance”, “precision”.
    
    OUTPUT:
        [fc_matrix, z_score_matrix]
        
        fc_matrix = N functional connectivity matrix, where N is the number of participants
        z_score_matrix = N transformed Z-score of functional connectivity matrix using Fishers r-to-z transformation,
                         where N is the number of participants

    
    """
    
    from nilearn.connectome import ConnectivityMeasure
    import numpy as np
    
    # Initialize matrix 
    fc_matrix = []
    z_score_matrix = []
    
    for i in range(len(ts_matr_filt)):
        
        time_series = ts_matr_filt[i]
        
        # Compute and display a correlation matrix ###
        correlation_measure = ConnectivityMeasure(kind=kind)
        fc_matrix.append(correlation_measure.fit_transform([time_series])[0])
        np.fill_diagonal(fc_matrix[i], 0)
        
        # Transform correlation to Z-score using Fisher's r-to-z transformation
        z_score_matrix.append(0.5 * np.log((1+fc_matrix[i])/(1-fc_matrix[i])))
        
    return (fc_matrix, z_score_matrix)
        