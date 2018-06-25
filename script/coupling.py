import numpy as np
import pandas as pd

def coupling(data_all,window):
    """
        creates a functional coupling metric from 'data'
        data_all: input timeseries of list size n containing a m x m matrix, where n is the number of subjects, and m is the number of nodes.
        smooth: smoothing parameter for dynamic coupling score
    """
    
    # Initialize matrix
    mtd_all = [None]*len(data_all)
    sma_all = [None]*len(data_all)
    
    for qq in range(len(data_all)): 
        
        #define variables
        data = data_all[qq]
        [tr,nodes] = data.shape
        der = tr-1
        td = np.zeros((der,nodes))
        td_std = np.zeros((der,nodes))
        data_std = np.zeros(nodes)
        mtd = np.zeros((der,nodes,nodes))
        sma = np.zeros((der,nodes*nodes))
        
        
        #calculate temporal derivative
        for i in range(0,nodes):
            for t in range(0,der):
                td[t,i] = data[t+1,i] - data[t,i]
    
    
        #standardize data
        for i in range(0,nodes):
            data_std[i] = np.std(td[:,i])
    
        td_std = td / data_std
    
    
        #functional coupling score
        for t in range(0,der):
            for i in range(0,nodes):
                for j in range(0,nodes):
                    mtd[t,i,j] = td_std[t,i] * td_std[t,j]
    
    
        #temporal smoothing
        temp = np.reshape(mtd,[der,nodes*nodes])
        #sma = pd.rolling_mean(temp,window)
        df = pd.DataFrame(temp)
        sma = np.asarray(df.rolling(window).mean())
        sma = np.reshape(sma,[der,nodes,nodes])
        
        mtd_all[qq] = mtd
        sma_all[qq] = sma
        
    return (mtd_all, sma_all)
