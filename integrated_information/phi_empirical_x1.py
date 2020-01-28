import numpy as np
import itertools

def binarise(X, threshold):
    """"Binarises data in array X according to threshold."""
    Xbin = np.copy((np.where(X >= threshold , 1, 0)))
    return(Xbin)

def probabilities(X):
    """Calculates empirical probability of each state (row) in X, returned as list of probabilities."""
    uniq, counts = np.unique((X), return_counts=True, axis=0)
    probs = np.divide(counts, np.sum(counts)).copy()
    return probs

def create_mask(X):
    """Creates mask on which partitions are generated."""
    num_channel   =  X.shape[1]
    array_channel =  range(num_channel)
    stop = num_channel // 2
    mask     = []
    if num_channel%2==0:
        for L in range(1, num_channel-(stop-1)):
            for subset in itertools.combinations(array_channel, L):    
                mask.append(list(subset))      
        sub_mask = []       
        for i in mask:
            if len(i)== int(num_channel/2):
                sub_mask.append(i)         
        sub_mask = sub_mask[0:int(len(sub_mask)/2)]    
        mask     = mask[0:(len(mask)-len(sub_mask))]           
    else:  
        for L in range(1, num_channel-(stop)):
            for subset in itertools.combinations(array_channel, L):    
                mask.append(list(subset))       
    return mask

def partitions(X, tspan, tau):
    """Creates all possible bipartition sets."""
    T = len(tspan)  
    t_range1 = list(np.arange(0, int(T-tau),1))
    t_range2 = list(np.arange(int(tau), int(T),1))
    mask = create_mask(X)
    num_channel = X.shape[1]
    labelled_partitions = []
    for i_mask in range(len(mask)):        
        XA  = X[:, mask[i_mask]]
        XB  = X[:, [col for col in range(0,num_channel) if col not in mask[i_mask]]]
        X1  = X[t_range1, :]       
        X2  = X[t_range2, :]
        XA1 = XA[t_range1,:]     
        XA2 = XA[t_range2,:]
        XB1 = XB[t_range1,:]
        XB2 = XB[t_range2,:]
        X1X2 = np.concatenate((X1,X2),axis=1)
        XA1XA2 = np.concatenate((XA1,XA2),axis=1)
        XB1XB2 = np.concatenate((XB1,XB2),axis=1)
        XA2XB2 = np.concatenate((XA2,XB2),axis=1)
        maskA = str(mask[i_mask])
        lsets = [X1, X2, XA1, XA2, XB1, XB2, X1X2, XA1XA2, XB1XB2, XA2XB2]
        labelled_partitions.append((maskA, lsets))
    return labelled_partitions
        
def probs_state(X, tau, tspan, code,  thrs):
    """Returns probabilities for each of the states across all partitions."""
    all_states = {}
    if code == 'binary':
        Xbin = binarise(X, thrs)
    partition_sets = partitions(Xbin, tspan, tau)
    for maskA, partition in partition_sets:
        table_freq_mask = []
        for set in partition:
            table_freq_mask.append(list(probabilities(set)))
        all_states.update({maskA:table_freq_mask})
    return(all_states)

def information_metrics(states):     
    part_info  = {} 
    for key, value in states.items() :   
        information_metrics = {}
        for i in range(0, len(states[str(key)])):
            df_states = states[str(key)][i]
            states_labels= ['X1', 'X2','XA1', 'XA2', 'XB1', 'XB2', 'X1X2', 'XA1XA2', 'XB1XB2', 'XA2XB2']
            H = (-1.0)*sum(np.log2(np.array(df_states))*df_states)
            information_metrics.update({'H'+states_labels[i]  :H}) 
        part_info.update({str(key):information_metrics})  
    return(part_info)

def eff_information(part_info):  
    part_eff_info = {} 
    for key, value in part_info.items():      
        eff_information = {}
        IA   = part_info[str(key)]['HXA1'] + part_info[str(key)]['HXA2'] - part_info[str(key)]['HXA1XA2']
        IB   = part_info[str(key)]['HXB1'] + part_info[str(key)]['HXB2'] - part_info[str(key)]['HXB1XB2']
        I    = part_info[str(key)]['HX1']  + part_info[str(key)]['HX2']  - part_info[str(key)]['HX1X2']
        IAB  = part_info[str(key)]['HXA2']  + part_info[str(key)]['HXB2']  - part_info[str(key)]['HXA2XB2']
        Ieff = I- (IA+IB) 
        IeffD = I- (IA+IB) + IAB
        IeffNorm = Ieff/ min(part_info[str(key)]['HXA2'], part_info[str(key)]['HXB2'])
        
        eff_information.update({'Eff_inf' :Ieff})
        eff_information.update({'Eff_inf_norm' :IeffNorm})
        eff_information.update({'Eff_infD' :IeffD})
        part_eff_info.update({str(key):eff_information})  
    return(part_eff_info, I)  

def int_inf(X, tau, tspan, code,  thrs):
    states                = probs_state(X, tau, tspan, code,  thrs)
    partition_inf_metrics = information_metrics(states)
    I_output = eff_information(partition_inf_metrics)
    I_eff    = I_output[0]
    I        = I_output[1]
    II_n   = {str(key): I_eff[str(key)]['Eff_inf_norm'] for key, value in I_eff.items()  }
    ABmin  = min(II_n.items(), key=lambda x: x[1])    
    II     = {str(key): I_eff[str(key)]['Eff_inf'] for key, value in I_eff.items()}[ABmin[0]]
    IID     = {str(key): I_eff[str(key)]['Eff_infD'] for key, value in I_eff.items()}[ABmin[0]]
    print( 'Integrated information is', II )
    print( 'Integrated information (modified) is', IID )
    print( 'Mutual information (system) is', I )
    print( 'Minimum information bipartition is', ABmin)
    return(states,partition_inf_metrics, I_eff, ABmin, II, I, IID)

