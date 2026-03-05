import numpy as np

def rel_err(tru,inv,cut_id=1):
    return (np.var(tru[cut_id:-cut_id,cut_id:-cut_id]-inv[cut_id:-cut_id,cut_id:-cut_id])/np.var(tru[cut_id:-cut_id,cut_id:-cut_id]))**(1/2)