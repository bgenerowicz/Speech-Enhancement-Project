import numpy as np
import copy

### First half: demonstration of averaging with random values
##

windowlength=3
arrtosearch=np.random.rand(20,20)
#arrtosearch = np.arange(64).reshape((8,8))
#arrtosearchreduced = np.arange(64).reshape((8,8))
k=arrtosearch.shape[1]
numrows=arrtosearch.shape[0]
arrtosearchreduced=copy.copy(arrtosearch)

for rowstart,rowend in zip(range(0,numrows-windowlength,1),range(windowlength-1,numrows,1)):
    #for k in range(0,arrtosearch.shape[1]):
    #   arrtosearch[list(range(rowstart, rowend+1)), k] = min(arrtosearch[list(range(rowstart, rowend+1)), k])
    arrtosearchreduced[list(range(rowstart, rowend+1)), 0:k+1] = np.amin( arrtosearch[list(range(rowstart, rowend+1)), :],axis=0)


end=1


## 2nd half, Some not so important np.nan calculations
#
# Qprev=np.random.rand(1,10)
# alpha=np.random.rand(1,10)
# alpha[0,list(range(0,10))]=np.nan
# psd_row=np.random.rand(1,10)
# result=Qprev/alpha
# onevector=np.array(np.ones(10))
# result2=onevector-alpha
#
# Q = alpha*Qprev + (onevector-alpha) * psd_row

end=1