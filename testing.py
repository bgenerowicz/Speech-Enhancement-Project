import numpy as np


### First half: demonstration of averaging with random values
##

# windowlength=5
# arrtosearch=np.random.rand(30,5)
# numrows=arrtosearch.shape[0]
#
# for rowstart,rowend in zip(range(0,numrows-windowlength,windowlength),range(windowlength-1,numrows,windowlength)):
#     print(rowstart)
#     print(rowend)
#     for k in range(0,arrtosearch.shape[1]):
#         arrtosearch[list(range(rowstart, rowend+1)), k] = min(arrtosearch[list(range(rowstart, rowend+1)), k])
#
#
#
# end=1
#


## 2nd half, Some not so important np.nan calculations
#
Qprev=np.random.rand(1,10)
alpha=np.random.rand(1,10)
alpha[0,list(range(0,10))]=np.nan
psd_row=np.random.rand(1,10)
result=Qprev/alpha
onevector=np.array(np.ones(10))
result2=onevector-alpha

Q = alpha*Qprev + (onevector-alpha) * psd_row

end=1