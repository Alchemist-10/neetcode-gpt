import numpy as np
from numpy.typing import NDArray
import math

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        ans=[]
        for x in z:
            ans.append(round(1/(1+math.e**(-x)),5))
        return np.array(ans)
        

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        ans=[]
        for x in z:
            ans.append(round(float(max(0,x)),5))
        return np.array(ans)
