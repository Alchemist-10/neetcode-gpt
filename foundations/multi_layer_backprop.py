import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x, dtype=float)
        w1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        w2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        yLinear1=np.dot(w1,x)+b1
        yLinear1[yLinear1<0]=0

        #after relu
        ylinear2=np.dot(w2,yLinear1)+b2
        loss=np.mean((ylinear2-y_true)**2)
        
        dylinear2 = 2 * (ylinear2 - y_true) / len(y_true)
      
        dw2 = np.outer(dylinear2, yLinear1)
        db2 = dylinear2

      
        dyLinear1 = np.dot(w2.T, dylinear2)

        
        dyLinear1[yLinear1 <= 0] = 0

        dw1 = np.outer(dyLinear1, x)
        db1 = dyLinear1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dw1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dw2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),}





        pass
