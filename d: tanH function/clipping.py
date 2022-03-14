import numpy as np

a = np.array([-2.5,2.1,1])
a = np.clip(a, -2+1e-4, 2)
print(a)