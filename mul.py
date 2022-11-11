import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from docutils.nodes import inline
# if using a Jupyter notebook, include:


S_w = np.arange(0, 40.0, 0.01)
T_w = np.arange(-15.0, 15.0, 0.01)

X, Y = np.meshgrid(S_w, T_w)

Z_neg = ((-(-3974.0 * 1.0e-4 * (Y - 9.39e-2 - (-7.53e-8) * 1.0e+7) + 2009.0 * 5.05e-7 * (
            -15 - 9.39e-2 - (-7.53e-8) * 1.0e+7 + (-6e-2) * X) - 5.05e-7 * 3.348e+5)
          - np.sqrt((-3974.0 * 1.0e-4 * (Y - 9.39e-2 - (-7.53e-8) * 1.0e+7) + 2009.0 * 5.05e-7 * (
                    -15 - 9.39e-2 - (-7.53e-8) * 1.0e+7 + (-6e-2) * X) - 5.05e-7 * 3.348e+5) ** 2 + 0.0951325092 * (
                                5.05e-7 * 3.348e+5 * X - 2009.0 * 5.05e-7 * X * (-15 - 9.39e-2 - (-7.53e-8) * 1.0e+7))))
         / (-0.0475662546))
Z_pos = ((-(-3974.0 * 1.0e-4 * (Y - 9.39e-2 - (-7.53e-8) * 1.0e+7) + 2009.0 * 5.05e-7 * (
            -15 - 9.39e-2 - (-7.53e-8) * 1.0e+7 + (-6e-2) * X) - 5.05e-7 * 3.348e+5)
          + np.sqrt((-3974.0 * 1.0e-4 * (Y - 9.39e-2 - (-7.53e-8) * 1.0e+7) + 2009.0 * 5.05e-7 * (
                    -15 - 9.39e-2 - (-7.53e-8) * 1.0e+7 + (-6e-2) * X) - 5.05e-7 * 3.348e+5) ** 2 + 0.0951325092 * (
                                5.05e-7 * 3.348e+5 * X - 2009.0 * 5.05e-7 * X * (-15 - 9.39e-2 - (-7.53e-8) * 1.0e+7))))
         / (-0.0475662546))

vmin = 0
vmax = 15

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))


cf1 = ax1.contourf(X,Y,Z_neg)

fig.colorbar(cf1, ax=ax1,shrink=0.5)



cf2 = ax2.contourf(X,Y,Z_pos)

fig.colorbar(cf2, ax=ax2,shrink=0.5)

print(X)
plt.show()
