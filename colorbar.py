import numpy as np

# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np

feature_x = np.linspace(0, 4,2)
feature_y = np.linspace(0, 5,2)


# Creating 2-D grid of features
[X, Y] = np.meshgrid(feature_x, feature_y)

fig, ax = plt.subplots(1, 1)

Z = X**2+Y**2
print(np.meshgrid(feature_x, feature_y))
print("==============")
print(Z)
# plots contour lines
ax.contourf(X, Y, Z)

ax.set_title('Contour Plot')
ax.set_xlabel('feature_x')
ax.set_ylabel('feature_y')

plt.show()


