import matplotlib
import numpy as np
from matplotlib import pyplot as plt

a = -6e-2
b = 9.39e-2  # Constant coefficient of freezing equation(˚C)
c = -7.53e-8  # Pressure coefficient of freezing equation(˚C Pa^{-1})
c_i = 2009.0  # Specific heat capacity ice(J kg^-1 K^-1)
c_a = 3974.0  # Specific heat capacity water(J kg^-1 K^-1)
Lf = 3.348e+5  # Latent heat fusion(J kg^-1)
# T_w = 5.4         # Temperature of water(˚C)
T_i = -15  # Temperature of ice(˚C)
# Sw = 35          # Salinity of water(psu)
Sc = 2500  # Schmidt number
Pr = 14  # Prandtl number
mu = 1.95e-6  # Kinematic viscosity of sea water(m^2 s^-1)
p_b = 1.0e+7  # Pressure at ice interface(Pa)
kT = mu / Pr  # Thermal diffusivity(m^2 s^-1)
kS = mu / Sc  # Salinity diffusivity(m^2 s^-1)
rhoI = 920  # density of ice(kg m^-3)
rhoW = 1025  # density of sea water(kg m^-3)
gammaS = 5.05e-7  # Salinity exchange velocity(m s^-1)
gammaT = 1.0e-4

Sw = np.linspace(0, 35, 1000)
T_w = np.linspace(-5, 10, 1000)
levels = np.linspace(0, 2, 3)
levels_1 = np.linspace(0, 20, 100)

# Creating 2-D grid of features
[X, Y] = np.meshgrid(Sw, T_w)

# fig, ax = plt.subplots(2, 3)
# A= -0.0237831273
A = a * (c_a * gammaT - c_i * gammaS)
B = -c_a * gammaT * (Y - b - c * p_b) + c_i * gammaS * (T_i - b - c * p_b + a * X) - gammaS * Lf
C = gammaS * Lf * X - c_i * gammaS * X * (T_i - b - c * p_b)
# ASb**2+BSb+C=0

# Sb= (-B(+/-)(B**2-4AC)**1/2)/2A

Z_1 = (-B + (B ** 2 - 4 * A * C) ** (1 / 2)) / (2 * A)
Z_2 = (-B - (B ** 2 - 4 * A * C) ** (1 / 2)) / (2 * A)


def Checksign(i, j, a, b, z_1pos=[], z_2pos=[], z_bothpos=[], z_bothneg=[]):
    if a >= 0:
        if b < 0:
            z_1pos.append((X[i][j], Y[i][j], 1))
            # z_1pos[i][j].append(1)
            # z_1pos[i][j] += 1
        else:
            z_bothpos.append((X[i][j], Y[i][j], 2))
            # z_bothpos[i][j].append(2)
            # z_bothpos[i][j] += 2

    else:
        if b >= 0:
            z_2pos.append((X[i][j], Y[i][j], 1))
            # z_2pos[i][j].append(1)
            # z_2pos[i][j] += 1
        else:
            z_bothneg.append((X[i][j], Y[i][j], -2))
            # z_bothneg[i][j].append(-2)
            # z_bothneg[i][j] += -2


positiveZ1 = []
positiveZ2 = []
negativeZ1 = []
negativeZ2 = []
bothposz = []
bothnegz = []

A_with_index = []
A_with_Sw_Tw = []
B_with_index = []
B_with_Sw_Tw = []
B = []
Collection = {}

for i in range(1000):
    for j in range(1000):
        # A_with_index.append((Z_1[i][j], i, j))
        # A_with_Sw_Tw.append((Z_1[i][j], X[i][j], Y[i][j]))
        # B_with_index.append((Z_1[i][j], i, j))
        # B_with_Sw_Tw.append((Z_1[i][j], X[i][j], Y[i][j]))
        A_with_index.append((i, j, Z_1[i][j]))
        A_with_Sw_Tw.append((X[i][j], Y[i][j], Z_1[i][j]))
        B_with_index.append((i, j, Z_2[i][j]))
        B_with_Sw_Tw.append((X[i][j], Y[i][j], Z_2[i][j]))
        Collection[X[i][j], Y[i][j]] = Z_1[i][j], Z_2[i][j]
        Checksign(i, j, Z_1[i][j], Z_2[i][j], positiveZ1, positiveZ2, bothposz, bothnegz)

for i in range(1000):
    for j in range(1000):
        if Z_1[i][j] >= 0:
            if Z_2[i][j] < 0:
                # changeZ_1[i][j] = 1
                Z_1[i][j] = 1
                #Z_2[i][j] = 0
            else:
                # changeZ_1[i][j] = 2
                # changeZ_2[i][j] = 2
                Z_1[i][j] = 2
                #Z_2[i][j] = 2
        else:
            if Z_2[i][j] >= 0:
                # changeZ_2[i][j] = 1
                Z_1[i][j] = 1
                #Z_1[i][j] = 0
            else:
                # changeZ_1[i][j] = -2
                # changeZ_2[i][j] = -2
                Z_1[i][j] = -2
                #Z_2[i][j] = -2

# print(Collection)


# print(A_with_index[0])
# print(B_with_index[0])

# print(positiveZ1)
# print(positiveZ2)
print(bothposz)
# print(bothnegz)
# print(Z_1)
# plt.contourf(X, Y, Z_2)
# plt.contourf(X, Y, Z_1)
# for i in range(1000):
#     for j in range(1000):
#         if Z_2[i][j] == 0:
#             print(Z_2[i][j])

plt.contourf(X, Y, Z_1,levels=levels)
#plt.contour(X, Y, Z_2)
plt.colorbar()
plt.show()
