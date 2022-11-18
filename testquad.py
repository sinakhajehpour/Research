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

Sw = np.linspace(0, 50,1000)
T_w= np.linspace(-20,20,1000)
levels = np.linspace(0,4,21)
levels_1 = np.linspace(0,10,21)

# Creating 2-D grid of features
[X, Y] = np.meshgrid(Sw, T_w)

fig, ax = plt.subplots(2,3)
#A= -0.0237831273
A= a * (c_a * gammaT - c_i * gammaS)
B = -c_a * gammaT * (Y - b - c * p_b) + c_i * gammaS * (T_i - b - c * p_b + a * X) - gammaS * Lf
C = gammaS * Lf * X - c_i * gammaS * X * (T_i - b - c * p_b)
#ASb**2+BSb+C=0

#Sb= (-B(+/-)(B**2-4AC)**1/2)/2A

Z_pos=(-B+(B**2-4*A*C)**1/2)/(2*A)
Z_neg=(-B-(B**2-4*A*C)**1/2)/(2*A)

ax[0,0].contourf(X, Y, Z_pos)
ax[0,0].set_xlabel('Salinity of water(psu)',fontsize=8)
ax[0,0].set_ylabel('Temperature of water(˚C)',fontsize=8,labelpad=-5)
fig.colorbar(ax[0,0].contourf(X, Y, Z_pos))

ax[0,1].contourf(X, Y, Z_neg)
ax[0,1].set_xlabel('Salinity of water(psu)',fontsize=8)
ax[0,1].set_ylabel('Temperature of water(˚C)',fontsize=8,labelpad=-5)
fig.colorbar(ax[0,1].contourf(X, Y, Z_pos,levels=levels))

ax[0,2].contourf(X, Y, Z_neg)
ax[0,2].set_xlabel('Salinity of water(psu)',fontsize=8)
ax[0,2].set_ylabel('Temperature of water(˚C)',fontsize=8,labelpad=-5)
fig.colorbar(ax[0,2].contourf(X, Y, Z_pos,levels=levels_1))


ax[1,0].contourf(X, Y, Z_neg)
ax[1,0].set_xlabel('Salinity of water(psu)',fontsize=8)
ax[1,0].set_ylabel('Temperature of water(˚C)',fontsize=8,labelpad=-5)
fig.colorbar(ax[1,0].contourf(X, Y, Z_neg))



ax[1,1].contourf(X, Y, Z_neg)
ax[1,1].set_xlabel('Salinity of water(psu)',fontsize=8)
ax[1,1].set_ylabel('Temperature of water(˚C)',fontsize=8,labelpad=-5)
fig.colorbar(ax[1,1].contourf(X, Y, Z_neg,levels=levels))

ax[1,2].contourf(X, Y, Z_neg)
ax[1,2].set_xlabel('Salinity of water(psu)',fontsize=8)
ax[1,2].set_ylabel('Temperature of water(˚C)',fontsize=8,labelpad=-5)
fig.colorbar(ax[1,2].contourf(X, Y, Z_neg,levels=levels_1))

plt.show()
