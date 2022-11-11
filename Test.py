import cmath
from decimal import Decimal

import matplot as matplot
import matplotlib.pyplot as plt
import numpy as np


class MyComplex(complex):
    def __repr__(self):
        if self.imag == 0:
            return str(self.real)
        elif self.real == 0:
            return str(self.imag) + 'j'
        else:
            return super(MyComplex, self).__repr__()


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


# A=a*(c_a*gammaT-c_i*gammaS)
# B=-c_a*gammaT*(T_w-b-c*p_b)+c_i*gammaS*(T_i-b-c*p_b+a*Sw)-gammaS*Lf
# C= gammaS*Lf*Sw-c_i*gammaS*Sw*(T_i-b-c*p_b)

def decimal_range(start, stop, increment):
    while start < stop:  # and not math.isclose(start, stop): Py>3.5
        yield start
        start += increment


listT_w = []
listSw = []
listA = []
listB = []
listC = []
listSolSb = []

# A=-0.0237831273

for i in decimal_range(-15.0, 15.0, 0.1):
    for j in range(0, 40):
        listT_w.append(i)
        listSw.append(j)
        T_w = i
        Sw = j
        A = a * (c_a * gammaT - c_i * gammaS)
        B = -c_a * gammaT * (T_w - b - c * p_b) + c_i * gammaS * (T_i - b - c * p_b + a * Sw) - gammaS * Lf
        C = gammaS * Lf * Sw - c_i * gammaS * Sw * (T_i - b - c * p_b)
        listA.append(A)
        listB.append(B)
        listC.append(C)
        d = (B ** 2) - (4 * A * C)
        # find two solutions
        sol1 = (-B - cmath.sqrt(d)) / (2 * A)
        sol2 = (-B + cmath.sqrt(d)) / (2 * A)
        listSolSb.append(sol1.real)
        listSolSb.append(sol2.real)
X, Y = np.meshgrid(Sw, T_w)
C = 5.05e-7 * 3.348e+5 * Sw - 2009.0 * 5.05e-7 * Sw * (-15 - 9.39e-2 - (-7.53e-8) * 1.0e+7)
# Z= ((-(-3974.0*1.0e-4*(T_w-9.39e-2-(-7.53e-8 )*1.0e+7)+2009.0*5.05e-7*(-15-9.39e-2-(-7.53e-8)*1.0e+7+(-6e-2)*Sw)-5.05e-7*3.348e+5)+cmath.sqrt((-3974.0*1.0e-4*(T_w-9.39e-2-(-7.53e-8 )*1.0e+7)+2009.0*5.05e-7*(-15-9.39e-2-(-7.53e-8)*1.0e+7+(-6e-2)*Sw)-5.05e-7*3.348e+5)**2+-0.0951325092*(5.05e-7*3.348e+5*Sw-2009.0*5.05e-7*Sw*(-15-9.39e-2 -(-7.53e-8)*1.0e+7))))/(-0.0475662546)).real
# Z= ((-(-3974.0*1.0e-4*(Y-9.39e-2-(-7.53e-8 )*1.0e+7)+2009.0*5.05e-7*(-15-9.39e-2-(-7.53e-8)*1.0e+7+(-6e-2)*X)-5.05e-7*3.348e+5)+np.sqrt((-3974.0*1.0e-4*(Y-9.39e-2-(-7.53e-8 )*1.0e+7)+2009.0*5.05e-7*(-15-9.39e-2-(-7.53e-8)*1.0e+7+(-6e-2)*X)-5.05e-7*3.348e+5)**2+-0.0951325092*(5.05e-7*3.348e+5*X-2009.0*5.05e-7*X*(-15-9.39e-2 -(-7.53e-8)*1.0e+7))))/(-0.0475662546)).real

# if (((-3974.0*1.0e-4*(Y-9.39e-2-(-7.53e-8 )*1.0e+7)+2009.0*5.05e-7*(-15-9.39e-2-(-7.53e-8)*1.0e+7+(-6e-2)*X)-5.05e-7*3.348e+5)**2+-0.0951325092*(5.05e-7*3.348e+5*X-2009.0*5.05e-7*X*(-15-9.39e-2 -(-7.53e-8)*1.0e+7))).all()>0):
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

print(Sw,"  ",T_w)
print(-3974.0 * 1.0e-4 * (-14 - 9.39e-2 - (-7.53e-8) * 1.0e+7) + 2009.0 * 5.05e-7 * (-15 - 9.39e-2 - (-7.53e-8) * 1.0e+7 + (-6e-2) * 1) - 5.05e-7 * 3.348e+5)
print(-3974.0 * 1.0e-4 * (-14 - 9.39e-2 - (-7.53e-8) * 1.0e+7) + 2009.0 * 5.05e-7 * (-15 - 9.39e-2 - (-7.53e-8) * 1.0e+7 + (-6e-2) * 35) - 5.05e-7 * 3.348e+5)
print(-3974.0 * 1.0e-4 * (14 - 9.39e-2 - (-7.53e-8) * 1.0e+7) + 2009.0 * 5.05e-7 * (-15 - 9.39e-2 - (-7.53e-8) * 1.0e+7 + (-6e-2) * 1) - 5.05e-7 * 3.348e+5)
print(-3974.0 * 1.0e-4 * (14 - 9.39e-2 - (-7.53e-8) * 1.0e+7) + 2009.0 * 5.05e-7 * (-15 - 9.39e-2 - (-7.53e-8) * 1.0e+7 + (-6e-2) * 35) - 5.05e-7 * 3.348e+5)
print(B)

T_w = np.linspace(-15, 15, 1000)
Sw = np.linspace(0, 40, 1000)
X1, Y1 = np.meshgrid(Sw, T_w)
B = -3974.0 * 1.0e-4 * (T_w - 9.39e-2 - (-7.53e-8) * 1.0e+7) + 2009.0 * 5.05e-7 * (-15 - 9.39e-2 - (-7.53e-8) * 1.0e+7 + (-6e-2) * Sw) - 5.05e-7 * 3.348e+5
#print(Z_pos)

fig, axs = plt.subplots(nrows=2)
fig.subplots_adjust(hspace=0.5)
axs[0].contourf(X, Y, Z_pos)
axs[0].set_xlabel("Salinity of water(psu)")
axs[0].set_ylabel("Temperature of water(˚C)")
axs[1].contourf(X, Y, Z_neg,vmin=-4, vmax=4)
axs[1].set_xlabel("Salinity of water(psu)")
axs[1].set_ylabel("Temperature of water(˚C)")
fig.colorbar(axs[0].contourf(X, Y, Z_pos), ax=axs[0])
fig.colorbar(axs[1].contourf(X, Y, Z_neg), ax=axs[1])
plt.show()
