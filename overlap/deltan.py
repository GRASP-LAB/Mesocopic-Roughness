import numpy as np

R=0.0025
nu = 0.35
E = 2.5*10**9
G = E/(2*(1+nu))

Rs = R**2/(2*R)
Es = (2*(1-nu**2)/E)**(-1)
Gs = (2*(1-nu)/G)**(-1)
Kt = 8*Gs*np.sqrt(Rs) 
Kn = (4/3)*Es*np.sqrt(Rs)

print("Kn:", Kn)
print("Kt:", Kt)

g = 9.81
m_rod = 0.3139*10**(-3)
m_cross = 0.5277*10**(-3)
m_star = 0.7301*10**(-3)
m_sphere = 1200*(4/3)*np.pi*R**3

# Ft = Kn * deltan**(3/2)
# Fn = Kt * np.sqrt(deltan) * deltat

# Normalize height of pile by length of rod
Length = 16

for name, mass in zip(["rod", "cross", "star", "sphere"], [m_rod, m_cross, m_star, m_sphere]):
    mass_ratio = mass/m_sphere
    print(f"{name.capitalize()}: Mass ratio ({name}/sphere): {mass_ratio:.4e}")
    delta = (Length*mass*g/Kn)**(2/3)
    delta_over_R = delta/R
    print(f"{name.capitalize()}: deltan/R: {delta_over_R:.4e}")
    print()