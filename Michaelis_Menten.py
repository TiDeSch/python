import numpy as np
import matplotlib.pyplot as plt

def michaelis_menten(S, Vmax, Km):
    """
    Michaelis-Menten kinetics: v = (Vmax * S) / (Km + S)
    
    Parameters:
    S: substrate concentration
    Vmax: maximum reaction velocity
    Km: Michaelis constant (substrate concentration at half Vmax)
    """
    return (Vmax * S) / (Km + S)

def hill_equation(S, Vmax, K, n):
    """
    Hill equation: v = (Vmax * S^n) / (K^n + S^n)
    
    Parameters:
    S: substrate concentration
    Vmax: maximum reaction velocity
    K: dissociation constant (substrate concentration at half Vmax)
    n: Hill coefficient (cooperativity)
    """
    return (Vmax * S**n) / (K**n + S**n)

# Parameters
Vmax = 100  # Maximum velocity (arbitrary units)
Km = 10     # Michaelis constant (μM)
K = 10      # Hill constant (μM)

# Substrate concentration range
S = np.linspace(0, 100, 1000)

# Calculate reaction velocities
v_mm = michaelis_menten(S, Vmax, Km)
v_hill_n1 = hill_equation(S, Vmax, K, n=1)  # Hill coefficient = 1
v_hill_n2 = hill_equation(S, Vmax, K, n=3)  # Hill coefficient = 2

# Create the plot
plt.figure(figsize=(10, 7))
plt.plot(S, v_mm, 'b-', linewidth=2, label='Michaelis-Menten')
#plt.plot(S, v_hill_n1, 'r--', linewidth=2, label='Hill (n=1)')
plt.plot(S, v_hill_n2, 'g-', linewidth=2, label='Hill (n=3)')

# Add horizontal line at Vmax/2
plt.axhline(y=Vmax/2, color='gray', linestyle=':', alpha=0.7)

# Add vertical lines at K50 values
plt.axvline(x=Km, color='gray', linestyle=':', alpha=0.7)
#plt.axvline(x=K, color='red', linestyle=':', alpha=0.5)
#plt.axvline(x=K, color='green', linestyle=':', alpha=0.5)

plt.xlabel('S', fontsize=12)
plt.ylabel('dP/dt', fontsize=12)
plt.title('Comparison of Hill and Michaelis-Menten Kinetics\nE + S ⇌ ES → P + E', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 50)
plt.ylim(0, 105)

# Add text annotations
#plt.text(25, 20, f'Km = {Km} μM', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
#plt.text(25, 10, f'K = {K} μM', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

plt.tight_layout()
plt.show()

# Print some key values for comparison
print("Reaction Mechanism: E + S ⇌ ES → P + E")
print("=" * 50)
print(f"Parameters used:")
print(f"  Vmax = {Vmax} units")
print(f"  Km (Michaelis-Menten) = {Km} μM")
print(f"  K (Hill equation) = {K} μM")
print()

# Calculate velocities at specific substrate concentrations
test_concentrations = [1, 5, 10, 20, 50]
print("Reaction velocities at different [S]:")
print("=" * 50)
print(f"{'[S] (μM)':<10} {'M-M':<10} {'Hill n=1':<10} {'Hill n=2':<10}")
print("-" * 45)

for conc in test_concentrations:
    v_mm_test = michaelis_menten(conc, Vmax, Km)
    v_h1_test = hill_equation(conc, Vmax, K, 1)
    v_h2_test = hill_equation(conc, Vmax, K, 2)
    print(f"{conc:<10} {v_mm_test:<10.1f} {v_h1_test:<10.1f} {v_h2_test:<10.1f}")

print()
print("Key observations:")
print("- Hill equation with n=1 is identical to Michaelis-Menten when K=Km")
print("- Hill coefficient n>1 shows positive cooperativity (sigmoidal curve)")
print("- Higher Hill coefficient = steeper transition around K50")
print("- All curves reach the same Vmax at saturating [S]")