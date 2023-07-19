"""Compare stress data to COMSOL."""
import pandas as pd
import matplotlib.pyplot as plt

# normal stress data from COMSOL
stress_x_df = pd.read_csv('tests/data/COMSOL_cylinder_stress_x.csv')
stress_y_df = pd.read_csv('tests/data/COMSOL_cylinder_stress_y.csv')
theta = stress_x_df['theta']
stress_x = stress_x_df['stress_x']
stress_y = stress_y_df['stress_y']
fig, ax = plt.subplots()
sample = 10
ax.plot(theta[::10], stress_x[::10], label='COMSOL Stress X')
ax.plot(theta[::10], stress_y[::10], label='COMSOL Stress Y')
plt.show()

