import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to generate data
def gendata(c):
    x = np.linspace(0, 10, 100)
    p_x = c[0] * np.exp(-c[1] * (x - c[2]) ** 2) 
    return x, p_x

# Generate 10 synthetic distributions and store coefficients
np.random.seed(0)
distributions = []
coefficients = []

for _ in range(1000):
    a1 = np.random.uniform(0, 3)
    b1 = np.random.uniform(0, 5)
    c1 = np.random.uniform(0, 10)
    coeff = [a1, b1, c1]
    x, p_x = gendata(coeff)
    distributions.append(p_x)
    coefficients.append(coeff)

# Convert to DataFrame
df_distributions = pd.DataFrame(distributions, columns=[f'x_{i}' for i in range(100)])
df_coefficients = pd.DataFrame(coefficients, columns=['a1', 'b1', 'c1'])

# Concatenate both DataFrames
df = pd.concat([df_distributions, df_coefficients], axis=1)

# Plot the generated distributions
plt.figure(figsize=(10, 6))
for i, p_x in enumerate(distributions):
    plt.plot(x, p_x, label=f'Distribution {i+1}')
plt.legend()
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Generated Distributions')
plt.show()

# Display the DataFrame
print(df.head())

# Save the DataFrame for later use
df.to_csv('dataforCVAE3.csv', index=False)
