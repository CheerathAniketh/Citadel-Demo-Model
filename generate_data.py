"""
Citadel AI — Demo dataset generator
Generates a synthetic hiring dataset with deliberate gender bias baked in.
Does nothing else — training lives in train_model.py, so you can regenerate
data and retrain independently without re-running both every time.

Usage:
    pip install pandas numpy
    python generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # training set size — plenty for a stable biased pattern

# ---------------------------------------------------------------------------
# Generate synthetic applicants
# ---------------------------------------------------------------------------
genders = np.random.choice(['M', 'F'], size=N)
ages = np.random.randint(22, 55, size=N)
experience_years = np.random.randint(0, 20, size=N)
education = np.random.choice(['bachelors', 'masters', 'phd'], size=N, p=[0.6, 0.3, 0.1])
salary_expectation = np.random.randint(40000, 120000, size=N)

# ---------------------------------------------------------------------------
# Bake in deliberate bias: gender directly influences hire probability,
# on top of (weaker) legitimate signal from experience/education.
# This mirrors a real-world scenario where historical hiring data
# encodes discrimination that a naively-trained model then learns.
# ---------------------------------------------------------------------------
base_prob = 0.3 + (experience_years / 20) * 0.3  # legitimate-ish signal
gender_bonus = np.where(genders == 'M', 0.35, -0.15)  # the bias
hire_prob = np.clip(base_prob + gender_bonus, 0.02, 0.98)
hired = np.random.binomial(1, hire_prob)

df = pd.DataFrame({
    'age': ages,
    'experience_years': experience_years,
    'education': education,
    'salary_expectation': salary_expectation,
    'gender': genders,
    'hired': hired
})

df.to_csv('hiring_data.csv', index=False)

print(f"Generated {N} synthetic applicants -> hiring_data.csv")
print("\nHire rate by gender (sanity check — should show a clear gap):")
print(df.groupby('gender')['hired'].mean())