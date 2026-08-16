"""
generate.py

Generates the 2000-row synthetic "hiring" dataset with a deliberate gender
bias baked into the ground-truth labels. No training happens here — this
script only produces the CSV.

Belongs in venv_train (requirements1.txt: numpy, pandas).

Output:
    artifacts/synthetic_hiring_data.csv

Run:
    python3 generate.py

    some commands right now 
"""

import os

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_ROWS = 2000
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n=N_ROWS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    gender = rng.integers(0, 2, size=n)  # 0=male, 1=female, roughly balanced
    years_experience = np.clip(rng.normal(6, 4, n), 0, 25)
    education_level = rng.choice([0, 1, 2, 3], size=n, p=[0.15, 0.45, 0.30, 0.10])
    test_score = np.clip(rng.normal(70, 15, n), 0, 100)
    num_previous_companies = np.clip(rng.poisson(2, n), 0, 10)
    age = np.clip(rng.normal(32, 8, n), 21, 65)

    exp_n = years_experience / 25
    edu_n = education_level / 3
    score_n = test_score / 100
    prev_n = num_previous_companies / 10

    merit_score = 0.35 * exp_n + 0.20 * edu_n + 0.35 * score_n + 0.10 * prev_n

    # Deliberate bias: flat penalty for being female, independent of merit.
    gender_penalty = 0.42 * gender

    noise = rng.normal(0, 0.12, n)
    logit_input = (merit_score - gender_penalty - 0.45) * 6 + noise
    prob_hired = sigmoid(logit_input)
    hired = (rng.random(n) < prob_hired).astype(int)

    df = pd.DataFrame({
        "years_experience": np.round(years_experience, 1),
        "education_level": education_level,
        "test_score": np.round(test_score, 1),
        "num_previous_companies": num_previous_companies,
        "age": np.round(age, 0).astype(int),
        "gender": gender,
        "hired": hired,
    })
    return df


def disparate_impact(df, label_col="hired", group_col="gender", favored_group=0):
    rates = df.groupby(group_col)[label_col].mean()
    unfavored_group = 1 - favored_group
    if favored_group not in rates or unfavored_group not in rates or rates[favored_group] == 0:
        return None
    return rates[unfavored_group] / rates[favored_group]


def statistical_parity_difference(df, label_col="hired", group_col="gender"):
    rates = df.groupby(group_col)[label_col].mean()
    return float(rates.max() - rates.min())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Generating {N_ROWS} synthetic hiring rows (seed={RANDOM_SEED})...")
    df = generate_dataset()

    csv_path = os.path.join(OUT_DIR, "synthetic_hiring_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset -> {csv_path}")

    di = disparate_impact(df)
    spd = statistical_parity_difference(df)
    male_rate = df[df.gender == 0].hired.mean()
    female_rate = df[df.gender == 1].hired.mean()
    print(f"Ground-truth bias check: DI={di:.3f}  SPD={spd:.3f}  "
          f"male_hire_rate={male_rate:.3f}  female_hire_rate={female_rate:.3f}")
    if di is not None and di < 0.8:
        print("Bias confirmed: DI < 0.80 (EEOC 4/5ths rule threshold).")
    else:
        print("WARNING: dataset did not come out biased as expected -- check gender_penalty.")

    print("\nDone. Next: run train.py (in the same venv) to train on this CSV.")


if __name__ == "__main__":
    main()