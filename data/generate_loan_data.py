"""Generate synthetic loan approval dataset with realistic bias patterns.

Creates a dataset where race and gender influence approval in a way that
mimics real-world bias — used as a case study for the ethics dashboard.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger


def generate_loan_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic loan approval data with bias.

    Features:
        income, credit_score, debt_to_income, employment_years,
        loan_amount, loan_term, gender, race, age, education

    The label 'approved' is generated with intentional bias on
    race and gender to demonstrate fairness issues.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating synthetic loan data: {} samples", n_samples)

    # Demographics
    gender = rng.choice(["male", "female"], n_samples, p=[0.55, 0.45])
    race = rng.choice(["white", "black", "hispanic", "asian"], n_samples, p=[0.6, 0.15, 0.15, 0.1])
    age = rng.integers(22, 65, n_samples)
    education = rng.choice(["high_school", "bachelors", "masters", "phd"], n_samples, p=[0.35, 0.40, 0.20, 0.05])

    # Financial features (correlated with demographics to be realistic)
    income_base = rng.lognormal(10.8, 0.6, n_samples)
    # Simulated income gap
    income_factor = np.where(gender == "male", 1.0, 0.85)
    income_factor *= np.where(race == "white", 1.0, np.where(race == "asian", 1.05, 0.80))
    income = np.round(income_base * income_factor, 0).astype(int)

    credit_score = rng.normal(700, 80, n_samples).clip(300, 850).astype(int)
    debt_to_income = rng.uniform(0.05, 0.6, n_samples).round(2)
    employment_years = rng.exponential(7, n_samples).clip(0, 40).round(1)
    loan_amount = (income * rng.uniform(1, 5, n_samples)).round(0).astype(int)
    loan_term = rng.choice([12, 24, 36, 48, 60], n_samples)

    # Approval logic: primarily based on financial merit, but with bias
    logit = (
        0.004 * (credit_score - 600)
        + 0.00002 * income
        - 2.0 * debt_to_income
        + 0.05 * employment_years
        - 0.000005 * loan_amount
        + np.where(education == "phd", 0.3, np.where(education == "masters", 0.2,
                   np.where(education == "bachelors", 0.1, 0.0)))
    )
    # Inject bias: lower approval probability for certain groups
    bias_term = np.where(race == "black", -0.5, np.where(race == "hispanic", -0.3, 0.0))
    bias_term += np.where(gender == "female", -0.2, 0.0)
    logit += bias_term
    logit += rng.normal(0, 0.5, n_samples)

    prob = 1 / (1 + np.exp(-logit))
    approved = (rng.uniform(0, 1, n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "income": income, "credit_score": credit_score,
        "debt_to_income": debt_to_income, "employment_years": employment_years,
        "loan_amount": loan_amount, "loan_term": loan_term,
        "gender": gender, "race": race, "age": age, "education": education,
        "approved": approved,
    })

    approval_rate = df["approved"].mean()
    logger.info("Generated: {:.1%} approval rate, {} samples", approval_rate, len(df))
    logger.info("Approval by race: {}", df.groupby("race")["approved"].mean().round(3).to_dict())
    logger.info("Approval by gender: {}", df.groupby("gender")["approved"].mean().round(3).to_dict())
    return df


if __name__ == "__main__":
    df = generate_loan_dataset()
    df.to_csv("loan_approval.csv", index=False)
    print(f"Saved {len(df)} records to loan_approval.csv")
