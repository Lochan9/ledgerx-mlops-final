import pandas as pd
from synthetic_failures import corrupt_invoice_record

def generate_synthetic_failures(df, n_samples=500):
    valid = df[df.failure == 0]

    if len(valid) == 0:
        return pd.DataFrame([])

    sampled = valid.sample(min(n_samples, len(valid)), replace=True)

    corrupted_rows = []
    for _, row in sampled.iterrows():
        corrupted_rows.append(corrupt_invoice_record(row))

    return pd.DataFrame(corrupted_rows)
