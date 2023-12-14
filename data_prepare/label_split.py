import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()


def label_transformer(filename, test_size, seed):
    df = pd.read_csv(filename)
    train_data, test_data = train_test_split(df, test_size=test_size, shuffle=True, random_state=seed)
    train_data, val_data = train_test_split(train_data, test_size=test_size, shuffle=True, random_state=seed)

    train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    scaler.fit(train_data['income'].values.reshape(-1, 1))
    train_data["income_std"] = scaler.transform(train_data["income"].values.reshape(-1, 1))
    val_data["income_std"] = scaler.transform(val_data["income"].values.reshape(-1, 1))
    test_data["income_std"] = scaler.transform(test_data["income"].values.reshape(-1, 1))

    train_data["income_std"] = train_data["income_std"].astype(np.float32)
    val_data["income_std"] = val_data["income_std"].astype(np.float32)
    test_data["income_std"] = test_data["income_std"].astype(np.float32)

    return train_data, val_data, test_data
