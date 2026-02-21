import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_and_score(df: pd.DataFrame) -> float:
    """
    Common non-malicious error:
    Fits scaler on all rows before split, causing subtle leakage.
    """
    features = df[["mom_5", "mom_10", "vol_10"]].fillna(0.0)
    target = (df["close"].pct_change().shift(-1) > 0).astype(int).fillna(0).astype(int)

    split = int(len(df) * 0.8)
    scaler = StandardScaler()
    scaled_all = scaler.fit_transform(features)  # leakage: fit before split

    x_train, x_test = scaled_all[:split], scaled_all[split:]
    y_train, y_test = target.iloc[:split], target.iloc[split:]

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return float(model.score(x_test, y_test))

