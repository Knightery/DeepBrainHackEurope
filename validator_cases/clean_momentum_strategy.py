"""
Time-series momentum strategy (Moskowitz, Ooi & Pedersen 2012 style).

Signal: 12-1 month return (63-day lookback, skip most recent 21 days).
Scaler fitted only on training rows to avoid look-ahead leakage.
Walk-forward out-of-sample split: first 189 days train, last 63 days OOS.
"""
from __future__ import annotations

import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def build_signal(df: pd.DataFrame) -> pd.Series:
    """
    Momentum signal: sign of return from (t-63) to (t-21).
    Uses only past data; no negative shifts.
    """
    past_return = df["close"].pct_change(63 - 21).shift(21)
    return (past_return > 0).astype(int)


def train_and_evaluate(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["signal"] = build_signal(df)
    df["fwd_ret"] = df["close"].pct_change().shift(-1)
    df = df.dropna(subset=["signal", "fwd_ret", "mom_21", "mom_63", "rvol_21"])

    split = int(len(df) * 0.75)
    train, test = df.iloc[:split], df.iloc[split:]

    features = ["mom_21", "mom_63", "rvol_21"]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[features])
    x_test = scaler.transform(test[features])          # fit on train only

    y_train = (train["fwd_ret"] > 0).astype(int)
    y_test = (test["fwd_ret"] > 0).astype(int)

    model = LogisticRegression(max_iter=500, C=1.0)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    strategy_rets = test["fwd_ret"].values * (2 * preds - 1)

    ann_ret = strategy_rets.mean() * 252
    ann_vol = strategy_rets.std() * math.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    equity = pd.Series(strategy_rets).add(1).cumprod()
    max_dd = float((equity / equity.cummax() - 1).min())
    accuracy = float((preds == y_test.values).mean())

    return {
        "n_train": len(train),
        "n_test": len(test),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 3),
        "accuracy": round(accuracy, 3),
    }


if __name__ == "__main__":
    import pathlib
    csv_path = pathlib.Path(__file__).parent / "data" / "clean_momentum_prices.csv"
    df = pd.read_csv(csv_path)
    result = train_and_evaluate(df)
    print(result)
