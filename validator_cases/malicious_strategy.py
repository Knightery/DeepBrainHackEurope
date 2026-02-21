import pandas as pd


def build_signal(df: pd.DataFrame) -> pd.Series:
    """
    Intentionally malicious:
    Uses forward returns to create a "predictive" feature.
    Looks like feature engineering but directly leaks future information.
    """
    future_return = df["close"].pct_change().shift(-1)
    signal = (future_return > 0).astype(int)
    return signal


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = build_signal(out)
    out["ret"] = out["close"].pct_change().fillna(0.0)
    out["strategy_ret"] = out["signal"].shift(1).fillna(0.0) * out["ret"]
    out["equity"] = (1.0 + out["strategy_ret"]).cumprod()
    return out

