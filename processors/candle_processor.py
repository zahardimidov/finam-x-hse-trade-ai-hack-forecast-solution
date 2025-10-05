import numpy as np
import pandas as pd


class CandleProcessor:
    def __init__(self, candles: pd.DataFrame):
        self.candles = candles

        self.candles['begin'] = pd.to_datetime(candles['begin'])
        self.candles.sort_values(by='begin', inplace=True)
        self.candles['weekday'] = candles['begin'].dt.weekday

    def drop_duplicates(self):
        duplicates = self.candles[self.candles.duplicated(subset=["ticker", "begin"], keep=False)]
        self.candles = self.candles.drop_duplicates(subset=["ticker", "begin"], keep="first")
        print("Количество удаленных дубликатов:", len(duplicates))

    def get_continuous_weeks(self) -> pd.DataFrame:
        """
        Эта функция возвращает интервалы, состоящие из полных недель (пн-пт)
        """
        filtered_candles = self.candles[self.candles["weekday"] < 5]

        intervals = []
        for ticker, group in filtered_candles.groupby("ticker"):
            group = group.sort_values("begin").reset_index(drop=True)
            day_diff = group["begin"].diff().dt.days.fillna(1)
            interval_id = (day_diff != 1).cumsum()
            group["interval_id"] = interval_id

            for _, interval_group in group.groupby("interval_id"):
                if len(interval_group) < 5:
                    continue

                intervals.append(
                    {
                        "ticker": ticker,
                        "dates": list(interval_group["begin"].dt.date),
                        "length": len(interval_group),
                    }
                )

        return pd.DataFrame(intervals)

    def get_intervals(self):
        """
        Эта функция сохраняет интервалы, состоящие из подряд идущих полных недель (пн-пт)
        """
        weeks = self.get_continuous_weeks()

        for ticker, group in weeks.groupby("ticker"):
            weeks = list(group.itertuples())
            merged = [weeks[0]]

            for week in weeks[1:]:
                start = week.dates[0]
                end = merged[-1].dates[-1]

                merged[-1].dates.extend(week.dates)

                if (start - end).days == 3:
                    merged[-1].dates.extend(week.dates)
                else:
                    merged.append(week)

            interval_id = 0
            for interval_group in merged:
                if len(interval_group.dates) < 30:
                    continue

                interval_id += 1
                dates = interval_group.dates

                mask = (self.candles["ticker"] == ticker) & (
                    self.candles["begin"].dt.date.isin(dates)
                )
                candles_interval = self.candles[mask].sort_values("begin")

                if not candles_interval.empty:
                    yield ticker, interval_id, candles_interval

    @classmethod
    def add_price_features(cls, candles: pd.DataFrame) -> pd.DataFrame:
        # Функция для расчёта индикаторов по каждому тикеру
        def calc_features(tdf: pd.DataFrame) -> pd.DataFrame:
            tdf = tdf.sort_values("begin").reset_index(drop=True)

            # --- Доходности с лагами ---
            for lag in [1, 2, 3, 5, 10, 20, 50, 100]:
                tdf[f"return_lag_{lag}"] = tdf["close"].pct_change(lag)

            # --- Скользящие средние и EMA ---
            for w in [10, 20, 30, 50, 100, 130]:
                tdf[f"SMA_{w}"] = tdf["close"].rolling(w).mean()
                tdf[f"EMA_{w}"] = tdf["close"].ewm(span=w, adjust=False).mean()

            # --- Волатильность ---
            for w in [10, 20, 50, 100]:
                tdf[f"volatility_{w}"] = tdf["close"].pct_change().rolling(w).std()

            # --- Моментум ---
            for w in [5, 10, 20, 50, 100]:
                tdf[f"momentum_{w}"] = tdf["close"].pct_change(w)

            # --- RSI ---
            def compute_rsi(series, window=14):
                delta = series.diff()
                up = np.where(delta > 0, delta, 0)
                down = np.where(delta < 0, -delta, 0)
                roll_up = pd.Series(up).rolling(window).mean()
                roll_down = pd.Series(down).rolling(window).mean()
                RS = roll_up / (roll_down + 1e-9)
                return 100 - (100 / (1 + RS))

            for w in [14, 20, 30]:
                tdf[f"RSI_{w}"] = compute_rsi(tdf["close"], window=w)

            # --- MACD ---
            ema_pairs = [(12, 26), (26, 52)]
            for short_span, long_span in ema_pairs:
                EMA_short = tdf["close"].ewm(span=short_span, adjust=False).mean()
                EMA_long = tdf["close"].ewm(span=long_span, adjust=False).mean()
                tdf[f"MACD_{short_span}_{long_span}"] = EMA_short - EMA_long
                tdf[f"MACD_signal_{short_span}_{long_span}"] = tdf[f"MACD_{short_span}_{long_span}"].ewm(span=9, adjust=False).mean()

            # --- Bollinger Bands ---
            for w in [20, 50, 100]:
                sma = tdf["close"].rolling(w).mean()
                std = tdf["close"].rolling(w).std()
                tdf[f"bollinger_upper_{w}"] = sma + 2 * std
                tdf[f"bollinger_lower_{w}"] = sma - 2 * std
                tdf[f"bollinger_bandwidth_{w}"] = (tdf[f"bollinger_upper_{w}"] - tdf[f"bollinger_lower_{w}"]) / sma

            # --- ATR ---
            high_low = tdf["high"] - tdf["low"]
            high_close = np.abs(tdf["high"] - tdf["close"].shift())
            low_close = np.abs(tdf["low"] - tdf["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            for w in [14, 20, 50, 100]:
                tdf[f"ATR_{w}"] = true_range.rolling(w).mean()

            # --- Объёмные признаки ---
            for w in [5, 10, 20, 50, 100]:
                tdf[f"volume_mean_{w}"] = tdf["volume"].rolling(w).mean()
                tdf[f"volume_change_{w}"] = tdf["volume"].pct_change(w)

            # --- Расстояние от скользящих средних ---
            for w in [10, 20, 50, 100, 130]:
                ma = tdf[f"SMA_{w}"]
                tdf[f"distance_from_SMA_{w}"] = (tdf["close"] - ma) / ma

            tdf.fillna(0, inplace=True)

            return tdf

        df = candles.sort_values(["ticker", "begin"]).copy()

        # Применяем к каждому тикеру
        df = df.groupby("ticker", group_keys=False).apply(
            calc_features, include_groups=True
        )
        features = [
            col
            for col in df.columns
            if col not in ["begin", "ticker", "R_t+1", "R_t+20"]
        ]

        return df, features