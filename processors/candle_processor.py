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

                if (start - end).days == 3:
                    merged[-1].dates.extend(week.dates)
                else:
                    merged.append(week)

            interval_id = 0
            for interval_group in merged:
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
            # Базовые доходности
            tdf["return_1d"] = tdf["close"].pct_change()
            tdf["log_return"] = np.log(tdf["close"]) - np.log(tdf["close"].shift(1))

            # Скользящие средние
            for w in [3, 5, 10]:
                tdf[f"SMA_{w}"] = tdf["close"].rolling(w).mean()
                tdf[f"EMA_{w}"] = tdf["close"].ewm(span=w, adjust=False).mean()

                tdf[f"volatility_{w}"] = tdf["return_1d"].rolling(w).std()
                tdf[f"momentum_{w}"] = tdf["close"].pct_change(w)

            # RSI (Relative Strength Index)
            def compute_rsi(series, window=14):
                delta = series.diff()
                up = np.where(delta > 0, delta, 0)
                down = np.where(delta < 0, -delta, 0)
                roll_up = pd.Series(up).rolling(window).mean()
                roll_down = pd.Series(down).rolling(window).mean()
                RS = roll_up / (roll_down + 1e-9)
                return 100 - (100 / (1 + RS))

            tdf["RSI_7"] = compute_rsi(tdf["close"], window=7)

            # MACD (EMA6 - EMA13) и сигнальная линия
            EMA6 = tdf["close"].ewm(span=6, adjust=False).mean()
            EMA13 = tdf["close"].ewm(span=13, adjust=False).mean()
            tdf["MACD"] = EMA6 - EMA13
            tdf["MACD_signal"] = tdf["MACD"].ewm(span=5, adjust=False).mean()

             # Bollinger Bands (10-дневные)
            sma10 = tdf["close"].rolling(10).mean()
            std10 = tdf["close"].rolling(10).std()
            tdf["bollinger_upper"] = sma10 + 2 * std10
            tdf["bollinger_lower"] = sma10 - 2 * std10
            tdf["bollinger_bandwidth"] = (tdf["bollinger_upper"] - tdf["bollinger_lower"]) / sma10

            # ATR (Average True Range, 7 дней)
            high_low = tdf["high"] - tdf["low"]
            high_close = np.abs(tdf["high"] - tdf["close"].shift())
            low_close = np.abs(tdf["low"] - tdf["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            tdf["ATR_7"] = true_range.rolling(7).mean()

            # Объемные признаки
            for w in [3, 5, 10]:
                tdf[f"volume_mean_{w}"] = tdf["volume"].rolling(w).mean()
                tdf[f"volume_change_{w}"] = tdf["volume"].pct_change(w)

            # Расстояние от скользящих средних (нормализованное)
            for w in [3, 5]:
                ma = tdf[f"SMA_{w}"]
                tdf[f"distance_from_SMA_{w}"] = (tdf["close"] - ma) / ma

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