import argparse
import os
import warnings

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from processors.candle_processor import CandleProcessor
from processors.news_processor import NewsProcessor

warnings.simplefilter(action="ignore", category=FutureWarning)

SEED = 42


class AiForecast:
    def __init__(self, artifact_dir="./artifacts"):
        self.model = None

        self.artifact_dir = artifact_dir

    def load_model(self, model_path=None):
        """Загружает сохранённую модель CatBoost"""
        if model_path is None:
            model_path = os.path.join(self.artifact_dir, "model_multi.cbm")
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)
        print(f"Модель загружена из {model_path}")

    def merge_candles_news(self, df: pd.DataFrame, news: pd.DataFrame):
        # Инициализируем колонки новостей
        df["news_count_1"] = 0

        df["news_novelty_mean_1"] = 0.0
        df["news_novelty_max_1"] = 0.0
        df["news_novelty_std_1"] = 0.0

        df["news_count_5"] = 0

        df["news_novelty_mean_5"] = 0.0
        df["news_novelty_max_5"] = 0.0
        df["news_novelty_std_5"] = 0.0

        # Сопоставляем новости
        for i, row in df.iterrows():
            ticker = row["ticker"]
            date = row["begin"].date()

            correlate_news_mask = (
                news["tickers"].str.split(",").apply(lambda lst: ticker in lst)
            )
            # correlate_news_mask = news["first_ticker"] == ticker
            correlate_news = news[correlate_news_mask]

            # print(ticker, len(correlate_news))

            # Новости за 1 день
            news_1d = correlate_news[correlate_news["publish_date"].dt.date == date]
            # print(ticker, len(news_1d), '\n')

            df.at[i, "news_count_1"] = len(news_1d)
            if len(news_1d) > 0:
                df.at[i, "news_novelty_mean_1"] = news_1d["novelty"].mean()
                df.at[i, "news_novelty_max_1"] = news_1d["novelty"].max()
                df.at[i, "news_novelty_std_1"] = news_1d["novelty"].std()

            # Новости за последние 5 дней
            start_5 = date - pd.Timedelta(days=4)
            news_5d = correlate_news[
                (correlate_news["publish_date"].dt.date >= start_5)
                & (correlate_news["publish_date"].dt.date <= date)
            ]
            df.at[i, "news_count_5"] = len(news_5d)
            if len(news_5d) > 0:
                df.at[i, "news_novelty_mean_5"] = news_5d["novelty"].mean()
                df.at[i, "news_novelty_max_5"] = news_5d["novelty"].max()
                df.at[i, "news_novelty_std_5"] = news_5d["novelty"].std()

        return df

    def train(self, train_candles: pd.DataFrame, train_news: pd.DataFrame):
        print("====== Training ======")

        news_processor = NewsProcessor(train_news)
        news_processor.add_tickers()
        news_processor.shift_publish_dates()
        news_processor.add_novelty()

        candle_processor = CandleProcessor(train_candles)
        candle_processor.drop_duplicates()
        candle_processor.get_intervals()

        train_intervals = []

        for ticker, interval_id, candles_interval in candle_processor.get_intervals():
            interval, features = candle_processor.add_price_features(candles_interval)

            train_intervals.append((ticker, interval))

        train_dfs = []

        for ticker, df in tqdm(train_intervals):
            df = self.merge_candles_news(df, news_processor.df)
            train_dfs.append(df)

        train_data = pd.concat(train_dfs, ignore_index=True)

        # создаем таргет на все горизонты
        for i in range(1, 21):
            train_data[f"R_t+{i}"] = (
                train_data["close"].shift(-i) / train_data["close"] - 1
            )
        target_cols = [f"R_t+{i}" for i in range(1, 21)]
        train_data.dropna(subset=target_cols, inplace=True)

        X = train_data[features]
        y = train_data[target_cols]

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )

        catboost_params = {
            "iterations": 2000,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "MultiRMSE",  # для многовыходной регрессии
            "eval_metric": "MultiRMSE",
            "random_seed": SEED,
            "verbose": 50,
        }

        self.model = CatBoostRegressor(**catboost_params)
        self.model.fit(X_train, y_train, eval_set=(X_val, y_val))

        os.makedirs(self.artifact_dir, exist_ok=True)
        self.model.save_model(f"{self.artifact_dir}/model_multi.cbm")

    def predict(self, test_candles: pd.DataFrame, test_news: pd.DataFrame):
        print("====== Predicting ======")
        candle_processor = CandleProcessor(candles=test_candles)

        news_processor = NewsProcessor(news_df=test_news)

        news_processor.add_tickers()
        news_processor.shift_publish_dates()
        news_processor.add_novelty()

        news = news_processor.df

        submission_list = []

        for ticker, group in candle_processor.candles.groupby("ticker"):
            candles = group.sort_values("begin").reset_index(drop=True)
            candles, features = candle_processor.add_price_features(candles)
            df = self.merge_candles_news(candles, news)
            last_row = df.iloc[-1:]
            X_test = last_row[features]

            preds = self.model.predict(X_test)[0]  # сразу массив из 20 элементов
            submission_list.append([ticker] + list(preds))

        submission_df = pd.DataFrame(
            submission_list, columns=["ticker"] + [f"p{i}" for i in range(1, 21)]
        )
        return submission_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Forecast train or predict")
    parser.add_argument(
        "mode",
        choices=["train", "predict"],
        help="Operation mode: 'train' to train the model, 'predict' to generate prediction",
    )
    args = parser.parse_args()

    ai_forecast = AiForecast()

    if args.mode == "train":
        train_candles = pd.read_csv("data/candles.csv", parse_dates=["begin"])
        train_news = pd.read_csv("data/news.csv", parse_dates=["publish_date"])

        ai_forecast.train(
            train_candles=train_candles,
            train_news=train_news,
        )
    elif args.mode == "predict":
        test_candles = pd.read_csv("data/candles_2.csv", parse_dates=["begin"])
        test_news = pd.read_csv("data/news_2.csv", parse_dates=["publish_date"])

        ai_forecast.load_model()

        submission = ai_forecast.predict(
            test_candles=test_candles,
            test_news=test_news,
        )
        submission.to_csv("submission.csv", index=False)
