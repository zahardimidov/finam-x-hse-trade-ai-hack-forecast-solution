import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Явные тикеры
ticker_keywords = {
    # Энергетика / нефть и газ
    "GAZP": ["газпром", "gazprom", "gazp", "газ", "газо", "энерг", "топлив", "добыч", "углеводород"],
    "ROSN": ["роснефт", "rosneft", "rosn", "нефт", "нефтя", "топлив", "бензин", "нефтепром", "добыч"],
    "LKOH": ["лукойл", "lukoil", "lkoh", "нефт", "топлив", "бензин", "азс", "заправк", "топливн"],
    "TATN": ["татнефт", "tatneft", "tatn", "тат", "нефт", "топлив", "казан"],
    "SIBN": ["сибур", "sibur", "sibn", "нефтехим", "энерг", "полимер", "пластик"],
    "BANEP": ["башнефт", "bashneft", "banep", "нефт", "топлив", "добыч"],
    "NVTK": ["новатэк", "novatek", "nvtk", "газ", "lng", "метан", "жижен"],
    "RNFT": ["руснефть", "russneft", "rnft", "нефт", "топлив", "бурен"],
    "TRNFP": ["транснефть", "transneft", "trnfp", "трубопровод", "нефтепровод", "транспортиров"],
    "UPRO": ["юнипро", "unipro", "upro", "энерг", "генерац", "электро"],
    "FEES": ["русгидро", "rushydro", "fees", "гидро", "энерг", "электростанц", "водоэлектро"],
    "HYDR": ["русгидро", "rushydro", "hydr", "гидро", "энерг", "станц"],
    "IRAO": ["интеррао", "interrao", "irao", "энерг", "электро", "генерац"],
    "MSNG": ["мосэнерго", "mosenergo", "msng", "энерг", "электро", "станц"],
    "OGKB": ["огк-2", "ogk-2", "ogkb", "энерг", "генерац"],
    "MCX": ["мосэнерг", "mcx", "mosenergo", "энерг"],

    # Банки и финансы
    "SBER": ["сбер", "сбербанк", "sber", "sberbank", "банк", "ипотек", "кредит", "вклад", "финанс", "карта", "платеж", "дебет", "займ"],
    "VTBR": ["втб", "vtb", "vtbr", "банк", "ипотек", "финанс", "инвест"],
    "BSPB": ["банк санкт-петербург", "bspb", "петербург", "банк", "кредит", "вклад"],
    "TCSG": ["тинькофф", "tinkoff", "tcsg", "банк", "финанс", "кредит", "карта", "финтех", "онлайн банк"],
    "CBOM": ["московский кредитный банк", "mkb", "cbom", "банк", "ипотек", "финанс"],
    "ABRD": ["абрау", "abrau", "abrd", "вино", "шампан", "винодельн"],

    # Металлургия / добыча
    "CHMF": ["северстал", "severstal", "chmf", "сталь", "металл", "металлург", "прокат", "завод"],
    "MAGN": ["ммк", "магнитогор", "magnitogorsk", "magn", "металл", "металлург"],
    "NLMK": ["нлмк", "nlmk", "новолипецк", "липецк", "металл", "сталь"],
    "RUAL": ["русал", "rusal", "rual", "алюмин", "металл", "завод"],
    "GMKN": ["норникел", "norilsk", "gmkn", "никел", "металл", "рудник", "добыч", "цветмет"],
    "POLY": ["полиметалл", "polymetal", "poly", "золото", "добыч", "рудник", "драгметалл"],
    "PLZL": ["полюс", "polyus", "plzl", "золото", "рудник", "добыч", "геологоразведк"],
    "SELG": ["селигдар", "seligdar", "selg", "золото", "добыч", "рудник"],
    "ALRS": ["алроса", "alrosa", "alrs", "алмаз", "добыч", "бриллиант"],
    "POGR": ["полиметалл", "pogr", "золото", "металл", "добыч"],

    # Промышленность / машиностроение
    "KMAZ": ["камаз", "kamaz", "kmaz", "грузовик", "автомоб", "транспорт", "машин", "двигател"],
    "AVAZ": ["ваз", "lada", "avtovaz", "avaz", "автоваз", "машин", "автомоб", "лада"],
    "IRKT": ["иркут", "irkut", "irkt", "самолет", "авиа", "авиастроен", "сухой", "мс-21", "боинг", "airbus"],
    "AFLT": ["аэрофлот", "aeroflot", "aflt", "авиа", "аэро", "самолет", "рейс", "перевозк", "полет", "туризм", "путешеств", "билет"],
    "AZMT": ["азимут", "azimut", "azmt", "авиа", "рейс", "полет", "перевозк"],
    "KZOSP": ["кузнецк", "оцм", "kzosp", "металл", "завод"],
    "KROT": ["красный октябр", "krot", "металл", "сталелитейн"],

    # Строительство / недвижимость
    "PIKK": ["пик", "pik", "pikk", "стройк", "девелопмент", "недвижим", "жиль", "квартир", "ипотек", "дом", "строител", "стройматериал"],
    "LSRG": ["лср", "lsr", "lsrg", "строител", "недвижим", "цемент", "бетон", "стройматериал"],
    "ETLN": ["эталон", "etalon", "etln", "стройк", "девелопмент", "недвижим", "жилой комплекс"],

    # Ритейл / потребительский сектор
    "MGNT": ["магнит", "magnit", "mgnt", "ритейл", "розниц", "магазин", "продукт", "торгов", "гипермаркет"],
    "FIVE": ["x5", "пятерочк", "перекресток", "five", "ритейл", "розниц", "магазин", "торгов", "продукт"],
    "MVID": ["мвидео", "mvideo", "mvid", "электроник", "техника", "ритейл", "магазин"],
    "DSKY": ["детский мир", "detsky mir", "dsky", "ритейл", "детск", "игрушк", "товар"],
    "FIXP": ["fix price", "fixprice", "fixp", "ритейл", "магазин", "дешев", "товар"],

    # Химия / удобрения / фарма
    "PHOR": ["фосагро", "phosagro", "phor", "удобрени", "химия", "фосфат", "агро", "сельхоз"],
    "AKRN": ["акрон", "acron", "akrn", "удобрени", "химия", "нитрат", "агро"],
    "URKA": ["уралкалий", "uralkali", "urka", "удобрени", "калий", "агро"],
    "PRTK": ["протек", "protek", "prtk", "фарма", "медиц", "аптек", "здоров"],
    "AFKS": ["афк", "система", "afks", "инвест", "холдинг", "портфель", "группа"],

    # IT / технологии / интернет
    "MAIL": ["vk", "mail", "mailru", "вк", "вконтакт", "технол", "соцсет", "интернет", "мессендж", "онлайн", "платформ"],
    "YNDX": ["яндекс", "yandex", "yndx", "поиск", "интернет", "такси", "технол", "доставка", "маркет", "бизнес", "облако"],
    "SFIN": ["софтлайн", "softline", "sfin", "it", "технол", "айти", "облачн", "цифров"],
    "QIWI": ["киви", "qiwi", "платеж", "кошелек", "терминал", "финтех", "оплат"],
    "OZON": ["ozon", "озон", "маркетплейс", "интернет", "доставка", "товар", "ритейл", "онлайн", "покупк"],
    "CIAN": ["cian", "циан", "недвижим", "портал", "аренд", "продаж"],

    # Транспорт / логистика
    "FLOT": ["совкомфлот", "sovcomflot", "flot", "морск", "танкер", "транспорт", "перевозк", "нефт", "логист"],
    "RZSB": ["ржд", "rzd", "железнодор", "поезд", "вагон", "транспорт", "логист"],
    "GLTR": ["глобалтранс", "globaltrans", "gltr", "логист", "перевозк", "жд", "контейнер"],
    "GTRK": ["гтлк", "gtlk", "gtrk", "лизинг", "авиа", "самолет", "финанс", "флот"],
}


def find_related_tickers(text: str):
    """Ищет все активы, которые упоминаются в тексте и ранжирует по количеству упоминаний"""
    related = []
    text_lower = text.lower()

    for ticker, kw_list in ticker_keywords.items():
        ticker_mentioned_cnt = 0

        for kw in kw_list:
            ticker_mentioned_cnt += text_lower.count(kw.lower())

        if ticker_mentioned_cnt > 0:
            related.append((ticker, ticker_mentioned_cnt))

    related.sort(key=lambda x: x[1], reverse=True)
    related = [x[0] for x in related]

    return ",".join(related)

    # for word, tickers in word_to_tickers.items():
    #    if word in text_lower:
    #        for ticker in tickers:
    #            if ticker not in related:
    #                related.append(ticker)
    # return ",".join(related)


class NewsProcessor:
    def __init__(self, news_df: pd.DataFrame):
        self.df = news_df

        self.df["publish_date"] = pd.to_datetime(self.df["publish_date"])
        self.df.sort_values(by="publish_date", inplace=True)
        self.df["weekday"] = self.df["publish_date"].dt.weekday

        self.df["text"] = self.df["title"] + " " + self.df["publication"]
        self.df = self.df.drop(columns=["title", "publication"])

    def add_tickers(self):
        self.df["tickers"] = self.df.apply(
            lambda row: find_related_tickers(row["text"]), axis=1
        )

        self.df["first_ticker"] = self.df["tickers"].str.split(",", n=1).str[0]

        news_with_tickers = self.df[self.df["tickers"] != ""]
        news_without_tickers = self.df[self.df["tickers"] == ""]

        print(f"Пустых tickers: {len(news_without_tickers)}")
        print(f"Заполненных tickers: {len(news_with_tickers)}")

        self.df = news_with_tickers

    def shift_publish_dates(self):
        self.df = self.df[self.df["weekday"] < 5]

        def shift_publish_date(row):
            if row["weekday"] == 4:
                return row["publish_date"] + pd.Timedelta(days=3)
            else:
                return row["publish_date"] + pd.Timedelta(days=1)

        self.df["publish_date"] = self.df.apply(shift_publish_date, axis=1)
        self.df["weekday"] = self.df["publish_date"].dt.weekday

        print("Сдвиг по дате выполнен")

    def add_novelty(self):
        # --- 1. TF-IDF ---
        vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(self.df["text"])

        # --- 2. SVD (сжатие признаков) ---
        svd = TruncatedSVD(n_components=50, random_state=42)
        svd_matrix = svd.fit_transform(tfidf_matrix)

        # Добавляем векторы в DataFrame
        data_svd = pd.DataFrame(svd_matrix, columns=[f"feat_{i}" for i in range(50)])
        data = pd.concat([self.df.reset_index(drop=True), data_svd], axis=1)

        # --- 3. Агрегация по ticker и publish_date ---
        aggregated = data.groupby(["first_ticker", "publish_date"]).agg(
            {f"feat_{i}": "mean" for i in range(50)}
        )
        aggregated.reset_index(inplace=True)

        # --- 4. Вычисляем динамику (новизну) ---
        # Функция косинусного расстояния
        def cosine_distance(a, b):
            return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

        novelty_list = []

        for ticker in aggregated["first_ticker"].unique():
            df_ticker = aggregated[aggregated["first_ticker"] == ticker].sort_values(
                "publish_date"
            )
            # среднее векторов за предыдущие дни
            prev_vectors = []
            for idx, row in df_ticker.iterrows():
                today_vector = row[[f"feat_{i}" for i in range(50)]].values
                if len(prev_vectors) == 0:
                    novelty = 0  # нет предыдущих дней
                else:
                    mean_prev = np.mean(prev_vectors, axis=0)
                    novelty = cosine_distance(today_vector, mean_prev)
                novelty_list.append(novelty)
                prev_vectors.append(today_vector)

        aggregated["novelty"] = novelty_list

        self.df["publish_date"] = pd.to_datetime(self.df["publish_date"])
        aggregated["publish_date"] = pd.to_datetime(aggregated["publish_date"])

        # Берём только ключи и novelty из агрегированных данных
        novelty_df = aggregated[["first_ticker", "publish_date", "novelty"]]

        # Мержим
        self.df = self.df.merge(
            novelty_df, on=["first_ticker", "publish_date"], how="left"
        )

        print("Новизна добавлена")
