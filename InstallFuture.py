import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import holidays


def preprocess_data(df):
    df["DOWNLOAD"] = df["DOWNLOAD"].astype(str).str.replace(",", ".").astype(float)
    df["UPLOAD"] = df["UPLOAD"].astype(str).str.replace(",", ".").astype(float)
    df["weekday"] = df["TIME_STAMP"].dt.weekday
    df["hour"] = df["TIME_STAMP"].dt.hour
    df["minute"] = df["TIME_STAMP"].dt.minute
    df["is_weekend"] = df["weekday"] >= 5

    # 'TIME_STAMP' sütunundaki tarihler için Türkiye'deki resmi tatilleri kontrol edip, 
    # sonucu 'is_holiday'(boolean) sütununa ata.
    df["is_holiday"] = df["TIME_STAMP"].dt.date.isin(holidays.Turkey(years=df["TIME_STAMP"].dt.year.unique()))
    return df


def analyze_traffic(df):
    df["total_traffic"] = df["DOWNLOAD"] + df["UPLOAD"]
    traffic_by_hour = df.groupby("hour")["total_traffic"].sum()
    # En yoğun saat dilimlerini bul 
    sorted_traffic = traffic_by_hour.sort_values(ascending=False)
    # 23.00 En yoğun olan saat dilimi
    top_hours = sorted_traffic.head(24)
    return top_hours


def detect_anomalies(df):
    # Sütunlardaki eksik değerleri ortalamaya göre doldur
    imputer = SimpleImputer(strategy="mean")
    df[["DOWNLOAD", "UPLOAD", "hour", "minute", "is_weekend", "is_holiday"]] = imputer.fit_transform(df[["DOWNLOAD", "UPLOAD", "hour", "minute", "is_weekend", "is_holiday"]])
    features = df[["DOWNLOAD", "UPLOAD", "hour", "minute", "is_weekend", "is_holiday"]]
    
    # Isolation Forest algoritması ile anomali tespiti yap ( contamination=0.01 )
    model = IsolationForest(contamination=0.01, random_state=42)
    df["anomaly"] = model.fit_predict(features)
    df["anomaly"] = df["anomaly"] == -1
    specific_data = df[(df["TIME_STAMP"] == "2024-03-05 18:25:00") & (df["SERVER_NAME"] == "10.0.901.xx Server 1")]
    is_anomaly = specific_data["anomaly"].values[0] if not specific_data.empty else False
    print(f"2024-03-05 18:25:00 tarihinde ve 10.0.901.xx Server 1 için veri noktası anomali: {is_anomaly}")
    return df

# Time series forecasting
def time_series_forecasting(df):
    results = {}
    for column in ["DOWNLOAD", "UPLOAD"]:
        prophet_df = df[["TIME_STAMP", column]].rename(columns={"TIME_STAMP": "ds", column: "y"})
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=0, freq="5T")  # 5 dakikalık aralıklarla tahmin yapması
        forecast = model.predict(future)
        prophet_df["forecast"] = forecast["yhat"][: len(prophet_df)]
        prophet_df["residual"] = prophet_df["y"] - prophet_df["forecast"]
        threshold = 2 * prophet_df["residual"].std()
        prophet_df["anomaly"] = np.abs(prophet_df["residual"]) > threshold
        specific_data = prophet_df[prophet_df["ds"] == "2024-03-05 18:25:00"]
        is_anomaly = specific_data["anomaly"].values[0] if not specific_data.empty else False
        print(f"{column} - 2024-03-05 18:25:00 tarihinde veri noktası anomali: {is_anomaly}")
        results[column] = prophet_df
    return results
    
def plot_forecast(prophet_df, column):
    plt.figure(figsize=(12, 6))
    plt.plot(prophet_df["ds"], prophet_df["y"], label=f"Gerçek {column.capitalize()} Hızı")
    plt.plot(prophet_df["ds"], prophet_df["forecast"], label=f"Tahmin Edilen {column.capitalize()} Hızı", linestyle="--")
    plt.scatter(prophet_df[prophet_df["anomaly"]]["ds"], prophet_df[prophet_df["anomaly"]]["y"], color="red", label="Anomali")
    plt.legend()
    plt.xlabel("Zaman")
    plt.ylabel(f"{column.capitalize()} Hızı (Kbps)")
    plt.title(f"Zaman Serisi Tahmini ve Anomali Tespiti (Prophet Kullanarak) - {column.capitalize()}")
    plt.show()

def main():
    df = pd.read_excel("dataset.xlsx", parse_dates=["TIME_STAMP"])
    df = preprocess_data(df)
    top_hours = analyze_traffic(df)
    df = detect_anomalies(df)


    plt.figure(figsize=(15, 12))

    plt.subplot(311)
    plt.bar(top_hours.index, top_hours.values)
    plt.xlabel("Saat Dilimi")
    plt.ylabel("Toplam Trafik (Kbps)")
    plt.title("En Yoğun Saat Dilimleri")
    plt.xticks(range(24))
    plt.grid(axis="y")

    plt.subplot(312)
    plt.plot(df["TIME_STAMP"], df["DOWNLOAD"], label="İndirme Hızı (Kbps)")
    plt.scatter(df[df["anomaly"]]["TIME_STAMP"], df[df["anomaly"]]["DOWNLOAD"], color="red", label="Anomali")
    plt.legend()
    plt.title("Anomali Tespiti - İndirme Hızı")

    plt.subplot(313)
    plt.plot(df["TIME_STAMP"], df["UPLOAD"], label="Yükleme Hızı (Kbps)")
    plt.scatter(df[df["anomaly"]]["TIME_STAMP"], df[df["anomaly"]]["UPLOAD"], color="red", label="Anomali")
    plt.legend()
    plt.title("Anomali Tespiti - Yükleme Hızı")

    plt.tight_layout()

    results = time_series_forecasting(df)
    for column in results:
        plot_forecast(results[column], column)

if __name__ == "__main__":
    main()
 