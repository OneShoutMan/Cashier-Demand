import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import io
import math
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from itertools import product
import warnings
import optuna
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("Model Prediksi Jumlah Kasir Mingguan")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    col_map_raw = {c.strip().lower(): c for c in df.columns}
    def getcol(*cands):
        for c in cands:
            key = c.lower()
            if key in col_map_raw:
                return col_map_raw[key]
        return None

    col_tanggal = getcol("tanggal","date","tgl")
    col_jam     = getcol("jam","hour","hours")
    col_tx      = getcol("trx total","trx_total","trx","tx")
    col_var     = getcol("var total","var_total","var","items")

    required = [col_tanggal, col_jam, col_tx, col_var]
    if any(x is None for x in required):
        st.error("Kolom wajib tidak ditemukan. Pastikan ada: 'tanggal', 'jam', 'trx TOTAL', 'var TOTAL'.")
        st.stop()

    df = df.rename(columns={
        col_tanggal: "Date",
        col_jam: "Hour",
        col_tx: "Tx",
        col_var: "Var"
    })

    def parse_date_any(s):
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt).any() if isinstance(dt, pd.Series) else pd.isna(dt):
            return pd.to_datetime(s, dayfirst=False, errors="coerce")
        return dt
    df["Date"] = parse_date_any(df["Date"])

    if pd.api.types.is_numeric_dtype(df["Hour"]):
        df["Hour"] = df["Hour"].astype(int)
    else:
        df["Hour"] = (
            df["Hour"]
            .astype(str)
            .str.extract(r"(\d{1,2})", expand=False)
            .astype(float)
            .fillna(0)
            .astype(int)
        )

    for c in ["Tx","Var"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    df = df.dropna(subset=["Date"]).copy()

    df = df.groupby(["Date","Hour"], as_index=False)[["Tx","Var"]].sum()

    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.strftime("%A")

    tanggal_mulai = st.date_input("Pilih tanggal mulai prediksi", value=df["Date"].min().date())
    tanggal_akhir = tanggal_mulai + timedelta(days=6)

    with st.sidebar:
        st.header("⚙️ Pengaturan Model")

        min_payday = st.number_input("Tanggal Awal Payday", min_value=1, max_value=31)
        max_payday = st.number_input("Tanggal Akhir Payday", min_value=1, max_value=31)

        durasi_transisi = st.number_input(
            "Durasi Transisi (menit)", min_value=0.0, value=2.0
        )
        durasi_per_barang = st.number_input(
            "Durasi Scan/Item (menit)", min_value=0.01, value=0.05
        )
        waktu_tunggu_maks = st.number_input(
            "Waktu Tunggu Maksimum (menit)", min_value=1, value=15
        )

        st.markdown("---")
        st.header("📅 Pengaturan Scheduling")

        shift_jam = st.number_input(
            "Durasi Shift (jam)", min_value=4, max_value=12, value=8
        )
        min_istirahat = st.number_input(
            "Min Jam Setelah Masuk (untuk Istirahat)", min_value=1, max_value=6, value=3
        )
        max_istirahat = st.number_input(
            "Max Jam Setelah Masuk (untuk Istirahat)", min_value=2, max_value=8, value=5
        )

        sidebar_hari = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pilihan_hari = st.selectbox("Pilih hari untuk scheduling", sidebar_hari, index=0)

        tombol_prediksi = st.button("✅ Lakukan Analisis")


    if tombol_prediksi:
        df["Is_Payday"] = df["Day"].between(min_payday, max_payday)

        target_dates = pd.date_range(start=tanggal_mulai, end=tanggal_akhir)
        info_hari = pd.DataFrame({
            "Date": target_dates,
            "Day": target_dates.day,
            "Weekday": target_dates.strftime("%A")
        })
        info_hari["Is_Payday"] = info_hari["Day"].between(min_payday, max_payday)

        payday_days  = info_hari[info_hari["Is_Payday"]]["Weekday"].tolist()
        reguler_days = info_hari[~info_hari["Is_Payday"]]["Weekday"].tolist()

        def hitung_kasir_realistis(data, durasi_transisi, durasi_per_barang, max_wait_time):
            hasil = []
            for _, row in data.iterrows():
                jam = row["Hour"]
                transaksi = row["Tx"]
                total_barang = row["Var"]

                barang_per_transaksi = np.ceil(total_barang / transaksi) if transaksi else 0
                waktu_per_customer = durasi_transisi + (barang_per_transaksi * durasi_per_barang)

                kapasitas_per_kasir = 60 / waktu_per_customer if waktu_per_customer else 1
                optimal_kasir = int(np.ceil(transaksi / kapasitas_per_kasir)) if kapasitas_per_kasir else 1
                rasio_antrian = 0.30
                avg_wait_time = ((transaksi * rasio_antrian) * waktu_per_customer) / optimal_kasir if optimal_kasir else 0

                while avg_wait_time > max_wait_time:
                    optimal_kasir += 1
                    avg_wait_time = ((transaksi * rasio_antrian) * waktu_per_customer) / optimal_kasir

                if optimal_kasir > 1:
                    optimal_kasir -= 1

                hasil.append({
                    "Hour": jam,
                    "Date": row["Date"],
                    "Tx": transaksi,
                    "Var": total_barang,
                    "Avg Item/Tx": round(barang_per_transaksi, 2),
                    "Waktu/Customer (min)": round(waktu_per_customer, 2),
                    "Optimal Kasir": optimal_kasir,
                    "Waktu Antre (min)": avg_wait_time,
                    "Gajian": row["Is_Payday"]
                })
            return pd.DataFrame(hasil)

        df["Weekday"] = df["Date"].dt.strftime("%A")
        df_payday  = df[df["Is_Payday"] & df["Weekday"].isin(payday_days)].copy()
        df_reguler = df[~df["Is_Payday"] & df["Weekday"].isin(reguler_days)].copy()

        hasil_payday = hitung_kasir_realistis(df_payday, durasi_transisi, durasi_per_barang, waktu_tunggu_maks)
        hasil_reguler = hitung_kasir_realistis(df_reguler, durasi_transisi, durasi_per_barang, waktu_tunggu_maks)

        hasil_all = pd.concat([hasil_payday, hasil_reguler], ignore_index=True)
        hasil_all["Weekday"] = hasil_all["Date"].dt.strftime("%A")

        hasil_all["Safety Net"] = (
            hasil_all["Optimal Kasir"] - np.ceil(
                hasil_all["Tx"] / (60 / hasil_all["Waktu/Customer (min)"])
            ).astype(int)
        )

        grouped = hasil_all.groupby(["Weekday", "Hour"]).agg({
            "Optimal Kasir": "mean",
            "Tx": "mean",
            "Var": "mean"
        }).reset_index()
        grouped["Optimal Kasir"] = grouped["Optimal Kasir"].apply(np.ceil).astype(int)

        weekday_to_date = dict(zip(info_hari["Weekday"], info_hari["Date"]))
        grouped["Date"] = grouped["Weekday"].map(weekday_to_date)

        ordered_days = info_hari["Weekday"].tolist()
        grouped["Weekday"] = pd.Categorical(
            grouped["Weekday"],
            categories=ordered_days,
            ordered=True
        )

        WORK_HOURS = list(range(7, 22))
        def reindex_hours(pvt):
            return pvt.reindex(index=[f"{h}:00" for h in WORK_HOURS])

        st.markdown("### Visualisasi Demand Kasir")

        safety_df = hasil_all.copy()
        safety_df["Safety Net"] = (
            safety_df["Optimal Kasir"] - np.ceil(
                safety_df["Tx"] / (60 / safety_df["Waktu/Customer (min)"])
            )
        ).clip(lower=0).astype(int)

        safety_df["Hour"] = safety_df["Hour"].astype(int).astype(str) + ":00"
        pivot_safety = safety_df.pivot_table(
            index="Hour",
            columns="Weekday",
            values="Safety Net",
            aggfunc="mean"
        ).reindex(columns=ordered_days)
        pivot_safety = reindex_hours(pivot_safety)

        fig_safety, ax_safety = plt.subplots(figsize=(9, 5))
        sns.heatmap(
            pivot_safety,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar_kws={'label': 'Safety Net (Kasir Lebih)'},
            ax=ax_safety
        )
        ax_safety.set_title("Safety Net Kasir per Jam & Hari", fontsize=11)
        st.pyplot(fig_safety)

        final_result = grouped[["Date", "Weekday", "Hour", "Tx", "Var", "Optimal Kasir"]].sort_values(["Date", "Hour"])
        final_result["Hour"] = final_result["Hour"].astype(int)

        pivot_data = final_result.assign(Hour=final_result["Hour"].astype(str) + ":00").pivot_table(
            index="Hour",
            columns="Weekday",
            values="Optimal Kasir",
            aggfunc="mean"
        ).reindex(columns=ordered_days)
        pivot_data = reindex_hours(pivot_data)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            cbar_kws={'label': 'Optimal Kasir'},
            ax=ax
        )
        ax.set_title("Jumlah Kasir Optimal per Jam & Hari", fontsize = 11)
        st.pyplot(fig)

        st.markdown("### Tren Jumlah Kasir")
        fig_line, ax_line = plt.subplots(figsize=(9, 5))
        for day in ordered_days:
            day_data = final_result[final_result["Weekday"] == day].sort_values("Hour")
            ax_line.plot(day_data["Hour"], day_data["Optimal Kasir"], marker="o", label=day)

        ax_line.set_title("Tren Jumlah Kasir per Hari", fontsize=11)
        ax_line.set_xlabel("Jam")
        ax_line.set_ylabel("Optimal Kasir")
        ax_line.legend(title="Hari")
        ax_line.grid(True, linestyle="--", alpha=0.8)
        st.pyplot(fig_line)

        st.markdown("### Tabel Prediksi Jumlah Kasir")
        st.dataframe(final_result.sort_values(["Date","Hour"]), use_container_width=True)

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            final_result.to_excel(writer, index=False, sheet_name='Prediksi_Kasir')

            img_buffer_heatmap = io.BytesIO()
            fig.savefig(img_buffer_heatmap, format='png', dpi=150, bbox_inches='tight')
            img_buffer_heatmap.seek(0)

            img_buffer_trend = io.BytesIO()
            fig_line.savefig(img_buffer_trend, format='png', dpi=150, bbox_inches='tight')
            img_buffer_trend.seek(0)

            workbook = writer.book
            worksheet = writer.sheets['Prediksi_Kasir']
            worksheet.insert_image('H2', 'heatmap.png', {'image_data': img_buffer_heatmap})
            worksheet.insert_image('H26', 'trend.png', {'image_data': img_buffer_trend})

            excel_buffer.seek(0)

        st.markdown("""
        <style>
        .big-download-button button {
            background-color: #4CAF50;
            color: white;
            font-size: 20px !important;
            font-weight: bold;
            padding: 15px 0px;
            border-radius: 10px;
        }
        .big-download-button button:hover {
            background-color: #45a049;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="big-download-button">', unsafe_allow_html=True) 
            st.download_button(
                label="Download Excel",
                data=excel_buffer,
                file_name="Demand Kasir Mingguan.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True) 
        

        st.markdown("### Tabel Scheduling")
        selected_day = pilihan_hari

        if selected_day not in ordered_days:
            st.warning(f"Hari '{selected_day}' tidak ada dalam minggu yang dipilih ({tanggal_mulai} - {tanggal_akhir}). Pilih hari yang sesuai minggu tersebut.")
        else:
            final_result_day = final_result[final_result["Weekday"] == selected_day].copy()

            def buat_schedule(final_result_subset, shift_jam=8, min_istirahat=3, max_istirahat=5):
                kebutuhan = final_result_subset.groupby("Hour")["Optimal Kasir"].max().to_dict()
                schedule = []
                total_kasir = 0
                aktif = 0
                istirahat_jadwal = {}

                for jam in range(7, 22):
                    masuk = 0
                    istirahat = 0
                    pulang = 0

                    req = kebutuhan.get(jam, 0)

                    if req > aktif:
                        masuk = req - aktif
                        total_kasir += masuk
                        aktif += masuk

                        for _ in range(masuk):
                            jam_ist = jam + np.random.randint(min_istirahat, max_istirahat + 1)
                            if jam_ist < 22:
                                istirahat_jadwal[jam_ist] = istirahat_jadwal.get(jam_ist, 0) + 1

                    if jam in istirahat_jadwal:
                        istirahat = istirahat_jadwal[jam]
                        aktif -= istirahat

                    schedule.append({
                        "Hour": float(jam),
                        "Req Kasir": int(req),
                        "M": int(masuk),
                        "I": int(istirahat),
                        "P": int(pulang),
                        "C": int(aktif)
                    })

                df_sched = pd.DataFrame(schedule)

                total_row = {
                    "Hour": "Total",
                    "Req Kasir": df_sched["Req Kasir"].sum(),
                    "M": df_sched["M"].sum(),
                    "I": df_sched["I"].sum(),
                    "P": df_sched["P"].sum(),
                    "C": df_sched["C"].sum(),
                }
                df_sched = pd.concat([df_sched, pd.DataFrame([total_row])], ignore_index=True)
                return df_sched, int(total_kasir)

            if not final_result_day.empty:
                df_sched, total_kasir = buat_schedule(
                    final_result_day,
                    shift_jam=shift_jam,
                    min_istirahat=min_istirahat,
                    max_istirahat=max_istirahat
                )
                st.write(f"**Hari terpilih:** {selected_day} — Tanggal: {final_result_day['Date'].iloc[0].date()}")
                st.dataframe(df_sched, use_container_width=True)
                st.success(f"Total Kasir yang dijadwalkan: {total_kasir}")
            else:
                st.warning("Tidak ada data untuk hari yang dipilih pada minggu tersebut.")

        st.markdown("### Forecasting Data Harian (XGBoost)")

        df['tanggal'] = pd.to_datetime(df['Date'])
        data_harian = df.groupby('tanggal').agg({
            'Tx': 'sum',
            'Var': 'sum'
        }).reset_index()

        def create_features(df,col,n_lags=14):
            df_feat = df.copy()
            df_feat['dayofweek'] = df_feat['tanggal'].dt.dayofweek
            df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)
            df_feat['day'] = df_feat['tanggal'].dt.day
            df_feat['month'] = df_feat['tanggal'].dt.month

            for lag in range(1, n_lags+1):
                df_feat[f"{col}_lag{lag}"] = df_feat[col].shift(lag)

            # capture pola data historis
            for win in [7, 14, 30]:
                df_feat[f"{col}_rollmean{win}"] = df_feat[col].shift(1).rolling(win).mean()
            for win in [7, 14]:
                df_feat[f"{col}_rollstd{win}"] = df_feat[col].shift(1).rolling(win).std()

            return df_feat.dropna()

        tx_feat = create_features(data_harian[['tanggal','Tx']], 'Tx')
        X_tx = tx_feat.drop(columns=['tanggal','Tx'])
        y_tx = tx_feat['Tx']

        var_feat = create_features(data_harian[['tanggal','Var']], 'Var')
        X_var = var_feat.drop(columns=['tanggal','Var'])
        y_var = var_feat['Var']

        X_train_tx, X_test_tx, y_train_tx, y_test_tx = train_test_split(X_tx, y_tx, test_size=0.2, shuffle=False)
        X_train_var, X_test_var, y_train_var, y_test_var = train_test_split(X_var, y_var, test_size=0.2, shuffle=False)

        def objective_tx(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "n_jobs": -1
            }
            model = XGBRegressor(**params)
            model.fit(X_train_tx, y_train_tx)
            preds = model.predict(X_test_tx)
            return mean_absolute_error(y_test_tx, preds)

        def objective_var(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "random_state": 42,
                "n_jobs": -1
            }
            model = XGBRegressor(**params)
            model.fit(X_train_var, y_train_var)
            preds = model.predict(X_test_var)
            return mean_absolute_error(y_test_var, preds)

        st.write("Sedang melakukan tuning hyperparameter (TX)...")
        study_tx = optuna.create_study(direction="minimize")
        study_tx.optimize(objective_tx, n_trials=30, show_progress_bar=True)

        st.write("Sedang melakukan tuning hyperparameter (VAR)...")
        study_var = optuna.create_study(direction="minimize")
        study_var.optimize(objective_var, n_trials=30, show_progress_bar=True)

        best_params_tx = study_tx.best_params
        best_params_var = study_var.best_params

        #pake ini kalo mau tuning
        use_tuned = False  

        if use_tuned:
            st.write("Menggunakan parameter hasil tuning Optuna")
            xgb_tx = XGBRegressor(**best_params_tx)
            xgb_var = XGBRegressor(**best_params_var)
        else:
            st.write("Menggunakan parameter default")
            xgb_tx = XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=7,
                subsample=0.9, colsample_bytree=0.9, random_state=42
            )
            xgb_var = XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=7,
                subsample=0.9, colsample_bytree=0.9, random_state=42
            )

        xgb_tx.fit(X_train_tx,y_train_tx)
        xgb_var.fit(X_train_var, y_train_var)

        tx_pred_test = xgb_tx.predict(X_test_tx)
        var_pred_test = xgb_var.predict(X_test_var)

        mae_tx = mean_absolute_error(y_test_tx, tx_pred_test)
        mse_tx = mean_squared_error(y_test_tx, tx_pred_test)
        mape_tx = mean_absolute_percentage_error(y_test_tx, tx_pred_test) * 100

        mae_var = mean_absolute_error(y_test_var, var_pred_test)
        mse_var = mean_squared_error(y_test_var, var_pred_test)
        mape_var = mean_absolute_percentage_error(y_test_var, var_pred_test) * 100

        st.write("**Evaluasi Model XGBoost (dengan fitur kalender & lag panjang)**")
        st.write(f"Tx - MAE: {mae_tx:.2f}, MAPE: {mape_tx:.2f}%")
        st.write(f"Var - MAE: {mae_var:.2f}, MAPE: {mape_var:.2f}%")

        forecast_horizon = 30
        last_date=data_harian['tanggal'].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

        def forecast_future(df,col,model,horizon = 30):
            hist = df.copy()
            preds = []
            for i in range(horizon):
                feat = create_features(hist[['tanggal',col]], col).iloc[-1:].drop(columns=['tanggal',col])
                pred = model.predict(feat)[0]
                next_date = hist['tanggal'].iloc[-1] + pd.Timedelta(days=1)
                hist = pd.concat([hist, pd.DataFrame({'tanggal':[next_date], col:[pred]})], ignore_index=True)
                preds.append((next_date, pred))
            return pd.DataFrame(preds, columns=['tanggal', col])

        tx_forecast = forecast_future(data_harian[['tanggal','Tx']], 'Tx', xgb_tx, forecast_horizon)
        var_forecast = forecast_future(data_harian[['tanggal','Var']], 'Var', xgb_var, forecast_horizon)

        forecast_df = pd.merge(tx_forecast, var_forecast, on='tanggal')

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(data_harian['tanggal'], data_harian['Tx'], label='Tx Historis', color='blue')
        axes[0].plot(forecast_df['tanggal'], forecast_df['Tx'], label='Tx Forecast', linestyle='--', color='orange')
        axes[0].set_title('Tx Forecast (Harian)')
        axes[0].set_ylabel('Nilai Tx')
        axes[0].grid(True)
        axes[0].legend()

        axes[1].plot(data_harian['tanggal'], data_harian['Var'], label='Var Historis', color='green')
        axes[1].plot(forecast_df['tanggal'], forecast_df['Var'], label='Var Forecast', linestyle='--', color='red')
        axes[1].set_title('Var Forecast (Harian)')
        axes[1].set_ylabel('Nilai Var')
        axes[1].set_xlabel('Tanggal')
        axes[1].grid(True)
        axes[1].legend()

        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=2))

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### Hasil Forecast 30 Hari ke Depan")
        st.dataframe(forecast_df.style.format({
            "Tx": "{:.0f}",
            "Var": "{:.0f}"
        }))

        forecast_buffer = io.BytesIO()
        with pd.ExcelWriter(forecast_buffer, engine="xlsxwriter") as writer:
            forecast_df.to_excel(writer, index=False, sheet_name="Forecast_30Hari")

            img_buffer_forecast = io.BytesIO()
            fig.savefig(img_buffer_forecast, format="png", dpi=150, bbox_inches="tight")
            img_buffer_forecast.seek(0)

            workbook = writer.book
            worksheet = writer.sheets["Forecast_30Hari"]
            worksheet.insert_image("E4", "forecast.png", {"image_data": img_buffer_forecast})

            forecast_buffer.seek(0)

        st.markdown("""
        <style>
        .big-download-button-forecast button {
            background-color: #2196F3;
            color: white;
            font-size: 18px !important;
            font-weight: bold;
            padding: 12px 0px;
            border-radius: 10px;
        }
        .big-download-button-forecast button:hover {
            background-color: #1976D2;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="big-download-button-forecast">', unsafe_allow_html=True)
            st.download_button(
                label="Download Forecast Excel",
                data=forecast_buffer,
                file_name="Forecast_DataToko.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

