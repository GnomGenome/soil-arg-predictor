import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Прогноз состояния почвы по ARG", layout="wide")

st.title("🌱 Прогноз состояния почвы по ARG")
st.markdown("""
Загрузите Excel-файл с колонками:
`Sample, Mn, Zn, Pb, Cu, Cr, Ni, PAH, ARG, total_contigs`
""")

# --- Настройка порогов через боковую панель ---
st.sidebar.header("Настройка порогов ARG")
threshold_clean = st.sidebar.slider("Порог для чистой почвы", min_value=0.0, max_value=0.05, value=0.0001, step=0.0001)
threshold_moderate = st.sidebar.slider("Порог для умеренно загрязнённой почвы", min_value=0.0, max_value=0.1, value=0.0005, step=0.0001)

uploaded_file = st.file_uploader("Выберите Excel файл", type="xlsx")

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    # --- Чистка данных ---
    numeric_cols = ["Mn","Zn","Pb","Cu","Cr","Ni","PAH","ARG","total_contigs"]
    for col in numeric_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace("\u00A0","",regex=False)
            .str.replace(" ","",regex=False)
            .str.replace(",",".",regex=False)
            .replace("-", "0")
            .astype(float)
        )

    # --- Доля ARG ---
    df["total_contigs"] = df["total_contigs"].replace(0,np.nan)
    df["ARG_fraction"] = df["ARG"]/df["total_contigs"]

    # --- Удаление NaN/Inf ---
    df = df.replace([np.inf, -np.inf], np.nan)
    df_model = df.dropna().copy()

    # --- Логарифмирование ---
    features = ["Mn","Zn","Pb","Cu","Cr","Ni","PAH"]
    for col in features:
        df_model[f"log_{col}"] = np.log(df_model[col]+1)

    X = df_model[[f"log_{c}" for c in features]].values
    y = df_model["ARG_fraction"].values

    # --- Масштабирование ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Модель ---
    model = LinearRegression()
    model.fit(X_scaled, y)

    # --- Прогноз ---
    y_pred = model.predict(X_scaled)
    y_pred = np.clip(y_pred, 0, 1)
    df_model["ARG_fraction_pred"] = y_pred

    # --- Функция описания почвы ---
    def describe_soil(arg_fraction, thresh_clean, thresh_moderate):
        if arg_fraction < thresh_clean:
            return f"Чистая 🌿. Доля ARG: {arg_fraction:.4f}"
        elif arg_fraction < thresh_moderate:
            return f"Умеренно загрязнённая 🟠. Доля ARG: {arg_fraction:.4f}"
        else:
            return f"Грязная 🔴. Доля ARG: {arg_fraction:.4f}"

    df_model["soil_description"] = df_model["ARG_fraction"].apply(
        lambda x: describe_soil(x, threshold_clean, threshold_moderate)
    )

    # --- Таблица ---
    st.subheader("📊 Результаты")
    st.dataframe(df_model[["Sample","ARG_fraction","ARG_fraction_pred","soil_description"]])

    # --- График ---
    st.subheader("📈 График доли ARG")
    fig, ax = plt.subplots(figsize=(12,6))
    bar_width = 0.4
    indices = np.arange(len(df_model))

    ax.bar(indices - bar_width/2, df_model["ARG_fraction"], width=bar_width, color='orange', alpha=0.7, label='Фактическая ARG_fraction')
    ax.bar(indices + bar_width/2, df_model["ARG_fraction_pred"], width=bar_width, color='blue', alpha=0.5, label='Прогнозная ARG_fraction')

    for i, val in enumerate(df_model["ARG_fraction"]):
        if val < threshold_clean:
            ax.text(i-bar_width/2, val+0.002, "Чистая 🌿", ha='center', fontsize=8, color='green')
        elif val < threshold_moderate:
            ax.text(i-bar_width/2, val+0.002, "Умеренно 🟠", ha='center', fontsize=8, color='orange')
        else:
            ax.text(i-bar_width/2, val+0.002, "Грязная 🔴", ha='center', fontsize=8, color='red')

    ax.set_xticks(indices)
    ax.set_xticklabels(df_model["Sample"], rotation=90)
    ax.set_ylabel("Доля ARG")
    ax.set_title("Фактическая и прогнозная доля ARG")
    ax.legend()
    st.pyplot(fig)

    # --- Скачивание ---
    st.subheader("💾 Скачать результаты")
    df_model.to_excel("soil_ARG_results.xlsx", index=False)
    st.markdown("[Скачать таблицу с результатами](soil_ARG_results.xlsx)")
