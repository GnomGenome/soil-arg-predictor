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
threshold_clean = st.sidebar.slider("Порог для чистой почвы", min_value=0.0000, max_value=0.0050, value=0.0001, step=0.0001, format="%.4f")
threshold_moderate = st.sidebar.slider("Порог для умеренно загрязнённой почвы", min_value=0.0000, max_value=0.0050, value=0.0005, step=0.0001, format="%.4f")

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
    st.subheader("📈 Доля антибиотикорезистентных генов (ARG)")

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    
    x = np.arange(len(df_model))
    
    # --- Столбцы: фактическая доля ARG ---
    bars = ax.bar(
        x,
        df_model["ARG_fraction"],
        color="tab:orange",
        alpha=0.7,
        label="Фактическая доля ARG"
    )
    
    # --- Точки: прогноз модели ---
    ax.scatter(
        x,
        df_model["ARG_fraction_pred"],
        color="tab:blue",
        s=40,
        zorder=3,
        label="Прогноз модели"
    )
    
    # --- Пороговые линии ---
    ax.axhline(
        threshold_clean,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label=f"Порог чистой почвы ({threshold_clean:.4f})"
    )
    
    ax.axhline(
        threshold_moderate,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Порог загрязнения ({threshold_moderate:.4f})"
    )
    
    # --- Подписи классов над столбцами ---
    for i, val in enumerate(df_model["ARG_fraction"]):
        if val < threshold_clean:
            label = "Чистая"
            color = "green"
        elif val < threshold_moderate:
            label = "Умеренно загрязнённая"
            color = "orange"
        else:
            label = "Загрязнённая"
            color = "red"
    
        ax.annotate(
            label,
            xy=(i, val),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color
        )
    
    # --- Оформление осей ---
    ax.set_xticks(x)
    ax.set_xticklabels(df_model["Sample"], rotation=90, fontsize=8)
    ax.set_ylabel("Доля ARG во всём метагеноме", fontsize=10)
    ax.set_xlabel("Образцы почвы", fontsize=10)
    
    ax.set_title(
        "Антибиотикорезистентные гены в почвах\n"
        "Фактические значения, прогноз модели и пороговые уровни",
        fontsize=12
    )
    
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    
    plt.tight_layout()
    st.pyplot(fig)

    # --- Скачивание ---
    st.subheader("💾 Скачать результаты")
    df_model.to_excel("soil_ARG_results.xlsx", index=False)
    st.markdown("[Скачать таблицу с результатами](soil_ARG_results.xlsx)")
