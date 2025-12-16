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
`Sample, Mn, Zn, Pb, Cu, Cr, Ni, PAH`

Модель использует референсный файл `ref_tab.xlsx` для обучения и предсказывает долю ARG для новых образцов.
""")

# --- Настройка порогов через боковую панель ---
st.sidebar.header("Настройка порогов ARG")
threshold_clean = st.sidebar.slider("Порог для чистой почвы", min_value=0.0000, max_value=0.0050, value=0.0001, step=0.0001, format="%.4f")
threshold_moderate = st.sidebar.slider("Порог для умеренно загрязнённой почвы", min_value=0.0000, max_value=0.0050, value=0.0005, step=0.0001, format="%.4f")

uploaded_file = st.file_uploader("Выберите Excel файл с новыми образцами", type="xlsx")

# --- Подгружаем референс локально из того же репозитория ---
ref_file_path = "ref_tab.xlsx"
df_ref = pd.read_excel(ref_file_path)

# --- Чистка данных референса ---
numeric_cols_ref = ["Mn","Zn","Pb","Cu","Cr","Ni","PAH","ARG","total_contigs"]
for col in numeric_cols_ref:
    df_ref[col] = (
        df_ref[col].astype(str)
        .str.replace("\u00A0","",regex=False)
        .str.replace(" ","",regex=False)
        .str.replace(",",".",regex=False)
        .replace("-", "0")
        .astype(float)
    )

# --- Доля ARG ---
df_ref["ARG_fraction"] = df_ref["ARG"] / df_ref["total_contigs"]

# --- Логарифмирование ---
features = ["Mn","Zn","Pb","Cu","Cr","Ni","PAH"]
for col in features:
    df_ref[f"log_{col}"] = np.log(df_ref[col]+1)

# --- Подготовка данных для модели ---
X_ref = df_ref[[f"log_{c}" for c in features]].values
y_ref = df_ref["ARG_fraction"].values

# --- Масштабирование ---
scaler = StandardScaler()
X_ref_scaled = scaler.fit_transform(X_ref)

# --- Модель ---
model = LinearRegression()
model.fit(X_ref_scaled, y_ref)

if uploaded_file:
    df_new = pd.read_excel(uploaded_file)

    # --- Чистка новых образцов ---
    numeric_cols_new = ["Mn","Zn","Pb","Cu","Cr","Ni","PAH"]
    for col in numeric_cols_new:
        df_new[col] = (
            df_new[col].astype(str)
            .str.replace("\u00A0","",regex=False)
            .str.replace(" ","",regex=False)
            .str.replace(",",".",regex=False)
            .replace("-", "0")
            .astype(float)
        )

    # --- Логарифмирование новых данных ---
    for col in features:
        df_new[f"log_{col}"] = np.log(df_new[col]+1)

    # --- Масштабирование новых данных ---
    X_new_scaled = scaler.transform(df_new[[f"log_{c}" for c in features]].values)

    # --- Прогноз ---
    y_pred_new = model.predict(X_new_scaled)
    y_pred_new = np.clip(y_pred_new, 0, 1)
    df_new["ARG_fraction_pred"] = y_pred_new

    # --- Функция описания почвы ---
    def describe_soil(arg_fraction, thresh_clean, thresh_moderate):
        if arg_fraction < thresh_clean:
            return f"Чистая 🌿. Доля ARG: {arg_fraction:.4f}"
        elif arg_fraction < thresh_moderate:
            return f"Умеренно загрязнённая 🟠. Доля ARG: {arg_fraction:.4f}"
        else:
            return f"Грязная 🔴. Доля ARG: {arg_fraction:.4f}"

    df_new["soil_description"] = df_new["ARG_fraction_pred"].apply(
        lambda x: describe_soil(x, threshold_clean, threshold_moderate)
    )

    # --- Таблица ---
    st.subheader("📊 Прогноз для новых образцов")
    st.dataframe(df_new[["Sample","ARG_fraction_pred","soil_description"]])

    # --- График ---
    st.subheader("📈 Прогноз доли ARG")
    fig, ax = plt.subplots(figsize=(14,6), dpi=150)
    x = np.arange(len(df_new))
    ax.bar(x, df_new["ARG_fraction_pred"], color="tab:blue", alpha=0.7, label="Прогноз модели")
    
    ax.axhline(threshold_clean, color="green", linestyle="--", linewidth=1.5, label=f"Порог чистой почвы ({threshold_clean:.4f})")
    ax.axhline(threshold_moderate, color="red", linestyle="--", linewidth=1.5, label=f"Порог загрязнения ({threshold_moderate:.4f})")

    for i, val in enumerate(df_new["ARG_fraction_pred"]):
        if val < threshold_clean:
            label = "Чистая"
            color = "green"
        elif val < threshold_moderate:
            label = "Умеренно загрязнённая"
            color = "orange"
        else:
            label = "Загрязнённая"
            color = "red"
        ax.annotate(label, xy=(i,val), xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontsize=7, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(df_new["Sample"], rotation=90, fontsize=8)
    ax.set_ylabel("Доля ARG (предсказание)", fontsize=10)
    ax.set_xlabel("Образцы почвы", fontsize=10)
    ax.set_title("Прогноз антибиотикорезистентных генов в новых образцах", fontsize=12)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Скачивание ---
    st.subheader("💾 Скачать результаты")
    df_new.to_excel("soil_ARG_predictions.xlsx", index=False)
    st.markdown("[Скачать таблицу с результатами](soil_ARG_predictions.xlsx)")
