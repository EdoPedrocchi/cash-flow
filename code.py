import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# Titolo dell'app
st.title("Previsione dei Flussi di Cassa per Startup")

# Upload file CSV
uploaded_file = st.file_uploader("Carica il file CSV con i dati finanziari", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["data"])  # Assumi che ci sia una colonna 'data'
    st.write("### Dati caricati:")
    st.dataframe(df.tail())

    # Visualizzazione iniziale
    st.write("### Andamento Storico del Flusso di Cassa")
    plt.figure(figsize=(10, 5))
    plt.plot(df["data"], df["flusso_cassa"], marker='o', linestyle='-')
    plt.xlabel("Data")
    plt.ylabel("Flusso di Cassa")
    plt.grid()
    st.pyplot(plt)

    # Preparazione dati per il modello
    df["data_ordinal"] = df["data"].map(datetime.toordinal)
    X = df[["data_ordinal"]]
    y = df["flusso_cassa"]
    
    # Split training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Addestramento modello
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Previsione per i prossimi 3 mesi
    future_dates = [datetime.today() + timedelta(days=i*30) for i in range(1, 4)]
    future_ordinal = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_preds = model.predict(future_ordinal)
    
    # Visualizzazione previsione
    st.write("### Previsione dei Prossimi 3 Mesi")
    future_df = pd.DataFrame({"Data": future_dates, "Flusso Cassa Previsto": future_preds})
    st.dataframe(future_df)
    
    # Grafico previsioni
    plt.figure(figsize=(10, 5))
    plt.plot(df["data"], df["flusso_cassa"], marker='o', linestyle='-', label="Storico")
    plt.plot(future_dates, future_preds, marker='s', linestyle='--', color='red', label="Previsione")
    plt.xlabel("Data")
    plt.ylabel("Flusso di Cassa")
    plt.legend()
    plt.grid()
    st.pyplot(plt)
else:
    st.write("Carica un dataset per iniziare.")
