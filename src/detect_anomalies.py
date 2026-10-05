import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


def load_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df


def filter_state(df, state_name):
    df_state = df[df['state_name'] == state_name].copy()
    df_state = df_state.sort_values('date').reset_index(drop=True)
    df_state = df_state.dropna(subset=['actual', 'normal'])
    return df_state


def detect_anomalies(df_state, contamination=0.03):
    model = IsolationForest(contamination=contamination, random_state=42)
    df_state = df_state.copy()
    df_state['anomaly'] = model.fit_predict(df_state[['actual', 'normal']])
    df_state['anomaly'] = df_state['anomaly'].map({1: 'normal', -1: 'anomaly'})
    return df_state


def plot_anomalies(df_state, state_name, save_path=None):
    plt.figure(figsize=(14, 5))
    plt.plot(df_state['date'], df_state['actual'], label='Actual Rainfall', alpha=0.6)
    plt.plot(df_state['date'], df_state['normal'], label='Normal Rainfall', alpha=0.5, color='orange')

    anomalies = df_state[df_state['anomaly'] == 'anomaly']
    plt.scatter(anomalies['date'], anomalies['actual'], color='red', label='Anomaly', zorder=5, s=15)

    plt.legend()
    plt.title(f'Rainfall Anomaly Detection — {state_name}')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    DATA_PATH = "../data/daily-rainfall-at-state-level.csv"   # update with your actual filename
    STATE = "Andhra Pradesh"
    OUTPUT_PATH = f"../outputs/anomaly_chart_{STATE.lower().replace(' ', '_')}.png"

    df = load_data(DATA_PATH)
    df_state = filter_state(df, STATE)
    df_result = detect_anomalies(df_state)
    plot_anomalies(df_result, STATE, save_path=OUTPUT_PATH)

    print(df_result['anomaly'].value_counts())