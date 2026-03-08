"""
=======================================================
  ANÁLISIS TÉCNICO + PREDICCIONES — NYSE/NASDAQ
=======================================================
Requisitos:
    pip install yfinance pandas matplotlib scikit-learn statsmodels

Uso:
    python analisis_bolsa_prediccion.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
TICKER    = "NVDA"   # Símbolo de la acción (ej. MSFT, TSLA, GOOGL)
PERIODO   = "6mo"    # 1mo, 3mo, 6mo, 1y, 2y, 5y
INTERVALO = "1d"     # 1d, 1wk, 1mo
DIAS_PRED = 30       # Días hábiles a predecir


# ─────────────────────────────────────────────
#  1. DESCARGA DE DATOS
# ─────────────────────────────────────────────
def descargar_datos(ticker, periodo, intervalo):
    print(f"\n📥 Descargando datos de {ticker} ({periodo})...")
    df = yf.download(ticker, period=periodo, interval=intervalo,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No se encontraron datos para '{ticker}'.")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    print(f"✅ {len(df)} registros  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────
#  2. INDICADORES TÉCNICOS
# ─────────────────────────────────────────────
def calcular_indicadores(df):
    df["SMA_20"]   = df["Close"].rolling(20).mean()
    df["SMA_50"]   = df["Close"].rolling(50).mean()
    df["EMA_12"]   = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"]   = df["Close"].ewm(span=26, adjust=False).mean()

    bb_media       = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_upper"] = bb_media + 2 * bb_std
    df["BB_lower"] = bb_media - 2 * bb_std

    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    delta            = df["Close"].diff()
    gain             = delta.clip(lower=0).rolling(14).mean()
    loss             = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]        = 100 - (100 / (1 + gain / loss))
    df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()

    df["Señal"] = "Neutral"
    df.loc[(df["SMA_20"] > df["SMA_50"]) & (df["RSI"] < 70), "Señal"] = "COMPRA"
    df.loc[(df["SMA_20"] < df["SMA_50"]) & (df["RSI"] > 30), "Señal"] = "VENTA"
    return df


# ─────────────────────────────────────────────
#  3. MODELOS DE PREDICCIÓN
# ─────────────────────────────────────────────
def predecir(df, dias_pred):
    precios = df["Close"].values.astype(float)
    n       = len(precios)

    ultima_fecha   = df.index[-1]
    fechas_futuras = pd.bdate_range(
        start=ultima_fecha + pd.Timedelta(days=1), periods=dias_pred
    )

    X_hist   = np.arange(n).reshape(-1, 1)
    X_futuro = np.arange(n, n + dias_pred).reshape(-1, 1)

    # Modelo 1: Regresión polinomial grado 3
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    poly_model.fit(X_hist, precios)
    pred_poly  = poly_model.predict(X_futuro)

    # Modelo 2: Holt-Winters
    try:
        hw_fit  = ExponentialSmoothing(
            precios, trend="add", seasonal=None,
            initialization_method="estimated"
        ).fit(optimized=True, remove_bias=True)
        pred_hw = hw_fit.forecast(dias_pred)
    except Exception:
        pred_hw = np.full(dias_pred, precios[-1])

    # Modelo 3: EMA extrapolada
    tendencia_diaria = (precios[-1] - precios[-min(10, n)]) / min(10, n)
    pred_ema = np.array([precios[-1] + tendencia_diaria * (i + 1)
                         for i in range(dias_pred)])

    # Ensemble ponderado
    pred_ensemble = 0.20 * pred_poly + 0.55 * pred_hw + 0.25 * pred_ema

    # Bandas de confianza
    vol_diaria = np.std(np.diff(precios) / precios[:-1])
    horizonte  = np.arange(1, dias_pred + 1)
    sigma      = vol_diaria * np.sqrt(horizonte) * precios[-1]

    return {
        "fechas":      fechas_futuras,
        "poly":        pred_poly,
        "hw":          pred_hw,
        "ema":         pred_ema,
        "ensemble":    pred_ensemble,
        "banda_1_sup": pred_ensemble + sigma,
        "banda_1_inf": pred_ensemble - sigma,
        "banda_2_sup": pred_ensemble + 2 * sigma,
        "banda_2_inf": pred_ensemble - 2 * sigma,
        "vol_diaria":  vol_diaria,
    }


# ─────────────────────────────────────────────
#  4. RESUMEN EN CONSOLA
# ─────────────────────────────────────────────
def imprimir_resumen(df, ticker, pred):
    ult  = df.iloc[-1]
    prev = df.iloc[-2]
    cambio   = float(ult["Close"]) - float(prev["Close"])
    cambio_p = (cambio / float(prev["Close"])) * 100
    p_final  = pred["ensemble"][-1]
    retorno  = ((p_final - float(ult["Close"])) / float(ult["Close"])) * 100

    print(f"""
╔══════════════════════════════════════════╗
  {'📈' if cambio >= 0 else '📉'}  ANÁLISIS TÉCNICO: {ticker}
╠══════════════════════════════════════════╣
  Precio actual  : ${float(ult['Close']):.2f}
  Cambio diario  : {'▲' if cambio >= 0 else '▼'} ${abs(cambio):.2f}  ({cambio_p:+.2f}%)
  RSI (14)       : {float(ult['RSI']):.1f}
  Señal actual   : {ult['Señal']}
──────────────────────────────────────────
  🔮 PREDICCIÓN ({DIAS_PRED} días hábiles)
  Precio estimado: ${p_final:.2f}
  Retorno est.   : {'🟢' if retorno > 0 else '🔴'} {retorno:+.2f}%
  Volatilidad    : {pred['vol_diaria']*100:.2f}% diaria
  Banda 68%      : ${pred['banda_1_inf'][-1]:.2f} – ${pred['banda_1_sup'][-1]:.2f}
  Banda 95%      : ${pred['banda_2_inf'][-1]:.2f} – ${pred['banda_2_sup'][-1]:.2f}
╚══════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────
#  5. GRÁFICA UNIFICADA
# ─────────────────────────────────────────────
def graficar(df, pred, ticker, dias_pred):
    CF = "#0d1117"
    CT = "#c9d1d9"
    CG = "#21262d"
    CS = "#30363d"
    CH = "#f0883e"

    precio_actual = float(df["Close"].iloc[-1])
    fechas_f      = pred["fechas"]
    p_final       = pred["ensemble"][-1]
    retorno       = ((p_final - precio_actual) / precio_actual) * 100
    color_ret     = "#3fb950" if retorno >= 0 else "#f85149"

    fig = plt.figure(figsize=(18, 14), facecolor=CF)
    fig.suptitle(
        f"Análisis Técnico + Predicción — {ticker}  "
        f"|  {df.index[-1].strftime('%d %b %Y')}  →  {fechas_f[-1].strftime('%d %b %Y')}",
        color="white", fontsize=14, fontweight="bold", y=0.99
    )

    gs = GridSpec(
        4, 1, figure=fig,
        hspace=0.9, height_ratios=[3, 1, 1, 1],
        top=0.95, bottom=0.06, left=0.07, right=0.97
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    paneles   = [ax1, ax2, ax3, ax4]
    etiquetas = ["Precio, Bollinger & Predicción", "Volumen", "MACD", "RSI (14)"]

    for ax, etiq in zip(paneles, etiquetas):
        ax.set_facecolor(CF)
        ax.tick_params(colors=CT, labelsize=8)
        ax.yaxis.label.set_color(CT)
        for sp in ax.spines.values():
            sp.set_edgecolor(CS)
        ax.grid(which="major", color="#2d333b", linewidth=0.8)
        ax.grid(which="minor", color=CG, linewidth=0.4, linestyle=":")
        ax.minorticks_on()
        ax.text(0.005, 0.97, etiq, transform=ax.transAxes,
                color="#8b949e", fontsize=7.5, va="top", ha="left")

    for ax in [ax2, ax3, ax4]:
        ax.spines["top"].set_edgecolor("#58a6ff")
        ax.spines["top"].set_linewidth(0.8)

    # Sombreado zona predicción en todos los paneles
    for ax in paneles:
        ax.axvspan(df.index[-1], fechas_f[-1], alpha=0.045, color="#bc8cff", zorder=0)
        ax.axvline(df.index[-1], color=CH, lw=1.1, ls="--", alpha=0.7, zorder=3)

    # ════ PANEL 1: PRECIO + BOLLINGER + PREDICCIÓN ════

    # Histórico
    ax1.plot(df.index, df["Close"],   color="#58a6ff", lw=1.8, label="Precio",  zorder=4)
    ax1.plot(df.index, df["SMA_20"],  color="#f0883e", lw=1.0, label="SMA 20",  ls="--")
    ax1.plot(df.index, df["SMA_50"],  color="#bc8cff", lw=1.0, label="SMA 50",  ls="--")
    ax1.fill_between(df.index, df["BB_upper"], df["BB_lower"],
                     alpha=0.08, color="#58a6ff", label="Bollinger")
    ax1.plot(df.index, df["BB_upper"], color="#58a6ff", lw=0.5, alpha=0.4)
    ax1.plot(df.index, df["BB_lower"], color="#58a6ff", lw=0.5, alpha=0.4)

    # Señales
    compras = df[df["Señal"] == "COMPRA"]
    ventas  = df[df["Señal"] == "VENTA"]
    ax1.scatter(compras.index, compras["Close"],
                marker="^", color="#3fb950", s=55, zorder=5, label="Compra")
    ax1.scatter(ventas.index,  ventas["Close"],
                marker="v", color="#f85149", s=55, zorder=5, label="Venta")

    # Bandas de confianza (predicción)
    ax1.fill_between(fechas_f, pred["banda_2_inf"], pred["banda_2_sup"],
                     alpha=0.10, color="#bc8cff", label="Banda 95%")
    ax1.fill_between(fechas_f, pred["banda_1_inf"], pred["banda_1_sup"],
                     alpha=0.22, color="#bc8cff", label="Banda 68%")

    # Modelos individuales (punteados sutiles)
    ax1.plot(fechas_f, pred["hw"],
             color="#f0883e", lw=1.0, ls=":", alpha=0.75, label="Holt-Winters")
    ax1.plot(fechas_f, pred["poly"],
             color="#3fb950", lw=1.0, ls=":", alpha=0.75, label="Reg. Polinomial")
    ax1.plot(fechas_f, pred["ema"],
             color="#ffa657", lw=1.0, ls=":", alpha=0.75, label="EMA Extrap.")

    # Ensemble (línea principal — conectada al último precio histórico)
    fechas_cx   = pd.DatetimeIndex([df.index[-1]]).append(fechas_f)
    valores_cx  = np.concatenate([[precio_actual], pred["ensemble"]])
    ax1.plot(fechas_cx, valores_cx,
             color="#ffffff", lw=2.2, label="Ensemble", zorder=6)

    ax1.scatter([df.index[-1]], [precio_actual], color=CH, s=70, zorder=7)

    # Anotación precio final predicho
    ax1.annotate(
        f"  ${p_final:.2f}  ({retorno:+.1f}%)",
        xy=(fechas_f[-1], p_final),
        color=color_ret, fontsize=8.5, fontweight="bold", va="center", zorder=8
    )
    ax1.annotate(f" ${pred['banda_2_sup'][-1]:.0f}",
                 xy=(fechas_f[-1], pred["banda_2_sup"][-1]),
                 color="#bc8cff", fontsize=7, va="bottom", alpha=0.85)
    ax1.annotate(f" ${pred['banda_2_inf'][-1]:.0f}",
                 xy=(fechas_f[-1], pred["banda_2_inf"][-1]),
                 color="#bc8cff", fontsize=7, va="top", alpha=0.85)

    # Anotación HOY
    ax1.annotate(
        f"HOY  {df.index[-1].strftime('%d %b')}",
        xy=(df.index[-1], precio_actual),
        xytext=(8, 14), textcoords="offset points",
        color=CH, fontsize=7.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=CH, lw=0.8)
    )

    # Caja de métricas
    metrics = (
        f"Volatilidad:  {pred['vol_diaria']*100:.2f}%/día\n"
        f"Precio hoy:   ${precio_actual:.2f}\n"
        f"Pred {dias_pred}d:    ${p_final:.2f}  ({retorno:+.1f}%)\n"
        f"Banda 68%: ${pred['banda_1_inf'][-1]:.0f} – ${pred['banda_1_sup'][-1]:.0f}\n"
        f"Banda 95%: ${pred['banda_2_inf'][-1]:.0f} – ${pred['banda_2_sup'][-1]:.0f}"
    )
    ax1.text(0.998, 0.97, metrics,
             transform=ax1.transAxes, color=CT, fontsize=7.5,
             va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="#161b22",
                       edgecolor="#30363d", alpha=0.92))

    ax1.set_ylabel("Precio (USD)", color=CT)
    ax1.legend(facecolor="#161b22", labelcolor=CT, fontsize=7.5,
               loc="upper left", framealpha=0.9, ncol=2)

    # ════ PANEL 2: VOLUMEN ════
    colores_vol = ["#3fb950" if df["Close"].iloc[i] >= df["Close"].iloc[i - 1]
                   else "#f85149" for i in range(len(df))]
    ax2.bar(df.index, df["Volume"], color=colores_vol, alpha=0.7, width=0.8)
    ax2.plot(df.index, df["Vol_SMA_20"], color=CH, lw=1, label="Vol SMA 20")
    ax2.set_ylabel("Volumen", color=CT)

    # ════ PANEL 3: MACD ════
    colores_hist = ["#3fb950" if v >= 0 else "#f85149" for v in df["MACD_hist"]]
    ax3.bar(df.index, df["MACD_hist"], color=colores_hist, alpha=0.6, width=0.8)
    ax3.plot(df.index, df["MACD"],        color="#58a6ff", lw=1, label="MACD")
    ax3.plot(df.index, df["MACD_signal"], color=CH,        lw=1, label="Señal")
    ax3.axhline(0, color=CS, lw=0.8)
    ax3.set_ylabel("MACD", color=CT)
    ax3.legend(facecolor="#161b22", labelcolor=CT, fontsize=7, loc="upper left")

    # ════ PANEL 4: RSI ════
    ax4.plot(df.index, df["RSI"], color="#bc8cff", lw=1.2)
    ax4.axhline(70, color="#f85149", lw=0.8, ls="--", alpha=0.7)
    ax4.axhline(30, color="#3fb950", lw=0.8, ls="--", alpha=0.7)
    ax4.fill_between(df.index, df["RSI"], 70,
                     where=(df["RSI"] >= 70), alpha=0.2, color="#f85149")
    ax4.fill_between(df.index, df["RSI"], 30,
                     where=(df["RSI"] <= 30), alpha=0.2, color="#3fb950")
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("RSI", color=CT)

    # ── Eje X compartido ──────────────────────
    for ax in paneles:
        ax.set_xlim(df.index[0], fechas_f[-1])
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), color=CT, fontsize=7.5)

    for ax in [ax1, ax2, ax3]:
        plt.setp(ax.xaxis.get_majorticklabels(), alpha=0.4, fontsize=6.5)

    archivo = f"{ticker}_analisis_completo.png"
    plt.savefig(archivo, dpi=150, bbox_inches="tight", facecolor=CF)
    print(f"📊 Gráfica guardada: {archivo}")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df   = descargar_datos(TICKER, PERIODO, INTERVALO)
    df   = calcular_indicadores(df)
    pred = predecir(df, DIAS_PRED)
    imprimir_resumen(df, TICKER, pred)
    graficar(df, pred, TICKER, DIAS_PRED)