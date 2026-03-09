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
import matplotlib.patches as mpatches
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
TICKER    = "NVDA"     # Símbolo de la acción (ej. MSFT, TSLA, GOOGL)
PERIODO   = "6mo"      # 1mo, 3mo, 6mo, 1y, 2y, 5y
INTERVALO = "1d"       # 1d, 1wk, 1mo
DIAS_PRED = 360         # Días a predecir en el futuro


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

    delta         = df["Close"].diff()
    gain          = delta.clip(lower=0).rolling(14).mean()
    loss          = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]     = 100 - (100 / (1 + gain / loss))
    df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()

    df["Señal"] = "Neutral"
    df.loc[(df["SMA_20"] > df["SMA_50"]) & (df["RSI"] < 70), "Señal"] = "COMPRA"
    df.loc[(df["SMA_20"] < df["SMA_50"]) & (df["RSI"] > 30), "Señal"] = "VENTA"
    return df


# ─────────────────────────────────────────────
#  3. MODELOS DE PREDICCIÓN
# ─────────────────────────────────────────────
def predecir(df, dias_pred):
    """
    Genera predicciones con 3 modelos y devuelve fechas futuras
    más los precios estimados con bandas de confianza.
    """
    precios = df["Close"].values.astype(float)
    n       = len(precios)

    # Fechas futuras (solo días hábiles)
    ultima_fecha  = df.index[-1]
    fechas_futuras = pd.bdate_range(start=ultima_fecha + pd.Timedelta(days=1),
                                    periods=dias_pred)

    X_hist  = np.arange(n).reshape(-1, 1)
    X_futuro = np.arange(n, n + dias_pred).reshape(-1, 1)

    # ── Modelo 1: Regresión polinomial (grado 3) ──
    poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    poly_model.fit(X_hist, precios)
    pred_poly  = poly_model.predict(X_futuro)

    # ── Modelo 2: Suavizado exponencial triple (Holt-Winters) ──
    try:
        hw_model  = ExponentialSmoothing(precios, trend="add", seasonal=None,
                                          initialization_method="estimated")
        hw_fit    = hw_model.fit(optimized=True, remove_bias=True)
        pred_hw   = hw_fit.forecast(dias_pred)
        resid_std = np.std(hw_fit.resid)
    except Exception:
        pred_hw   = np.full(dias_pred, precios[-1])
        resid_std = np.std(precios[-20:]) * 0.5

    # ── Modelo 3: Media móvil exponencial extrapolada ──
    alpha     = 2 / (dias_pred + 1)
    ema_val   = precios[-1]
    tendencia = precios[-1] - precios[-min(10, n)]
    tendencia_diaria = tendencia / min(10, n)
    pred_ema  = np.array([precios[-1] + tendencia_diaria * (i + 1)
                          for i in range(dias_pred)])

    # ── Ensemble: promedio ponderado ──────────
    #   Holt-Winters suele ser el más preciso para series financieras
    pred_ensemble = 0.20 * pred_poly + 0.55 * pred_hw + 0.25 * pred_ema

    # ── Bandas de confianza (±1σ y ±2σ basadas en volatilidad histórica) ──
    volatilidad_diaria = np.std(np.diff(precios) / precios[:-1])
    horizonte          = np.arange(1, dias_pred + 1)
    sigma_acumulada    = volatilidad_diaria * np.sqrt(horizonte) * precios[-1]

    banda_1_sup = pred_ensemble + sigma_acumulada
    banda_1_inf = pred_ensemble - sigma_acumulada
    banda_2_sup = pred_ensemble + 2 * sigma_acumulada
    banda_2_inf = pred_ensemble - 2 * sigma_acumulada

    return {
        "fechas":      fechas_futuras,
        "poly":        pred_poly,
        "hw":          pred_hw,
        "ema":         pred_ema,
        "ensemble":    pred_ensemble,
        "banda_1_sup": banda_1_sup,
        "banda_1_inf": banda_1_inf,
        "banda_2_sup": banda_2_sup,
        "banda_2_inf": banda_2_inf,
        "vol_diaria":  volatilidad_diaria,
    }


# ─────────────────────────────────────────────
#  4. RESUMEN EN CONSOLA
# ─────────────────────────────────────────────
def imprimir_resumen(df, ticker, pred):
    ult   = df.iloc[-1]
    prev  = df.iloc[-2]
    cambio   = float(ult["Close"]) - float(prev["Close"])
    cambio_p = (cambio / float(prev["Close"])) * 100
    arrow    = "▲" if cambio >= 0 else "▼"
    emoji    = "📈" if cambio >= 0 else "📉"

    precio_pred_final = pred["ensemble"][-1]
    retorno_est = ((precio_pred_final - float(ult["Close"])) / float(ult["Close"])) * 100
    tend_emoji  = "🟢" if retorno_est > 0 else "🔴"

    print(f"""
╔══════════════════════════════════════════╗
  {emoji}  ANÁLISIS TÉCNICO: {ticker}
╠══════════════════════════════════════════╣
  Precio actual  : ${float(ult['Close']):.2f}
  Cambio diario  : {arrow} ${abs(cambio):.2f}  ({cambio_p:+.2f}%)
  RSI (14)       : {float(ult['RSI']):.1f}
  Señal actual   : {ult['Señal']}
──────────────────────────────────────────
  🔮 PREDICCIÓN ({DIAS_PRED} días hábiles)
  Precio estimado: ${precio_pred_final:.2f}
  Retorno est.   : {tend_emoji} {retorno_est:+.2f}%
  Volatilidad    : {pred['vol_diaria']*100:.2f}% diaria
  Banda 68%      : ${pred['banda_1_inf'][-1]:.2f} – ${pred['banda_1_sup'][-1]:.2f}
  Banda 95%      : ${pred['banda_2_inf'][-1]:.2f} – ${pred['banda_2_sup'][-1]:.2f}
╚══════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────
#  5. GRÁFICA DE ANÁLISIS TÉCNICO (original)
# ─────────────────────────────────────────────
def graficar_tecnico(df, ticker):
    CF = "#0d1117"; CT = "#c9d1d9"; CG = "#21262d"
    CS = "#30363d"; CH = "#f0883e"

    fig = plt.figure(figsize=(16, 13), facecolor=CF)
    fig.suptitle(f"Análisis Técnico — {ticker}  |  {df.index[-1].strftime('%d %b %Y')}",
                 color="white", fontsize=15, fontweight="bold", y=0.99)

    gs = GridSpec(4, 1, figure=fig, hspace=1, height_ratios=[3, 1, 1, 1],
                  top=0.95, bottom=0.07, left=0.07, right=0.97)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    paneles = [ax1, ax2, ax3, ax4]
    etiquetas = ["Precio & Bollinger", "Volumen", "MACD", "RSI (14)"]

    for ax, etiq in zip(paneles, etiquetas):
        ax.set_facecolor(CF)
        ax.tick_params(colors=CT, labelsize=8)
        ax.yaxis.label.set_color(CT)
        for sp in ax.spines.values(): sp.set_edgecolor(CS)
        ax.grid(which="major", color="#2d333b", linewidth=0.8)
        ax.grid(which="minor", color=CG, linewidth=0.4, linestyle=":")
        ax.minorticks_on()
        ax.text(0.005, 0.97, etiq, transform=ax.transAxes,
                color="#8b949e", fontsize=7.5, va="top", ha="left")

    hoy = df.index[-1]
    for ax in paneles:
        ax.axvline(hoy, color=CH, linewidth=1.0, linestyle="--", alpha=0.6, zorder=3)
    for ax in [ax2, ax3, ax4]:
        ax.spines["top"].set_edgecolor("#58a6ff"); ax.spines["top"].set_linewidth(0.8)

    ax1.plot(df.index, df["Close"],   color="#58a6ff", lw=1.5, label="Precio")
    ax1.plot(df.index, df["SMA_20"],  color="#f0883e", lw=1,   label="SMA 20", ls="--")
    ax1.plot(df.index, df["SMA_50"],  color="#bc8cff", lw=1,   label="SMA 50", ls="--")
    ax1.fill_between(df.index, df["BB_upper"], df["BB_lower"], alpha=0.1, color="#58a6ff", label="Bollinger")
    ax1.plot(df.index, df["BB_upper"], color="#58a6ff", lw=0.5, alpha=0.5)
    ax1.plot(df.index, df["BB_lower"], color="#58a6ff", lw=0.5, alpha=0.5)
    compras = df[df["Señal"] == "COMPRA"]
    ventas  = df[df["Señal"] == "VENTA"]
    ax1.scatter(compras.index, compras["Close"], marker="^", color="#3fb950", s=60, zorder=5, label="Compra")
    ax1.scatter(ventas.index,  ventas["Close"],  marker="v", color="#f85149", s=60, zorder=5, label="Venta")
    ax1.set_ylabel("Precio (USD)", color=CT)
    ax1.legend(facecolor="#161b22", labelcolor=CT, fontsize=8, loc="upper left")

    colores_vol = ["#3fb950" if df["Close"].iloc[i] >= df["Close"].iloc[i-1]
                   else "#f85149" for i in range(len(df))]
    ax2.bar(df.index, df["Volume"], color=colores_vol, alpha=0.7, width=0.8)
    ax2.plot(df.index, df["Vol_SMA_20"], color="#f0883e", lw=1)
    ax2.set_ylabel("Volumen", color=CT)

    colores_hist = ["#3fb950" if v >= 0 else "#f85149" for v in df["MACD_hist"]]
    ax3.bar(df.index, df["MACD_hist"], color=colores_hist, alpha=0.6, width=0.8)
    ax3.plot(df.index, df["MACD"],        color="#58a6ff", lw=1, label="MACD")
    ax3.plot(df.index, df["MACD_signal"], color="#f0883e", lw=1, label="Señal")
    ax3.axhline(0, color="#30363d", lw=0.8)
    ax3.set_ylabel("MACD", color=CT)
    ax3.legend(facecolor="#161b22", labelcolor=CT, fontsize=7, loc="upper left")

    ax4.plot(df.index, df["RSI"], color="#bc8cff", lw=1.2)
    ax4.axhline(70, color="#f85149", lw=0.8, ls="--", alpha=0.7)
    ax4.axhline(30, color="#3fb950", lw=0.8, ls="--", alpha=0.7)
    ax4.fill_between(df.index, df["RSI"], 70, where=(df["RSI"] >= 70), alpha=0.2, color="#f85149")
    ax4.fill_between(df.index, df["RSI"], 30, where=(df["RSI"] <= 30), alpha=0.2, color="#3fb950")
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("RSI", color=CT)

    for ax in paneles:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), color=CT, fontsize=7.5)

    ax1.annotate(f"HOY\n{hoy.strftime('%d %b')}",
                 xy=(hoy, df["Close"].iloc[-1]),
                 xytext=(10, 10), textcoords="offset points",
                 color=CH, fontsize=7.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=CH, lw=0.8))

    archivo = f"{ticker}_analisis_tecnico.png"
    plt.savefig(archivo, dpi=150, bbox_inches="tight", facecolor=CF)
    print(f"📊 Gráfica técnica guardada: {archivo}")
    plt.show()


# ─────────────────────────────────────────────
#  6. GRÁFICA DE PREDICCIONES (nueva imagen)
# ─────────────────────────────────────────────
def graficar_predicciones(df, pred, ticker, dias_pred):
    CF = "#0d1117"; CT = "#c9d1d9"; CS = "#30363d"

    # Tomamos los últimos 60 días históricos como contexto
    contexto = min(60, len(df))
    df_ctx   = df.iloc[-contexto:]

    fig = plt.figure(figsize=(16, 10), facecolor=CF)
    fig.suptitle(
        f"🔮 Predicción de Precio — {ticker}  |  Horizonte: {dias_pred} días hábiles",
        color="white", fontsize=15, fontweight="bold", y=0.98
    )

    gs = GridSpec(2, 1, figure=fig, hspace=0.55,
                  height_ratios=[3, 1], top=0.93, bottom=0.08,
                  left=0.07, right=0.97)

    ax_main = fig.add_subplot(gs[0])   # Precio + predicciones
    ax_ret  = fig.add_subplot(gs[1])   # Retorno esperado acumulado

    for ax in [ax_main, ax_ret]:
        ax.set_facecolor(CF)
        ax.tick_params(colors=CT, labelsize=8)
        ax.yaxis.label.set_color(CT)
        for sp in ax.spines.values(): sp.set_edgecolor(CS)
        ax.grid(which="major", color="#2d333b", linewidth=0.8)
        ax.grid(which="minor", color="#1c2128", linewidth=0.4, linestyle=":")
        ax.minorticks_on()

    fechas_f = pred["fechas"]
    precio_actual = float(df["Close"].iloc[-1])

    # ── Precio histórico (contexto) ──────────
    ax_main.plot(df_ctx.index, df_ctx["Close"],
                 color="#58a6ff", lw=2, label="Histórico", zorder=4)

    # ── Bandas de confianza ──────────────────
    ax_main.fill_between(fechas_f,
                         pred["banda_2_inf"], pred["banda_2_sup"],
                         alpha=0.12, color="#bc8cff",
                         label="Banda 95% (±2σ)")
    ax_main.fill_between(fechas_f,
                         pred["banda_1_inf"], pred["banda_1_sup"],
                         alpha=0.25, color="#bc8cff",
                         label="Banda 68% (±1σ)")

    # ── Modelos individuales ─────────────────
    ax_main.plot(fechas_f, pred["hw"],
                 color="#f0883e", lw=1.2, ls="--", alpha=0.8, label="Holt-Winters")
    ax_main.plot(fechas_f, pred["poly"],
                 color="#3fb950", lw=1.2, ls="--", alpha=0.8, label="Regresión Polinomial")
    ax_main.plot(fechas_f, pred["ema"],
                 color="#ffa657", lw=1.2, ls="--", alpha=0.8, label="EMA Extrapolada")

    # ── Ensemble (predicción principal) ──────
    ax_main.plot(fechas_f, pred["ensemble"],
                 color="#ffffff", lw=2.5, label="Ensemble (predicción)", zorder=5)

    # ── Punto de inicio de predicción ────────
    ax_main.axvline(df.index[-1], color="#f0883e", lw=1.2, ls="--", alpha=0.7)
    ax_main.scatter([df.index[-1]], [precio_actual],
                    color="#f0883e", s=80, zorder=6, label="Hoy")

    # ── Anotaciones precio final predicho ────
    p_final = pred["ensemble"][-1]
    retorno = ((p_final - precio_actual) / precio_actual) * 100
    color_ret = "#3fb950" if retorno >= 0 else "#f85149"
    ax_main.annotate(
        f"  ${p_final:.2f}\n  {retorno:+.1f}%",
        xy=(fechas_f[-1], p_final),
        color=color_ret, fontsize=9, fontweight="bold",
        va="center"
    )

    # ── Anotaciones bandas extremas ──────────
    ax_main.annotate(f"${pred['banda_2_sup'][-1]:.2f}",
                     xy=(fechas_f[-1], pred["banda_2_sup"][-1]),
                     color="#bc8cff", fontsize=7.5, va="bottom", alpha=0.8)
    ax_main.annotate(f"${pred['banda_2_inf'][-1]:.2f}",
                     xy=(fechas_f[-1], pred["banda_2_inf"][-1]),
                     color="#bc8cff", fontsize=7.5, va="top", alpha=0.8)

    ax_main.set_ylabel("Precio (USD)", color=CT)
    ax_main.legend(facecolor="#161b22", labelcolor=CT, fontsize=8,
                   loc="upper left", framealpha=0.9)

    # ── Etiqueta de separación ────────────────
    ymin, ymax = ax_main.get_ylim()
    ax_main.text(df.index[-1], ymax * 0.97, "  ◄ HIST  PRED ►",
                 color="#f0883e", fontsize=7, va="top", ha="center")

    # ── Panel inferior: Retorno acumulado esperado ──
    retornos_acum = ((pred["ensemble"] - precio_actual) / precio_actual) * 100
    colores_ret   = ["#3fb950" if r >= 0 else "#f85149" for r in retornos_acum]
    ax_ret.bar(fechas_f, retornos_acum, color=colores_ret, alpha=0.7, width=0.8)
    ax_ret.axhline(0, color="#58a6ff", lw=0.8)
    ax_ret.set_ylabel("Retorno Est. (%)", color=CT)
    ax_ret.text(0.005, 0.97, "Retorno Acumulado Esperado (Ensemble)",
                transform=ax_ret.transAxes, color="#8b949e", fontsize=7.5,
                va="top", ha="left")

    # ── Eje X ─────────────────────────────────
    for ax in [ax_main, ax_ret]:
        all_dates = df_ctx.index.append(fechas_f)
        ax.set_xlim(all_dates[0], all_dates[-1])
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), color=CT, fontsize=7.5)

    # ── Caja de métricas ──────────────────────
    metrics_text = (
        f"Volatilidad diaria: {pred['vol_diaria']*100:.2f}%\n"
        f"Precio actual:   ${precio_actual:.2f}\n"
        f"Pred. {dias_pred}d:      ${p_final:.2f}  ({retorno:+.1f}%)\n"
        f"Banda 68%: ${pred['banda_1_inf'][-1]:.2f} – ${pred['banda_1_sup'][-1]:.2f}\n"
        f"Banda 95%: ${pred['banda_2_inf'][-1]:.2f} – ${pred['banda_2_sup'][-1]:.2f}"
    )
    ax_main.text(0.997, 0.97, metrics_text, transform=ax_main.transAxes,
                 color=CT, fontsize=7.5, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#161b22",
                           edgecolor="#30363d", alpha=0.9))

    archivo = f"{ticker}_prediccion_{dias_pred}d.png"
    plt.savefig(archivo, dpi=150, bbox_inches="tight", facecolor=CF)
    print(f"🔮 Gráfica de predicción guardada: {archivo}")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df   = descargar_datos(TICKER, PERIODO, INTERVALO)
    df   = calcular_indicadores(df)
    pred = predecir(df, DIAS_PRED)
    imprimir_resumen(df, TICKER, pred)
    graficar_tecnico(df, TICKER)
    graficar_predicciones(df, pred, TICKER, DIAS_PRED)