"""
==============================================
  ANÁLISIS TÉCNICO DE ACCIONES - NYSE/NASDAQ
==============================================
Requisitos:
    pip install yfinance pandas matplotlib plotly ta

Uso:
    python analisis_bolsa.py
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
TICKER   = "NVDA"      # Símbolo de la acción (ej. MSFT, TSLA, GOOGL)
PERIODO  = "6mo"       # 1mo, 3mo, 6mo, 1y, 2y, 5y
INTERVALO = "1d"       # 1d, 1wk, 1mo


# ─────────────────────────────────────────────
#  1. DESCARGA DE DATOS
# ─────────────────────────────────────────────
def descargar_datos(ticker: str, periodo: str, intervalo: str) -> pd.DataFrame:
    print(f"\n📥 Descargando datos de {ticker} ({periodo})...")
    df = yf.download(ticker, period=periodo, interval=intervalo, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No se encontraron datos para el ticker '{ticker}'.")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    print(f"✅ {len(df)} registros descargados  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────
#  2. INDICADORES TÉCNICOS
# ─────────────────────────────────────────────
def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    # ── Medias Móviles ──────────────────────
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["EMA_12"]  = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = df["Close"].ewm(span=26, adjust=False).mean()

    # ── Bandas de Bollinger ──────────────────
    bb_media      = df["Close"].rolling(20).mean()
    bb_std        = df["Close"].rolling(20).std()
    df["BB_upper"] = bb_media + 2 * bb_std
    df["BB_lower"] = bb_media - 2 * bb_std

    # ── MACD ────────────────────────────────
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # ── RSI (14 períodos) ────────────────────
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Volumen promedio ─────────────────────
    df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()

    # ── Señales básicas ──────────────────────
    df["Señal"] = "Neutral"
    df.loc[(df["SMA_20"] > df["SMA_50"]) & (df["RSI"] < 70), "Señal"] = "COMPRA"
    df.loc[(df["SMA_20"] < df["SMA_50"]) & (df["RSI"] > 30), "Señal"] = "VENTA"

    return df


# ─────────────────────────────────────────────
#  3. RESUMEN EN CONSOLA
# ─────────────────────────────────────────────
def imprimir_resumen(df: pd.DataFrame, ticker: str):
    ult = df.iloc[-1]
    prev = df.iloc[-2]

    cambio    = ult["Close"] - prev["Close"]
    cambio_p  = (cambio / prev["Close"]) * 100
    arrow     = "▲" if cambio >= 0 else "▼"
    color_txt = "📈" if cambio >= 0 else "📉"

    print(f"""
╔══════════════════════════════════════════╗
  {color_txt}  ANÁLISIS TÉCNICO: {ticker}
╠══════════════════════════════════════════╣
  Precio actual : ${ult['Close']:.2f}
  Cambio diario : {arrow} ${abs(cambio):.2f}  ({cambio_p:+.2f}%)
  Máximo (sesión): ${ult['High']:.2f}
  Mínimo (sesión): ${ult['Low']:.2f}
  Volumen       : {int(ult['Volume']):,}
──────────────────────────────────────────
  SMA 20        : ${ult['SMA_20']:.2f}
  SMA 50        : ${ult['SMA_50']:.2f}
  RSI (14)      : {ult['RSI']:.1f}  {'⚠️ Sobrecomprado' if ult['RSI']>70 else '⚠️ Sobrevendido' if ult['RSI']<30 else '✅ Normal'}
  MACD          : {ult['MACD']:.3f}
  MACD Señal    : {ult['MACD_signal']:.3f}
──────────────────────────────────────────
  🔔 Señal      : {ult['Señal']}
╚══════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────
#  4. GRÁFICA COMPLETA
# ─────────────────────────────────────────────
def graficar(df: pd.DataFrame, ticker: str):
    color_fondo  = "#0d1117"
    color_texto  = "#c9d1d9"
    color_grid   = "#21262d"
    color_sep    = "#30363d"
    color_hoy    = "#f0883e"

    fig = plt.figure(figsize=(16, 13), facecolor=color_fondo)
    fig.suptitle(f"Análisis Técnico — {ticker}  |  {df.index[-1].strftime('%d %b %Y')}",
                 color="white", fontsize=15, fontweight="bold", y=0.99)

    # Mayor hspace para separar paneles visualmente
    gs = GridSpec(4, 1, figure=fig, hspace=1, height_ratios=[3, 1, 1, 1],
                  top=0.95, bottom=0.07, left=0.07, right=0.97)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    paneles = [ax1, ax2, ax3, ax4]
    etiquetas_panel = ["Precio & Bollinger", "Volumen", "MACD", "RSI (14)"]

    for ax, etiq in zip(paneles, etiquetas_panel):
        ax.set_facecolor(color_fondo)
        ax.tick_params(colors=color_texto, labelsize=8)
        ax.yaxis.label.set_color(color_texto)
        for spine in ax.spines.values():
            spine.set_edgecolor(color_sep)
        # Grid mayor (meses) más visible; grid menor sutil
        ax.grid(which="major", color="#2d333b", linewidth=0.8, linestyle="-")
        ax.grid(which="minor", color=color_grid, linewidth=0.4, linestyle=":")
        ax.minorticks_on()
        # Etiqueta del panel en esquina superior izquierda
        ax.text(0.005, 0.97, etiq, transform=ax.transAxes,
                color="#8b949e", fontsize=7.5, va="top", ha="left")

    # ── Línea vertical "HOY" en todos los paneles ──
    hoy = df.index[-1]
    for ax in paneles:
        ax.axvline(hoy, color=color_hoy, linewidth=1.0, linestyle="--", alpha=0.6, zorder=3)

    # ── Separadores horizontales entre paneles ──────
    # (línea superior de cada panel excepto el primero)
    for ax in [ax2, ax3, ax4]:
        ax.spines["top"].set_edgecolor("#58a6ff")
        ax.spines["top"].set_linewidth(0.8)

    # ── Panel 1: Precio ──────────────────────
    ax1.plot(df.index, df["Close"],    color="#58a6ff", linewidth=1.5, label="Precio")
    ax1.plot(df.index, df["SMA_20"],   color="#f0883e", linewidth=1,   label="SMA 20", linestyle="--")
    ax1.plot(df.index, df["SMA_50"],   color="#bc8cff", linewidth=1,   label="SMA 50", linestyle="--")
    ax1.fill_between(df.index, df["BB_upper"], df["BB_lower"],
                     alpha=0.1, color="#58a6ff", label="Bollinger")
    ax1.plot(df.index, df["BB_upper"], color="#58a6ff", linewidth=0.5, alpha=0.5)
    ax1.plot(df.index, df["BB_lower"], color="#58a6ff", linewidth=0.5, alpha=0.5)

    # Señales de compra/venta
    compras = df[df["Señal"] == "COMPRA"]
    ventas  = df[df["Señal"] == "VENTA"]
    ax1.scatter(compras.index, compras["Close"], marker="^", color="#3fb950", s=60, zorder=5, label="Compra")
    ax1.scatter(ventas.index,  ventas["Close"],  marker="v", color="#f85149", s=60, zorder=5, label="Venta")

    ax1.set_ylabel("Precio (USD)", color=color_texto)
    legend = ax1.legend(facecolor="#161b22", labelcolor=color_texto, fontsize=8, loc="upper left")

    # ── Panel 2: Volumen ─────────────────────
    colores_vol = ["#3fb950" if df["Close"].iloc[i] >= df["Close"].iloc[i-1] else "#f85149"
                   for i in range(len(df))]
    ax2.bar(df.index, df["Volume"], color=colores_vol, alpha=0.7, width=0.8)
    ax2.plot(df.index, df["Vol_SMA_20"], color="#f0883e", linewidth=1, label="Vol SMA 20")
    ax2.set_ylabel("Volumen", color=color_texto)

    # ── Panel 3: MACD ────────────────────────
    colores_hist = ["#3fb950" if v >= 0 else "#f85149" for v in df["MACD_hist"]]
    ax3.bar(df.index, df["MACD_hist"], color=colores_hist, alpha=0.6, width=0.8)
    ax3.plot(df.index, df["MACD"],        color="#58a6ff", linewidth=1, label="MACD")
    ax3.plot(df.index, df["MACD_signal"], color="#f0883e", linewidth=1, label="Señal")
    ax3.axhline(0, color="#30363d", linewidth=0.8)
    ax3.set_ylabel("MACD", color=color_texto)
    ax3.legend(facecolor="#161b22", labelcolor=color_texto, fontsize=7, loc="upper left")

    # ── Panel 4: RSI ─────────────────────────
    ax4.plot(df.index, df["RSI"], color="#bc8cff", linewidth=1.2)
    ax4.axhline(70, color="#f85149", linewidth=0.8, linestyle="--", alpha=0.7)
    ax4.axhline(30, color="#3fb950", linewidth=0.8, linestyle="--", alpha=0.7)
    ax4.fill_between(df.index, df["RSI"], 70, where=(df["RSI"] >= 70), alpha=0.2, color="#f85149")
    ax4.fill_between(df.index, df["RSI"], 30, where=(df["RSI"] <= 30), alpha=0.2, color="#3fb950")
    ax4.set_ylim(0, 100)
    ax4.set_ylabel("RSI", color=color_texto)

    # ── Eje X: etiquetas de mes en TODOS los paneles ──
    for ax in paneles:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), color=color_texto, fontsize=7.5)

    plt.setp(ax1.xaxis.get_majorticklabels(), fontsize=6.5, alpha=0.5)
    plt.setp(ax2.xaxis.get_majorticklabels(), fontsize=6.5, alpha=0.5)
    plt.setp(ax3.xaxis.get_majorticklabels(), fontsize=6.5, alpha=0.5)
    plt.setp(ax4.xaxis.get_majorticklabels(), fontsize=7.5, alpha=1.0)

    # ── Anotación "HOY" en panel de precio ──
    ax1.annotate(f"HOY\n{hoy.strftime('%d %b')}",
                 xy=(hoy, df["Close"].iloc[-1]),
                 xytext=(10, 10), textcoords="offset points",
                 color=color_hoy, fontsize=7.5, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=color_hoy, lw=0.8))

    archivo = f"{ticker}_analisis_tecnico.png"
    plt.savefig(archivo, dpi=150, bbox_inches="tight", facecolor=color_fondo)
    print(f"📊 Gráfica guardada como: {archivo}")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = descargar_datos(TICKER, PERIODO, INTERVALO)
    df = calcular_indicadores(df)
    imprimir_resumen(df, TICKER)
    graficar(df, TICKER)