"""
╔══════════════════════════════════════════════════════╗
   STOCK MONITOR — TIEMPO REAL
   Consola estilizada + Gráfica en vivo
╚══════════════════════════════════════════════════════╝

Requisitos:
    pip install yfinance pandas matplotlib rich

Uso:
    python stock_realtime.py
    python stock_realtime.py AAPL
    python stock_realtime.py TSLA 10
"""

import sys
import time
import threading
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from rich import box
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
TICKER       = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
INTERVALO_S  = int(sys.argv[2]) if len(sys.argv) > 2 else 8   # segundos entre updates
HISTORIAL    = 120       # máximo de puntos en la gráfica en vivo
PERIODO_HIST = "5d"      # historia previa para calcular RSI/MACD al inicio


# ─────────────────────────────────────────────
#  ESTADO COMPARTIDO (hilo consola ↔ hilo gráfica)
# ─────────────────────────────────────────────
class Estado:
    def __init__(self):
        self.lock       = threading.Lock()
        self.tiempos    = deque(maxlen=HISTORIAL)
        self.precios    = deque(maxlen=HISTORIAL)
        self.volumenes  = deque(maxlen=HISTORIAL)
        self.rsi        = deque(maxlen=HISTORIAL)
        self.macd       = deque(maxlen=HISTORIAL)
        self.macd_sig   = deque(maxlen=HISTORIAL)
        self.macd_hist  = deque(maxlen=HISTORIAL)
        self.ultimo     = {}
        self.ticker_obj = None
        self.info       = {}
        self.running    = True

estado = Estado()
console = Console()


# ─────────────────────────────────────────────
#  INDICADORES TÉCNICOS
# ─────────────────────────────────────────────
def calcular_rsi(precios_arr, periodo=14):
    if len(precios_arr) < periodo + 1:
        return 50.0
    delta = np.diff(precios_arr)
    gain  = np.where(delta > 0, delta, 0)
    loss  = np.where(delta < 0, -delta, 0)
    # Wilder smoothing
    avg_gain = np.mean(gain[:periodo])
    avg_loss = np.mean(loss[:periodo])
    for i in range(periodo, len(gain)):
        avg_gain = (avg_gain * (periodo - 1) + gain[i]) / periodo
        avg_loss = (avg_loss * (periodo - 1) + loss[i]) / periodo
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def calcular_macd(precios_arr):
    if len(precios_arr) < 26:
        return 0.0, 0.0, 0.0
    s = pd.Series(precios_arr)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - sig
    return round(float(macd.iloc[-1]), 4), \
           round(float(sig.iloc[-1]),  4), \
           round(float(hist.iloc[-1]), 4)


# ─────────────────────────────────────────────
#  CARGA INICIAL (histórico para indicadores)
# ─────────────────────────────────────────────
def cargar_historico():
    console.print(f"\n[bold cyan]📥 Cargando histórico de {TICKER}...[/bold cyan]")
    t = yf.Ticker(TICKER)
    estado.ticker_obj = t

    try:
        estado.info = t.info
    except Exception:
        estado.info = {}

    df = t.history(period=PERIODO_HIST, interval="1m", auto_adjust=True)
    if df.empty:
        console.print("[red]⚠ No se pudo cargar el histórico. Verifica el ticker.[/red]")
        return False

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    with estado.lock:
        for ts, row in df.iterrows():
            t_naive = ts.replace(tzinfo=None) if hasattr(ts, "tzinfo") else ts
            estado.tiempos.append(t_naive)
            estado.precios.append(float(row["Close"]))
            estado.volumenes.append(float(row["Volume"]))

            arr = np.array(estado.precios)
            rsi = calcular_rsi(arr)
            m, ms, mh = calcular_macd(arr)
            estado.rsi.append(rsi)
            estado.macd.append(m)
            estado.macd_sig.append(ms)
            estado.macd_hist.append(mh)

    console.print(f"[green]✅ {len(df)} velas de 1m cargadas como base[/green]\n")
    return True


# ─────────────────────────────────────────────
#  FETCH DE PRECIO EN TIEMPO REAL
# ─────────────────────────────────────────────
def fetch_precio():
    """Obtiene el último precio disponible vía fast_info / history de 1d."""
    t = estado.ticker_obj
    try:
        fi = t.fast_info
        precio  = float(fi.last_price)
        volumen = float(fi.three_month_average_volume or 0)
        return precio, volumen
    except Exception:
        pass
    try:
        df = t.history(period="1d", interval="1m", auto_adjust=True)
        if not df.empty:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return float(df["Close"].iloc[-1]), float(df["Volume"].iloc[-1])
    except Exception:
        pass
    return None, None


# ─────────────────────────────────────────────
#  HILO DE DATOS (actualiza estado cada N seg)
# ─────────────────────────────────────────────
def hilo_datos():
    while estado.running:
        precio, volumen = fetch_precio()
        if precio:
            now = datetime.now().replace(microsecond=0)
            with estado.lock:
                estado.tiempos.append(now)
                estado.precios.append(precio)
                estado.volumenes.append(volumen or 0)

                arr = np.array(estado.precios)
                rsi = calcular_rsi(arr)
                m, ms, mh = calcular_macd(arr)
                estado.rsi.append(rsi)
                estado.macd.append(m)
                estado.macd_sig.append(ms)
                estado.macd_hist.append(mh)

                prev = estado.precios[-2] if len(estado.precios) >= 2 else precio
                cambio   = precio - prev
                cambio_p = (cambio / prev * 100) if prev else 0

                estado.ultimo = {
                    "precio":   precio,
                    "prev":     prev,
                    "cambio":   cambio,
                    "cambio_p": cambio_p,
                    "volumen":  volumen or 0,
                    "rsi":      rsi,
                    "macd":     m,
                    "macd_sig": ms,
                    "macd_hist":mh,
                    "hora":     now.strftime("%H:%M:%S"),
                    "fecha":    now.strftime("%d %b %Y"),
                }

        time.sleep(INTERVALO_S)


# ─────────────────────────────────────────────
#  CONSOLA RICH — render
# ─────────────────────────────────────────────
def construir_panel():
    d = estado.ultimo
    if not d:
        return Panel("[yellow]Cargando datos...[/yellow]", title="[bold]STOCK MONITOR[/bold]")

    precio   = d["precio"]
    cambio   = d["cambio"]
    cambio_p = d["cambio_p"]
    rsi      = d["rsi"]
    macd     = d["macd"]
    macd_s   = d["macd_sig"]
    macd_h   = d["macd_hist"]

    color_precio = "green" if cambio >= 0 else "red"
    arrow        = "▲" if cambio >= 0 else "▼"

    # RSI color
    if rsi > 70:
        rsi_color, rsi_label = "red",    "Sobrecomprado ⚠"
    elif rsi < 30:
        rsi_color, rsi_label = "green",  "Sobrevendido ⚠"
    else:
        rsi_color, rsi_label = "white",  "Neutral ✓"

    # MACD señal
    if macd > macd_s:
        macd_color, macd_label = "green", "Alcista ↑"
    else:
        macd_color, macd_label = "red",   "Bajista ↓"

    # Nombre de la empresa
    nombre = estado.info.get("shortName", TICKER)

    # ── Tabla principal ──────────────────────
    tabla = Table(box=box.SIMPLE_HEAVY, show_header=False,
                  padding=(0, 2), expand=True,
                  style="on #0d1117")

    tabla.add_column("Campo",  style="bold #8b949e", min_width=18)
    tabla.add_column("Valor",  min_width=22)
    tabla.add_column("Campo2", style="bold #8b949e", min_width=18)
    tabla.add_column("Valor2", min_width=22)

    tabla.add_row(
        "Empresa",   f"[bold white]{nombre}[/bold white]",
        "Ticker",    f"[bold cyan]{TICKER}[/bold cyan]"
    )
    tabla.add_row(
        "Precio",
        f"[bold {color_precio}]${precio:,.2f}[/bold {color_precio}]",
        "Cambio",
        f"[{color_precio}]{arrow} ${abs(cambio):.2f}  ({cambio_p:+.2f}%)[/{color_precio}]"
    )
    tabla.add_row(
        "RSI (14)",
        f"[{rsi_color}]{rsi:.1f}  {rsi_label}[/{rsi_color}]",
        "MACD",
        f"[{macd_color}]{macd:.4f}  {macd_label}[/{macd_color}]"
    )
    tabla.add_row(
        "MACD Señal", f"[white]{macd_s:.4f}[/white]",
        "MACD Hist",  f"[{'green' if macd_h >= 0 else 'red'}]{macd_h:+.4f}[/]"
    )
    tabla.add_row(
        "Volumen",    f"[white]{int(d['volumen']):,}[/white]",
        "Actualizado", f"[#58a6ff]{d['hora']}[/]  [#8b949e]{d['fecha']}[/]"
    )

    # Puntos en memoria
    n = len(estado.precios)
    tabla.add_row(
        "Intervalo",  f"[#8b949e]{INTERVALO_S}s[/]",
        "Muestras",   f"[#8b949e]{n}/{HISTORIAL}[/]"
    )

    # Mini sparkline ASCII
    if n >= 8:
        arr    = list(estado.precios)[-40:]
        lo, hi = min(arr), max(arr)
        chars  = " ▁▂▃▄▅▆▇█"
        spark  = ""
        for v in arr:
            idx    = int((v - lo) / (hi - lo + 1e-9) * 8)
            spark += chars[idx]
        spark_color = "green" if arr[-1] >= arr[0] else "red"
        tabla.add_row(
            "Tendencia",
            f"[{spark_color}]{spark}[/{spark_color}]",
            "", ""
        )

    titulo = (
        f"[bold white] 📡 STOCK MONITOR  [/bold white]"
        f"[bold cyan]{TICKER}[/bold cyan]"
        f"[#8b949e]  ·  refresh {INTERVALO_S}s[/]"
    )

    return Panel(tabla, title=titulo, border_style="#30363d",
                 subtitle=f"[#8b949e]Ctrl+C para salir[/]")


# ─────────────────────────────────────────────
#  HILO DE CONSOLA (Rich Live)
# ─────────────────────────────────────────────
def hilo_consola():
    with Live(construir_panel(), refresh_per_second=1,
              screen=False, console=console) as live:
        while estado.running:
            live.update(construir_panel())
            time.sleep(1)


# ─────────────────────────────────────────────
#  GRÁFICA EN VIVO (matplotlib FuncAnimation)
# ─────────────────────────────────────────────
def iniciar_grafica():
    CF = "#0d1117"
    CT = "#c9d1d9"
    CG = "#21262d"
    CS = "#30363d"

    fig = plt.figure(figsize=(14, 9), facecolor=CF)
    fig.suptitle(f"📡 {TICKER} — Monitor en Tiempo Real",
                 color="white", fontsize=13, fontweight="bold")

    gs  = GridSpec(3, 1, figure=fig, hspace=0.8,
                   height_ratios=[3, 1, 1],
                   top=0.93, bottom=0.07, left=0.08, right=0.97)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    paneles   = [ax1, ax2, ax3]
    etiquetas = ["Precio en Vivo", "MACD", "RSI (14)"]

    for ax, et in zip(paneles, etiquetas):
        ax.set_facecolor(CF)
        ax.tick_params(colors=CT, labelsize=8)
        ax.yaxis.label.set_color(CT)
        for sp in ax.spines.values():
            sp.set_edgecolor(CS)
        ax.grid(which="major", color="#2d333b", lw=0.8)
        ax.grid(which="minor", color=CG, lw=0.4, ls=":")
        ax.minorticks_on()
        ax.text(0.005, 0.97, et, transform=ax.transAxes,
                color="#8b949e", fontsize=7.5, va="top")

    for ax in [ax2, ax3]:
        ax.spines["top"].set_edgecolor("#58a6ff")
        ax.spines["top"].set_linewidth(0.8)

    def actualizar(frame):
        with estado.lock:
            if len(estado.tiempos) < 2:
                return
            tiempos   = list(estado.tiempos)
            precios   = list(estado.precios)
            rsi_vals  = list(estado.rsi)
            macd_vals = list(estado.macd)
            macd_sigs = list(estado.macd_sig)
            macd_hist = list(estado.macd_hist)

        # Calcular cambio desde inicio de sesión
        p0      = precios[0]
        p_ult   = precios[-1]
        cambio  = p_ult - p0
        cambio_p = (cambio / p0 * 100) if p0 else 0
        color_l = "#3fb950" if cambio >= 0 else "#f85149"

        for ax in paneles:
            ax.cla()
            ax.set_facecolor(CF)
            ax.tick_params(colors=CT, labelsize=8)
            ax.yaxis.label.set_color(CT)
            for sp in ax.spines.values():
                sp.set_edgecolor(CS)
            ax.grid(which="major", color="#2d333b", lw=0.8)
            ax.grid(which="minor", color=CG, lw=0.4, ls=":")
            ax.minorticks_on()

        # ── Panel 1: Precio ──────────────────
        ax1.text(0.005, 0.97, "Precio en Vivo", transform=ax1.transAxes,
                 color="#8b949e", fontsize=7.5, va="top")

        ax1.plot(tiempos, precios, color=color_l, lw=1.8, zorder=4)
        ax1.fill_between(tiempos, precios, min(precios),
                         alpha=0.12, color=color_l)

        # Precio mínimo / máximo
        idx_min = int(np.argmin(precios))
        idx_max = int(np.argmax(precios))
        ax1.scatter([tiempos[idx_min]], [precios[idx_min]],
                    color="#f85149", s=50, zorder=5)
        ax1.scatter([tiempos[idx_max]], [precios[idx_max]],
                    color="#3fb950", s=50, zorder=5)
        ax1.annotate(f"Min ${precios[idx_min]:.2f}",
                     xy=(tiempos[idx_min], precios[idx_min]),
                     xytext=(5, -14), textcoords="offset points",
                     color="#f85149", fontsize=7)
        ax1.annotate(f"Max ${precios[idx_max]:.2f}",
                     xy=(tiempos[idx_max], precios[idx_max]),
                     xytext=(5, 6), textcoords="offset points",
                     color="#3fb950", fontsize=7)

        # Último precio
        arrow = "▲" if cambio >= 0 else "▼"
        ax1.annotate(
            f"  ${p_ult:.2f}  {arrow} {cambio_p:+.2f}%",
            xy=(tiempos[-1], p_ult),
            color=color_l, fontsize=10, fontweight="bold", va="center"
        )
        ax1.set_ylabel("Precio (USD)", color=CT)

        # Hora actualización
        ax1.text(0.998, 0.97,
                 f"Actualizado: {tiempos[-1].strftime('%H:%M:%S')}",
                 transform=ax1.transAxes, color="#58a6ff",
                 fontsize=7.5, va="top", ha="right")

        # ── Panel 2: MACD ────────────────────
        ax2.text(0.005, 0.97, "MACD", transform=ax2.transAxes,
                 color="#8b949e", fontsize=7.5, va="top")
        colores_h = ["#3fb950" if v >= 0 else "#f85149" for v in macd_hist]
        ax2.bar(tiempos, macd_hist, color=colores_h, alpha=0.6, width=0.00035)
        ax2.plot(tiempos, macd_vals, color="#58a6ff", lw=1.2, label="MACD")
        ax2.plot(tiempos, macd_sigs, color="#f0883e", lw=1.2, label="Señal")
        ax2.axhline(0, color=CS, lw=0.8)
        ax2.set_ylabel("MACD", color=CT)
        ax2.legend(facecolor="#161b22", labelcolor=CT, fontsize=7, loc="upper left")
        for sp in [ax2.spines["top"]]:
            sp.set_edgecolor("#58a6ff"); sp.set_linewidth(0.8)

        # ── Panel 3: RSI ─────────────────────
        ax3.text(0.005, 0.97, "RSI (14)", transform=ax3.transAxes,
                 color="#8b949e", fontsize=7.5, va="top")
        ax3.plot(tiempos, rsi_vals, color="#bc8cff", lw=1.2)
        ax3.axhline(70, color="#f85149", lw=0.8, ls="--", alpha=0.7)
        ax3.axhline(30, color="#3fb950", lw=0.8, ls="--", alpha=0.7)
        ax3.axhline(50, color=CS, lw=0.5, ls=":")
        ax3.fill_between(tiempos, rsi_vals, 70,
                         where=[r >= 70 for r in rsi_vals],
                         alpha=0.2, color="#f85149")
        ax3.fill_between(tiempos, rsi_vals, 30,
                         where=[r <= 30 for r in rsi_vals],
                         alpha=0.2, color="#3fb950")
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("RSI", color=CT)
        rsi_now = rsi_vals[-1] if rsi_vals else 50
        rsi_col = "#f85149" if rsi_now > 70 else "#3fb950" if rsi_now < 30 else "#bc8cff"
        ax3.text(0.998, 0.5, f"{rsi_now:.1f}",
                 transform=ax3.transAxes, color=rsi_col,
                 fontsize=9, fontweight="bold", va="center", ha="right")
        for sp in [ax3.spines["top"]]:
            sp.set_edgecolor("#58a6ff"); sp.set_linewidth(0.8)

        # Eje X
        for ax in paneles:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), color=CT, fontsize=7.5, rotation=0)

        fig.suptitle(
            f"📡 {TICKER} — Monitor en Tiempo Real  |  "
            f"${p_ult:.2f}  {'▲' if cambio >= 0 else '▼'} {cambio_p:+.2f}%",
            color="white", fontsize=13, fontweight="bold"
        )

    ani = animation.FuncAnimation(
        fig, actualizar,
        interval=INTERVALO_S * 1000,
        cache_frame_data=False
    )

    plt.show()
    estado.running = False


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    console.print(Rule(f"[bold cyan]📡 STOCK MONITOR  ·  {TICKER}[/bold cyan]"))
    console.print(f"[#8b949e]Intervalo de actualización: {INTERVALO_S}s  ·  "
                  f"Historial máximo: {HISTORIAL} puntos[/]\n")

    if not cargar_historico():
        sys.exit(1)

    # Primer fetch de precio real
    hilo_datos_inst = threading.Thread(target=hilo_datos, daemon=True)
    hilo_datos_inst.start()

    # Esperar el primer dato real
    console.print("[yellow]⏳ Esperando primer precio en tiempo real...[/yellow]")
    for _ in range(20):
        if estado.ultimo:
            break
        time.sleep(0.5)

    # Hilo de consola Rich (background)
    hilo_con = threading.Thread(target=hilo_consola, daemon=True)
    hilo_con.start()

    # Gráfica en el hilo principal (matplotlib requiere el main thread)
    try:
        iniciar_grafica()
    except KeyboardInterrupt:
        pass
    finally:
        estado.running = False
        console.print("\n[bold red]⏹ Monitor detenido.[/bold red]")