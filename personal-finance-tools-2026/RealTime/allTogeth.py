"""
╔══════════════════════════════════════════════════════════════╗
   STOCK MONITOR + NEWS — Tiempo Real
   Gráficas en vivo + Panel lateral de noticias y sentimiento
╚══════════════════════════════════════════════════════════════╝

Requisitos:
    pip install yfinance feedparser textblob rich requests pandas matplotlib numpy
    python -m textblob.download_corpora

Uso:
    python stock_monitor_fixed.py
    python stock_monitor_fixed.py AAPL
    python stock_monitor_fixed.py TSLA 10 120
      (ticker, segundos entre precio, segundos entre noticias)
"""

import sys
import time
import threading
import warnings
import textwrap
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import deque
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

import feedparser
from textblob import TextBlob
from rich.console import Console
from rich.rule import Rule

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
TICKER         = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
INTERVALO_S     = int(sys.argv[2])   if len(sys.argv) > 2 else 60  # fetch datos (velas de 1m)
REFRESCO_GRAF_S = 2                  # refresco visual independiente (segundos)
REFRESCO_NEWS  = int(sys.argv[3])    if len(sys.argv) > 3 else 30
HISTORIAL      = 390          # máx puntos en deque (1 día de minutos)
PERIODO_HIST   = "1d"         # solo cargar hoy para no mezclar días
VENTANA_MINS   = 60           # ventana visible en el eje X (minutos)
MAX_NOTICIAS   = 12
RESUMEN_CHARS  = 80

console = Console()


# ─────────────────────────────────────────────
#  INDICADORES TÉCNICOS
# ─────────────────────────────────────────────
def calcular_rsi(precios_arr, periodo=14):
    if len(precios_arr) < periodo + 1:
        return 50.0
    delta    = np.diff(precios_arr)
    gain     = np.where(delta > 0, delta, 0)
    loss     = np.where(delta < 0, -delta, 0)
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
    s     = pd.Series(precios_arr)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9, adjust=False).mean()
    hist  = macd - sig
    return round(float(macd.iloc[-1]), 4), \
           round(float(sig.iloc[-1]),  4), \
           round(float(hist.iloc[-1]), 4)


# ─────────────────────────────────────────────
#  SENTIMIENTO
# ─────────────────────────────────────────────
PALABRAS_POSITIVAS = {
    "surge","soar","rally","gain","rise","beat","record","high",
    "profit","growth","strong","upgrade","buy","bullish","boost",
    "outperform","positive","exceeds","top","win","success","up",
}
PALABRAS_NEGATIVAS = {
    "fall","drop","slump","loss","decline","miss","cut","low",
    "risk","concern","warn","downgrade","sell","bearish","crash",
    "plunge","fear","weak","below","down","deficit","layoff",
}

def analizar_sentimiento(texto: str):
    blob     = TextBlob(texto)
    polarity = blob.sentiment.polarity
    tl       = texto.lower()
    bonus    = sum(0.08 for p in PALABRAS_POSITIVAS if p in tl) \
             - sum(0.08 for p in PALABRAS_NEGATIVAS if p in tl)
    score    = max(-1.0, min(1.0, polarity + bonus))
    if score > 0.12:
        return "POSITIVE", score, "#3fb950"
    elif score < -0.12:
        return "NEGATIVE", score, "#f85149"
    else:
        return "NEUTRAL",  score, "#8b949e"


# ─────────────────────────────────────────────
#  RSS FEEDS
# ─────────────────────────────────────────────
def construir_feeds(ticker):
    nombre = ticker
    try:
        info   = yf.Ticker(ticker).info
        nombre = info.get("shortName", ticker)
        for s in [" Inc."," Corp."," Ltd."," LLC"," Inc"," Corp"]:
            nombre = nombre.replace(s, "").strip()
    except Exception:
        pass
    qt = quote(ticker)
    qn = quote(nombre)
    return [
        {"nombre": "Yahoo Finance",
         "url": f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={qt}&region=US&lang=en-US"},
        {"nombre": "Google News",
         "url": f"https://news.google.com/rss/search?q={qt}+stock&hl=en-US&gl=US&ceid=US:en"},
        {"nombre": "Google News Co.",
         "url": f"https://news.google.com/rss/search?q={qn}&hl=en-US&gl=US&ceid=US:en"},
        {"nombre": "Reuters",
         "url": "https://feeds.reuters.com/reuters/businessNews"},
    ]


def parsear_fecha(entry):
    for campo in ("published_parsed", "updated_parsed"):
        t = getattr(entry, campo, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def hace_cuanto(dt):
    s = int((datetime.now(timezone.utc) - dt).total_seconds())
    if s < 60:    return f"{s}s"
    if s < 3600:  return f"{s//60}m"
    if s < 86400: return f"{s//3600}h"
    return f"{s//86400}d"


def fetch_noticias(ticker):
    feeds  = construir_feeds(ticker)
    todas  = []
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed["url"])
            for entry in parsed.entries[:8]:
                titulo  = getattr(entry, "title",   "No title")
                resumen = getattr(entry, "summary", "") or ""
                link    = getattr(entry, "link", "")
                fecha   = parsear_fecha(entry)
                for tag in ["<b>","</b>","<p>","</p>","<br>","<li>","</li>",
                            "&amp;","&lt;","&gt;","&quot;"]:
                    resumen = resumen.replace(tag, " ")
                resumen = " ".join(resumen.split())[:RESUMEN_CHARS]
                etiq, score, color = analizar_sentimiento(f"{titulo}. {resumen}")
                todas.append({
                    "titulo":  titulo,
                    "resumen": resumen,
                    "fuente":  feed["nombre"],
                    "link":    link,
                    "fecha":   fecha,
                    "etiq":    etiq,
                    "score":   score,
                    "color":   color,
                })
        except Exception:
            pass

    vistos, unicas = set(), []
    for n in sorted(todas, key=lambda x: x["fecha"], reverse=True):
        clave = n["titulo"][:60].lower()
        if clave not in vistos:
            vistos.add(clave)
            unicas.append(n)
        if len(unicas) >= MAX_NOTICIAS:
            break

    conteo = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
    for n in unicas:
        conteo[n["etiq"]] += 1
    return unicas, conteo


# ─────────────────────────────────────────────
#  ESTADO COMPARTIDO
# ─────────────────────────────────────────────
class Estado:
    def __init__(self):
        self.lock        = threading.Lock()
        self.tiempos     = deque(maxlen=HISTORIAL)
        self.precios     = deque(maxlen=HISTORIAL)
        self.volumenes   = deque(maxlen=HISTORIAL)
        self.rsi         = deque(maxlen=HISTORIAL)
        self.macd        = deque(maxlen=HISTORIAL)
        self.macd_sig    = deque(maxlen=HISTORIAL)
        self.macd_hist   = deque(maxlen=HISTORIAL)
        self.ultimo      = {}
        self.ticker_obj  = None
        self.info        = {}
        self.running     = True
        # noticias
        self.noticias    = []
        self.conteo_sent = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        self.news_ultima = None
        self.news_cargando = True

estado = Estado()


# ─────────────────────────────────────────────
#  CARGA INICIAL
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
        console.print("[red]⚠ No se pudo cargar el histórico.[/red]")
        return False

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    with estado.lock:
        for ts, row in df.iterrows():
            # Convertir a hora local (igual que fetch_precio) para que historico
            # y datos en tiempo real usen la misma referencia de tiempo.
            # .replace(tzinfo=None) DESCARTA la TZ sin convertir → salto de horas.
            ts_dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if getattr(ts_dt, "tzinfo", None) is not None:
                t_naive = ts_dt.astimezone().replace(tzinfo=None)
            else:
                t_naive = ts_dt
            estado.tiempos.append(t_naive)
            estado.precios.append(float(row["Close"]))
            estado.volumenes.append(float(row["Volume"]))
            arr = np.array(estado.precios)
            estado.rsi.append(calcular_rsi(arr))
            m, ms, mh = calcular_macd(arr)
            estado.macd.append(m); estado.macd_sig.append(ms); estado.macd_hist.append(mh)

    console.print(f"[green]✅ {len(df)} velas cargadas[/green]")
    return True


# ─────────────────────────────────────────────
#  HILOS DE DATOS
# ─────────────────────────────────────────────
def fetch_precio():
    # Crear nuevo objeto Ticker cada vez para evitar caché interno de yfinance.
    # fast_info reutiliza datos obsoletos si se llama sobre el mismo objeto.
    t = yf.Ticker(TICKER)
    try:
        # Jalamos el último minuto real del intraday — es la fuente más fresca.
        df = t.history(period="1d", interval="1m", auto_adjust=True)
        if not df.empty:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            fila = df.iloc[-1]
            ts_raw = df.index[-1]
            # Convertir timestamp de la vela a naive datetime local
            ts = ts_raw.to_pydatetime()
            if ts.tzinfo is not None:
                ts = ts.astimezone().replace(tzinfo=None)
            return float(fila["Close"]), float(fila["Volume"]), ts
    except Exception:
        pass
    return None, None, None


def hilo_datos():
    while estado.running:
        precio, volumen, ts_vela = fetch_precio()
        if precio and ts_vela:
            with estado.lock:
                # Deduplicar: solo agregar si es una vela nueva (timestamp distinto)
                ultimo_ts = estado.tiempos[-1] if estado.tiempos else None
                if ultimo_ts is None or ts_vela > ultimo_ts:
                    estado.tiempos.append(ts_vela)
                    estado.precios.append(precio)
                    estado.volumenes.append(volumen or 0)
                    arr  = np.array(estado.precios)
                    rsi  = calcular_rsi(arr)
                    m, ms, mh = calcular_macd(arr)
                    estado.rsi.append(rsi)
                    estado.macd.append(m); estado.macd_sig.append(ms); estado.macd_hist.append(mh)
                # Siempre actualizar estado.ultimo para el título aunque no haya vela nueva
                prev     = estado.precios[-2] if len(estado.precios) >= 2 else precio
                cambio   = precio - prev
                cambio_p = (cambio / prev * 100) if prev else 0
                estado.ultimo = {
                    "precio": precio, "prev": prev, "cambio": cambio,
                    "cambio_p": cambio_p, "volumen": volumen or 0,
                    "rsi": estado.rsi[-1] if estado.rsi else 50,
                    "hora": ts_vela.strftime("%H:%M:%S"),
                    "fecha": ts_vela.strftime("%d %b %Y"),
                }
        time.sleep(INTERVALO_S)


def hilo_noticias():
    while estado.running:
        try:
            noticias, conteo = fetch_noticias(TICKER)
            with estado.lock:
                estado.noticias      = noticias
                estado.conteo_sent   = conteo
                estado.news_ultima   = datetime.now()
                estado.news_cargando = False
        except Exception as e:
            with estado.lock:
                estado.news_cargando = False
        time.sleep(REFRESCO_NEWS)


# ─────────────────────────────────────────────
#  GRÁFICA INTEGRADA
# ─────────────────────────────────────────────
def iniciar_grafica():
    # ── Colores ─────────────────────────────
    CF  = "#0d1117"   # fondo
    CT  = "#c9d1d9"   # texto
    CG  = "#21262d"   # grid menor
    CS  = "#30363d"   # grid mayor / bordes
    CAZ = "#58a6ff"   # azul acento
    CVE = "#3fb950"   # verde
    CRO = "#f85149"   # rojo
    CMO = "#bc8cff"   # morado (RSI)
    CNA = "#f0883e"   # naranja

    fig = plt.figure(figsize=(20, 10), facecolor=CF)
    fig.suptitle(
        f"📡  {TICKER}  —  Stock Monitor + News  ",
        color="white", fontsize=13, fontweight="bold", x=0.42
    )

    # Layout: 3 columnas (gráficas | separador | noticias)
    gs = GridSpec(
        3, 3, figure=fig,
        hspace=0.65, wspace=0.05,
        height_ratios=[3, 1, 1],
        width_ratios=[5, 0.02, 2],
        top=0.93, bottom=0.06, left=0.05, right=0.98
    )

    # ── IMPORTANTE: NO usar sharex para evitar conflictos de límites ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax_news = fig.add_subplot(gs[:, 2])

    def _estilo_ax(ax, etiq):
        ax.set_facecolor(CF)
        ax.tick_params(colors=CT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(CS)
        ax.grid(which="major", color="#2d333b", lw=0.8)
        ax.grid(which="minor", color=CG, lw=0.4, ls=":")
        ax.minorticks_on()
        ax.text(0.005, 0.97, etiq, transform=ax.transAxes,
                color="#8b949e", fontsize=7.5, va="top")

    # ── Separador vertical ──────────────────
    ax_sep = fig.add_subplot(gs[:, 1])
    ax_sep.set_facecolor("#1c2128")
    ax_sep.axis("off")

    # ── Eje de noticias ─────────────────────
    ax_news.set_facecolor(CF)
    ax_news.axis("off")

    paneles_graf = [ax1, ax2, ax3]

    def dibujar_noticias(ax):
        ax.cla()
        ax.set_facecolor(CF)
        ax.axis("off")
        ax.set_xlim(0, 1)

        with estado.lock:
            noticias   = list(estado.noticias)
            conteo     = dict(estado.conteo_sent)
            ultima     = estado.news_ultima
            cargando   = estado.news_cargando

        y = 0.985  # cursor vertical

        # ── Título del panel ────────────────
        ax.text(0.5, y, "📰  NEWS & SENTIMENT",
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=CAZ, transform=ax.transAxes)
        y -= 0.028

        hora_str = ultima.strftime("%H:%M:%S") if ultima else "—"
        ax.text(0.5, y, f"refresh {REFRESCO_NEWS}s  ·  {hora_str}",
                ha="center", va="top", fontsize=6.5, color="#8b949e",
                transform=ax.transAxes)
        y -= 0.018

        # Línea divisoria
        ax.plot([0.02, 0.98], [y, y], color=CS, lw=0.8,
                transform=ax.transAxes, clip_on=False)
        y -= 0.020

        if cargando and not noticias:
            ax.text(0.5, y, "⟳ Loading news...",
                    ha="center", va="top", fontsize=8, color="yellow",
                    transform=ax.transAxes)
            return

        total = sum(conteo.values()) or 1

        # ── Distribución de sentimiento ─────
        ax.text(0.04, y, "SENTIMENT DISTRIBUTION",
                va="top", fontsize=6.5, fontweight="bold",
                color="#8b949e", transform=ax.transAxes)
        y -= 0.022

        for etiq, color_b, label in [
            ("POSITIVE", CVE, "Positive"),
            ("NEUTRAL",  "#8b949e", "Neutral"),
            ("NEGATIVE", CRO, "Negative"),
        ]:
            count  = conteo.get(etiq, 0)
            ratio  = count / total
            bar_w  = 0.55 * ratio
            ax.text(0.04, y, label,
                    va="top", fontsize=6.5, color=color_b,
                    transform=ax.transAxes)
            rect = mpatches.FancyBboxPatch(
                (0.30, y - 0.009), bar_w, 0.013,
                boxstyle="round,pad=0.001",
                facecolor=color_b, alpha=0.75,
                transform=ax.transAxes, clip_on=False
            )
            ax.add_patch(rect)
            ax.text(0.30 + bar_w + 0.02, y - 0.001,
                    f"{count}",
                    va="top", fontsize=6.5, color=color_b,
                    transform=ax.transAxes)
            y -= 0.022

        # Señal global
        if conteo.get("POSITIVE", 0) > conteo.get("NEGATIVE", 0) * 1.5:
            sig_txt, sig_col = "↑ BULLISH BIAS",  CVE
        elif conteo.get("NEGATIVE", 0) > conteo.get("POSITIVE", 0) * 1.5:
            sig_txt, sig_col = "↓ BEARISH BIAS",  CRO
        else:
            sig_txt, sig_col = "→ MIXED / NEUTRAL", CT

        ax.text(0.5, y, sig_txt,
                ha="center", va="top", fontsize=8, fontweight="bold",
                color=sig_col, transform=ax.transAxes)
        y -= 0.025

        ax.plot([0.02, 0.98], [y, y], color=CS, lw=0.7,
                transform=ax.transAxes, clip_on=False)
        y -= 0.016

        # ── Lista de noticias ───────────────
        ax.text(0.04, y, "LATEST HEADLINES",
                va="top", fontsize=6.5, fontweight="bold",
                color="#8b949e", transform=ax.transAxes)
        y -= 0.022

        for n in noticias:
            if y < 0.01:
                break

            color_n = n["color"]
            score   = n["score"]
            etiq    = n["etiq"]

            emoji = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "⚪"}[etiq]
            ax.text(0.04, y,
                    f"{emoji} {etiq[:3]}  {score:+.2f}",
                    va="top", fontsize=6, color=color_n,
                    transform=ax.transAxes)

            bx    = 0.40
            bw    = 0.52
            mitad = bx + bw / 2
            pos_x = bx + (score + 1) / 2 * bw
            ax.plot([bx, bx + bw], [y - 0.005, y - 0.005],
                    color=CS, lw=1.2, transform=ax.transAxes, clip_on=False)
            ax.plot([mitad, mitad], [y - 0.009, y - 0.001],
                    color="#444c56", lw=0.8, transform=ax.transAxes, clip_on=False)
            ax.plot(pos_x, y - 0.005, "o",
                    ms=4, color=color_n, transform=ax.transAxes, clip_on=False)
            y -= 0.020

            titulo = n["titulo"]
            if len(titulo) > 52:
                titulo = titulo[:50] + "…"
            ax.text(0.04, y, titulo,
                    va="top", fontsize=6.8, color=CT, fontweight="bold",
                    transform=ax.transAxes, wrap=False)
            y -= 0.020

            age = hace_cuanto(n["fecha"])
            ax.text(0.04, y,
                    f"  {n['fuente']}  ·  {age} ago",
                    va="top", fontsize=6, color="#8b949e",
                    transform=ax.transAxes)
            y -= 0.014

            ax.plot([0.02, 0.98], [y, y], color="#1c2128", lw=0.5,
                    transform=ax.transAxes, clip_on=False)
            y -= 0.010

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

        # ── Ventana rodante: siempre mostramos los últimos VENTANA_MINS ──
        # t_max = último dato + pequeño padding para que no quede pegado al borde
        # t_min = t_max - ventana fija
        # Así el eje X SIEMPRE avanza hacia la derecha con cada nuevo dato.
        t_max = tiempos[-1] + timedelta(seconds=INTERVALO_S * 3)
        t_min = t_max - timedelta(minutes=VENTANA_MINS)

        # Filtrar solo los puntos dentro de la ventana visible
        indices_vis   = [i for i, t in enumerate(tiempos) if t >= t_min]
        if not indices_vis:
            indices_vis = list(range(len(tiempos)))
        tiempos_vis   = [tiempos[i]   for i in indices_vis]
        precios_vis   = [precios[i]   for i in indices_vis]
        rsi_vis       = [rsi_vals[i]  for i in indices_vis]
        macd_vis      = [macd_vals[i] for i in indices_vis]
        macd_sig_vis  = [macd_sigs[i] for i in indices_vis]
        macd_hist_vis = [macd_hist[i] for i in indices_vis]

        # Precio actual y cambio (siempre del dato más reciente vs apertura de ventana)
        p_ult    = precios[-1]
        p0_vis   = precios_vis[0]
        cambio   = p_ult - p0_vis
        cambio_p = (cambio / p0_vis * 100) if p0_vis else 0
        color_l  = CVE if cambio >= 0 else CRO

        for ax, etiq in zip(paneles_graf, ["Precio en Vivo", "MACD", "RSI (14)"]):
            ax.cla()
            _estilo_ax(ax, etiq)

        # ── Panel 1: Precio ─────────────────
        ax1.plot(tiempos_vis, precios_vis, color=color_l, lw=1.8, zorder=4)
        ax1.fill_between(tiempos_vis, precios_vis, min(precios_vis),
                         alpha=0.12, color=color_l)
        idx_min = int(np.argmin(precios_vis))
        idx_max = int(np.argmax(precios_vis))
        ax1.scatter([tiempos_vis[idx_min]], [precios_vis[idx_min]], color=CRO, s=45, zorder=5)
        ax1.scatter([tiempos_vis[idx_max]], [precios_vis[idx_max]], color=CVE, s=45, zorder=5)
        ax1.annotate(f"Min ${precios_vis[idx_min]:.2f}",
                     xy=(tiempos_vis[idx_min], precios_vis[idx_min]),
                     xytext=(5, -13), textcoords="offset points",
                     color=CRO, fontsize=7)
        ax1.annotate(f"Max ${precios_vis[idx_max]:.2f}",
                     xy=(tiempos_vis[idx_max], precios_vis[idx_max]),
                     xytext=(5, 6), textcoords="offset points",
                     color=CVE, fontsize=7)
        arrow = "▲" if cambio >= 0 else "▼"
        ax1.annotate(f"  ${p_ult:.2f}  {arrow} {cambio_p:+.2f}%",
                     xy=(tiempos_vis[-1], p_ult),
                     color=color_l, fontsize=7, fontweight="bold", va="center")
        ax1.set_ylabel("Price (USD)", color=CT)
        ax1.text(0.998, 0.97, f"Updated: {tiempos[-1].strftime('%H:%M:%S')}",
                 transform=ax1.transAxes, color=CAZ,
                 fontsize=7.5, va="top", ha="right")

        # ── Panel 2: MACD ───────────────────
        colores_h = [CVE if v >= 0 else CRO for v in macd_hist_vis]
        ax2.bar(tiempos_vis, macd_hist_vis, color=colores_h, alpha=0.6, width=0.00035)
        ax2.plot(tiempos_vis, macd_vis, color=CAZ, lw=1.2, label="MACD")
        ax2.plot(tiempos_vis, macd_sig_vis, color=CNA, lw=1.2, label="Signal")
        ax2.axhline(0, color=CS, lw=0.8)
        ax2.set_ylabel("MACD", color=CT)
        ax2.legend(facecolor="#161b22", labelcolor=CT, fontsize=7, loc="upper left")

        # ── Panel 3: RSI ────────────────────
        ax3.plot(tiempos_vis, rsi_vis, color=CMO, lw=1.2)
        ax3.axhline(70, color=CRO, lw=0.8, ls="--", alpha=0.7)
        ax3.axhline(30, color=CVE, lw=0.8, ls="--", alpha=0.7)
        ax3.axhline(50, color=CS,  lw=0.5, ls=":")
        ax3.fill_between(tiempos_vis, rsi_vis, 70,
                         where=[r >= 70 for r in rsi_vis],
                         alpha=0.2, color=CRO)
        ax3.fill_between(tiempos_vis, rsi_vis, 30,
                         where=[r <= 30 for r in rsi_vis],
                         alpha=0.2, color=CVE)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("RSI", color=CT)
        rsi_now = rsi_vis[-1] if rsi_vis else 50
        rsi_col = CRO if rsi_now > 70 else CVE if rsi_now < 30 else CMO
        ax3.text(0.998, 0.5, f"{rsi_now:.1f}",
                 transform=ax3.transAxes, color=rsi_col,
                 fontsize=9, fontweight="bold", va="center", ha="right")

        # ── Ventana rodante fija en los 3 paneles ───────────────────────
        # set_xlim explícito DESPUÉS de dibujar — esto es lo que hace que
        # el timeline avance hacia la derecha en vez de resetearse.
        for ax in paneles_graf:
            ax.set_xlim(t_min, t_max)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), color=CT, fontsize=7.5, rotation=0)

        fig.suptitle(
            f"📡  {TICKER}  —  ${p_ult:.2f}  "
            f"{'▲' if cambio >= 0 else '▼'} {cambio_p:+.2f}%"
            f"   |   Stock Monitor + News",
            color="white", fontsize=13, fontweight="bold", x=0.42
        )

        # Noticias
        dibujar_noticias(ax_news)

    ani = animation.FuncAnimation(
        fig, actualizar,
        interval=REFRESCO_GRAF_S * 1000,
        cache_frame_data=False
    )

    plt.show()
    estado.running = False


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    console.print(Rule(f"[bold cyan]📡 STOCK MONITOR + NEWS  ·  {TICKER}[/bold cyan]"))
    console.print(
        f"[#8b949e]Precio cada {INTERVALO_S}s  ·  "
        f"Noticias cada {REFRESCO_NEWS}s  ·  "
        f"Historial: {HISTORIAL} pts[/]\n"
    )

    if not cargar_historico():
        sys.exit(1)

    threading.Thread(target=hilo_datos, daemon=True).start()
    threading.Thread(target=hilo_noticias, daemon=True).start()

    console.print("[yellow]⏳ Esperando primer precio...[/yellow]")
    for _ in range(20):
        if estado.ultimo:
            break
        time.sleep(0.5)

    console.print("[yellow]⏳ Cargando noticias iniciales...[/yellow]")
    for _ in range(30):
        with estado.lock:
            listo = not estado.news_cargando
        if listo:
            break
        time.sleep(1)

    try:
        iniciar_grafica()
    except KeyboardInterrupt:
        pass
    finally:
        estado.running = False
        console.print("\n[bold red]⏹ Monitor detenido.[/bold red]")