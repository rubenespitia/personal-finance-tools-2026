"""
╔══════════════════════════════════════════════════════╗
   STOCK NEWS MONITOR — Noticias en Tiempo Real
   Fuentes: Yahoo Finance RSS + Google News RSS
   Sentimiento: TextBlob (local, sin API)
╚══════════════════════════════════════════════════════╝

Requisitos:
    pip install yfinance feedparser textblob rich requests
    python -m textblob.download_corpora

Uso:
    python stock_news.py
    python stock_news.py AAPL
    python stock_news.py TSLA 120    (refresco cada 120 segundos)
"""

import sys
import time
import threading
import textwrap
import webbrowser
from datetime import datetime, timezone
from urllib.parse import quote

import feedparser
import yfinance as yf
from textblob import TextBlob
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# ─────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────
TICKER        = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
REFRESCO_S    = int(sys.argv[2]) if len(sys.argv) > 2 else 180   # segundos entre fetch
MAX_NOTICIAS  = 20       # máximo de noticias a mostrar
RESUMEN_CHARS = 160      # caracteres del resumen por noticia

console = Console()


# ─────────────────────────────────────────────
#  FUENTES RSS — múltiples feeds por ticker
# ─────────────────────────────────────────────
def construir_feeds(ticker: str) -> list[dict]:
    nombre_empresa = ""
    try:
        info = yf.Ticker(ticker).info
        nombre_empresa = info.get("shortName", ticker)
        # Limpiar sufijos legales para mejor búsqueda
        for sufijo in [" Inc.", " Corp.", " Ltd.", " LLC", " Inc", " Corp"]:
            nombre_empresa = nombre_empresa.replace(sufijo, "").strip()
    except Exception:
        nombre_empresa = ticker

    q_ticker  = quote(ticker)
    q_empresa = quote(nombre_empresa)

    return [
        {
            "nombre": "Yahoo Finance",
            "url": f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={q_ticker}&region=US&lang=en-US",
            "icono": "🟡"
        },
        {
            "nombre": "Google News (Ticker)",
            "url": f"https://news.google.com/rss/search?q={q_ticker}+stock&hl=en-US&gl=US&ceid=US:en",
            "icono": "🔵"
        },
        {
            "nombre": "Google News (Empresa)",
            "url": f"https://news.google.com/rss/search?q={q_empresa}&hl=en-US&gl=US&ceid=US:en",
            "icono": "🔵"
        },
        {
            "nombre": "Google News (Mercados)",
            "url": "https://news.google.com/rss/search?q=stock+market+wall+street&hl=en-US&gl=US&ceid=US:en",
            "icono": "🌐"
        },
        {
            "nombre": "Reuters (Mercados)",
            "url": "https://feeds.reuters.com/reuters/businessNews",
            "icono": "📰"
        },
    ]


# ─────────────────────────────────────────────
#  ANÁLISIS DE SENTIMIENTO
# ─────────────────────────────────────────────
PALABRAS_POSITIVAS = {
    "surge", "soar", "rally", "gain", "rise", "beat", "record", "high",
    "profit", "growth", "strong", "upgrade", "buy", "bullish", "boost",
    "outperform", "positive", "exceeds", "top", "win", "success", "up",
    "sube", "gana", "récord", "positivo", "alza", "beneficio", "creció"
}
PALABRAS_NEGATIVAS = {
    "fall", "drop", "slump", "loss", "decline", "miss", "cut", "low",
    "risk", "concern", "warn", "downgrade", "sell", "bearish", "crash",
    "plunge", "fear", "weak", "below", "down", "deficit", "layoff",
    "baja", "pierde", "caída", "negativo", "riesgo", "recorte", "cae"
}

def analizar_sentimiento(texto: str) -> tuple[str, float, str]:
    """
    Retorna: (etiqueta, score, emoji)
    score: -1.0 (muy negativo) → +1.0 (muy positivo)
    """
    blob        = TextBlob(texto)
    polarity    = blob.sentiment.polarity      # -1 a +1
    subjectivity = blob.sentiment.subjectivity  # 0 a 1

    # Refuerzo con palabras clave del dominio financiero
    texto_lower = texto.lower()
    bonus = 0.0
    for p in PALABRAS_POSITIVAS:
        if p in texto_lower:
            bonus += 0.08
    for p in PALABRAS_NEGATIVAS:
        if p in texto_lower:
            bonus -= 0.08

    score = max(-1.0, min(1.0, polarity + bonus))

    if score > 0.12:
        return "POSITIVO", score, "🟢"
    elif score < -0.12:
        return "NEGATIVO", score, "🔴"
    else:
        return "NEUTRO",   score, "⚪"


def barra_sentimiento(score: float, ancho: int = 12) -> str:
    """Barra visual  ◄────●────►  centrada en 0."""
    mitad   = ancho // 2
    pos     = int((score + 1) / 2 * ancho)
    pos     = max(0, min(ancho - 1, pos))
    barra   = ["─"] * ancho
    barra[mitad] = "┼"
    barra[pos]   = "●"
    return "◄" + "".join(barra) + "►"


# ─────────────────────────────────────────────
#  PARSEO DE FECHA RSS
# ─────────────────────────────────────────────
def parsear_fecha(entry) -> datetime:
    for campo in ("published_parsed", "updated_parsed"):
        t = getattr(entry, campo, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def hace_cuanto(dt: datetime) -> str:
    delta = datetime.now(timezone.utc) - dt
    s     = int(delta.total_seconds())
    if s < 60:    return f"{s}s ago"
    if s < 3600:  return f"{s//60}m ago"
    if s < 86400: return f"{s//3600}h ago"
    return f"{s//86400}d ago"


# ─────────────────────────────────────────────
#  ESTADO COMPARTIDO
# ─────────────────────────────────────────────
class EstadoNoticias:
    def __init__(self):
        self.lock        = threading.Lock()
        self.noticias    = []          # lista de dicts procesados
        self.ultima_act  = None
        self.cargando    = True
        self.error       = ""
        self.conteo_sent = {"POSITIVO": 0, "NEUTRO": 0, "NEGATIVO": 0}
        self.nombre_emp  = TICKER
        self.feeds_ok    = 0

estado = EstadoNoticias()


# ─────────────────────────────────────────────
#  FETCH DE NOTICIAS
# ─────────────────────────────────────────────
def fetch_noticias():
    feeds    = construir_feeds(TICKER)
    todas    = []
    feeds_ok = 0

    for feed in feeds:
        try:
            parsed = feedparser.parse(feed["url"])
            for entry in parsed.entries[:10]:
                titulo  = getattr(entry, "title",   "Sin título")
                resumen = getattr(entry, "summary", "") or \
                          getattr(entry, "description", "")
                link    = getattr(entry, "link", "")
                fuente  = feed["nombre"]
                icono   = feed["icono"]
                fecha   = parsear_fecha(entry)

                # Limpiar resumen de HTML básico
                for tag in ["<b>", "</b>", "<p>", "</p>", "<br>", "<br/>",
                            "<ul>", "</ul>", "<li>", "</li>", "&amp;", "&lt;",
                            "&gt;", "&quot;", "&#39;"]:
                    resumen = resumen.replace(tag, " ")
                resumen = " ".join(resumen.split())  # normalizar espacios
                resumen = resumen[:RESUMEN_CHARS + 50]  # trim raw

                # Sentimiento sobre título + resumen
                texto_analisis = f"{titulo}. {resumen}"
                etiq, score, emoji = analizar_sentimiento(texto_analisis)

                todas.append({
                    "titulo":   titulo,
                    "resumen":  resumen[:RESUMEN_CHARS],
                    "fuente":   fuente,
                    "icono":    icono,
                    "link":     link,
                    "fecha":    fecha,
                    "etiq":     etiq,
                    "score":    score,
                    "emoji":    emoji,
                    "barra":    barra_sentimiento(score),
                })
            feeds_ok += 1
        except Exception as e:
            pass  # feed no disponible, continuar con los demás

    # Deduplicar por título similar y ordenar por fecha desc
    vistos = set()
    unicas = []
    for n in sorted(todas, key=lambda x: x["fecha"], reverse=True):
        clave = n["titulo"][:60].lower()
        if clave not in vistos:
            vistos.add(clave)
            unicas.append(n)
        if len(unicas) >= MAX_NOTICIAS:
            break

    conteo = {"POSITIVO": 0, "NEUTRO": 0, "NEGATIVO": 0}
    for n in unicas:
        conteo[n["etiq"]] += 1

    return unicas, conteo, feeds_ok


# ─────────────────────────────────────────────
#  HILO DE ACTUALIZACIÓN
# ─────────────────────────────────────────────
def hilo_fetch():
    while True:
        with estado.lock:
            estado.cargando = True
        try:
            noticias, conteo, feeds_ok = fetch_noticias()
            with estado.lock:
                estado.noticias    = noticias
                estado.conteo_sent = conteo
                estado.ultima_act  = datetime.now()
                estado.cargando    = False
                estado.feeds_ok    = feeds_ok
                estado.error       = ""
        except Exception as e:
            with estado.lock:
                estado.error    = str(e)
                estado.cargando = False
        time.sleep(REFRESCO_S)


# ─────────────────────────────────────────────
#  RENDER RICH
# ─────────────────────────────────────────────
def color_sentimiento(etiq: str) -> str:
    return {"POSITIVO": "green", "NEGATIVO": "red", "NEUTRO": "white"}[etiq]


def render_header() -> Panel:
    """Barra superior con ticker, hora y resumen de sentimiento."""
    with estado.lock:
        conteo   = estado.conteo_sent.copy()
        ultima   = estado.ultima_act
        cargando = estado.cargando
        feeds_ok = estado.feeds_ok
        total    = len(estado.noticias)

    hora_str = ultima.strftime("%H:%M:%S  %d %b %Y") if ultima else "—"

    txt = Text()
    txt.append(f"  📡 STOCK NEWS MONITOR  ", style="bold white")
    txt.append(f"[ {TICKER} ]", style="bold cyan")
    txt.append("   |   ", style="#30363d")
    txt.append(f"🟢 {conteo['POSITIVO']} positivas  ", style="green")
    txt.append(f"⚪ {conteo['NEUTRO']} neutras  ",     style="white")
    txt.append(f"🔴 {conteo['NEGATIVO']} negativas",   style="red")
    txt.append(f"   |   {total} noticias  ", style="#8b949e")
    txt.append(f"feeds: {feeds_ok}  ", style="#8b949e")
    if cargando:
        txt.append("⟳ actualizando...", style="yellow")
    else:
        txt.append(f"actualizado: {hora_str}", style="#58a6ff")
    txt.append(f"  ·  refresh: {REFRESCO_S}s", style="#8b949e")

    return Panel(txt, border_style="#30363d", padding=(0, 1))


def render_noticia(n: dict, idx: int) -> Panel:
    """Renderiza una noticia individual como Panel."""
    etiq      = n["etiq"]
    score     = n["score"]
    color_s   = color_sentimiento(etiq)
    hace      = hace_cuanto(n["fecha"])
    hora_fmt  = n["fecha"].strftime("%d %b  %H:%M")

    # Cabecera: índice + sentimiento
    cab = Text()
    cab.append(f" #{idx+1:02d} ", style="bold #8b949e")
    cab.append(f" {n['emoji']} {etiq} ", style=f"bold {color_s}")
    cab.append(f"  {n['barra']}  ", style=color_s)
    cab.append(f"score: {score:+.2f}", style=f"dim {color_s}")

    # Título
    titulo = Text(n["titulo"], style="bold white")

    # Resumen
    resumen_wrap = textwrap.fill(n["resumen"], width=90) if n["resumen"] else "[dim]Sin resumen disponible[/dim]"
    resumen_txt  = Text(resumen_wrap, style="#c9d1d9")

    # Pie: fuente + hora + link
    pie = Text()
    pie.append(f"{n['icono']} {n['fuente']}", style="#58a6ff")
    pie.append(f"  ·  {hora_fmt}  ({hace})", style="#8b949e")
    if n["link"]:
        pie.append(f"  ·  🔗 {n['link'][:70]}{'…' if len(n['link'])>70 else ''}",
                   style="dim #8b949e")

    contenido = Text.assemble(cab, "\n", titulo, "\n", resumen_txt, "\n", pie)

    border = {"POSITIVO": "#238636", "NEGATIVO": "#da3633", "NEUTRO": "#30363d"}[etiq]
    return Panel(contenido, border_style=border, padding=(0, 2))


def render_resumen_sentimiento() -> Panel:
    """Panel lateral con distribución de sentimiento."""
    with estado.lock:
        c     = estado.conteo_sent.copy()
        total = sum(c.values()) or 1
        nots  = estado.noticias[:8]

    tabla = Table(box=box.SIMPLE, show_header=False, padding=(0, 1),
                  style="on #0d1117", expand=True)
    tabla.add_column("", style="#8b949e")
    tabla.add_column("", min_width=14)
    tabla.add_column("", style="bold")

    def barra_dist(n, total, color, ancho=10):
        filled = int(n / total * ancho)
        return f"[{color}]{'█' * filled}{'░' * (ancho - filled)}[/{color}]"

    tabla.add_row("🟢 Positivo",
                  barra_dist(c["POSITIVO"], total, "green"),
                  f"[green]{c['POSITIVO']}[/green]")
    tabla.add_row("⚪ Neutro",
                  barra_dist(c["NEUTRO"], total, "white"),
                  f"[white]{c['NEUTRO']}[/white]")
    tabla.add_row("🔴 Negativo",
                  barra_dist(c["NEGATIVO"], total, "red"),
                  f"[red]{c['NEGATIVO']}[/red]")

    # Señal general del mercado
    tabla.add_row("", "", "")
    if c["POSITIVO"] > c["NEGATIVO"] * 1.5:
        signal = "[bold green]↑ SESGO ALCISTA[/bold green]"
    elif c["NEGATIVO"] > c["POSITIVO"] * 1.5:
        signal = "[bold red]↓ SESGO BAJISTA[/bold red]"
    else:
        signal = "[bold white]→ MIXTO / NEUTRO[/bold white]"
    tabla.add_row("Señal:", signal, "")

    # Scores recientes
    if nots:
        tabla.add_row("", "", "")
        tabla.add_row("[bold #8b949e]Scores recientes[/bold #8b949e]", "", "")
        for n in nots:
            c_s = color_sentimiento(n["etiq"])
            tabla.add_row(
                f"[{c_s}]{n['emoji']}[/{c_s}]",
                f"[{c_s}]{n['score']:+.2f}[/{c_s}]",
                f"[dim white]{n['titulo'][:28]}…[/dim white]"
            )

    return Panel(tabla, title="[bold #8b949e]Distribución Sentimiento[/bold #8b949e]",
                 border_style="#30363d", padding=(0, 1))


def construir_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="noticias", ratio=3),
        Layout(name="sidebar",  ratio=1),
    )
    return layout


def actualizar_layout(layout: Layout):
    with estado.lock:
        noticias  = list(estado.noticias)
        cargando  = estado.cargando
        error_msg = estado.error

    layout["header"].update(render_header())
    layout["sidebar"].update(render_resumen_sentimiento())

    if cargando and not noticias:
        layout["noticias"].update(
            Panel("[yellow]⟳ Cargando noticias...[/yellow]",
                  border_style="#30363d")
        )
        return

    if error_msg and not noticias:
        layout["noticias"].update(
            Panel(f"[red]⚠ Error: {error_msg}[/red]",
                  border_style="red")
        )
        return

    # Renderizar noticias apiladas
    from rich.console import Group
    paneles = [render_noticia(n, i) for i, n in enumerate(noticias)]
    layout["noticias"].update(Panel(
        Group(*paneles),
        title=f"[bold white]Noticias — {TICKER}[/bold white]",
        border_style="#21262d",
        padding=(0, 0),
    ))


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    console.print(Rule(f"[bold cyan]📰 STOCK NEWS MONITOR  ·  {TICKER}[/bold cyan]"))
    console.print(
        f"[#8b949e]Fuentes: Yahoo Finance RSS + Google News RSS + Reuters\n"
        f"Sentimiento: TextBlob (análisis local)\n"
        f"Refresco: cada {REFRESCO_S}s  ·  Máx. noticias: {MAX_NOTICIAS}[/]\n"
    )

    # Lanzar hilo de fetch
    t = threading.Thread(target=hilo_fetch, daemon=True)
    t.start()

    # Esperar primera carga
    console.print("[yellow]⏳ Obteniendo noticias...[/yellow]")
    for _ in range(30):
        with estado.lock:
            listo = not estado.cargando
        if listo:
            break
        time.sleep(1)

    # Live display
    layout = construir_layout()
    try:
        with Live(layout, refresh_per_second=0.5,
                  screen=True, console=console) as live:
            while True:
                actualizar_layout(layout)
                time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[bold red]⏹ Monitor de noticias detenido.[/bold red]")