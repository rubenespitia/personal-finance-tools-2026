# 📡 Stock Monitor + News

> Panel financiero en tiempo real — Precio en vivo · Análisis Técnico · Sentimiento de Noticias

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-orange?style=flat-square)
![yfinance](https://img.shields.io/badge/yfinance-0.2%2B-green?style=flat-square)
![License](https://img.shields.io/badge/uso-local%20personal-lightgrey?style=flat-square)

---

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Distribución de la Ventana](#-distribución-de-la-ventana)
- [Instalación desde Cero](#-instalación-desde-cero)
- [Uso](#-uso)
- [Indicadores Técnicos](#-indicadores-técnicos)
- [Panel de Noticias y Sentimiento](#-panel-de-noticias-y-sentimiento)
- [Temporizadores y Hilos](#-temporizadores-de-refresco-y-hilos)
- [Configuración](#-referencia-rápida-de-configuración)
- [Referencia de Imports](#-referencia-de-imports)
- [Glosario](#-glosario-de-acrónimos-y-términos)

---

## 🧭 Descripción General

**Stock Monitor + News** combina datos de precio en vivo, indicadores técnicos y sentimiento de noticias puntuado por IA, todo en una sola ventana de **Matplotlib**. Se ejecuta completamente de forma local — sin APIs de pago, sin navegador, sin dependencias en la nube.

```
┌─────────────────────────────────┬─────────────────┐
│  Panel Precio (3/5)             │                 │
├─────────────────────────────────│  Noticias  +    │
│  Panel MACD  (1/5)              │  Sentimiento    │
├─────────────────────────────────│  (2/7 ancho)    │
│  Panel RSI   (1/5)              │                 │
└─────────────────────────────────┴─────────────────┘
```

---

## 🚀 Instalación desde Cero

### 1. Requisitos del sistema

| Requisito | Versión mínima |
|-----------|---------------|
| Python    | 3.10+         |
| pip       | 22+           |
| Git       | cualquiera    |

### 2. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/personal-finance-tools-2026.git
cd personal-finance-tools-2026/RealTime
```

### 3. Crear entorno virtual (recomendado)

```bash
# Crear
python -m venv venv

# Activar — Windows
venv\Scripts\activate

# Activar — macOS / Linux
source venv/bin/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

Si no tienes el archivo `requirements.txt`, instala manualmente:

```bash
pip install yfinance feedparser textblob rich requests numpy pandas matplotlib
```

### 5. Descargar corpus de TextBlob

TextBlob necesita descargar su base de datos lingüística la primera vez:

```bash
python -m textblob.download_corpora
```

> ⚠️ Este paso es obligatorio. Sin él, el análisis de sentimiento fallará con un error de corpus faltante.

### 6. Verificar instalación

```bash
python -c "import yfinance, feedparser, textblob, matplotlib, rich; print('✅ Todo OK')"
```

---

## 📦 requirements.txt

```txt
yfinance>=0.2.0
feedparser>=6.0.0
textblob>=0.17.0
rich>=13.0.0
requests>=2.28.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
```

---

## ▶️ Uso

```bash
# Configuración por defecto — NVDA, precio cada 8s, noticias cada 180s
python stock_monitor_integrated.py

# Ticker personalizado
python stock_monitor_integrated.py AAPL

# Ticker + intervalo de precio
python stock_monitor_integrated.py TSLA 10

# Control total
python stock_monitor_integrated.py MSFT 5 120
#                                   │    │  └── noticias cada 120s
#                                   │    └───── precio cada 5s
#                                   └────────── ticker
```

---

## 📊 Indicadores Técnicos

### RSI — Índice de Fuerza Relativa

Mide el **momentum**: qué tan rápido y cuánto se ha movido un precio. Oscila entre 0 y 100.

| Zona | Valor RSI | Significado |
|------|-----------|-------------|
| 🔴 Sobrecomprado | > 70 | El precio subió muy rápido — posible reversión |
| ⚪ Neutral | 30–70 | Rango normal, sin señal extrema |
| 🟢 Sobrevendido | < 30 | El precio cayó muy rápido — posible rebote |

- Período: **14 velas** (suavizado de Wilder)
- Se dibuja en **morado** en el panel inferior
- Líneas de referencia discontinuas en 30 y 70

---

### MACD — Convergencia/Divergencia de Medias Móviles

Rastrea la **dirección de tendencia y momentum** comparando dos EMAs.

| Componente | Cálculo | Qué indica |
|------------|---------|------------|
| Línea MACD | EMA(12) − EMA(26) | Tendencia corto vs largo plazo |
| Línea Señal | EMA(9) del MACD | Cruce = posible entrada/salida |
| Histograma | MACD − Señal | 🟢 verde = alcista · 🔴 rojo = bajista |

**Señales clave:**
- MACD cruza **por encima** de la señal → 📈 alcista
- MACD cruza **por debajo** de la señal → 📉 bajista

---

### EMA — Media Móvil Exponencial

Da **mayor peso a precios recientes**, respondiendo más rápido que una SMA. Todo el MACD se construye con EMAs (períodos 12, 26 y 9).

---

### Anotaciones del Panel de Precio

| Elemento | Descripción |
|----------|-------------|
| Línea verde / roja | Traza del precio — color según ganancia o pérdida vs apertura |
| Área sombreada | Relleno entre la curva y el mínimo de sesión |
| `●` Puntos Mín / Máx | Mínimo 🔴 y máximo 🟢 de sesión con etiquetas |
| Etiqueta de precio | Último precio + flecha + % de cambio |
| `Actualizado: HH:MM:SS` | Timestamp del fetch más reciente (esquina superior derecha) |

---

## 📰 Panel de Noticias y Sentimiento

### Fuentes RSS

La app consulta **4 feeds** cada `REFRESCO_NEWS` segundos:

| Feed | Contenido |
|------|-----------|
| Yahoo Finance RSS | Titulares específicos del ticker |
| Google News (Ticker) | Búsqueda amplia por símbolo |
| Google News (Empresa) | Búsqueda por nombre de empresa |
| Reuters Business | Noticias financieras generales |

Los duplicados se eliminan automáticamente. Resultados ordenados del más reciente al más antiguo.

---

### Análisis de Sentimiento

Calculado **100% localmente** con [TextBlob](https://textblob.readthedocs.io/), sin llamadas externas:

1. **TextBlob** calcula una puntuación de polaridad de `-1.0` a `+1.0`
2. Un **bono de dominio** suma/resta `0.08` por cada palabra clave financiera encontrada  
   *(ej. `surge`, `rally`, `crash`, `layoff`, `upgrade`, `downgrade`...)*
3. El score final se limita al rango `[-1.0, +1.0]`

| Etiqueta | Rango | Color | Significado |
|----------|-------|-------|-------------|
| `POSITIVE` | > +0.12 | 🟢 Verde | Titular alcista |
| `NEUTRAL` | -0.12 a +0.12 | ⚪ Gris | Sin señal clara |
| `NEGATIVE` | < -0.12 | 🔴 Rojo | Titular bajista |

### Barra de Score `◄──●──►`

```
◄────────────┼────────────►
-1.0        0.0        +1.0
```

El punto `●` se posiciona según el score. El centro `┼` representa neutralidad exacta.

### Señal de Sesgo Global

| Señal | Condición |
|-------|-----------|
| `↑ SESGO ALCISTA` | Positivos > Negativos × 1.5 |
| `↓ SESGO BAJISTA` | Negativos > Positivos × 1.5 |
| `→ MIXTO / NEUTRO` | Ningún umbral se cumple |

---

## 🧵 Temporizadores de Refresco y Hilos

La app corre **3 hilos concurrentes** que comparten el objeto `Estado` protegido por `threading.Lock()`:

| Hilo | Variable | Por defecto | Controla |
|------|----------|-------------|----------|
| `hilo_datos()` | `INTERVALO_S` | `8 s` | Fetch de precio + recálculo de indicadores |
| `hilo_noticias()` | `REFRESCO_NEWS` | `180 s` | Fetch RSS + puntuación de sentimiento |
| Hilo principal | `INTERVALO_S` | `8 s × 1000 ms` | Redibujado de la animación Matplotlib |

> El `Lock` garantiza que ningún hilo lea datos a mitad de una escritura, evitando condiciones de carrera.

---

## ⚙️ Referencia Rápida de Configuración

Edita estas constantes al inicio del archivo `stock_monitor_integrated.py`:

| Constante | Por Defecto | Efecto |
|-----------|-------------|--------|
| `TICKER` | `NVDA` | Símbolo bursátil a monitorear |
| `INTERVALO_S` | `8` | Segundos entre cada fetch de precio y redibujado |
| `REFRESCO_NEWS` | `180` | Segundos entre fetches de noticias RSS |
| `HISTORIAL` | `120` | Máximo de velas en memoria (las más antiguas se descartan) |
| `PERIODO_HIST` | `5d` | Período de yfinance para la carga histórica inicial |
| `MAX_NOTICIAS` | `12` | Máximo de titulares únicos en el panel |
| `RESUMEN_CHARS` | `80` | Límite de caracteres del resumen por titular |

---

## 📦 Referencia de Imports

### Librería Estándar

| Import | Propósito |
|--------|-----------|
| `sys` | Lee argumentos de línea de comandos (`argv`) para ticker e intervalos |
| `time` | `time.sleep()` — pausa los hilos entre ciclos de fetch |
| `threading` | `Thread` y `Lock` — ejecución concurrente sin bloquear la UI |
| `warnings` | Suprime advertencias de deprecación de yfinance |
| `datetime / timezone` | Marcas de tiempo para velas y parseo de fechas RSS |
| `collections.deque` | Buffer circular — descarta automáticamente datos al superar `HISTORIAL` |
| `urllib.parse.quote` | Codifica en URL el ticker/empresa para queries RSS seguros |

### Datos y Cálculo Numérico

| Import | Propósito |
|--------|-----------|
| `numpy` | Arrays y matemáticas para RSI/MACD: `diff`, `mean`, `where`, `argmin/argmax` |
| `pandas` | `Series.ewm()` para EMA del MACD; parseo de DataFrames de yfinance |
| `yfinance` | Descarga velas OHLCV históricas y precio en vivo vía `Ticker` y `fast_info` |
| `feedparser` | Parsea feeds XML RSS/Atom de Yahoo Finance, Google News y Reuters |
| `textblob` | Sentimiento NLP local: `.sentiment.polarity` devuelve float de -1.0 a +1.0 |

### Visualización

| Import | Propósito |
|--------|-----------|
| `matplotlib.pyplot` | Crea figura/ejes y abre la ventana interactiva con `plt.show()` |
| `matplotlib.dates` | Formatea el eje X como horas legibles (`HH:MM`) |
| `matplotlib.animation` | `FuncAnimation` — llama a `actualizar()` en un temporizador |
| `matplotlib.patches` | `FancyBboxPatch` — barras redondeadas para la distribución de sentimiento |
| `matplotlib.gridspec` | `GridSpec` — define la cuadrícula 3×3 con proporciones personalizadas |

### Consola (Rich)

> Solo se usa en la terminal durante el arranque, no en la ventana de Matplotlib.

| Import | Propósito |
|--------|-----------|
| `rich.console.Console` | Salida estilizada en terminal con colores |
| `rich.rule.Rule` | Imprime una línea divisoria con título centrado al iniciar |

---

## 📖 Glosario de Acrónimos y Términos

| Término | Nombre Completo | Contexto |
|---------|-----------------|----------|
| RSI | Relative Strength Index | Oscilador de momentum, 0–100 |
| MACD | Moving Average Convergence/Divergence | Indicador de tendencia y momentum |
| EMA | Exponential Moving Average | Media móvil ponderada a precios recientes |
| SMA | Simple Moving Average | Promedio de precios sin ponderación |
| RSS | Really Simple Syndication | Formato XML de feed para noticias |
| NLP | Natural Language Processing | Campo al que pertenece TextBlob |
| Polaridad | Sentiment Polarity Score | Salida de TextBlob: -1 a +1 |
| Ticker | Símbolo Bursátil | Ej. `NVDA`, `AAPL`, `TSLA` |
| Vela / Candle | Punto de datos OHLCV | Open / High / Low / Close / Volume |
| `fast_info` | Objeto de metadatos rápidos de yfinance | Método más rápido para obtener el precio |
| `deque` | Double-Ended Queue | Buffer circular de longitud fija para el historial |
| `FuncAnimation` | Clase de animación de Matplotlib | Llama a `actualizar()` cada N ms |
| `GridSpec` | Gestor de layout de Matplotlib | Controla posiciones y proporciones de subplots |
| `transAxes` | Sistema de coordenadas de ejes | Coordenadas 0→1 relativas al área del eje |
| `Thread` | Hilo de Ejecución | Tarea paralela vía `threading.Thread` |
| `Lock` | Mutex de Threading | Evita lectura/escritura simultánea entre hilos |
| `daemon` | Bandera de Hilo Daemon | El hilo muere cuando termina el programa principal |

---

## ⚠️ Notas y Limitaciones

- **Mercado cerrado:** yfinance puede devolver el último precio de cierre si el mercado no está en horario regular. Los indicadores seguirán siendo válidos con los datos históricos cargados.
- **Rate limits:** Las consultas frecuentes a yfinance o Google News RSS pueden resultar en bloqueos temporales. Aumenta `INTERVALO_S` o `REFRESCO_NEWS` si ocurre.
- **TextBlob en español:** El análisis de sentimiento está optimizado para titulares en **inglés**. Los feeds RSS de Yahoo Finance y Reuters publican en inglés por defecto.
- **Uso personal:** Este proyecto es para uso local y educativo. Revisa los términos de uso de Yahoo Finance y Google News antes de cualquier uso en producción.

---

*Stock Monitor + News · personal-finance-tools-2026 · Solo uso local*
