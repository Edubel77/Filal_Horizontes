"""\
MÓDULO: new_estadistica_descriptiva.py
ETAPA CRISP-DM: Comprensión de los Datos (Data Understanding)
PROPÓSITO: Primera implementación de Estadística Descriptiva para el dataset consolidado 2022-2024.
BASE: estadistica_descriptiva_II.py
"""\

import unicodedata  # PARCHE 1: normalización de tildes (MACROSECTOR)

import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots  # PARCHE 13
from scipy.stats import norm

import plotly.express as px

COLS_FIN_2022 = [
    "INGRESOS OPERACIONALES 2022*",
    "GANANCIA (PERDIDA) 2022",
    "TOTAL ACTIVOS 2022",
    "TOTAL PASIVOS 2022",
    "TOTAL PATRIMONIO 2022",
]

COLS_FIN_2023 = [
    "INGRESOS OPERACIONALES 2023*",
    "GANANCIA (PERDIDA) 2023",
    "TOTAL ACTIVOS 2023",
    "TOTAL PASIVOS 2023",
    "TOTAL PATRIMONIO 2023",
]

COLS_FIN_2024 = [
    "INGRESOS OPERACIONALES 2024*",
    "GANANCIA (PERDIDA) 2024",
    "TOTAL ACTIVOS 2024",
    "TOTAL PASIVOS 2024",
    "TOTAL PATRIMONIO 2024",
]

ETIQUETAS = {
    # "INGRESOS OPERACIONALES 2022*": "Ingresos Op. 2022",
    "INGRESOS OPERACIONALES 2022*": "Ingresos Op. 2022",
    "INGRESOS OPERACIONALES 2022": "Ingresos Op. 2022",
    "GANANCIA (PERDIDA) 2022": "Ganancia/Pérdida 2022",
    "TOTAL ACTIVOS 2022": "Total Activos 2022",
    "TOTAL PASIVOS 2022": "Total Pasivos 2022",
    "TOTAL PATRIMONIO 2022": "Total Patrimonio 2022",

    "INGRESOS OPERACIONALES 2023*": "Ingresos Op. 2023",
    "GANANCIA (PERDIDA) 2023": "Ganancia/Pérdida 2023",
    "TOTAL ACTIVOS 2023": "Total Activos 2023",
    "TOTAL PASIVOS 2023": "Total Pasivos 2023",
    "TOTAL PATRIMONIO 2023": "Total Patrimonio 2023",

    "INGRESOS OPERACIONALES 2024*": "Ingresos Op. 2024",
    "GANANCIA (PERDIDA) 2024": "Ganancia/Pérdida 2024",
    "TOTAL ACTIVOS 2024": "Total Activos 2024",
    "TOTAL PASIVOS 2024": "Total Pasivos 2024",
    "TOTAL PATRIMONIO 2024": "Total Patrimonio 2024",
}


def _resolver_columnas(df: pd.DataFrame, columnas: list[str]) -> list[str]:
    """Resuelve nombres financieros permitiendo diferencias como el * de 2022."""
    disponibles = {str(c).strip(): c for c in df.columns}
    resueltas = []

    for col in columnas:
        if col in disponibles:
            resueltas.append(disponibles[col])
            continue

        # Algunas fuentes conservan el asterisco (*) y otras no.
        alterna = col.replace("*", "")
        if alterna in disponibles:
            resueltas.append(disponibles[alterna])
            continue

        # Comparación adicional ignorando espacios extremos y asteriscos.
        objetivo = alterna.strip().upper()
        encontrada = next(
            (original for normalizada, original in
             ((str(c).replace("*", "").strip().upper(), c) for c in df.columns)
             if normalizada == objetivo),
            None,
        )
        if encontrada is not None:
            resueltas.append(encontrada)

    return resueltas


def mostrar_estadistica(df: pd.DataFrame):
    """
    Etapa CRISP-DM: Comprensión de los Datos.

    En esta primera versión de la nueva estadística descriptiva se implementa
    únicamente la pestaña "Tabla de Estadísticos", preparada para 2022, 2023
    y 2024. Las demás pestañas se incorporarán progresivamente.
    """
    st.header("📊 Análisis Estadístico Descriptivo")
    st.markdown(
        """
        La estadística descriptiva resume las características de las variables
        financieras del dataset. Esta versión incorpora los períodos
        **2022, 2023 y 2024**, manteniendo la estructura metodológica utilizada
        en `estadistica_descriptiva_II.py`.
        """
    )

    # tabs = st.tabs(["📐 Tabla de Estadísticos"])

    # with tabs[0]:
    #     _tabla_estadisticos(df)
    
    tabs = st.tabs([
        "📐 Tabla de Estadísticos",
        "🔔 Distribución y Prueba de Normalidad",
        "📦 Percentiles y Cuartiles",
        "📏 Métricas Financieras Derivadas",
        "🔍 Análisis por Segmento",
    ])

    with tabs[0]:
        _tabla_estadisticos(df)

    with tabs[1]:
        _distribucion_normalidad(df)
        
    with tabs[2]:
        _percentiles_cuartiles(df)
        
    with tabs[3]:
        _metricas_derivadas(df)
        
    with tabs[4]:
        _analisis_segmento(df)
    

def _tabla_estadisticos(df: pd.DataFrame):
    """Construye la tabla completa de estadísticos para 2022-2024."""

    st.subheader("📐 Tabla Completa de Estadísticos Descriptivos")

    anio = st.radio(
        "Seleccionar período:",
        ["2024", "2023", "2022", "Los tres años"],
        horizontal=True,
        key="estadistica_anio_2022_2024",
    )

    columnas_por_anio = {
        "2022": COLS_FIN_2022,
        "2023": COLS_FIN_2023,
        "2024": COLS_FIN_2024,
    }

    if anio == "Los tres años":
        cols = COLS_FIN_2022 + COLS_FIN_2023 + COLS_FIN_2024
    else:
        cols = columnas_por_anio[anio]

    # Solo se utilizan columnas efectivamente disponibles en el DataFrame.
    # cols = _resolver_columnas(df, cols)
    
    # Solo se utilizan columnas efectivamente disponibles en el DataFrame.
    # Se admiten las dos denominaciones posibles de Ingresos Operacionales.
    cols_disponibles = []

    for c in cols:
        if c in df.columns:
            cols_disponibles.append(c)

        elif c == "INGRESOS OPERACIONALES 2022*" and "INGRESOS OPERACIONALES 2022" in df.columns:
            cols_disponibles.append("INGRESOS OPERACIONALES 2022")

        elif c == "INGRESOS OPERACIONALES 2023*" and "INGRESOS OPERACIONALES 2023" in df.columns:
            cols_disponibles.append("INGRESOS OPERACIONALES 2023")

        elif c == "INGRESOS OPERACIONALES 2024*" and "INGRESOS OPERACIONALES 2024" in df.columns:
            cols_disponibles.append("INGRESOS OPERACIONALES 2024")

    cols = cols_disponibles    

    if not cols:
        st.error(
            "No se encontraron las columnas financieras correspondientes al "
            "período seleccionado. Verifica los nombres de las columnas del "
            "dataset consolidado 2022-2024."
        )
        return

    df_num = df[cols].apply(pd.to_numeric, errors="coerce")

    filas = []

    # for col in cols:
    #     s = df_num[col].dropna()
    
    for col in cols:
        s = df_num[col].dropna()

        # ==============================================================
        # TRATAMIENTO DE OUTLIERS — MISMO CRITERIO DE new_eda_VII.py
        # Solo se aplica a Ingresos Operacionales 2022.
        # El rango de new_eda_VII.py para Ingresos Operacionales es:
        # 0.004 a 500 Miles de Millones COP.
        # Los datos originales están expresados en miles de COP.
        # 500 Miles de Millones COP = 500.000.000 millones de COP
        #                         = 500.000.000.000.000 COP.
        # ==============================================================
        if col == "INGRESOS OPERACIONALES 2022":
            # limite_inferior = 0.004 * 1_000_000
            # limite_superior = 500 * 1_000_000

            # s = s[
            #     (s >= limite_inferior) &
            #     (s <= limite_superior)
            # ]
        
            s = s[
                (s / 1_000_000 >= 0.004) &
                (s / 1_000_000 <= 500)
            ]        

        if len(s) == 0:

        # if len(s) == 0:
            filas.append({
                "Variable": ETIQUETAS.get(col, col),
                "N": 0,
                "Media (Miles de Millones COP)": None,
                "Mediana (Miles de Millones COP)": None,
                "Moda (Miles de Millones COP)": None,
                "Desv. Estándar (Miles de Millones COP)": None,
                "Varianza": None,
                "CV (%)": None,
                "Mínimo (Miles de Millones COP)": None,
                "Máximo (Miles de Millones COP)": None,
                "Rango (Miles de Millones COP)": None,
                "Asimetría": None,
                "Curtosis": None,
                "P-valor Shapiro": None,
                "¿Normal?": "Sin datos",
            })
            continue

        # Shapiro-Wilk: se conserva la lógica del módulo II.
        # SciPy requiere al menos 3 observaciones.
        if len(s) >= 3:
            muestra = s.sample(min(5000, len(s)), random_state=42)
            _, p_valor = stats.shapiro(muestra)
            p_valor_txt = f"{p_valor:.4f}"
            normal_txt = "✅ Sí" if p_valor > 0.05 else "❌ No"
        else:
            p_valor = np.nan
            p_valor_txt = None
            normal_txt = "No evaluable"

        skew = s.skew()
        kurt = s.kurtosis()
        media = s.mean()

        filas.append({
            "Variable": ETIQUETAS.get(col, col),
            "N": len(s),
            "Media (Miles de Millones COP)": round(media / 1e6, 2),
            "Mediana (Miles de Millones COP)": round(s.median() / 1e6, 2),
            "Moda (Miles de Millones COP)": (
                round(float(s.mode().iloc[0]) / 1e6, 2)
                if len(s.mode()) > 0 else None
            ),
            "Desv. Estándar (Miles de Millones COP)": round(s.std() / 1e6, 2),
            
            "Varianza": f"{s.var():.2e}",
            "CV (%)": round(s.std() / abs(s.mean()) * 100, 1) if s.mean() != 0 else None,
            
            # "Varianza": f"{s.var() / 1e18:.6g}",
            # "CV (%)": (
            #     round(s.std() / abs(media) * 100, 1)
            #     if media != 0 else None
            # ),
            "Mínimo (Miles de Millones COP)": round(s.min() / 1e6, 2),
            "Máximo (Miles de Millones COP)": round(s.max() / 1e6, 2),
            "Rango (Miles de Millones COP)": round((s.max() - s.min()) / 1e6, 2),
            "Asimetría": round(skew, 3),
            "Curtosis": round(kurt, 3),
            "P-valor Shapiro": p_valor_txt,
            "¿Normal?": normal_txt,
        })

    tabla = pd.DataFrame(filas)

    # st.dataframe(
    #     tabla.set_index("Variable"),
    #     use_container_width=True,
    #     height=620,
    # )
    
    # Altura dinámica:
    # 5 filas para un solo año y 15 filas para "Los tres años".
    altura_tabla = 260 if anio != "Los tres años" else 620

    st.dataframe(
        tabla.set_index("Variable"),
        use_container_width=True,
        height=altura_tabla,
    )

    with st.expander("ℹ️ Glosario de estadísticos"):
        st.markdown(
            """
            | Estadístico | Descripción |
            |---|---|
            | **Media** | Promedio aritmético |
            | **Mediana** | Valor central que divide el 50% de los datos |
            | **Moda** | Valor más frecuente |
            | **Desv. Estándar** | Dispersión promedio respecto a la media |
            | **CV (%)** | Coeficiente de variación: dispersión relativa |
            | **Asimetría** | >0 cola derecha, <0 cola izquierda |
            | **Curtosis** | >0 más puntiaguda que normal, <0 más plana |
            | **P-valor Shapiro** | >0.05 indica distribución compatible con normalidad |
            """
        )

    # Control de trazabilidad: informa qué años fueron realmente encontrados.
    anios_encontrados = []
    for anio_check, cols_check in columnas_por_anio.items():
        if any(c in df.columns for c in cols_check):
            anios_encontrados.append(anio_check)

    st.caption(
        "Valores monetarios expresados en Miles de Millones COP. La columna de Ingresos Op. 2022 se resuelve con o sin asterisco (*) según el nombre presente en el DataFrame."
    )

    st.caption(
        "Períodos financieros disponibles en el DataFrame: "
        + ", ".join(anios_encontrados)
    )
    
# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA: 📈 DISTRIBUCIÓN Y NORMALIDAD
# ═══════════════════════════════════════════════════════════════════════════

# def _distribucion_normalidad(df: pd.DataFrame):
#     """
#     Analiza la distribución de las variables financieras por Macrosector
#     para 2022, 2023 y 2024.

#     La visualización utiliza violines KDE para facilitar la interpretación
#     de la forma, concentración y dispersión de los datos.
#     """

#     st.subheader("📈 Distribución y Normalidad")

#     st.markdown(
#         """
#         Esta sección permite observar la **forma de la distribución** de las
#         variables financieras por **Macrosector**, para cada uno de los años
#         disponibles. La representación mediante violines KDE facilita la
#         identificación visual de concentración, dispersión y asimetría.
#         """
#     )

#     anio = st.radio(
#         "Seleccionar período:",
#         ["2024", "2023", "2022"],
#         horizontal=True,
#         key="distribucion_normalidad_anio",
#     )

#     variables = {
#         "Ingresos Operacionales": {
#             "2022": "INGRESOS OPERACIONALES 2022*",
#             "2023": "INGRESOS OPERACIONALES 2023*",
#             "2024": "INGRESOS OPERACIONALES 2024*",
#         },
#         "Ganancia/Pérdida": {
#             "2022": "GANANCIA (PERDIDA) 2022",
#             "2023": "GANANCIA (PERDIDA) 2023",
#             "2024": "GANANCIA (PERDIDA) 2024",
#         },
#         "Total Activos": {
#             "2022": "TOTAL ACTIVOS 2022",
#             "2023": "TOTAL ACTIVOS 2023",
#             "2024": "TOTAL ACTIVOS 2024",
#         },
#         "Total Pasivos": {
#             "2022": "TOTAL PASIVOS 2022",
#             "2023": "TOTAL PASIVOS 2023",
#             "2024": "TOTAL PASIVOS 2024",
#         },
#         "Total Patrimonio": {
#             "2022": "TOTAL PATRIMONIO 2022",
#             "2023": "TOTAL PATRIMONIO 2023",
#             "2024": "TOTAL PATRIMONIO 2024",
#         },
#     }

#     columna = variables["Ingresos Operacionales"][anio]

#     # Selector de variable financiera
#     variable = st.selectbox(
#         "Variable financiera:",
#         list(variables.keys()),
#         key=f"distribucion_variable_{anio}",
#     )

#     columna = variables[variable][anio]

#     # Compatibilidad con la nomenclatura de 2022 sin asterisco.
#     if columna not in df.columns and columna.endswith("*"):
#         columna = columna[:-1]

#     if columna not in df.columns:
#         st.warning(
#             f"No se encontró la columna correspondiente a "
#             f"**{variable} — {anio}**."
#         )
#         return

#     if "MACROSECTOR" not in df.columns:
#         st.warning("El DataFrame no contiene la columna `MACROSECTOR`.")
#         return

#     # Preparación de datos
#     df_plot = df[["MACROSECTOR", columna]].copy()

#     df_plot[columna] = pd.to_numeric(
#         df_plot[columna],
#         errors="coerce",
#     )

#     df_plot = df_plot.dropna(
#         subset=["MACROSECTOR", columna]
#     )

#     # Conversión a Miles de Millones COP
#     df_plot["Valor"] = df_plot[columna] / 1_000_000_000

#     # Orden de los macrosectores por cantidad de empresas
#     orden_macrosector = (
#         df_plot["MACROSECTOR"]
#         .value_counts()
#         .index
#         .tolist()
#     )

#     # Violín KDE
#     fig = px.violin(
#         df_plot,
#         x="MACROSECTOR",
#         y="Valor",
#         category_orders={
#             "MACROSECTOR": orden_macrosector
#         },
#         box=True,
#         points=False,
#         title=(
#             f"Densidad (KDE) por Macrosector — "
#             f"{variable} — {anio}"
#         ),
#     )

#     fig.update_traces(
#         meanline_visible=True,
#         width=1.0,
#         opacity=0.80,
#     )

#     fig.update_layout(
#         height=760,
#         autosize=True,
#         xaxis_title="Macrosector",
#         yaxis_title="Miles de Millones COP",
#         xaxis=dict(
#             tickangle=-20,
#             tickfont=dict(size=14),
#         ),
#         yaxis=dict(
#             tickfont=dict(size=13),
#         ),
#         title=dict(
#             font=dict(size=20),
#         ),
#         margin=dict(
#             l=70,
#             r=40,
#             t=80,
#             b=120,
#         ),
#     )

#     st.plotly_chart(
#         fig,
#         use_container_width=True,
#     )

#     st.caption(
#         "La amplitud de cada violín representa la densidad de observaciones "
#         "en los diferentes niveles de la variable. La caja interior permite "
#         "visualizar la mediana y los cuartiles."
#     )

# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA: 🔔 DISTRIBUCIÓN Y PRUEBA DE NORMALIDAD
# ═══════════════════════════════════════════════════════════════════════════

# Conversión del dataset:
# Los valores financieros originales están expresados en MILES DE PESOS.
# Para presentarlos en MILES DE MILLONES DE COP se divide entre 1.000.000.
DIVISOR_MM_COP = 1_000_000



# ═══ PARCHE 7 (inicio) — Etiquetas emergentes legibles ═══════════════════
# Texto NEGRO sobre fondo BLANCO para todas las etiquetas que aparecen al
# pasar el cursor (histograma, curva normal, Q-Q y violines KDE).
# Se aplica a nivel de TRAZA y de LAYOUT para que ni el color de cada
# macrosector ni el tema de Streamlit cambien el fondo o el texto.
HOVERLABEL_LEGIBLE = dict(
    bgcolor="#FFFFFF",
    bordercolor="#333333",
    font=dict(
        color="#000000",
        size=14,
        family="Arial, sans-serif",
    ),
    align="left",
    namelength=-1,
)


def _aplicar_hover_legible(fig: go.Figure, tamano: int = 14) -> go.Figure:
    """Fuerza texto negro sobre fondo blanco en todas las etiquetas hover."""
    estilo = dict(HOVERLABEL_LEGIBLE)
    estilo["font"] = dict(HOVERLABEL_LEGIBLE["font"], size=tamano)
    fig.update_traces(hoverlabel=estilo)
    fig.update_layout(hoverlabel=estilo)
    return fig
# ═══ PARCHE 7 (fin) ══════════════════════════════════════════════════════


VARIABLES_DISTRIBUCION = {
    "Ingresos Op.": {
        2022: "INGRESOS OPERACIONALES 2022*",
        2023: "INGRESOS OPERACIONALES 2023*",
        2024: "INGRESOS OPERACIONALES 2024*",
    },
    "Ganancia/Pérdida": {
        2022: "GANANCIA (PERDIDA) 2022",
        2023: "GANANCIA (PERDIDA) 2023",
        2024: "GANANCIA (PERDIDA) 2024",
    },
    "Total Activos": {
        2022: "TOTAL ACTIVOS 2022",
        2023: "TOTAL ACTIVOS 2023",
        2024: "TOTAL ACTIVOS 2024",
    },
    "Total Pasivos": {
        2022: "TOTAL PASIVOS 2022",
        2023: "TOTAL PASIVOS 2023",
        2024: "TOTAL PASIVOS 2024",
    },
    "Total Patrimonio": {
        2022: "TOTAL PATRIMONIO 2022",
        2023: "TOTAL PATRIMONIO 2023",
        2024: "TOTAL PATRIMONIO 2024",
    },
}


# Mismos colores utilizados en new_eda_VII.py
COLORES_KDE_MACROSECTOR = {
    "COMERCIO": "#0072B2",
    "SERVICIOS": "#56B4E9",
    "MANUFACTURA": "#CC79A7",
    "CONSTRUCCION": "#E69F00",
    "AGROPECUARIO": "#009E73",
    "MINERO-HIDROCARBUROS": "#D55E00",
}

# ═══════════════════════════════════════════════════════════════════════════
# COLORES PARA RADAR — ANÁLISIS POR SEGMENTO
# ═══════════════════════════════════════════════════════════════════════════

# COLORES_RADAR_MACROSECTOR = {
#     "MANUFACTURA": "#1E88E5",
#     "SERVICIOS": "#FF9800",
#     "CONSTRUCCION": "#E53935",
#     "COMERCIO": "#FF69B4",
#     "MINERO-HIDROCARBUROS": "#00C853",
#     "AGROPECUARIO": "#FFFFFF",
# }


# COLORES_RADAR_REGION = {
#     "EJE CAFETERO": "#00FFFF",
#     "BOGOTA - CUNDINAMARCA": "#FF00FF",
#     "ANTIOQUIA": "#39FF14",
#     "SUR Y ORIENTE": "#FF3131",
#     "CENTRO - ORIENTE": "#FF9933",
#     "COSTA PACIFICA": "#7B2CBF",
#     "COSTA ATLANTICA": "#CCFF00",
# }

COLORES_RADAR_MACROSECTOR = {
    "COMERCIO": "#0072B2",
    "SERVICIOS": "#56B4E9",
    "MANUFACTURA": "#CC79A7",
    "CONSTRUCCION": "#E69F00",
    "AGROPECUARIO": "#009E73",
    "MINERO-HIDROCARBUROS": "#D55E00",
}


COLORES_RADAR_REGION = {
    "BOGOTA - CUNDINAMARCA": "#1D3557",
    "ANTIOQUIA": "#457B9D",
    "COSTA PACIFICA": "#2A9D8F",
    "COSTA ATLANTICA": "#E63946",
    "CENTRO - ORIENTE": "#6A4C93",
    "SUR Y ORIENTE": "#BC6C25",
    "EJE CAFETERO": "#606C38",
}
# Para los Supervisores se utiliza una paleta más sobria,
# diferenciada y compatible con fondo oscuro.
COLORES_RADAR_SUPERVISOR = {
    "SUPERSOCIEDADES": "#4E79A7",
    "SUPERTRANSPORTE": "#59A14F",
    "SUPERSERVICIOS": "#F28E2B",
    "SUPERSALUD": "#B07AA1",
    "SUPERFINANCIERA": "#76B7B2",
    "SUPERVIGILANCIA": "#E15759",
}

COLORES_RADAR_SEGMENTO = {
    "MACROSECTOR": COLORES_RADAR_MACROSECTOR,
    "REGION": COLORES_RADAR_REGION,
    "SUPERVISOR": COLORES_RADAR_SUPERVISOR,
}

def _normalizar_segmento_radar(valor) -> str:
    """
    Normaliza nombres de segmentos para que las etiquetas del DataFrame
    coincidan con las claves de las paletas de colores.
    """

    if pd.isna(valor):
        return ""

    texto = str(valor).strip().upper()

    texto = unicodedata.normalize(
        "NFKD",
        texto,
    ).encode(
        "ascii",
        "ignore",
    ).decode(
        "ascii"
    )

    # Normalización específica para separadores
    texto = texto.replace("–", "-")
    texto = texto.replace("—", "-")
    texto = " ".join(texto.split())

    return texto


# ═══ PARCHE 2 (inicio) ═════════════════════════════════════════════════════
# El dataset original trae "CONSTRUCCIÓN" (con tilde), mientras que la
# paleta usa "CONSTRUCCION" (sin tilde, igual que new_eda_VII.py). Por eso
# el violín de Construcción no se dibujaba. Se normaliza el texto antes de
# comparar: mayúsculas, sin tildes y sin espacios extremos.
def _normalizar_macrosector(texto) -> str:
    if pd.isna(texto):
        return texto
    texto = str(texto).strip().upper()
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


# Nombres legibles (con tilde) para mostrar en el eje X de la gráfica KDE.
NOMBRES_KDE_MACROSECTOR = {
    "COMERCIO": "COMERCIO",
    "SERVICIOS": "SERVICIOS",
    "MANUFACTURA": "MANUFACTURA",
    "CONSTRUCCION": "CONSTRUCCIÓN",
    "AGROPECUARIO": "AGROPECUARIO",
    "MINERO-HIDROCARBUROS": "MINERO-HIDROCARBUROS",
}
# ═══ PARCHE 2 (fin) ════════════════════════════════════════════════════════


# Mismos rangos establecidos en el EDA para evitar que valores extremos
# compriman visualmente las distribuciones.
RANGOS_DISTRIBUCION = {
    "Ingresos Op.": (0.004, 500),
    "Ganancia/Pérdida": (-250, 400),
    "Total Activos": (0.06, 995),
    "Total Pasivos": (0, 995),
    "Total Patrimonio": (-250, 400),
}


def _resolver_columna_distribucion(
    df: pd.DataFrame,
    columna: str,
) -> str | None:

    if columna in df.columns:
        return columna

    # Compatibilidad especial con Ingresos Operacionales 2022
    # cuando el DataFrame no contiene el asterisco.
    if columna.endswith("*"):
        columna_sin_asterisco = columna[:-1]
        if columna_sin_asterisco in df.columns:
            return columna_sin_asterisco

    return None


def _distribucion_normalidad(df: pd.DataFrame):

    st.subheader("🔔 Distribución y Prueba de Normalidad")

    st.markdown(
        """
        Análisis de la distribución de las variables financieras para los
        **7.533 registros empresariales**, mediante histograma, curva normal
        teórica, gráfico Q-Q y densidad KDE por Macrosector.

        **Todas las cifras financieras de las gráficas están expresadas en
        Miles de Millones de COP.**
        """
    )

    # ─────────────────────────────────────────────────────────────────────
    # SELECTOR
    # ─────────────────────────────────────────────────────────────────────

    variable_label = st.selectbox(
        "Seleccionar variable:",
        list(VARIABLES_DISTRIBUCION.keys()),
        key="distribucion_variable_financiera",
    )

    anio = st.radio(
        "Seleccionar año:",
        [2024, 2023, 2022],
        horizontal=True,
        key="distribucion_anio_financiero",
    )

    columna = _resolver_columna_distribucion(
        df,
        VARIABLES_DISTRIBUCION[variable_label][anio],
    )

    if columna is None:
        st.error(
            f"No se encontró la columna correspondiente a "
            f"**{variable_label} — {anio}**."
        )
        return

    # ─────────────────────────────────────────────────────────────────────
    # PREPARACIÓN DE DATOS
    # ─────────────────────────────────────────────────────────────────────

    columnas_necesarias = [columna]

    if "MACROSECTOR" in df.columns:
        columnas_necesarias.append("MACROSECTOR")

    df_plot = df[columnas_necesarias].copy()

    df_plot[columna] = pd.to_numeric(
        df_plot[columna],
        errors="coerce",
    )

    df_plot["Valor"] = (
        df_plot[columna] / DIVISOR_MM_COP
    )

    df_plot = df_plot.dropna(subset=["Valor"])

    # Mismo rango del EDA para evitar que outliers extremos
    # compriman las gráficas.
    limite_inferior, limite_superior = RANGOS_DISTRIBUCION[
        variable_label
    ]

    df_plot = df_plot[
        (df_plot["Valor"] >= limite_inferior)
        & (df_plot["Valor"] <= limite_superior)
    ].copy()

    serie = df_plot["Valor"]

    if len(serie) < 3:
        st.warning(
            "No existen suficientes observaciones para construir "
            "las gráficas de distribución."
        )
        return

    media = serie.mean()
    desviacion = serie.std()

    # ─────────────────────────────────────────────────────────────────────
    # 1. HISTOGRAMA + CURVA NORMAL
    # ─────────────────────────────────────────────────────────────────────

    st.markdown(
        f"### 1️⃣ Histograma + Curva Normal: "
        f"{variable_label} {anio}"
    )

    col_hist, col_qq = st.columns(2)

    with col_hist:

        fig_hist = go.Figure()

        # ═══ PARCHE 3 (inicio) ═════════════════════════════════════════
        # En un histograma de Plotly, %{x} SIN formato devuelve el RANGO
        # del bin ("inicio - fin"), igual que el hover por defecto de
        # estadistica_descriptiva_II.py. Con formato (%{x:,.2f}) Plotly
        # devuelve solo el centro del bin; por eso no aparecía el rango.
        # El formato numérico del rango se controla con xhoverformat.
        # Los bins se fijan explícitamente para que coincidan con el
        # escalamiento de la curva normal teórica.
        n_bins = 45
        ancho_bin = (limite_superior - limite_inferior) / n_bins

        fig_hist.add_trace(
            go.Histogram(
                x=serie,
                xbins=dict(
                    start=limite_inferior,
                    end=limite_superior,
                    size=ancho_bin,
                ),
                name="Distribución Real",
                marker_color="#457B9D",
                opacity=0.82,
                xhoverformat=",.2f",
                hovertemplate=(
                    "<b>Rango:</b> %{x} Miles de Millones COP"
                    "<br><b>Frecuencia:</b> %{y:,} empresas"
                    "<extra></extra>"
                ),
            )
        )
        # ═══ PARCHE 3 (fin) ════════════════════════════════════════════

        if desviacion > 0:

            x_normal = np.linspace(
                limite_inferior,
                limite_superior,
                500,
            )

            y_normal = (
                norm.pdf(
                    x_normal,
                    media,
                    desviacion,
                )
                * len(serie)
                * ancho_bin  # PARCHE 4: mismo ancho de bin del histograma
            )

            fig_hist.add_trace(
                go.Scatter(
                    x=x_normal,
                    y=y_normal,
                    mode="lines",
                    name="Curva Normal Teórica",
                    line=dict(
                        color="#E63946",
                        width=3,
                        dash="dash",
                    ),
                    hovertemplate=(
                        "Miles de Millones COP: %{x:,.2f}"
                        "<extra></extra>"
                    ),
                )
            )

        fig_hist.update_layout(
            height=470,
            title=dict(
                text=(
                    f"Histograma + Curva Normal: "
                    f"{variable_label} {anio}"
                ),
                font=dict(size=17),
            ),
            xaxis=dict(
                title="Miles de Millones COP",
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title="Frecuencia",
                tickfont=dict(size=12),
            ),
            legend=dict(
                font=dict(size=11),
            ),
            margin=dict(
                l=70,
                r=30,
                t=75,
                b=70,
            ),
            bargap=0.04,
            # PARCHE 5: el rango se muestra solo al pasar el cursor
            # sobre el área de cada barra.
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
            ),
        )        

        _aplicar_hover_legible(fig_hist, tamano=14)  # PARCHE 8        

        st.plotly_chart(
            fig_hist,
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 2. Q-Q PLOT
    # ─────────────────────────────────────────────────────────────────────

    with col_qq:

        valores_ordenados = np.sort(serie.values)

        n = len(valores_ordenados)

        cuantiles_teoricos = norm.ppf(
            (np.arange(1, n + 1) - 0.5) / n
        )

        media_qq = valores_ordenados.mean()
        desviacion_qq = valores_ordenados.std()

        cuantiles_ajustados = (
            media_qq
            + desviacion_qq * cuantiles_teoricos
        )

        fig_qq = go.Figure()

        fig_qq.add_trace(
            go.Scatter(
                x=cuantiles_teoricos,
                y=valores_ordenados,
                mode="markers",
                name="Datos",
                marker=dict(
                    size=4,
                    color="#457B9D",
                    opacity=0.65,
                ),
                hovertemplate=(
                    "Cuantil teórico: %{x:.3f}"
                    "<br>Valor: %{y:,.2f}"
                    " Miles de Millones COP"
                    "<extra></extra>"
                ),
            )
        )

        fig_qq.add_trace(
            go.Scatter(
                x=cuantiles_teoricos,
                y=cuantiles_ajustados,
                mode="lines",
                name="Línea Normal",
                line=dict(
                    color="#E63946",
                    width=2,
                    dash="dash",
                ),
            )
        )

        fig_qq.update_layout(
            height=470,
            title=dict(
                text=(
                    f"Q-Q Plot: "
                    f"{variable_label} {anio}"
                ),
                font=dict(size=17),
            ),
            xaxis=dict(
                title="Cuantiles Teóricos",
                tickfont=dict(size=12),
            ),
            yaxis=dict(
                title="Cuantiles Muestrales "
                      "(Miles de Millones COP)",
                tickfont=dict(size=12),
            ),
            legend=dict(
                font=dict(size=11),
            ),
            margin=dict(
                l=80,
                r=30,
                t=75,
                b=70,
            ),
        )        

        _aplicar_hover_legible(fig_qq, tamano=14)  # PARCHE 9

        st.plotly_chart(
            fig_qq,
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # 3. DENSIDAD KDE POR MACROSECTOR
    # ═══ PARCHE 6 (inicio) — KDE con CONSTRUCCIÓN + tamaño/legibilidad ════
    # ─────────────────────────────────────────────────────────────────────

    st.markdown(
        "### 3️⃣ Densidad (KDE) por Macrosector"
    )

    st.caption(
        "Cada violín representa la distribución de la variable dentro "
        "de un Macrosector. Los colores se mantienen constantes en todo "
        "el proyecto para facilitar la comparación. "
        "**Todas las cifras están en Miles de Millones COP.**"
    )

    if "MACROSECTOR" not in df_plot.columns:
        st.warning(
            "No existe la columna `MACROSECTOR`; "
            "no es posible construir la distribución sectorial."
        )
        return

    # Normalización: "CONSTRUCCIÓN" → "CONSTRUCCION" (clave de la paleta).
    df_plot["MACROSECTOR_NORM"] = df_plot["MACROSECTOR"].map(
        _normalizar_macrosector
    )

    orden_macrosector = [
        macro
        for macro in COLORES_KDE_MACROSECTOR
        if macro in df_plot["MACROSECTOR_NORM"].dropna().unique()
    ]

    # Controles de visualización (no alteran los datos ni las unidades).
    col_alto, col_zoom = st.columns(2)

    with col_alto:
        altura_kde = st.slider(
            "Altura de la gráfica (píxeles):",
            min_value=700,
            max_value=1500,
            value=1000,
            step=50,
            key="kde_altura_grafica",
        )

    with col_zoom:
        zoom_kde = st.select_slider(
            "Zoom del eje Y (Miles de Millones COP):",
            options=[
                "Rango completo",
                "Percentil 99",
                "Percentil 95",
                "Percentil 90",
            ],
            value="Rango completo",
            key="kde_zoom_eje_y",
            help=(
                "Solo acerca la vista del eje Y. La densidad KDE se calcula "
                "siempre con todas las observaciones y las cifras siguen "
                "expresadas en Miles de Millones COP."
            ),
        )

    fig_kde = go.Figure()

    for macrosector in orden_macrosector:

        datos = df_plot.loc[
            df_plot["MACROSECTOR_NORM"] == macrosector,
            "Valor",
        ].dropna()

        if len(datos) < 3:
            continue

        color = COLORES_KDE_MACROSECTOR[macrosector]
        nombre = NOMBRES_KDE_MACROSECTOR.get(macrosector, macrosector)
        etiqueta_x = f"{nombre}<br>(n = {len(datos):,})".replace(",", ".")

        fig_kde.add_trace(
            go.Violin(
                x=[etiqueta_x] * len(datos),
                y=datos,
                name=nombre,
                line=dict(
                    color=color,
                    width=2.5,
                ),
                fillcolor=color,
                opacity=0.75,
                width=0.9,
                box=dict(
                    visible=True,
                    width=0.25,
                    fillcolor="rgba(255,255,255,0.85)",
                    line=dict(color="#1D3557", width=1.5),
                ),
                meanline=dict(
                    visible=True,
                    color="#1D3557",
                    width=2,
                ),
                points=False,
                scalemode="width",
                spanmode="hard",
                hoveron="violins+kde",
                hovertemplate=(
                    f"<b>{nombre}</b>"
                    "<br>Valor: %{y:,.2f} Miles de Millones COP"
                    "<extra></extra>"
                ),
            )
        )

    # Rango del eje Y según el zoom elegido (solo vista).
    percentiles_zoom = {
        "Percentil 99": 0.99,
        "Percentil 95": 0.95,
        "Percentil 90": 0.90,
    }

    rango_y = None
    if zoom_kde in percentiles_zoom:
        q = percentiles_zoom[zoom_kde]
        y_min = serie.min() if serie.min() >= 0 else serie.quantile(1 - q)
        y_max = serie.quantile(q)
        margen = (y_max - y_min) * 0.05
        rango_y = [y_min - margen, y_max + margen]

    fig_kde.update_layout(
        height=altura_kde,
        title=dict(
            text=(
                f"Distribución por Macrosector: "
                f"{variable_label} {anio} "
                f"<span style='font-size:16px'>(Miles de Millones COP)</span>"
            ),
            font=dict(size=24),
            x=0.01,
        ),
        xaxis=dict(
            title=dict(
                text="MACROSECTOR",
                font=dict(size=18),
            ),
            tickangle=0,
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            title=dict(
                text=f"{variable_label} — Miles de Millones COP",
                font=dict(size=18),
            ),
            tickfont=dict(size=16),
            tickformat=",.2~f",
            separatethousands=True,
            gridcolor="rgba(0,0,0,0.10)",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.35)",
            range=rango_y,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=15,
        ),
        showlegend=False,
        margin=dict(
            l=110,
            r=40,
            t=100,
            b=110,
        ),
        violingap=0.08,
        violinmode="overlay",
        annotations=[
            dict(
                text="Cifras en Miles de Millones COP",
                xref="paper",
                yref="paper",
                x=1,
                y=1.035,
                xanchor="right",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=14, color="#555555"),
            )
        ],
    )    

    _aplicar_hover_legible(fig_kde, tamano=15)  # PARCHE 10

    st.plotly_chart(
        fig_kde,
        use_container_width=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"kde_macrosector_{variable_label}_{anio}",
                "scale": 3,
            },
        },
    )

    st.caption(
        "Consejo: usa el ícono de pantalla completa (⛶) en la esquina "
        "superior derecha de la gráfica para verla a mayor tamaño. "
        "El zoom del eje Y solo cambia la vista; los valores siguen en "
        "Miles de Millones COP."
    )
    # ═══ PARCHE 6 (fin) ═══════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────────────
    # PRUEBA DE NORMALIDAD
    # ─────────────────────────────────────────────────────────────────────

    if len(serie) >= 3:

        muestra_shapiro = serie.sample(
            min(5000, len(serie)),
            random_state=42,
        )

        _, p_valor = stats.shapiro(
            muestra_shapiro
        )

        if p_valor > 0.05:
            mensaje_normalidad = (
                "La prueba no rechaza la hipótesis de "
                "normalidad al 5%."
            )
        else:
            mensaje_normalidad = (
                "La prueba rechaza la hipótesis de "
                "normalidad al 5%."
            )

        st.info(
            f"**Prueba Shapiro-Wilk:** "
            f"p-valor = {p_valor:.6f} → "
            f"{mensaje_normalidad}"
        )

    st.caption(
        f"Rango utilizado para la visualización: "
        f"{limite_inferior:,.3f} a "
        f"{limite_superior:,.0f} Miles de Millones COP. "
        f"Observaciones incluidas: {len(serie):,}."
    )
    
# ═══════════════════════════════════════════════════════════════════════════
# ═══ PARCHE 12 (inicio) — PESTAÑA: 📦 PERCENTILES Y CUARTILES ══════════════
# Base: _percentiles_cuartiles() de estadistica_descriptiva_II.py, ampliada a
# las 5 variables financieras × 3 años (2022, 2023, 2024).
#
# Tratamiento de cifras (el mismo de new_eda_VII.py):
#   · Los valores originales están en MILES DE COP → se dividen entre
#     DIVISOR_MM_COP (1.000.000) para expresarlos en MILES DE MILLONES COP.
#   · Se excluyen los outliers extremos (empresas del orden de billones de
#     COP) con el mismo rango fijo por variable (RANGOS_DISTRIBUCION), p. ej.
#     las 3 empresas con Ingresos Operacionales 2022 de decenas de billones.
# ═══════════════════════════════════════════════════════════════════════════

PERCENTILES_TABLA = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]

# Colores fijos por año (sobrios, de menor a mayor intensidad).
COLORES_ANIO = {
    2022: "#8D99AE",
    2023: "#457B9D",
    2024: "#1D3557",
}


def _serie_percentiles(
    df: pd.DataFrame,
    variable_label: str,
    anio: int,
    excluir_outliers: bool = True,
) -> tuple[pd.Series, int]:
    """
    Serie de una variable/año en Miles de Millones COP.
    Retorna (serie_filtrada, total_registros_con_dato).
    """
    columna = _resolver_columna_distribucion(
        df, VARIABLES_DISTRIBUCION[variable_label][anio]
    )
    if columna is None:
        return pd.Series(dtype=float), 0

    serie = (
        pd.to_numeric(df[columna], errors="coerce") / DIVISOR_MM_COP
    ).dropna()
    total = len(serie)

    if excluir_outliers:
        lim_inf, lim_sup = RANGOS_DISTRIBUCION[variable_label]
        serie = serie[(serie >= lim_inf) & (serie <= lim_sup)]

    return serie, total


def _tabla_percentiles(
    df: pd.DataFrame,
    anios: list[int],
    excluir_outliers: bool,
) -> pd.DataFrame:
    """Tabla de percentiles, cuartiles e IQR para cada variable y año."""
    filas = []

    for anio in anios:
        for variable_label in VARIABLES_DISTRIBUCION:
            serie, total = _serie_percentiles(
                df, variable_label, anio, excluir_outliers
            )
            fila = {"Variable": f"{variable_label} {anio}"}

            if len(serie) == 0:
                fila.update({f"P{p}": np.nan for p in PERCENTILES_TABLA})
                fila.update({"Q1": np.nan, "Q2 (Mediana)": np.nan,
                             "Q3": np.nan, "IQR (Q3 − Q1)": np.nan,
                             "N incluidas": 0, "N total": total})
                filas.append(fila)
                continue

            valores = np.percentile(serie, PERCENTILES_TABLA)
            fila.update({
                f"P{p}": round(float(v), 2)
                for p, v in zip(PERCENTILES_TABLA, valores)
            })

            q1, q2, q3 = np.percentile(serie, [25, 50, 75])
            fila.update({
                "Q1": round(float(q1), 2),
                "Q2 (Mediana)": round(float(q2), 2),
                "Q3": round(float(q3), 2),
                "IQR (Q3 − Q1)": round(float(q3 - q1), 2),
                "N incluidas": int(len(serie)),
                "N total": int(total),
            })
            filas.append(fila)

    return pd.DataFrame(filas).set_index("Variable")

# ═══ PARCHE 14 (inicio) — Utilidades para Box Plots legibles ═════════════

# Tonalidad por año sobre el color base del Macrosector (new_eda_VII.py):
#   2022 → más claro · 2023 → color original · 2024 → más oscuro.
TONO_ANIO_MACROSECTOR = {
    2022: 0.45,   # mezcla 45% con blanco
    2023: 0.00,   # color base original
    2024: -0.35,  # mezcla 35% con negro
}


def _tono_color(hex_color: str, factor: float) -> str:
    """
    Aclara (factor > 0, hacia blanco) u oscurece (factor < 0, hacia negro)
    un color HEX conservando su matiz.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    if factor >= 0:
        r, g, b = (int(c + (255 - c) * factor) for c in (r, g, b))
    else:
        r, g, b = (int(c * (1 + factor)) for c in (r, g, b))
    return f"#{r:02X}{g:02X}{b:02X}"


def _limites_bigotes(serie: pd.Series) -> tuple[float, float] | None:
    """
    Extremos de los bigotes de Tukey (1,5 × IQR) en Miles de Millones COP.
    Se usan para acercar el eje Y y que las cajas se vean grandes.
    """
    serie = pd.Series(serie).dropna()
    if len(serie) == 0:
        return None
    q1, q3 = np.percentile(serie, [25, 75])
    iqr = q3 - q1
    inferior = serie[serie >= q1 - 1.5 * iqr].min()
    superior = serie[serie <= q3 + 1.5 * iqr].max()
    return float(inferior), float(superior)


def _rango_y_cajas(limites: list[tuple[float, float]]) -> list[float] | None:
    """Rango Y común a partir de varios bigotes, con 8% de margen."""
    limites = [l for l in limites if l is not None]
    if not limites:
        return None
    y_min = min(l[0] for l in limites)
    y_max = max(l[1] for l in limites)
    margen = (y_max - y_min) * 0.08 or 1
    return [y_min - margen, y_max + margen]
# ═══ PARCHE 14 (fin) ═════════════════════════════════════════════════════

def _percentiles_cuartiles(df: pd.DataFrame):

    st.subheader("📦 Percentiles y Cuartiles (Miles de Millones COP)")

    st.markdown(
        """
        Percentiles (P0 a P100) y cuartiles (Q1, Q2 y Q3) de las cinco
        variables financieras para **2022, 2023 y 2024**.

        **Todas las cifras están expresadas en Miles de Millones de COP**
        (valor original en miles de COP ÷ 1.000.000), con el mismo
        tratamiento de outliers extremos aplicado en el EDA.
        """
    )

    # ─────────────────────────────────────────────────────────────────────
    # SELECTORES
    # ─────────────────────────────────────────────────────────────────────

    col_sel_anio, col_sel_out = st.columns([3, 2])

    with col_sel_anio:
        periodo = st.radio(
            "Seleccionar período:",
            ["2024", "2023", "2022", "Los tres años"],
            horizontal=True,
            key="percentiles_periodo",
        )

    with col_sel_out:
        excluir_outliers = st.checkbox(
            "Excluir outliers extremos (rango fijo del EDA)",
            value=True,
            key="percentiles_excluir_outliers",
            help=(
                "Aplica los mismos límites por variable usados en el EDA "
                "(p. ej., Ingresos Op.: 0,004 a 500 Miles de Millones COP). "
                "Así, las empresas del orden de billones de COP no "
                "distorsionan los percentiles."
            ),
        )

    anios = [2022, 2023, 2024] if periodo == "Los tres años" else [int(periodo)]

    # ─────────────────────────────────────────────────────────────────────
    # 1. TABLA DE PERCENTILES Y CUARTILES
    # ─────────────────────────────────────────────────────────────────────

    st.markdown("### 1️⃣ Tabla de Percentiles y Cuartiles")

    tabla = _tabla_percentiles(df, anios, excluir_outliers)

    columnas_pct = [f"P{p}" for p in PERCENTILES_TABLA]
    columnas_cuartiles = ["Q1", "Q2 (Mediana)", "Q3", "IQR (Q3 − Q1)"]
    columnas_n = ["N incluidas", "N total"]

    tabla_estilizada = (
        tabla.style
        .format("{:,.2f}", subset=columnas_pct + columnas_cuartiles, na_rep="—")
        .format("{:,.0f}", subset=columnas_n)
        .background_gradient(cmap="Blues", axis=1, subset=columnas_pct)
        .set_properties(**{"color": "#000000"}, subset=columnas_cuartiles + columnas_n)
    )

    st.dataframe(
        tabla_estilizada,
        use_container_width=True,
        height=250 if len(anios) == 1 else 600,
    )

    st.caption(
        "Valores en Miles de Millones COP. P50 = Q2 = Mediana. "
        "IQR (rango intercuartílico) = Q3 − Q1: concentra el 50% central "
        "de las empresas. «N incluidas» son las empresas dentro del rango "
        "fijo del EDA; «N total» son las empresas con dato en esa variable."
    )

    # ─────────────────────────────────────────────────────────────────────
    # 2. BOX PLOTS COMPARATIVOS POR VARIABLE Y AÑO
    # ─────────────────────────────────────────────────────────────────────

    st.markdown(
        "### 2️⃣ Box Plots Comparativos — Variables Financieras "
        "(Miles de Millones COP)"
    )

    # fig_box = go.Figure()

    # for anio in anios:
    #     x_vals, y_vals = [], []
    #     for variable_label in VARIABLES_DISTRIBUCION:
    #         serie, _ = _serie_percentiles(
    #             df, variable_label, anio, excluir_outliers
    #         )
    #         x_vals.extend([variable_label] * len(serie))
    #         y_vals.extend(serie.tolist())

    #     fig_box.add_trace(
    #         go.Box(
    #             x=x_vals,
    #             y=y_vals,
    #             name=str(anio),
    #             marker_color=COLORES_ANIO[anio],
    #             line=dict(width=2),
    #             boxpoints=False,
    #             boxmean=True,
    #             hovertemplate=(
    #                 f"<b>Año {anio}</b>"
    #                 "<br>%{x}"
    #                 "<br>Valor: %{y:,.2f} Miles de Millones COP"
    #                 "<extra></extra>"
    #             ),
    #         )
    #     )

    # fig_box.update_layout(
    #     height=620,
    #     boxmode="group",
    #     title=dict(
    #         text=(
    #             "Distribución Comparativa de Variables Financieras — "
    #             + (", ".join(str(a) for a in anios))
    #         ),
    #         font=dict(size=20),
    #     ),
    #     xaxis=dict(
    #         title=dict(text="Variable financiera", font=dict(size=16)),
    #         tickfont=dict(size=14),
    #     ),
    #     yaxis=dict(
    #         title=dict(text="Miles de Millones COP", font=dict(size=16)),
    #         tickfont=dict(size=14),
    #         tickformat=",.2~f",
    #         zeroline=True,
    #         zerolinecolor="rgba(0,0,0,0.35)",
    #     ),
    #     legend=dict(
    #         title=dict(text="Año"),
    #         font=dict(size=13),
    #         orientation="h",
    #         yanchor="bottom",
    #         y=1.02,
    #         xanchor="right",
    #         x=1,
    #     ),
    #     margin=dict(l=90, r=30, t=100, b=80),
    # )

    # _aplicar_hover_legible(fig_box, tamano=14)

    # st.plotly_chart(fig_box, use_container_width=True)

    # st.caption(
    #     "La caja va de Q1 a Q3; la línea interior es la mediana (Q2) y la "
    #     "línea punteada, la media. Los bigotes llegan al mínimo y al máximo "
    #     "de las empresas incluidas (Miles de Millones COP)."
    # )
    
# ========================================================================================
    
    # ═══ PARCHE 15 (inicio) — Box Plots Comparativos grandes y legibles ══
    # Un panel por variable (cada una con su propio eje Y en Miles de
    # Millones COP), para que variables de escala pequeña como
    # Ganancia/Pérdida no queden aplastadas junto a Ingresos o Activos.
    # Con "Excluir outliers extremos" activo, los bigotes siguen la regla
    # de Tukey (1,5 × IQR) y el eje Y se acerca a las cajas.
    variables_box = list(VARIABLES_DISTRIBUCION.keys())

    fig_box = make_subplots(
        rows=1,
        cols=len(variables_box),
        # subplot_titles=variables_box,
        # PARCHE 17-A: sin subplot_titles (el nombre va en el eje X)
        horizontal_spacing=0.055,
    )

    for i, variable_label in enumerate(variables_box, start=1):
        limites_panel = []

        for anio in anios:
            serie, _ = _serie_percentiles(
                df, variable_label, anio, excluir_outliers
            )
            if len(serie) == 0:
                continue

            if excluir_outliers:
                limites_panel.append(_limites_bigotes(serie))

            fig_box.add_trace(
                go.Box(
                    x=[str(anio)] * len(serie),
                    y=serie,
                    name=str(anio),
                    legendgroup=str(anio),
                    showlegend=(i == 1),
                    marker=dict(
                        color=COLORES_ANIO[anio],
                        size=3,
                        opacity=0.35,
                    ),
                    fillcolor=_tono_color(COLORES_ANIO[anio], 0.25),
                    line=dict(color=COLORES_ANIO[anio], width=2.5),
                    boxpoints="outliers" if excluir_outliers else False,
                    boxmean=True,
                    width=0.75,
                    hovertemplate=(
                        f"<b>{variable_label} — {anio}</b>"
                        "<br>Valor: %{y:,.2f} Miles de Millones COP"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=i,
            )

        fig_box.update_yaxes(
            tickformat=",.2~f",
            tickfont=dict(size=14),
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.35)",
            gridcolor="rgba(0,0,0,0.10)",
            range=_rango_y_cajas(limites_panel) if excluir_outliers else None,
            row=1,
            col=i,
        )
        # fig_box.update_xaxes(tickfont=dict(size=14), row=1, col=i)
        
        # PARCHE 17-B: nombre de la variable debajo de sus tres años
        fig_box.update_xaxes(
            tickfont=dict(size=14),
            title=dict(
                text=f"<b>{variable_label}</b>",
                font=dict(size=16, color="#FFFFFF"),
                standoff=12,
            ),
            row=1,
            col=i,
        )

    fig_box.update_yaxes(
        title=dict(text="Miles de Millones COP", font=dict(size=16)),
        row=1,
        col=1,
    )
    # fig_box.update_annotations(font=dict(size=16, color="#000000"))
    
    # PARCHE 17-C: se eliminó update_annotations (ya no hay títulos arriba)  

    fig_box.update_layout(
        height=720,
        title=dict(
            text=(
                "Distribución Comparativa de Variables Financieras — "
                + ", ".join(str(a) for a in anios)
                + " (Miles de Millones COP)"
            ),
            font=dict(size=20),
        ),
        # legend=dict(
        #     title=dict(text="Año"),
        #     font=dict(size=14),
        #     orientation="h",
        #     yanchor="top",
        #     y=-0.08,
        #     xanchor="center",
        #     x=0.5,
        # ),
        # boxgap=0.15,
        # margin=dict(l=90, r=30, t=110, b=110),
        
        # PARCHE 17-D: leyenda sin el título "Año", debajo de los nombres
        legend=dict(
            font=dict(size=14),
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
        ),
        boxgap=0.15,
        margin=dict(l=90, r=30, t=90, b=140),
    )

    _aplicar_hover_legible(fig_box, tamano=14)

    st.plotly_chart(fig_box, use_container_width=True)

    if excluir_outliers:
        st.caption(
            "La caja va de Q1 a Q3; la línea interior es la mediana (Q2) y "
            "la línea punteada, la media. Los bigotes llegan hasta 1,5 × IQR "
            "(regla de Tukey) y el eje Y se acerca a las cajas para ganar "
            "legibilidad; los puntos tenues son valores atípicos dentro del "
            "rango del EDA (arrastra el eje Y o usa el zoom para explorarlos). "
            "Cifras en Miles de Millones COP."
        )
    else:
        st.caption(
            "La caja va de Q1 a Q3; la línea interior es la mediana (Q2) y "
            "la línea punteada, la media. Los bigotes llegan al mínimo y al "
            "máximo de todas las empresas (Miles de Millones COP)."
        )
    # ═══ PARCHE 15 (fin) ══════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────────────
    # 3. CUARTILES POR MACROSECTOR (variable seleccionada, 3 años)
    # ─────────────────────────────────────────────────────────────────────

    st.markdown("### 3️⃣ Cuartiles por Macrosector")

    if "MACROSECTOR" not in df.columns:
        st.warning(
            "No existe la columna `MACROSECTOR`; no es posible construir "
            "los cuartiles sectoriales."
        )
        return

    variable_macro = st.selectbox(
        "Seleccionar variable:",
        list(VARIABLES_DISTRIBUCION.keys()),
        key="percentiles_variable_macrosector",
    )

    macro_norm = df["MACROSECTOR"].map(_normalizar_macrosector)
 
    # ═══ PARCHE 16 (inicio) — Cuartiles por Macrosector: colores del EDA ═
    # Cada Macrosector conserva su color de new_eda_VII.py
    # (COLORES_KDE_MACROSECTOR) y cada año usa una tonalidad distinta:
    # 2022 más claro, 2023 color original y 2024 más oscuro.
    orden_nombres = [
        NOMBRES_KDE_MACROSECTOR[m] for m in COLORES_KDE_MACROSECTOR
    ]

    fig_macro = go.Figure()
    limites_macro = []

    for anio in anios:
        columna = _resolver_columna_distribucion(
            df, VARIABLES_DISTRIBUCION[variable_macro][anio]
        )
        if columna is None:
            continue

        df_m = pd.DataFrame({
            "MACRO": macro_norm,
            "Valor": pd.to_numeric(df[columna], errors="coerce") / DIVISOR_MM_COP,
        }).dropna()

        if excluir_outliers:
            lim_inf, lim_sup = RANGOS_DISTRIBUCION[variable_macro]
            df_m = df_m[(df_m["Valor"] >= lim_inf) & (df_m["Valor"] <= lim_sup)]

        factor_tono = TONO_ANIO_MACROSECTOR.get(anio, 0.0)

        for macro, color_base in COLORES_KDE_MACROSECTOR.items():
            datos = df_m.loc[df_m["MACRO"] == macro, "Valor"]
            if len(datos) == 0:
                continue

            if excluir_outliers:
                limites_macro.append(_limites_bigotes(datos))

            nombre = NOMBRES_KDE_MACROSECTOR[macro]
            color_anio = _tono_color(color_base, factor_tono)

            fig_macro.add_trace(
                go.Box(
                    x=[nombre] * len(datos),
                    y=datos,
                    name=f"{nombre} {anio}",
                    offsetgroup=str(anio),
                    alignmentgroup="macrosector",
                    showlegend=False,
                    marker=dict(color=color_anio, size=3, opacity=0.35),
                    fillcolor=color_anio,
                    opacity=0.90,
                    line=dict(
                        color=_tono_color(color_base, -0.45),
                        width=2,
                    ),
                    boxpoints="outliers" if excluir_outliers else False,
                    boxmean=True,
                    hovertemplate=(
                        f"<b>{nombre} — {anio}</b>"
                        "<br>Valor: %{y:,.2f} Miles de Millones COP"
                        "<extra></extra>"
                    ),
                )
            )

    # Leyenda de tonalidades por año (gris de referencia).
    for anio in anios:
        tono = {2022: "claro", 2023: "original", 2024: "oscuro"}.get(anio, "")
        fig_macro.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"{anio} (tono {tono})",
                marker=dict(
                    symbol="square",
                    size=14,
                    color=_tono_color("#6B6B6B", TONO_ANIO_MACROSECTOR.get(anio, 0.0)),
                    line=dict(color="#000000", width=1),
                ),
                hoverinfo="skip",
            )
        )

    fig_macro.update_layout(
        height=760,
        boxmode="group",
        boxgap=0.12,
        boxgroupgap=0.08,
        title=dict(
            text=f"Cuartiles por Macrosector: {variable_macro} (Miles de Millones COP)",
            font=dict(size=20),
        ),
        xaxis=dict(
            title=dict(text="MACROSECTOR", font=dict(size=16)),
            tickfont=dict(size=15),
            categoryorder="array",
            categoryarray=orden_nombres,
        ),
        yaxis=dict(
            title=dict(text=f"{variable_macro} — Miles de Millones COP", font=dict(size=16)),
            tickfont=dict(size=14),
            tickformat=",.2~f",
            zeroline=True,
            zerolinecolor="rgba(0,0,0,0.35)",
            gridcolor="rgba(0,0,0,0.10)",
            range=_rango_y_cajas(limites_macro) if excluir_outliers else None,
        ),
        legend=dict(
            title=dict(text="Año"),
            font=dict(size=14),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=90, r=30, t=110, b=80),
    )

    _aplicar_hover_legible(fig_macro, tamano=14)

    st.plotly_chart(fig_macro, use_container_width=True)

    st.caption(
        "Colores por Macrosector iguales a los del EDA; la tonalidad indica "
        "el año (claro = 2022, original = 2023, oscuro = 2024). "
        + ("Bigotes hasta 1,5 × IQR y eje Y acercado a las cajas. "
           if excluir_outliers else "")
        + "Cifras en Miles de Millones COP."
    )
    # ═══ PARCHE 16 (fin) ══════════════════════════════════════════════════

    # fig_macro = go.Figure()

    # for anio in anios:
    #     columna = _resolver_columna_distribucion(
    #         df, VARIABLES_DISTRIBUCION[variable_macro][anio]
    #     )
    #     if columna is None:
    #         continue

    #     df_m = pd.DataFrame({
    #         "MACRO": macro_norm,
    #         "Valor": pd.to_numeric(df[columna], errors="coerce") / DIVISOR_MM_COP,
    #     }).dropna()

    #     if excluir_outliers:
    #         lim_inf, lim_sup = RANGOS_DISTRIBUCION[variable_macro]
    #         df_m = df_m[(df_m["Valor"] >= lim_inf) & (df_m["Valor"] <= lim_sup)]

    #     df_m = df_m[df_m["MACRO"].isin(COLORES_KDE_MACROSECTOR.keys())]
    #     df_m["Macrosector"] = df_m["MACRO"].map(NOMBRES_KDE_MACROSECTOR)

    #     fig_macro.add_trace(
    #         go.Box(
    #             x=df_m["Macrosector"],
    #             y=df_m["Valor"],
    #             name=str(anio),
    #             marker_color=COLORES_ANIO[anio],
    #             line=dict(width=2),
    #             boxpoints=False,
    #             boxmean=True,
    #             hovertemplate=(
    #                 f"<b>Año {anio}</b>"
    #                 "<br>%{x}"
    #                 "<br>Valor: %{y:,.2f} Miles de Millones COP"
    #                 "<extra></extra>"
    #             ),
    #         )
    #     )

    # fig_macro.update_layout(
    #     height=620,
    #     boxmode="group",
    #     title=dict(
    #         text=f"Cuartiles por Macrosector: {variable_macro} (Miles de Millones COP)",
    #         font=dict(size=20),
    #     ),
    #     xaxis=dict(
    #         title=dict(text="MACROSECTOR", font=dict(size=16)),
    #         tickfont=dict(size=14),
    #         categoryorder="array",
    #         categoryarray=[
    #             NOMBRES_KDE_MACROSECTOR[m] for m in COLORES_KDE_MACROSECTOR
    #         ],
    #     ),
    #     yaxis=dict(
    #         title=dict(text=f"{variable_macro} — Miles de Millones COP", font=dict(size=16)),
    #         tickfont=dict(size=14),
    #         tickformat=",.2~f",
    #         zeroline=True,
    #         zerolinecolor="rgba(0,0,0,0.35)",
    #     ),
    #     legend=dict(
    #         title=dict(text="Año"),
    #         font=dict(size=13),
    #         orientation="h",
    #         yanchor="bottom",
    #         y=1.02,
    #         xanchor="right",
    #         x=1,
    #     ),
    #     margin=dict(l=90, r=30, t=100, b=80),
    # )

    # _aplicar_hover_legible(fig_macro, tamano=14)

    # st.plotly_chart(fig_macro, use_container_width=True)

    # Tabla de cuartiles por Macrosector y año (Miles de Millones COP)
    filas_macro = []
    for anio in anios:
        columna = _resolver_columna_distribucion(
            df, VARIABLES_DISTRIBUCION[variable_macro][anio]
        )
        if columna is None:
            continue
        valores = pd.to_numeric(df[columna], errors="coerce") / DIVISOR_MM_COP
        for macro in COLORES_KDE_MACROSECTOR:
            s = valores[macro_norm == macro].dropna()
            if excluir_outliers:
                lim_inf, lim_sup = RANGOS_DISTRIBUCION[variable_macro]
                s = s[(s >= lim_inf) & (s <= lim_sup)]
            if len(s) == 0:
                continue
            q1, q2, q3 = np.percentile(s, [25, 50, 75])
            filas_macro.append({
                "Macrosector": NOMBRES_KDE_MACROSECTOR[macro],
                "Año": anio,
                "N": len(s),
                "Q1": q1,
                "Q2 (Mediana)": q2,
                "Q3": q3,
                "IQR (Q3 − Q1)": q3 - q1,
            })

    if filas_macro:
        tabla_macro = pd.DataFrame(filas_macro).set_index(["Macrosector", "Año"])
        with st.expander("📋 Ver tabla de cuartiles por Macrosector (Miles de Millones COP)"):
            st.dataframe(
                tabla_macro.style.format(
                    {"N": "{:,.0f}", "Q1": "{:,.2f}", "Q2 (Mediana)": "{:,.2f}",
                     "Q3": "{:,.2f}", "IQR (Q3 − Q1)": "{:,.2f}"}
                ),
                use_container_width=True,
            )

    lim_inf, lim_sup = RANGOS_DISTRIBUCION[variable_macro]
    st.caption(
        f"Rango fijo del EDA para {variable_macro}: {lim_inf:,} a {lim_sup:,} "
        f"Miles de Millones COP"
        + (" (outliers extremos excluidos)." if excluir_outliers
           else " (desactivado: se incluyen todas las empresas).")
    )
# ═══ PARCHE 12 (fin) ══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# ═══ PARCHE 19 (inicio) — PESTAÑA: 📏 MÉTRICAS FINANCIERAS DERIVADAS ═══════
# Base: _metricas_derivadas() de estadistica_descriptiva_II.py, ampliada a
# 2022, 2023 y 2024 y con ROA, ROE, Margen Neto y Solvencia.
#
# Tratamiento de cifras (el mismo de new_eda_VII.py y de las pestañas
# anteriores):
#   · Cada componente se lleva a MILES DE MILLONES COP (÷ DIVISOR_MM_COP).
#   · Una empresa entra en el cálculo de un ratio solo si sus DOS
#     componentes están dentro del rango fijo del EDA (RANGOS_DISTRIBUCION);
#     así se excluyen las empresas del orden de billones de COP.
#   · Además se excluyen los ratios extremos (RANGOS_RATIOS), producidos por
#     denominadores muy cercanos a cero (p. ej. patrimonios casi nulos).
#   · Siempre se exige denominador > 0 (Activos, Ingresos y Patrimonio).
# ═══════════════════════════════════════════════════════════════════════════

ANIOS_RATIOS = [2022, 2023, 2024]

# RATIOS_FINANCIEROS = {
#     "ROA (%)": {
#         "numerador": "Ganancia/Pérdida",
#         "denominador": "Total Activos",
#         "factor": 100,
#         "formula": "Ganancia/Pérdida ÷ Total Activos × 100",
#     },
#     "ROE (%)": {
#         "numerador": "Ganancia/Pérdida",
#         "denominador": "Total Patrimonio",
#         "factor": 100,
#         "formula": "Ganancia/Pérdida ÷ Total Patrimonio × 100",
#     },
#     "Margen Neto (%)": {
#         "numerador": "Ganancia/Pérdida",
#         "denominador": "Ingresos Op.",
#         "factor": 100,
#         "formula": "Ganancia/Pérdida ÷ Ingresos Op. × 100",
#     },
#     "Solvencia (Pasivos / Patrimonio)": {
#         "numerador": "Total Pasivos",
#         "denominador": "Total Patrimonio",
#         "factor": 1,
#         "formula": "Total Pasivos ÷ Total Patrimonio (veces)",
#     },
# }

# # Rango fijo de visualización/cálculo para cada ratio (≈ P1–P99 del dataset).
# RANGOS_RATIOS = {
#     "ROA (%)": (-50, 50),
#     "ROE (%)": (-150, 150),
#     "Margen Neto (%)": (-100, 100),
#     "Solvencia (Pasivos / Patrimonio)": (0, 40),
# }

# # Tinte suave por indicador para facilitar la lectura del cuadro.
# FONDO_RATIO = {
#     "ROA (%)": "#EAF2FB",
#     "ROE (%)": "#EAF7F1",
#     "Margen Neto (%)": "#FDF3E6",
#     "Solvencia (Pasivos / Patrimonio)": "#F4EEF8",
# }

# ═══ PARCHE 20 (inicio) — 5 ratios: se agrega Solvencia y Apalancamiento ═
RATIOS_FINANCIEROS = {
    "ROA (%)": {
        "numerador": "Ganancia/Pérdida",
        "denominador": "Total Activos",
        "factor": 100,
        "formula": "Ganancia/Pérdida ÷ Total Activos × 100",
        "lectura": "Rentabilidad de los activos. Más alto = mejor.",
        "mayor_mejor": True,
    },
    "ROE (%)": {
        "numerador": "Ganancia/Pérdida",
        "denominador": "Total Patrimonio",
        "factor": 100,
        "formula": "Ganancia/Pérdida ÷ Total Patrimonio × 100",
        "lectura": "Rentabilidad del patrimonio de los socios. Más alto = mejor.",
        "mayor_mejor": True,
    },
    "Margen Neto (%)": {
        "numerador": "Ganancia/Pérdida",
        "denominador": "Ingresos Op.",
        "factor": 100,
        "formula": "Ganancia/Pérdida ÷ Ingresos Op. × 100",
        "lectura": "Utilidad por cada $100 de ingresos. Más alto = mejor.",
        "mayor_mejor": True,
    },
    "Solvencia (Activos / Pasivos)": {
        "numerador": "Total Activos",
        "denominador": "Total Pasivos",
        "factor": 1,
        "formula": "Total Activos ÷ Total Pasivos (veces)",
        "lectura": (
            "Pesos de activos por cada peso de deuda. Más alto = mejor. "
            "< 1: insolvencia técnica · 1–1,5: ajustada · 1,5–2: aceptable · "
            "> 2: holgada (umbrales orientativos, dependen del sector)."
        ),
        "mayor_mejor": True,
    },
    "Apalancamiento (Pasivos / Patrimonio)": {
        "numerador": "Total Pasivos",
        "denominador": "Total Patrimonio",
        "factor": 1,
        "formula": "Total Pasivos ÷ Total Patrimonio (veces)",
        "lectura": (
            "Pesos de deuda por cada peso aportado por los socios. "
            "Más alto = más riesgo (1 = pasivo igual al patrimonio)."
        ),
        "mayor_mejor": False,
    },
}

# Rango fijo de cálculo/visualización para cada ratio (≈ P1–P99 del dataset).
RANGOS_RATIOS = {
    "ROA (%)": (-50, 50),
    "ROE (%)": (-150, 150),
    "Margen Neto (%)": (-100, 100),
    "Solvencia (Activos / Pasivos)": (0, 50),
    "Apalancamiento (Pasivos / Patrimonio)": (0, 40),
}

# Estadísticos disponibles en las segmentaciones.
ESTADISTICOS_RATIOS = {
    "Mediana": "median",
    "Media": "mean",
    "Desv. Estándar": "std",
    "Mínimo": "min",
    "Máximo": "max",
}

# Expresión gramatical para los títulos ("Evolución de la Mediana"...).
ESTADISTICO_TITULO = {
    "Mediana": "de la Mediana",
    "Media": "de la Media",
    "Desv. Estándar": "de la Desviación Estándar",
    "Mínimo": "del Mínimo",
    "Máximo": "del Máximo",
}

# Comparaciones entre años para los Indicadores Clave (inicial, final).
COMPARACIONES_ANIOS = {
    "2023 → 2024": (2023, 2024),
    "2022 → 2023": (2022, 2023),
    "2022 → 2024": (2022, 2024),
}

# Fondo del cuadro de estadísticos: alto contraste con texto oscuro.
FONDO_RATIO = {
    "ROA (%)": "#DCEBFA",
    "ROE (%)": "#DDF3E8",
    "Margen Neto (%)": "#FCEBD5",
    "Solvencia (Activos / Pasivos)": "#E6E1F5",
    "Apalancamiento (Pasivos / Patrimonio)": "#F9DEDE",
}


def _estadistico_serie(serie: pd.Series, estadistico: str) -> float:
    """Calcula el estadístico elegido (Mediana, Media, Desv., Mín., Máx.)."""
    serie = serie.dropna()
    if len(serie) == 0 or (estadistico == "Desv. Estándar" and len(serie) < 2):
        return np.nan
    return float(getattr(serie, ESTADISTICOS_RATIOS[estadistico])())


def _nombre_ratio_html(ratio: str) -> str:
    """'Solvencia (Activos / Pasivos)' → '<b>Solvencia</b><br>(Activos / Pasivos)'."""
    if " (" in ratio and "/" in ratio:
        nombre, resto = ratio.split(" (", 1)
        return f"<b>{nombre}</b><br>({resto}"
    return f"<b>{ratio}</b>"
# ═══ PARCHE 20 (fin) ═════════════════════════════════════════════════════


def _es_porcentaje(ratio: str) -> bool:
    return "%" in ratio


def _fmt_ratio(valor: float, ratio: str) -> str:
    if pd.isna(valor):
        return "—"
    if _es_porcentaje(ratio):
        return f"{valor:,.2f} %"
    return f"{valor:,.2f} veces"


def _variable_mm_cop(
    df: pd.DataFrame,
    variable_label: str,
    anio: int,
    excluir_outliers: bool,
) -> pd.Series:
    """Variable financiera en Miles de Millones COP (NaN si queda excluida)."""
    columna = _resolver_columna_distribucion(
        df, VARIABLES_DISTRIBUCION[variable_label][anio]
    )
    if columna is None:
        return pd.Series(np.nan, index=df.index)

    serie = pd.to_numeric(df[columna], errors="coerce") / DIVISOR_MM_COP

    if excluir_outliers:
        lim_inf, lim_sup = RANGOS_DISTRIBUCION[variable_label]
        serie = serie.where((serie >= lim_inf) & (serie <= lim_sup))

    return serie


def _calcular_ratios(
    df: pd.DataFrame,
    anio: int,
    excluir_outliers: bool,
) -> pd.DataFrame:
    """
    Ratios por empresa para un año. Cada columna es un ratio; NaN cuando la
    empresa no cumple las condiciones de cálculo.
    """
    salida = pd.DataFrame(index=df.index)

    if "MACROSECTOR" in df.columns:
        salida["MACRO"] = df["MACROSECTOR"].map(_normalizar_macrosector)

    for ratio, cfg in RATIOS_FINANCIEROS.items():
        num = _variable_mm_cop(df, cfg["numerador"], anio, excluir_outliers)
        den = _variable_mm_cop(df, cfg["denominador"], anio, excluir_outliers)

        valores = (num / den.where(den > 0)) * cfg["factor"]
        valores = valores.replace([np.inf, -np.inf], np.nan)

        if excluir_outliers:
            lim_inf, lim_sup = RANGOS_RATIOS[ratio]
            valores = valores.where((valores >= lim_inf) & (valores <= lim_sup))

        salida[ratio] = valores

    return salida


def _tabla_ratios(ratios_por_anio: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Media, Mediana, Desv. Est., Mín., Máx. y N por ratio y año."""
    filas = []
    for ratio in RATIOS_FINANCIEROS:
        for anio in ANIOS_RATIOS:
            s = ratios_por_anio[anio][ratio].dropna()
            filas.append({
                "Indicador": ratio,
                "Año": anio,
                "Media": s.mean() if len(s) else np.nan,
                "Mediana": s.median() if len(s) else np.nan,
                "Desv. Estándar": s.std() if len(s) > 1 else np.nan,
                "Mínimo": s.min() if len(s) else np.nan,
                "Máximo": s.max() if len(s) else np.nan,
                "N empresas": len(s),
            })
    return pd.DataFrame(filas).set_index(["Indicador", "Año"])


# def _grafico_barras_ratio(
#     ratios_por_anio: dict[int, pd.DataFrame],
#     key_prefix: str,
#     ratio_defecto: int,
# ):
#     """Barras horizontales por Macrosector con segmentación propia."""
#     col_r, col_a = st.columns([3, 2])
#     with col_r:
#         ratio = st.selectbox(
#             "Ratio:",
#             list(RATIOS_FINANCIEROS.keys()),
#             index=ratio_defecto,
#             key=f"{key_prefix}_ratio",
#         )
#     with col_a:
#         anio = st.radio(
#             "Año:",
#             [2024, 2023, 2022],
#             horizontal=True,
#             key=f"{key_prefix}_anio",
#         )

#     datos = ratios_por_anio[anio][["MACRO", ratio]].dropna()
#     datos = datos[datos["MACRO"].isin(COLORES_KDE_MACROSECTOR.keys())]

#     resumen = (
#         datos.groupby("MACRO")[ratio]
#         .agg(["median", "count"])
#         .reset_index()
#         .sort_values("median")
#     )
#     resumen["Nombre"] = resumen["MACRO"].map(NOMBRES_KDE_MACROSECTOR)
#     resumen["Color"] = resumen["MACRO"].map(COLORES_KDE_MACROSECTOR)
#     resumen["Texto"] = resumen["median"].map(lambda v: _fmt_ratio(v, ratio))

#     mediana_total = datos[ratio].median()
#     unidad = "%" if _es_porcentaje(ratio) else "veces"

#     fig = go.Figure(
#         go.Bar(
#             x=resumen["median"],
#             y=resumen["Nombre"],
#             orientation="h",
#             marker=dict(
#                 color=resumen["Color"],
#                 line=dict(color="#333333", width=1),
#             ),
#             text=resumen["Texto"],
#             textposition="outside",
#             textfont=dict(size=14, color="#000000"),
#             cliponaxis=False,
#             customdata=np.stack([resumen["count"]], axis=-1),
#             hovertemplate=(
#                 "<b>%{y}</b>"
#                 f"<br>{ratio} mediano {anio}: " + "%{x:,.2f} " + unidad +
#                 "<br>Empresas: %{customdata[0]:,}"
#                 "<extra></extra>"
#             ),
#         )
#     )

#     if pd.notna(mediana_total):
#         fig.add_vline(
#             x=mediana_total,
#             line=dict(color="#1D3557", width=2, dash="dash"),
#             annotation_text=f"Total: {_fmt_ratio(mediana_total, ratio)}",
#             annotation_position="top",
#             annotation_font=dict(size=13, color="#000000"),
#         )

#     x_min = min(0, resumen["median"].min()) if len(resumen) else 0
#     x_max = max(0, resumen["median"].max()) if len(resumen) else 1
#     holgura = (x_max - x_min) * 0.25 or 1

#     fig.update_layout(
#         height=470,
#         title=dict(
#             text=f"{ratio} mediano por Macrosector — {anio}",
#             font=dict(size=18),
#         ),
#         xaxis=dict(
#             title=dict(text=ratio, font=dict(size=15)),
#             tickfont=dict(size=13),
#             range=[x_min - (holgura if x_min < 0 else 0), x_max + holgura],
#             zeroline=True,
#             zerolinecolor="rgba(0,0,0,0.45)",
#             gridcolor="rgba(0,0,0,0.10)",
#         ),
#         yaxis=dict(tickfont=dict(size=14), automargin=True),
#         showlegend=False,
#         margin=dict(l=10, r=40, t=80, b=60),
#     )

#     _aplicar_hover_legible(fig, tamano=14)
#     st.plotly_chart(fig, use_container_width=True)

# ═══ PARCHE 21 (inicio) — Barras por Macrosector: Mediana/Media y escala común ═
def _selector_barras(key_prefix: str, ratio_defecto: int) -> tuple[str, str, int]:
    """Segmentación propia de cada gráfica: ratio, estadístico y año."""
    ratio = st.selectbox(
        "Ratio:",
        list(RATIOS_FINANCIEROS.keys()),
        index=ratio_defecto,
        key=f"{key_prefix}_ratio",
    )
    col_e, col_a = st.columns(2)
    with col_e:
        estadistico = st.radio(
            "Estadístico:",
            ["Mediana", "Media"],
            horizontal=True,
            key=f"{key_prefix}_estadistico",
        )
    with col_a:
        anio = st.radio(
            "Año:",
            [2024, 2023, 2022],
            horizontal=True,
            key=f"{key_prefix}_anio",
        )
    return ratio, estadistico, anio


def _resumen_barras(
    ratios_por_anio: dict[int, pd.DataFrame],
    ratio: str,
    estadistico: str,
    anio: int,
) -> tuple[pd.DataFrame, float]:
    """Valor del estadístico por Macrosector y valor del total de empresas."""
    datos = ratios_por_anio[anio][["MACRO", ratio]].dropna()
    datos = datos[datos["MACRO"].isin(COLORES_KDE_MACROSECTOR.keys())]

    funcion = ESTADISTICOS_RATIOS[estadistico]
    resumen = (
        datos.groupby("MACRO")[ratio]
        .agg([funcion, "count"])
        .reset_index()
        .rename(columns={funcion: "valor"})
        .sort_values("valor")
    )
    resumen["Nombre"] = resumen["MACRO"].map(NOMBRES_KDE_MACROSECTOR)
    resumen["Color"] = resumen["MACRO"].map(COLORES_KDE_MACROSECTOR)
    resumen["Texto"] = resumen["valor"].map(lambda v: _fmt_ratio(v, ratio))

    return resumen, _estadistico_serie(datos[ratio], estadistico)


def _figura_barras(
    resumen: pd.DataFrame,
    valor_total: float,
    ratio: str,
    estadistico: str,
    anio: int,
    rango_x: list[float],
) -> go.Figure:
    """Barras horizontales por Macrosector con el rango X común recibido."""
    unidad = "%" if _es_porcentaje(ratio) else "veces"

    fig = go.Figure(
        go.Bar(
            x=resumen["valor"],
            y=resumen["Nombre"],
            orientation="h",
            marker=dict(
                color=resumen["Color"],
                line=dict(color="#333333", width=1),
            ),
            text=resumen["Texto"],
            textposition="outside",
            textfont=dict(size=14, color="#FFFFFF"),
            cliponaxis=False,
            customdata=np.stack([resumen["count"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b>"
                f"<br>{ratio} — {estadistico} {anio}: " + "%{x:,.2f} " + unidad +
                "<br>Empresas: %{customdata[0]:,}"
                "<extra></extra>"
            ),
        )
    )

    if pd.notna(valor_total):
        fig.add_vline(
            x=valor_total,
            line=dict(color="#FFFFFF", width=2, dash="dash"),
            annotation_text=f"Total: {_fmt_ratio(valor_total, ratio)}",
            annotation_position="top",
            annotation_font=dict(size=13, color="#FFFFFF"),
        )

    fig.update_layout(
        height=480,
        title=dict(
            text=f"{ratio} — {estadistico} por Macrosector ({anio})",
            font=dict(size=17),
        ),
        xaxis=dict(
            title=dict(text=ratio, font=dict(size=15)),
            tickfont=dict(size=13),
            range=rango_x,
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.45)",
            gridcolor="rgba(255,255,255,0.12)",
        ),
        yaxis=dict(tickfont=dict(size=14), automargin=True),
        showlegend=False,
        margin=dict(l=10, r=40, t=90, b=60),
    )

    _aplicar_hover_legible(fig, tamano=14)
    return fig


def _rango_comun_barras(valores: list[float]) -> list[float]:
    """Rango X compartido por las dos gráficas (incluye el 0)."""
    valores = [v for v in valores if pd.notna(v)]
    x_min = min(valores + [0])
    x_max = max(valores + [0])
    holgura = (x_max - x_min) * 0.28 or 1
    return [x_min - (holgura if x_min < 0 else 0), x_max + holgura]
# ═══ PARCHE 21 (fin) ═════════════════════════════════════════════════════


# def _opciones_scatter_ratios() -> dict[str, tuple[str, str, int]]:
#     """Etiqueta → (tipo, nombre, año). Ratios y variables en MM COP."""
#     opciones = {}
#     for anio in ANIOS_RATIOS:
#         for ratio in RATIOS_FINANCIEROS:
#             nombre = ratio.replace(" (%)", "")
#             etiqueta = (
#                 f"{nombre} {anio} (%)" if _es_porcentaje(ratio)
#                 else f"Solvencia {anio} (veces)"
#             )
#             opciones[etiqueta] = ("ratio", ratio, anio)
#     for anio in ANIOS_RATIOS:
#         for variable_label in VARIABLES_DISTRIBUCION:
#             opciones[f"{variable_label} {anio} (MM COP)"] = (
#                 "variable", variable_label, anio
#             )
#     return opciones

# ═══ PARCHE 22 (inicio) — Opciones del scatter con los 5 ratios ══════════
def _opciones_scatter_ratios() -> dict[str, tuple[str, str, int]]:
    """Etiqueta → (tipo, nombre, año). Ratios y variables en MM COP."""
    opciones = {}
    for anio in ANIOS_RATIOS:
        for ratio in RATIOS_FINANCIEROS:
            nombre = ratio.split(" (")[0]
            unidad = "%" if _es_porcentaje(ratio) else "veces"
            opciones[f"{nombre} {anio} ({unidad})"] = ("ratio", ratio, anio)
    for anio in ANIOS_RATIOS:
        for variable_label in VARIABLES_DISTRIBUCION:
            opciones[f"{variable_label} {anio} (MM COP)"] = (
                "variable", variable_label, anio
            )
    return opciones
# ═══ PARCHE 22 (fin) ═════════════════════════════════════════════════════


def _grafico_dispersion_ratios(
    df: pd.DataFrame,
    ratios_por_anio: dict[int, pd.DataFrame],
    excluir_outliers: bool,
):
    """
    Misma estructura y metodología de "Relación entre Dos Variables"
    (pestaña Correlaciones de new_eda_VII.py): un panel por Macrosector, con
    su color, su línea de tendencia (regresión lineal simple) y su r de
    Pearson.
    """
    opciones = _opciones_scatter_ratios()
    etiquetas = list(opciones.keys())

    col_x, col_y = st.columns(2)
    with col_x:
        etiqueta_x = st.selectbox(
            "Variable eje X:",
            etiquetas,
            index=etiquetas.index("ROA 2024 (%)"),
            key="ratios_scatter_x",
        )
    with col_y:
        etiqueta_y = st.selectbox(
            "Variable eje Y:",
            etiquetas,
            index=etiquetas.index("ROE 2024 (%)"),
            key="ratios_scatter_y",
        )

    def _serie(etiqueta: str) -> pd.Series:
        tipo, nombre, anio = opciones[etiqueta]
        if tipo == "ratio":
            return ratios_por_anio[anio][nombre]
        return _variable_mm_cop(df, nombre, anio, excluir_outliers)

    df_plot = pd.DataFrame({
        "MACRO": df["MACROSECTOR"].map(_normalizar_macrosector),
        "X": _serie(etiqueta_x),
        "Y": _serie(etiqueta_y),
    })
    total = len(df_plot)
    df_plot = df_plot.dropna()
    df_plot = df_plot[df_plot["MACRO"].isin(COLORES_KDE_MACROSECTOR.keys())]
    incluidas = len(df_plot)

    # Mismo orden del EDA: de mayor a menor número de empresas.
    orden_macro = (
        df["MACROSECTOR"].map(_normalizar_macrosector)
        .value_counts().index.tolist()
    )
    orden_macro = [m for m in orden_macro if m in COLORES_KDE_MACROSECTOR]
    titulos = [NOMBRES_KDE_MACROSECTOR[m] for m in orden_macro]

    columnas_grid = 3
    filas_grid = -(-len(orden_macro) // columnas_grid)

    fig = make_subplots(
        rows=filas_grid,
        cols=columnas_grid,
        subplot_titles=titulos,
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )

    for i, macro in enumerate(orden_macro):
        fila, col = i // columnas_grid + 1, i % columnas_grid + 1
        datos = df_plot[df_plot["MACRO"] == macro]
        color = COLORES_KDE_MACROSECTOR[macro]
        nombre = NOMBRES_KDE_MACROSECTOR[macro]

        fig.add_trace(
            go.Scatter(
                x=datos["X"],
                y=datos["Y"],
                mode="markers",
                showlegend=False,
                marker=dict(size=6, color=color, opacity=0.65),
                hovertemplate=(
                    f"<b>{nombre}</b>"
                    f"<br>{etiqueta_x}: " + "%{x:,.2f}" +
                    f"<br>{etiqueta_y}: " + "%{y:,.2f}" +
                    "<extra></extra>"
                ),
            ),
            row=fila,
            col=col,
        )

        if len(datos) > 2 and datos["X"].nunique() > 1:
            pendiente, intercepto = np.polyfit(datos["X"], datos["Y"], 1)
            x_linea = np.linspace(datos["X"].min(), datos["X"].max(), 50)
            r = datos["X"].corr(datos["Y"])
            fig.add_trace(
                go.Scatter(
                    x=x_linea,
                    y=pendiente * x_linea + intercepto,
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    # line=dict(color="#1a1a1a", width=2, dash="dash"),
                    line=dict(color="#FFFFFF", width=2, dash="dash"),  # PARCHE 23-A
                ),
                row=fila,
                col=col,
            )
            fig.layout.annotations[i].text = f"{nombre}  (r = {r:.2f})"

    for ann in fig.layout.annotations:
        # ann.font = dict(size=15, color="#000000")
        ann.font = dict(size=15, color="#FFFFFF")  # PARCHE 23-B: nombre visible en fondo oscuro

    fig.update_xaxes(
        title_text=etiqueta_x,
        tickfont=dict(size=12),
        title_font=dict(size=12),
        tickformat=",.2~f",
    )
    fig.update_yaxes(
        title_text=etiqueta_y,
        tickfont=dict(size=12),
        title_font=dict(size=12),
        tickformat=",.2~f",
    )

    fig.update_layout(
        title=dict(
            text=f"{etiqueta_x} vs. {etiqueta_y}, por Macrosector",
            font=dict(size=18),
        ),
        height=320 * filas_grid,
        showlegend=False,
        margin=dict(t=100),
    )

    _aplicar_hover_legible(fig, tamano=14)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Cada panel muestra únicamente las empresas de ese macrosector, con su "
        f"propia línea de tendencia y su propio coeficiente r de Pearson. "
        f"Incluye {incluidas:,} de {total:,} empresas — {total - incluidas:,} "
        f"quedaron fuera por no tener dato válido o por estar fuera del rango "
        f"fijo de alguna de las dos variables. Las variables en MM COP están en "
        f"Miles de Millones COP."
    )


# def _metricas_derivadas(df: pd.DataFrame):

#     st.subheader("📏 Métricas Financieras Derivadas")

#     st.markdown(
#         """
#         Ratios financieros estándar calculados por empresa para **2022, 2023
#         y 2024**. Los componentes de cada ratio se toman en **Miles de
#         Millones de COP**, con el mismo tratamiento de outliers extremos de
#         las pestañas anteriores.
#         """
#     )

#     with st.expander("ℹ️ Fórmulas de los ratios"):
#         st.markdown(
#             "\n".join(
#                 f"- **{ratio}** = {cfg['formula']}"
#                 for ratio, cfg in RATIOS_FINANCIEROS.items()
#             )
#         )

#     excluir_outliers = st.checkbox(
#         "Excluir outliers extremos (rango fijo del EDA y ratios extremos)",
#         value=True,
#         key="ratios_excluir_outliers",
#         help=(
#             "Componentes dentro del rango fijo del EDA en Miles de Millones "
#             "COP (se excluyen las empresas del orden de billones de COP) y "
#             "ratios dentro de: ROA ±50 %, ROE ±150 %, Margen Neto ±100 % y "
#             "Solvencia 0 a 40 veces."
#         ),
#     )

#     if "MACROSECTOR" not in df.columns:
#         st.warning("El DataFrame no contiene la columna `MACROSECTOR`.")
#         return

#     ratios_por_anio = {
#         anio: _calcular_ratios(df, anio, excluir_outliers)
#         for anio in ANIOS_RATIOS
#     }

#     # ─────────────────────────────────────────────────────────────────────
#     # 1. INDICADORES CLAVE (mediana 2024 y variación frente a 2023)
#     # ─────────────────────────────────────────────────────────────────────

#     st.markdown("### 1️⃣ Indicadores Clave — Mediana 2024 vs. 2023")

#     columnas_kpi = st.columns(len(RATIOS_FINANCIEROS))
#     for col_kpi, ratio in zip(columnas_kpi, RATIOS_FINANCIEROS):
#         med_2024 = ratios_por_anio[2024][ratio].median()
#         med_2023 = ratios_por_anio[2023][ratio].median()
#         delta = med_2024 - med_2023
#         sufijo = " p.p." if _es_porcentaje(ratio) else " veces"
#         col_kpi.metric(
#             label=ratio,
#             value=_fmt_ratio(med_2024, ratio),
#             delta=f"{delta:+,.2f}{sufijo}",
#             # Para Solvencia, un aumento implica más endeudamiento.
#             delta_color="normal" if _es_porcentaje(ratio) else "inverse",
#         )

#     # ─────────────────────────────────────────────────────────────────────
#     # 2. CUADRO DE ESTADÍSTICOS POR RATIO Y AÑO
#     # ─────────────────────────────────────────────────────────────────────

#     st.markdown("### 2️⃣ Estadísticos de los Ratios por Año")

#     tabla = _tabla_ratios(ratios_por_anio)
#     columnas_valor = ["Media", "Mediana", "Desv. Estándar", "Mínimo", "Máximo"]

#     estilo = tabla.style
#     for ratio in RATIOS_FINANCIEROS:
#         filas_ratio = pd.IndexSlice[ratio, :]
#         estilo = estilo.format(
#             lambda v, r=ratio: _fmt_ratio(v, r),
#             subset=pd.IndexSlice[filas_ratio, columnas_valor],
#         )
#         estilo = estilo.set_properties(
#             subset=pd.IndexSlice[filas_ratio, :],
#             **{"background-color": FONDO_RATIO[ratio], "color": "#000000"},
#         )
#     estilo = estilo.format("{:,.0f}", subset=["N empresas"])
#     estilo = estilo.set_properties(
#         subset=["Mediana"], **{"font-weight": "bold"}
#     )

#     st.dataframe(estilo, use_container_width=True, height=490)

#     st.caption(
#         "ROA, ROE y Margen Neto en %; Solvencia en veces (Pasivos ÷ "
#         "Patrimonio). Componentes en Miles de Millones COP, iguales a los de "
#         "las pestañas anteriores. Solo se incluyen empresas con denominador "
#         "positivo (Activos, Ingresos o Patrimonio > 0)."
#     )

#     # Evolución de la mediana (un panel por ratio, colores por año).
#     fig_evol = make_subplots(
#         rows=1,
#         cols=len(RATIOS_FINANCIEROS),
#         horizontal_spacing=0.07,
#     )
#     for i, ratio in enumerate(RATIOS_FINANCIEROS, start=1):
#         medianas = [ratios_por_anio[a][ratio].median() for a in ANIOS_RATIOS]
#         fig_evol.add_trace(
#             go.Bar(
#                 x=[str(a) for a in ANIOS_RATIOS],
#                 y=medianas,
#                 marker=dict(
#                     color=[COLORES_ANIO[a] for a in ANIOS_RATIOS],
#                     line=dict(color="#333333", width=1),
#                 ),
#                 text=[_fmt_ratio(m, ratio) for m in medianas],
#                 textposition="outside",
#                 textfont=dict(size=13, color="#000000"),
#                 cliponaxis=False,
#                 showlegend=False,
#                 hovertemplate=(
#                     f"<b>{ratio}</b><br>Año %{{x}}<br>Mediana: "
#                     + "%{y:,.2f}"
#                     + (" %" if _es_porcentaje(ratio) else " veces")
#                     + "<extra></extra>"
#                 ),
#             ),
#             row=1,
#             col=i,
#         )
#         maximo = max([m for m in medianas if pd.notna(m)] + [0])
#         fig_evol.update_yaxes(
#             range=[0, maximo * 1.3 or 1],
#             tickfont=dict(size=12),
#             tickformat=",.2~f",
#             gridcolor="rgba(0,0,0,0.10)",
#             row=1,
#             col=i,
#         )
#         fig_evol.update_xaxes(
#             tickfont=dict(size=13),
#             title=dict(
#                 text=f"<b>{ratio}</b>",
#                 font=dict(size=14, color="#000000"),
#                 standoff=10,
#             ),
#             row=1,
#             col=i,
#         )

#     fig_evol.update_layout(
#         height=430,
#         title=dict(
#             text="Evolución de la Mediana de cada Ratio — 2022, 2023 y 2024",
#             font=dict(size=18),
#         ),
#         margin=dict(l=60, r=30, t=80, b=90),
#     )
#     _aplicar_hover_legible(fig_evol, tamano=14)
#     st.plotly_chart(fig_evol, use_container_width=True)

#     # ─────────────────────────────────────────────────────────────────────
#     # 3. RATIOS POR MACROSECTOR (dos gráficas con segmentación propia)
#     # ─────────────────────────────────────────────────────────────────────

#     st.markdown("### 3️⃣ Ratios Medianos por Macrosector")
#     st.caption(
#         "Cada gráfica tiene su propia segmentación de ratio y año, para "
#         "comparar libremente. La línea punteada es la mediana del total de "
#         "empresas; los colores son los de cada Macrosector en el EDA."
#     )

#     col_izq, col_der = st.columns(2)
#     with col_izq:
#         _grafico_barras_ratio(ratios_por_anio, "ratios_barras_izq", 0)
#     with col_der:
#         _grafico_barras_ratio(ratios_por_anio, "ratios_barras_der", 1)

#     # ─────────────────────────────────────────────────────────────────────
#     # 4. RELACIÓN ENTRE DOS VARIABLES POR MACROSECTOR
#     # ─────────────────────────────────────────────────────────────────────

#     st.markdown("### 4️⃣ Relación entre Dos Variables por Macrosector")
#     st.caption(
#         "Cada punto es una empresa. La línea punteada es la tendencia lineal "
#         "(regresión simple); r es el coeficiente de correlación de Pearson "
#         "entre las dos variables elegidas."
#     )

#     _grafico_dispersion_ratios(df, ratios_por_anio, excluir_outliers)
# ═══ PARCHE 19 (fin) ══════════════════════════════════════════════════════

# ═══ PARCHE 24 (inicio) — _metricas_derivadas() ampliada ═════════════════
def _tabla_ratios_html(tabla: pd.DataFrame) -> str:
    """Cuadro HTML: números grandes, centrados y fondo de alto contraste."""
    columnas_valor = ["Media", "Mediana", "Desv. Estándar", "Mínimo", "Máximo"]
    encabezados = ["Indicador", "Año"] + columnas_valor + ["N empresas"]

    estilo_th = (
        "background:#1D3557;color:#FFFFFF;font-size:17px;font-weight:700;"
        "text-align:center;vertical-align:middle;padding:12px 10px;"
        "border:1px solid #9FB3C8;"
    )
    estilo_td = (
        "color:#0B0B0B;font-size:17px;text-align:center;"
        "vertical-align:middle;padding:10px 10px;border:1px solid #B8C4D2;"
    )

    html = [
        '<div style="overflow-x:auto;border-radius:10px;'
        'box-shadow:0 2px 10px rgba(0,0,0,0.35);margin-bottom:0.6rem;">',
        '<table style="width:100%;border-collapse:collapse;'
        'font-family:inherit;">',
        "<thead><tr>",
    ]
    html += [f'<th style="{estilo_th}">{h}</th>' for h in encabezados]
    html.append("</tr></thead><tbody>")

    for ratio in RATIOS_FINANCIEROS:
        fondo = FONDO_RATIO[ratio]
        for j, anio in enumerate(ANIOS_RATIOS):
            fila = tabla.loc[(ratio, anio)]
            # Filas alternas: blanco y gris muy claro para seguir la lectura.
            fondo_fila = "#FFFFFF" if j % 2 == 0 else "#EEF2F7"
            html.append("<tr>")
            if j == 0:
                html.append(
                    f'<td rowspan="{len(ANIOS_RATIOS)}" style="{estilo_td}'
                    f'background:{fondo};font-weight:700;font-size:17px;">'
                    f"{_nombre_ratio_html(ratio)}</td>"
                )
            html.append(
                f'<td style="{estilo_td}background:{fondo_fila};'
                f'font-weight:700;">{anio}</td>'
            )
            for col in columnas_valor:
                negrita = "font-weight:800;" if col == "Mediana" else ""
                html.append(
                    f'<td style="{estilo_td}background:{fondo_fila};{negrita}">'
                    f"{_fmt_ratio(fila[col], ratio)}</td>"
                )
            html.append(
                f'<td style="{estilo_td}background:{fondo_fila};">'
                f"{int(fila['N empresas']):,}</td>"
            )
            html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)


def _metricas_derivadas(df: pd.DataFrame):

    st.subheader("📏 Métricas Financieras Derivadas")

    st.markdown(
        """
        Ratios financieros estándar calculados por empresa para **2022, 2023
        y 2024**. Los componentes de cada ratio se toman en **Miles de
        Millones de COP**, con el mismo tratamiento de outliers extremos de
        las pestañas anteriores.
        """
    )

    # ── Cuadro de fórmulas (Solvencia y Apalancamiento diferenciados) ────
    with st.expander("ℹ️ Fórmulas y lectura de los ratios", expanded=False):
        st.markdown(
            "| Ratio | Fórmula | Cómo se lee |\n"
            "|---|---|---|\n"
            + "\n".join(
                f"| **{ratio}** | {cfg['formula']} | {cfg['lectura']} |"
                for ratio, cfg in RATIOS_FINANCIEROS.items()
            )
        )
        st.caption(
            "Solvencia y Apalancamiento se leen en sentidos opuestos: en "
            "Solvencia un valor más alto es mejor; en Apalancamiento un valor "
            "más alto implica más riesgo."
        )

    excluir_outliers = st.checkbox(
        "Excluir outliers extremos (rango fijo del EDA y ratios extremos)",
        value=True,
        key="ratios_excluir_outliers",
        help=(
            "Componentes dentro del rango fijo del EDA en Miles de Millones "
            "COP (se excluyen las empresas del orden de billones de COP) y "
            "ratios dentro de: ROA ±50 %, ROE ±150 %, Margen Neto ±100 %, "
            "Solvencia 0 a 50 veces y Apalancamiento 0 a 40 veces."
        ),
    )

    if "MACROSECTOR" not in df.columns:
        st.warning("El DataFrame no contiene la columna `MACROSECTOR`.")
        return

    ratios_por_anio = {
        anio: _calcular_ratios(df, anio, excluir_outliers)
        for anio in ANIOS_RATIOS
    }

    # ─────────────────────────────────────────────────────────────────────
    # 1. INDICADORES CLAVE (estadístico y comparación segmentables)
    # ─────────────────────────────────────────────────────────────────────

    st.markdown("### 1️⃣ Indicadores Clave")

    col_est, col_comp = st.columns([3, 2])
    with col_est:
        est_kpi = st.radio(
            "Estadístico:",
            list(ESTADISTICOS_RATIOS.keys()),
            horizontal=True,
            key="ratios_kpi_estadistico",
        )
    with col_comp:
        comparacion = st.radio(
            "Comparación:",
            list(COMPARACIONES_ANIOS.keys()),
            horizontal=True,
            key="ratios_kpi_comparacion",
        )
    anio_ini, anio_fin = COMPARACIONES_ANIOS[comparacion]

    # st.caption(
    #     f"Valor de **{est_kpi}** en **{anio_fin}** y variación frente a "
    #     f"**{anio_ini}**. Pase el cursor sobre el ícono (?) de cada tarjeta "
    #     f"para ver ambos valores."
    # )

    # columnas_kpi = st.columns(len(RATIOS_FINANCIEROS))
    # for col_kpi, (ratio, cfg) in zip(columnas_kpi, RATIOS_FINANCIEROS.items()):
    #     valor_ini = _estadistico_serie(ratios_por_anio[anio_ini][ratio], est_kpi)
    #     valor_fin = _estadistico_serie(ratios_por_anio[anio_fin][ratio], est_kpi)
    #     delta = valor_fin - valor_ini
    #     sufijo = " p.p." if _es_porcentaje(ratio) else " veces"

    #     # Verde/rojo solo cuando el sentido es claro (Media y Mediana);
    #     # en Desv., Mínimo y Máximo la variación se muestra en gris.
    #     if est_kpi in ("Mediana", "Media"):
    #         color_delta = "normal" if cfg["mayor_mejor"] else "inverse"
    #     else:
    #         color_delta = "off"

    #     col_kpi.metric(
    #         label=ratio,
    #         value=_fmt_ratio(valor_fin, ratio),
    #         delta=f"{delta:+,.2f}{sufijo}" if pd.notna(delta) else None,
    #         delta_color=color_delta,
    #         help=(
    #             f"{est_kpi} {anio_ini}: {_fmt_ratio(valor_ini, ratio)}  ·  "
    #             f"{est_kpi} {anio_fin}: {_fmt_ratio(valor_fin, ratio)}"
    #         ),
    #     )
    
    # ═══ PARCHE 25 (inicio) — Tarjetas con comparación explícita de años ══
    st.caption(
        f"**{est_kpi}**: variación de **{anio_fin}** con respecto a "
        f"**{anio_ini}** (valor {anio_fin} − valor {anio_ini})."
    )

    columnas_kpi = st.columns(len(RATIOS_FINANCIEROS))
    for col_kpi, (ratio, cfg) in zip(columnas_kpi, RATIOS_FINANCIEROS.items()):
        # Cada valor se calcula directamente con el año de la comparación.
        valor_ini = _estadistico_serie(
            ratios_por_anio[anio_ini][ratio], est_kpi
        )
        valor_fin = _estadistico_serie(
            ratios_por_anio[anio_fin][ratio], est_kpi
        )
        delta = valor_fin - valor_ini
        sufijo = " p.p." if _es_porcentaje(ratio) else " veces"

        # Verde/rojo solo cuando el sentido es claro (Media y Mediana);
        # en Desv., Mínimo y Máximo la variación se muestra en gris.
        if pd.isna(delta) or delta == 0 or est_kpi not in ("Mediana", "Media"):
            color = "#B0B7C3"
        else:
            mejora = (delta > 0) == cfg["mayor_mejor"]
            color = "#34C759" if mejora else "#FF5A5F"
        flecha = "▲" if pd.notna(delta) and delta > 0 else (
            "▼" if pd.notna(delta) and delta < 0 else "●"
        )
        texto_delta = (
            f"{flecha} {delta:+,.2f}{sufijo}" if pd.notna(delta) else "—"
        )

        # HTML en una sola cadena, sin sangría (Markdown trataría las líneas
        # sangradas como bloque de código).
        tarjeta = (
            '<div style="background:#1B2433;border:1px solid #3A4A63;'
            'border-radius:10px;padding:14px 12px;text-align:center;">'
            '<div style="color:#FFFFFF;font-size:15px;font-weight:700;'
            f'min-height:42px;">{_nombre_ratio_html(ratio)}</div>'
            '<div style="color:#C9D3E0;font-size:14px;margin-top:6px;">'
            f'{anio_ini}: <b style="color:#FFFFFF;">'
            f'{_fmt_ratio(valor_ini, ratio)}</b></div>'
            '<div style="color:#C9D3E0;font-size:14px;">'
            f'{anio_fin}: <b style="color:#FFFFFF;">'
            f'{_fmt_ratio(valor_fin, ratio)}</b></div>'
            f'<div style="color:{color};font-size:22px;font-weight:800;'
            f'margin-top:8px;">{texto_delta}</div>'
            '<div style="color:#C9D3E0;font-size:12px;">'
            f'{anio_fin} vs. {anio_ini}</div>'
            '</div>'
        )
        col_kpi.markdown(tarjeta, unsafe_allow_html=True)
    # ═══ PARCHE 25 (fin) ══════════════════════════════════════════════════

    # ─────────────────────────────────────────────────────────────────────
    # 2. CUADRO DE ESTADÍSTICOS POR RATIO Y AÑO
    # ─────────────────────────────────────────────────────────────────────

    st.markdown("### 2️⃣ Estadísticos de los Ratios por Año")

    tabla = _tabla_ratios(ratios_por_anio)
    st.markdown(_tabla_ratios_html(tabla), unsafe_allow_html=True)

    st.caption(
        "ROA, ROE y Margen Neto en %; Solvencia y Apalancamiento en veces. "
        "Componentes en Miles de Millones COP, iguales a los de las pestañas "
        "anteriores. Solo se incluyen empresas con denominador positivo "
        "(Activos, Ingresos, Pasivos o Patrimonio > 0)."
    )

    # ── Evolución del estadístico elegido (un panel por ratio) ───────────
    est_evol = st.radio(
        "Estadístico a visualizar en la evolución:",
        list(ESTADISTICOS_RATIOS.keys()),
        horizontal=True,
        key="ratios_evol_estadistico",
    )

    fig_evol = make_subplots(
        rows=1,
        cols=len(RATIOS_FINANCIEROS),
        horizontal_spacing=0.06,
    )
    for i, ratio in enumerate(RATIOS_FINANCIEROS, start=1):
        valores = [
            _estadistico_serie(ratios_por_anio[a][ratio], est_evol)
            for a in ANIOS_RATIOS
        ]
        fig_evol.add_trace(
            go.Bar(
                x=[str(a) for a in ANIOS_RATIOS],
                y=valores,
                marker=dict(
                    color=[COLORES_ANIO[a] for a in ANIOS_RATIOS],
                    line=dict(color="#DDDDDD", width=1),
                ),
                text=[
                    f"{v:,.4f} veces"
                    if pd.notna(v) and 0 < abs(v) < 0.01 and not _es_porcentaje(ratio)
                    else _fmt_ratio(v, ratio)
                    for v in valores
                ],
                textposition="outside",
                textfont=dict(size=13, color="#FFFFFF"),
                cliponaxis=False,
                showlegend=False,
                hovertemplate=(
                    f"<b>{ratio}</b><br>Año %{{x}}<br>{est_evol}: "
                    + "%{y:,.2f}"
                    + (" %" if _es_porcentaje(ratio) else " veces")
                    + "<extra></extra>"
                ),
            ),
            row=1,
            col=i,
        )
        validos = [v for v in valores if pd.notna(v)]
        y_min = min(validos + [0])
        y_max = max(validos + [0])
        holgura = (y_max - y_min) * 0.30 or 1
        fig_evol.update_yaxes(
            range=[y_min - (holgura if y_min < 0 else 0), y_max + holgura],
            tickformat=",.2~f" if (y_max - y_min) >= 0.05 else ",.4~f",
            tickfont=dict(size=12),
            gridcolor="rgba(255,255,255,0.12)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.45)",
            row=1,
            col=i,
        )
        fig_evol.update_xaxes(
            tickfont=dict(size=13),
            title=dict(
                text=_nombre_ratio_html(ratio),
                font=dict(size=14, color="#FFFFFF"),
                standoff=10,
            ),
            row=1,
            col=i,
        )

    fig_evol.update_layout(
        height=460,
        title=dict(
            text=(
                f"Evolución {ESTADISTICO_TITULO[est_evol]} de cada Ratio — "
                f"2022, 2023 y 2024"
            ),
            font=dict(size=18),
        ),
        margin=dict(l=60, r=30, t=80, b=110),
    )
    _aplicar_hover_legible(fig_evol, tamano=14)
    st.plotly_chart(fig_evol, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # 3. RATIOS POR MACROSECTOR (dos gráficas, segmentación propia y
    #    escala X común para que las barras sean comparables)
    # ─────────────────────────────────────────────────────────────────────

    st.markdown("### 3️⃣ Ratios por Macrosector — Mediana o Media")
    st.caption(
        "Cada gráfica tiene su propia segmentación de ratio, estadístico y "
        "año. Ambas comparten la misma escala en el eje X, de modo que un "
        "valor menor nunca se ve con una barra más larga que la de la otra "
        "gráfica. La línea punteada es el valor del total de empresas."
    )

    col_izq, col_der = st.columns(2)
    with col_izq:
        sel_izq = _selector_barras("ratios_barras_izq", 0)
    with col_der:
        sel_der = _selector_barras("ratios_barras_der", 1)

    resumen_izq, total_izq = _resumen_barras(ratios_por_anio, *sel_izq)
    resumen_der, total_der = _resumen_barras(ratios_por_anio, *sel_der)

    rango_x = _rango_comun_barras(
        resumen_izq["valor"].tolist() + resumen_der["valor"].tolist()
        + [total_izq, total_der]
    )

    with col_izq:
        st.plotly_chart(
            _figura_barras(resumen_izq, total_izq, *sel_izq, rango_x),
            use_container_width=True,
        )
    with col_der:
        st.plotly_chart(
            _figura_barras(resumen_der, total_der, *sel_der, rango_x),
            use_container_width=True,
        )

    if _es_porcentaje(sel_izq[0]) != _es_porcentaje(sel_der[0]):
        st.info(
            "Las dos gráficas usan unidades distintas (% y veces). La escala "
            "es común, pero conviene comparar ratios de la misma unidad."
        )

    # ─────────────────────────────────────────────────────────────────────
    # 4. RELACIÓN ENTRE DOS VARIABLES POR MACROSECTOR
    # ─────────────────────────────────────────────────────────────────────

    st.markdown("### 4️⃣ Relación entre Dos Variables por Macrosector")
    st.caption(
        "Cada punto es una empresa. La línea punteada es la tendencia lineal "
        "(regresión simple); r es el coeficiente de correlación de Pearson "
        "entre las dos variables elegidas."
    )

    _grafico_dispersion_ratios(df, ratios_por_anio, excluir_outliers)
# ═══ PARCHE 24 (fin) ═════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA: 🔍 ANÁLISIS POR SEGMENTO
# ═══════════════════════════════════════════════════════════════════════════


# VARIABLES_SEGMENTO = {
#     "Ingresos Op.": {
#         2022: "INGRESOS OPERACIONALES 2022*",
#         2023: "INGRESOS OPERACIONALES 2023*",
#         2024: "INGRESOS OPERACIONALES 2024*",
#     },
#     "Ganancia/Pérdida": {
#         2022: "GANANCIA (PERDIDA) 2022",
#         2023: "GANANCIA (PERDIDA) 2023",
#         2024: "GANANCIA (PERDIDA) 2024",
#     },
#     "Total Activos": {
#         2022: "TOTAL ACTIVOS 2022",
#         2023: "TOTAL ACTIVOS 2023",
#         2024: "TOTAL ACTIVOS 2024",
#     },
#     "Total Pasivos": {
#         2022: "TOTAL PASIVOS 2022",
#         2023: "TOTAL PASIVOS 2023",
#         2024: "TOTAL PASIVOS 2024",
#     },
#     "Total Patrimonio": {
#         2022: "TOTAL PATRIMONIO 2022",
#         2023: "TOTAL PATRIMONIO 2023",
#         2024: "TOTAL PATRIMONIO 2024",
#     },
# }


# def _analisis_segmento(df: pd.DataFrame):

#     st.subheader("🔍 Análisis por Segmento")

#     st.markdown(
#         """
#         Permite analizar el comportamiento financiero de las empresas
#         agrupándolas por **Macrosector, Región o Supervisor**.

#         Los estadísticos monetarios están expresados en
#         **Miles de Millones de COP**.
#         """
#     )

#     # ═══════════════════════════════════════════════════════════════════════
#     # 1. SELECTORES
#     # ═══════════════════════════════════════════════════════════════════════

#     col1, col2, col3 = st.columns([1.2, 1.8, 1])

#     with col1:
#         segmento = st.selectbox(
#             "Agrupar por:",
#             [
#                 "MACROSECTOR",
#                 "REGION",
#                 "SUPERVISOR",
#             ],
#             key="segmento_agrupacion_new",
#         )

#     with col2:
#         variable = st.selectbox(
#             "Variable a analizar:",
#             list(VARIABLES_SEGMENTO.keys()),
#             key="segmento_variable_new",
#         )

#     with col3:
#         anio = st.selectbox(
#             "Año:",
#             [2024, 2023, 2022],
#             key="segmento_anio_new",
#         )

#     columna = _resolver_columna_distribucion(
#         df,
#         VARIABLES_SEGMENTO[variable][anio],
#     )

#     if columna is None:
#         st.error(
#             f"No se encontró la columna correspondiente a "
#             f"{variable} — {anio}."
#         )
#         return

#     if segmento not in df.columns:
#         st.error(
#             f"No existe la columna de segmentación `{segmento}`."
#         )
#         return

#     # ═══════════════════════════════════════════════════════════════════════
#     # 2. PREPARACIÓN DE DATOS
#     # ═══════════════════════════════════════════════════════════════════════

#     datos = df[
#         [
#             segmento,
#             columna,
#         ]
#     ].copy()

#     datos[columna] = pd.to_numeric(
#         datos[columna],
#         errors="coerce",
#     )

#     datos = datos.dropna(
#         subset=[
#             segmento,
#             columna,
#         ]
#     )

#     datos["Valor_MM_COP"] = (
#         datos[columna] / DIVISOR_MM_COP
#     )

#     # Aplicar el mismo rango fijo utilizado en las visualizaciones
#     # de Distribución y Normalidad.
#     limite_inferior, limite_superior = RANGOS_DISTRIBUCION[
#         variable
#     ]

#     datos = datos[
#         (datos["Valor_MM_COP"] >= limite_inferior)
#         & (datos["Valor_MM_COP"] <= limite_superior)
#     ].copy()

#     if datos.empty:
#         st.warning(
#             "No existen observaciones válidas para realizar "
#             "el análisis seleccionado."
#         )
#         return

#     # ═══════════════════════════════════════════════════════════════════════
#     # 3. TABLA DE ESTADÍSTICOS POR SEGMENTO
#     # ═══════════════════════════════════════════════════════════════════════

#     st.markdown(
#         f"### 1️⃣ Estadísticos por {segmento.title()}"
#     )

#     tabla = (
#         datos
#         .groupby(segmento)["Valor_MM_COP"]
#         .agg(
#             N="count",
#             Media="mean",
#             Mediana="median",
#             Desv_Std="std",
#             Mínimo="min",
#             Máximo="max",
#             Q25=lambda x: x.quantile(0.25),
#             Q75=lambda x: x.quantile(0.75),
#         )
#         .reset_index()
#     )

#     columnas_numericas = [
#         "Media",
#         "Mediana",
#         "Desv_Std",
#         "Mínimo",
#         "Máximo",
#         "Q25",
#         "Q75",
#     ]

#     tabla[columnas_numericas] = (
#         tabla[columnas_numericas].round(2)
#     )

#     st.caption(
#         f"**{variable} — {anio}** | "
#         f"Valores monetarios en **Miles de Millones COP**"
#     )

#     # st.dataframe(
#     #     tabla.set_index(segmento),
#     #     use_container_width=True,
#     #     height=420,
#     # )
    
#     # Altura dinámica: solamente se reserva espacio para las filas
#     # realmente existentes en el dataset.
#     numero_segmentos = len(tabla)

#     altura_fila = 35
#     altura_encabezado = 38
#     altura_tabla = (
#         altura_encabezado
#         + (numero_segmentos * altura_fila)
#         + 8
#     )

#     st.dataframe(
#         tabla.set_index(segmento),
#         use_container_width=True,
#         height=altura_tabla,
#     )

#     # ═══════════════════════════════════════════════════════════════════════
#     # 4. RADAR / ARAÑA
#     # ═══════════════════════════════════════════════════════════════════════

#     st.markdown(
#         "### 2️⃣ Radar: Perfil Financiero Mediano por Segmento"
#     )

#     st.caption(
#         f"Perfil financiero normalizado para {anio}. "
#         "Cada eje representa una de las cinco variables financieras."
#     )

#     variables_radar = [
#         VARIABLES_SEGMENTO[v][anio]
#         for v in VARIABLES_SEGMENTO
#     ]

#     columnas_radar = []

#     for columna_radar in variables_radar:

#         columna_resuelta = _resolver_columna_distribucion(
#             df,
#             columna_radar,
#         )

#         if columna_resuelta is not None:
#             columnas_radar.append(
#                 columna_resuelta
#             )

#     if len(columnas_radar) < 2:
#         st.warning(
#             "No existen suficientes variables financieras "
#             "para construir el Radar."
#         )
#         return

#     radar_df = df[
#         [segmento] + columnas_radar
#     ].copy()

#     for col in columnas_radar:
#         radar_df[col] = pd.to_numeric(
#             radar_df[col],
#             errors="coerce",
#         ) / DIVISOR_MM_COP

#     # Normalización de cada variable.
#     medianas = radar_df.groupby(segmento)[
#         columnas_radar
#     ].median()

#     # Evitar división por cero cuando todos los segmentos
#     # tengan el mismo valor.
#     medianas_norm = medianas.copy()

#     for col in columnas_radar:

#         minimo = medianas[col].min()
#         maximo = medianas[col].max()

#         if pd.isna(minimo) or pd.isna(maximo):
#             medianas_norm[col] = 0
#         elif maximo == minimo:
#             medianas_norm[col] = 0.5
#         else:
#             medianas_norm[col] = (
#                 (medianas[col] - minimo)
#                 / (maximo - minimo)
#             )

#     # ═══════════════════════════════════════════════════════════════════════
#     # 5. RADAR
#     # ═══════════════════════════════════════════════════════════════════════

#     fig = go.Figure()

#     colores_actuales = COLORES_RADAR_SEGMENTO.get(
#         segmento,
#         {},
#     )

#     etiquetas_radar = [
#         "Ingresos Op.",
#         "Ganancia/Pérdida",
#         "Total Activos",
#         "Total Pasivos",
#         "Total Patrimonio",
#     ]

#     def _hex_to_rgba_segmento(
#         hex_color,
#         alpha=0.16,
#     ):
#         hex_color = hex_color.lstrip("#")

#         r = int(hex_color[0:2], 16)
#         g = int(hex_color[2:4], 16)
#         b = int(hex_color[4:6], 16)

#         return (
#             f"rgba({r},{g},{b},{alpha})"
#         )

#     for seg in medianas_norm.index:

#         llave_seg = _normalizar_segmento_radar(
#             seg
#         )

#         color_seg = colores_actuales.get(
#             llave_seg,
#             "#B0B7C3",
#         )

#         valores = medianas_norm.loc[
#             seg,
#             columnas_radar,
#         ].fillna(0).values

#         valores = np.asarray(
#             valores,
#             dtype=float,
#         )

#         # Cerrar el polígono.
#         r = np.append(
#             valores,
#             valores[0],
#         )

#         theta = (
#             etiquetas_radar[:len(valores)]
#             + [etiquetas_radar[0]]
#         )

#         fig.add_trace(
#             go.Scatterpolar(
#                 r=r,
#                 theta=theta,
#                 name=str(seg),
#                 mode="lines+markers",

#                 line=dict(
#                     color=color_seg,
#                     width=4,
#                 ),

#                 marker=dict(
#                     size=11,
#                     color=color_seg,
#                     line=dict(
#                         color="#FFFFFF",
#                         width=1.5,
#                     ),
#                 ),

#                 fill="toself",

#                 fillcolor=_hex_to_rgba_segmento(
#                     color_seg,
#                     0.18,
#                 ),

#                 opacity=0.90,

#                 hovertemplate=(
#                     f"<b>{seg}</b>"
#                     "<br>%{theta}: %{r:.3f}"
#                     "<extra></extra>"
#                 ),
#             )
#         )

#     # ═══════════════════════════════════════════════════════════════════════
#     # 6. CONFIGURACIÓN VISUAL DEL RADAR
#     # ═══════════════════════════════════════════════════════════════════════

#     fig.update_layout(

#         height=900,

#         title=dict(
#             text=(
#                 f"Perfil Financiero Mediano por "
#                 f"{segmento.title()} — {anio} "
#                 "<br><sup>Valores normalizados entre 0 y 1</sup>"
#             ),
#             x=0.02,
#             font=dict(
#                 size=24,
#                 color="#FFFFFF",
#             ),
#         ),

#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",

#         polar=dict(

#             bgcolor="rgba(0,0,0,0)",

#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1],
#                 tickfont=dict(
#                     size=13,
#                     color="#FFFFFF",
#                 ),
#                 gridcolor=(
#                     "rgba(255,255,255,0.22)"
#                 ),
#                 gridwidth=1,
#                 tickformat=".1f",
#             ),

#             angularaxis=dict(
#                 tickfont=dict(
#                     size=18,
#                     color="#FFFFFF",
#                 ),
#                 gridcolor=(
#                     "rgba(255,255,255,0.22)"
#                 ),
#                 linecolor=(
#                     "rgba(255,255,255,0.35)"
#                 ),
#                 linewidth=1,
#             ),
#         ),

#         legend=dict(
#             orientation="v",
#             yanchor="middle",
#             y=0.5,
#             xanchor="left",
#             x=1.03,
#             font=dict(
#                 size=16,
#                 color="#FFFFFF",
#             ),
#             bgcolor="rgba(0,0,0,0)",
#         ),

#         margin=dict(
#             l=100,
#             r=260,
#             t=120,
#             b=90,
#         ),
#     )

#     st.plotly_chart(
#         fig,
#         use_container_width=True,
#         config={
#             "displaylogo": False,
#             "toImageButtonOptions": {
#                 "format": "png",
#                 "filename": (
#                     f"radar_{segmento.lower()}_"
#                     f"{anio}"
#                 ),
#                 "scale": 3,
#             },
#         },
#     )

#     st.caption(
#         "El Radar utiliza la mediana de cada variable por segmento y "
#         "una normalización Min-Max independiente para cada variable. "
#         "Por tanto, los valores del Radar son índices relativos entre "
#         "0 y 1 y no Miles de Millones COP."
#     )

#     st.info(
#         "Las cifras monetarias de la tabla están expresadas en "
#         "**Miles de Millones COP**. El Radar es una representación "
#         "**normalizada**, por lo que sus ejes no representan valores "
#         "monetarios absolutos."
#     )

# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA: 🔍 ANÁLISIS POR SEGMENTO
# ═══════════════════════════════════════════════════════════════════════════


VARIABLES_SEGMENTO = {
    "Ingresos Op.": {
        2022: "INGRESOS OPERACIONALES 2022*",
        2023: "INGRESOS OPERACIONALES 2023*",
        2024: "INGRESOS OPERACIONALES 2024*",
    },
    "Ganancia/Pérdida": {
        2022: "GANANCIA (PERDIDA) 2022",
        2023: "GANANCIA (PERDIDA) 2023",
        2024: "GANANCIA (PERDIDA) 2024",
    },
    "Total Activos": {
        2022: "TOTAL ACTIVOS 2022",
        2023: "TOTAL ACTIVOS 2023",
        2024: "TOTAL ACTIVOS 2024",
    },
    "Total Pasivos": {
        2022: "TOTAL PASIVOS 2022",
        2023: "TOTAL PASIVOS 2023",
        2024: "TOTAL PASIVOS 2024",
    },
    "Total Patrimonio": {
        2022: "TOTAL PATRIMONIO 2022",
        2023: "TOTAL PATRIMONIO 2023",
        2024: "TOTAL PATRIMONIO 2024",
    },
}


def _analisis_segmento(df: pd.DataFrame):

    st.subheader("🔍 Análisis por Segmento")

    st.markdown(
        """
        Análisis estadístico de las variables financieras por
        **Macrosector, Región y Supervisor**.

        Las cifras monetarias de la tabla se expresan en
        **Miles de Millones COP**.
        """
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SELECTORES
    # ═══════════════════════════════════════════════════════════════════════

    col1, col2, col3 = st.columns([1.2, 1.8, 1])

    with col1:
        segmento = st.selectbox(
            "Agrupar por:",
            [
                "MACROSECTOR",
                "REGION",
                "SUPERVISOR",
            ],
            key="segmento_agrupacion_new_v2",
        )

    with col2:
        variable = st.selectbox(
            "Variable a analizar:",
            list(VARIABLES_SEGMENTO.keys()),
            key="segmento_variable_new_v2",
        )

    with col3:
        anio = st.selectbox(
            "Año:",
            [2024, 2023, 2022],
            key="segmento_anio_new_v2",
        )

    columna = _resolver_columna_distribucion(
        df,
        VARIABLES_SEGMENTO[variable][anio],
    )

    if columna is None:
        st.error(
            f"No se encontró la columna correspondiente a "
            f"{variable} — {anio}."
        )
        return

    if segmento not in df.columns:
        st.error(
            f"No existe la columna `{segmento}` en el DataFrame."
        )
        return

    # ═══════════════════════════════════════════════════════════════════════
    # DATOS
    # ═══════════════════════════════════════════════════════════════════════

    datos = df[
        [
            segmento,
            columna,
        ]
    ].copy()

    datos[columna] = pd.to_numeric(
        datos[columna],
        errors="coerce",
    )

    datos["Valor"] = (
        datos[columna] / DIVISOR_MM_COP
    )

    datos["SEGMENTO_NORMALIZADO"] = datos[
        segmento
    ].map(_normalizar_segmento_radar)

    datos = datos.dropna(
        subset=[
            "SEGMENTO_NORMALIZADO",
            "Valor",
        ]
    )

    # ═══════════════════════════════════════════════════════════════════════
    # TRATAMIENTO FINANCIERO — MISMO CRITERIO DE new_eda_VII.py
    # ═══════════════════════════════════════════════════════════════════════

    limite_inferior, limite_superior = RANGOS_DISTRIBUCION[
        variable
    ]

    datos = datos[
        (datos["Valor"] >= limite_inferior)
        & (datos["Valor"] <= limite_superior)
    ].copy()

    if datos.empty:
        st.warning(
            "No existen observaciones dentro del rango financiero "
            "establecido para esta variable."
        )
        return

    # ═══════════════════════════════════════════════════════════════════════
    # 1. TABLA
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown(
        f"### 1️⃣ Estadísticos por "
        f"{segmento.title()}"
    )

    tabla = (
        datos
        .groupby("SEGMENTO_NORMALIZADO")["Valor"]
        .agg(
            N="count",
            Media="mean",
            Mediana="median",
            Desv_Std="std",
            Mínimo="min",
            Máximo="max",
            Q25=lambda x: x.quantile(0.25),
            Q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
        .rename(
            columns={
                "SEGMENTO_NORMALIZADO": segmento
            }
        )
    )

    columnas_estadisticas = [
        "Media",
        "Mediana",
        "Desv_Std",
        "Mínimo",
        "Máximo",
        "Q25",
        "Q75",
    ]

    tabla[columnas_estadisticas] = (
        tabla[columnas_estadisticas].round(2)
    )

    st.caption(
        f"**{variable} — {anio}** | "
        "Valores en **Miles de Millones COP**"
    )

    # Altura EXACTA según el número de segmentos.
    numero_segmentos = len(tabla)

    altura_fila = 35
    altura_encabezado = 38

    altura_tabla = (
        altura_encabezado
        + numero_segmentos * altura_fila
        + 8
    )

    st.dataframe(
        tabla.set_index(segmento),
        use_container_width=True,
        height=altura_tabla,
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 2. CONSTRUCCIÓN DE LOS DOS RADARES
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown(
        "### 2️⃣ Perfil Financiero por Segmento"
    )

    columnas_financieras = []

    for variable_radar, mapa_anios in VARIABLES_SEGMENTO.items():

        columna_radar = _resolver_columna_distribucion(
            df,
            mapa_anios[anio],
        )

        if columna_radar is not None:
            columnas_financieras.append(
                (
                    variable_radar,
                    columna_radar,
                )
            )

    if len(columnas_financieras) < 2:
        st.warning(
            "No existen suficientes variables financieras "
            "para construir los Radar."
        )
        return

    # ═══════════════════════════════════════════════════════════════════════
    # CONSTRUIR DATASET DE LAS CINCO VARIABLES
    # ═══════════════════════════════════════════════════════════════════════

    radar_data = pd.DataFrame(
        index=df.index
    )

    radar_data["SEGMENTO"] = (
        df[segmento]
        .map(_normalizar_segmento_radar)
    )

    for variable_radar, columna_radar in columnas_financieras:

        serie = pd.to_numeric(
            df[columna_radar],
            errors="coerce",
        ) / DIVISOR_MM_COP

        lim_inf, lim_sup = RANGOS_DISTRIBUCION[
            variable_radar
        ]

        serie = serie.where(
            (serie >= lim_inf)
            & (serie <= lim_sup)
        )

        radar_data[variable_radar] = serie

    radar_data = radar_data.dropna(
        subset=["SEGMENTO"],
    )

    # ═══════════════════════════════════════════════════════════════════════
    # MEDIAS Y MEDIANAS POR SEGMENTO
    # ═══════════════════════════════════════════════════════════════════════

    medias = (
        radar_data
        .groupby("SEGMENTO")[
            list(VARIABLES_SEGMENTO.keys())
        ]
        .mean()
    )

    medianas = (
        radar_data
        .groupby("SEGMENTO")[
            list(VARIABLES_SEGMENTO.keys())
        ]
        .median()
    )

    # ═══════════════════════════════════════════════════════════════════════
    # NORMALIZACIÓN MIN-MAX
    # ═══════════════════════════════════════════════════════════════════════

    def _normalizar_radar(tabla_radar):

        resultado = tabla_radar.copy()

        for col in resultado.columns:

            minimo = resultado[col].min()
            maximo = resultado[col].max()

            if pd.isna(minimo) or pd.isna(maximo):
                resultado[col] = 0.0

            elif maximo == minimo:
                resultado[col] = 0.5

            else:
                resultado[col] = (
                    (resultado[col] - minimo)
                    / (maximo - minimo)
                )

        return resultado

    medias_norm = _normalizar_radar(medias)
    medianas_norm = _normalizar_radar(medianas)

    # ═══════════════════════════════════════════════════════════════════════
    # PALETA SEGÚN SEGMENTACIÓN
    # ═══════════════════════════════════════════════════════════════════════

    if segmento == "MACROSECTOR":
        paleta = COLORES_RADAR_MACROSECTOR

    elif segmento == "REGION":
        paleta = COLORES_RADAR_REGION

    else:
        paleta = COLORES_RADAR_SUPERVISOR

    etiquetas_radar = list(
        VARIABLES_SEGMENTO.keys()
    )

    def _color_segmento(nombre):

        nombre = _normalizar_segmento_radar(
            nombre
        )

        return paleta.get(
            nombre,
            "#A0A0A0",
        )

    def _rgba(hex_color, alpha=0.15):

        hex_color = hex_color.lstrip("#")

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        return (
            f"rgba({r},{g},{b},{alpha})"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # FUNCIÓN DE CONSTRUCCIÓN DEL RADAR
    # ═══════════════════════════════════════════════════════════════════════

    def _crear_radar(
        datos_radar,
        titulo,
    ):

        fig = go.Figure()

        for nombre_segmento in datos_radar.index:

            valores = datos_radar.loc[
                nombre_segmento,
                etiquetas_radar,
            ].fillna(0).values

            valores = np.asarray(
                valores,
                dtype=float,
            )

            r = np.append(
                valores,
                valores[0],
            )

            theta = (
                etiquetas_radar
                + [etiquetas_radar[0]]
            )

            color = _color_segmento(
                nombre_segmento
            )

            fig.add_trace(
                go.Scatterpolar(
                    r=r,
                    theta=theta,
                    name=str(nombre_segmento),
                    mode="lines+markers",

                    line=dict(
                        color=color,
                        width=5,
                    ),

                    marker=dict(
                        color=color,
                        size=12,
                        line=dict(
                            color="#FFFFFF",
                            width=1.5,
                        ),
                    ),

                    fill="toself",

                    fillcolor=_rgba(
                        color,
                        0.22,
                    ),

                    hovertemplate=(
                        f"<b>{nombre_segmento}</b>"
                        "<br>%{theta}: %{r:.3f}"
                        "<extra></extra>"
                    ),
                )
            )

        # fig.update_layout(

        #     height=850,

        #     title=dict(
        #         text=titulo,
        #         font=dict(
        #             size=23,
        #         ),
        #         x=0.02,
        #     ),

        #     polar=dict(

        #         radialaxis=dict(
        #             visible=True,
        #             range=[0, 1],
        #             tickfont=dict(
        #                 size=13,
        #             ),
        #             gridcolor=(
        #                 "rgba(255,255,255,0.22)"
        #             ),
        #         ),

        #         angularaxis=dict(
        #             tickfont=dict(
        #                 size=18,
        #             ),
        #             gridcolor=(
        #                 "rgba(255,255,255,0.22)"
        #             ),
        #         ),
        #     ),

        #     legend=dict(
        #         orientation="v",
        #         yanchor="middle",
        #         y=0.5,
        #         xanchor="left",
        #         x=1.03,
        #         font=dict(
        #             size=15,
        #         ),
        #     ),

        #     margin=dict(
        #         l=90,
        #         r=260,
        #         t=100,
        #         b=80,
        #     ),
        # )

        # return fig
        
        fig.update_layout(

            # ───────────────────────────────────────────────────────────────
            # FONDO NEGRO DEL RADAR
            # ───────────────────────────────────────────────────────────────
            paper_bgcolor="#000000",
            plot_bgcolor="#000000",

            height=850,

            title=dict(
                text=titulo,
                font=dict(
                    size=23,
                    color="#FFFFFF",
                ),
                x=0.02,
            ),
            
            # ───────────────────────────────────────────────────────────────
            # CONFIGURACIÓN DE LA "ARAÑA"
            # ───────────────────────────────────────────────────────────────
            polar=dict(

                bgcolor="#000000",

            radialaxis=dict(
                visible=True,

                # Escala normalizada
                range=[0, 1],

                # Valores 0.0 - 1.0 en BLANCO
                tickfont=dict(
                    size=16,
                    color="#FFFFFF",
                    family="Arial",
                ),

                tickformat=".1f",

                # Círculos concéntricos de la araña
                gridcolor="rgba(255,255,255,0.38)",
                gridwidth=1.5,

                # Eje radial central
                linecolor="rgba(255,255,255,0.55)",
                linewidth=1.5,

                # Mayor contraste
                ticks="outside",
            ),

            # ───────────────────────────────────────────────────────────
            # ETIQUETAS DE LAS VARIABLES
            # ───────────────────────────────────────────────────────────
            angularaxis=dict(

                tickfont=dict(
                    size=19,
                    color="#FFFFFF",
                    family="Arial",
                ),

                # Líneas radiales que forman la araña
                gridcolor="rgba(255,255,255,0.32)",
                gridwidth=1.5,

                linecolor="rgba(255,255,255,0.55)",
                linewidth=1.5,

                direction="clockwise",
            ),
        ),

            # ───────────────────────────────────────────────────────────────
            # LEYENDA
            # ───────────────────────────────────────────────────────────────
            legend=dict(
                orientation="v",

                yanchor="middle",
                y=0.5,

                xanchor="left",
                x=1.03,

                font=dict(
                    size=16,
                    color="#FFFFFF",
                    family="Arial",
                ),

                bgcolor="rgba(0,0,0,0.85)",

                bordercolor="rgba(255,255,255,0.30)",
                borderwidth=1,
        ),

            margin=dict(
                l=90,
                r=280,
                t=110,
                b=80,
            ),
        )
        
        return fig  
   
    # ═══════════════════════════════════════════════════════════════════════
    # RADAR 1 — MEDIANA
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown(
        "#### Radar: Perfil Financiero Mediano por Segmento"
    )

    fig_mediana = _crear_radar(
        medianas_norm,
        (
            f"Radar: Perfil Financiero Mediano "
            f"por {segmento.title()} — {anio}"
            "<br><sup>Índice normalizado 0–1</sup>"
        ),
    )

    st.plotly_chart(
        fig_mediana,
        use_container_width=True,
        config={
            "displaylogo": False,
        },
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RADAR 2 — MEDIA
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown(
        "#### Radar: Perfil Financiero de la Media por Segmento"
    )

    fig_media = _crear_radar(
        medias_norm,
        (
            f"Radar: Perfil Financiero de la Media "
            f"por {segmento.title()} — {anio}"
            "<br><sup>Índice normalizado 0–1</sup>"
        ),
    )

    st.plotly_chart(
        fig_media,
        use_container_width=True,
        config={
            "displaylogo": False,
        },
    )

    st.info(
        "Las cifras de la tabla están expresadas en "
        "**Miles de Millones COP**. "
        "Los dos Radar utilizan medias y medianas financieras "
        "normalizadas mediante Min-Max entre 0 y 1 para permitir "
        "comparar simultáneamente las cinco variables."
    )