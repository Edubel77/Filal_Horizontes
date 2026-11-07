# """
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  ARCHIVO PRINCIPAL: new_main_streamlit.py                            ║
# ║  PROYECTO: Modelo Financiero Predictivo con IA — REINICIO            ║
# ║  DATOS: 7,533 Empresas Colombianas — 2022, 2023 y 2024 (completo)    ║
# ║  METODOLOGÍA: CRISP-DM                                                ║
# ║  PLATAFORMA: Streamlit                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

# PUNTO DE PARTIDA
# ────────────────────────────────────────────────────────────────────────────
# Este archivo reinicia el proyecto sobre la base consolidada final:
# df_empresas_2022_2023_2024_consolidado.xlsx (7.533 empresas con las 5
# variables financieras completas en los 3 años, más RANKING 2022-2023 y
# RANKING 2023-2024 recalculados sobre ese universo).

# PATRÓN DE CRECIMIENTO INCREMENTAL — "REGISTRO + IMPORT DEFENSIVO"
# ────────────────────────────────────────────────────────────────────────────
# El proyecto se construye fase por fase. Para poder ver el resultado de cada
# módulo apenas se crea, sin que el archivo principal se rompa por fases que
# aún no existen, se usa este patrón:

#   1. Cada fase CRISP-DM vive en su propio módulo `new_<fase>.py`, y expone
#      UNA sola función pública `mostrar_<fase>(df)` que recibe el DataFrame
#      ya cargado (contrato uniforme, igual para todas las fases).

#   2. `_SECCIONES` es el ÚNICO lugar que hay que tocar al agregar una fase
#      nueva: se agrega una entrada con su import (protegido por try/except)
#      y su función de render. El sidebar y el router se generan a partir de
#      este diccionario — nunca hay que editar el sidebar a mano.

#   3. Si el módulo de una fase todavía no existe, esa sección se marca
#      automáticamente como "⏳ Pendiente" y muestra un placeholder, en vez
#      de reventar la aplicación. Así se puede repartir la construcción en
#      varias sesiones sin dejar la app en un estado roto.

#   4. El DataFrame se carga UNA sola vez (vía `data_loader_V.py`, que ya
#      apunta al archivo consolidado correcto) y se guarda en
#      `st.session_state["df"]`, disponible para todas las fases sin
#      recargarlo en cada clic.

# CÓMO AGREGAR UNA FASE NUEVA (ej. "new_eda.py")
# ────────────────────────────────────────────────────────────────────────────
#   1. Crear `new_eda.py` con una función `mostrar_eda(df)`.
#   2. En este archivo, dentro de `_SECCIONES`, agregar una entrada:
#          "eda": {
#              "label": "🔍 EDA",
#              "estado": "pendiente",   # se actualiza a "completa" al terminar
#              "modulo": "new_eda",
#              "funcion": "mostrar_eda",
#          },
#   3. Nada más cambia: el sidebar, el router y el placeholder se generan solos.
# """

# import importlib
# import os
# import sys

# import streamlit as st

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# from data_loader_V import mostrar_carga_ui
# from new_crisp_dm_overview import mostrar_crisp_dm

# # ── Configuración de la página ────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Predicción Financiera IA | Colombia 2022 a 2024",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"] {background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);}
#     [data-testid="stSidebar"] * {color: #e0e0e0 !important;}
#     [data-testid="stSidebar"] .stRadio label {color: #e0e0e0 !important;}
#     .main-title {
#         font-size: 2.2rem; font-weight: 800;
#         background: linear-gradient(90deg, #00b4d8, #0077b6);
#         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
#         margin-bottom: 0.2rem;
#     }
#     .sub-title {font-size: 1rem; color: #6c757d; margin-bottom: 1.5rem;}
#     [data-testid="stMetric"] {
#         background: #1E293B; border-radius: 10px;
#         padding: 12px; border-left: 4px solid #0077b6;
#     }
#     .stTabs [data-baseweb="tab"] {font-size: 0.9rem; font-weight: 600;}
#     .stTabs [aria-selected="true"] {border-bottom: 3px solid #0077b6 !important;}
#     .stButton > button {
#         width: 100%; border-radius: 8px;
#         background: linear-gradient(90deg, #0077b6, #00b4d8);
#         color: white; font-weight: 700; border: none;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# # ── Registro central de secciones ─────────────────────────────────────────────
# # Único lugar a editar para agregar una fase nueva. "estado" es solo
# # informativo (se sobrescribe automáticamente a "pendiente" si el módulo
# # indicado no puede importarse todavía).
# _SECCIONES = {
#     "inicio":      {"label": "🏠 Inicio",                    "estado": "completa", "modulo": None, "funcion": None},
#     "carga":       {"label": "📂 Carga de Datos",             "estado": "completa", "modulo": None, "funcion": None},
#     "crisp":       {"label": "🔄 Marco CRISP-DM",             "estado": "completa", "modulo": None, "funcion": None},
#     # "eda":         {"label": "🔍 EDA",                        "estado": "pendiente", "modulo": "new_eda",               "funcion": "mostrar_eda"},
#     "eda":         {"label": "🔍 EDA",                        "modulo": "new_eda_VII",             "funcion": "mostrar_eda"},
#     "estadistica": {"label": "📊 Estadística Descriptiva",    "estado": "pendiente", "modulo": "new_estadistica",       "funcion": "mostrar_estadistica"},
#     "preparacion": {"label": "🔧 Preparación de Datos",       "estado": "pendiente", "modulo": "new_preparacion",       "funcion": "mostrar_preparacion"},
#     "modelado":    {"label": "🤖 Modelado",                   "estado": "pendiente", "modulo": "new_modelado",          "funcion": "mostrar_modelado"},
#     "evaluacion":  {"label": "📐 Evaluación",                 "estado": "pendiente", "modulo": "new_evaluacion",        "funcion": "mostrar_evaluacion"},
#     "prediccion":  {"label": "🚀 Despliegue / Predicción",    "estado": "pendiente", "modulo": "new_prediccion",        "funcion": "mostrar_prediccion"},
# }


# def _resolver_funcion(clave: str):
#     """
#     Intenta importar el módulo de la sección y obtener su función de render.
#     Retorna (funcion, disponible: bool). Si el módulo no existe aún, no
#     lanza excepción: retorna disponible=False para mostrar el placeholder.
#     """
#     info = _SECCIONES[clave]
#     if info["modulo"] is None:
#         return None, True  # secciones "core" (inicio/carga/crisp) no pasan por aquí

#     try:
#         mod = importlib.import_module(info["modulo"])
#         fn = getattr(mod, info["funcion"])
#         return fn, True
#     except (ImportError, ModuleNotFoundError, AttributeError):
#         return None, False


# # ── Sidebar: navegación ───────────────────────────────────────────────────────
# def _sidebar():
#     with st.sidebar:
#         st.markdown(
#             """
#             <div style='text-align:center; padding: 20px 0 10px 0;'>
#                 <div style='font-size:2.5rem'>📈</div>
#                 <div style='font-size:1.1rem; font-weight:700; color:#00b4d8'>Predicción Financiera</div>
#                 <div style='font-size:0.8rem; color:#aaa'>con Inteligencia Artificial</div>
#                 <div style='font-size:0.75rem; color:#888; margin-top:4px'>Colombia · 2022 a 2024</div>
#             </div>
#             <hr style='border-color:#333; margin: 10px 0'>
#             """,
#             unsafe_allow_html=True,
#         )

#         st.markdown("### 📌 Navegación CRISP-DM")

#         opciones = []
#         etiquetas_a_clave = {}
#         for clave, info in _SECCIONES.items():
#             _, disponible = _resolver_funcion(clave)
#             marca = "" if disponible else " 🚧"
#             etiqueta = f"{info['label']}{marca}"
#             opciones.append(etiqueta)
#             etiquetas_a_clave[etiqueta] = clave

#         seleccion = st.radio("", opciones, label_visibility="collapsed")
#         clave_seleccionada = etiquetas_a_clave[seleccion]

#         st.markdown("<hr style='border-color:#333'>", unsafe_allow_html=True)

#         if "df" in st.session_state and st.session_state["df"] is not None:
#             df = st.session_state["df"]
#             st.markdown(
#                 f"""
#                 <div style='background:#1e3a5f; border-radius:8px; padding:10px; font-size:0.82rem'>
#                 <b style='color:#00b4d8'>📊 Dataset Activo</b><br>
#                 <span style='color:#ccc'>📁 {st.session_state.get("fuente_datos","—")}</span><br>
#                 <span style='color:#ccc'>🏢 {len(df):,} empresas</span><br>
#                 <span style='color:#ccc'>📋 {df.shape[1]} variables</span>
#                 </div>
#                 """,
#                 unsafe_allow_html=True,
#             )
#         else:
#             st.markdown(
#                 "<div style='background:#3a1a1a; border-radius:8px; padding:10px; font-size:0.82rem; color:#ffaaaa'>"
#                 "⚠️ Ningún dataset cargado.<br>Ve a <b>📂 Carga de Datos</b></div>",
#                 unsafe_allow_html=True,
#             )

#         st.markdown(
#             """
#             <hr style='border-color:#333'>
#             <div style='font-size:0.75rem; color:#666; text-align:center'>
#             Metodología: CRISP-DM<br>
#             Framework: Streamlit + Plotly<br>
#             <b style='color:#00b4d8'>Reinicio · 7,533 empresas · 2022-2024</b>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )

#         return clave_seleccionada


# # ── Página de inicio ──────────────────────────────────────────────────────────
# # def _pagina_inicio():
# #     st.markdown(
# #         "<div class='main-title'>📈 Modelo Financiero Predictivo con IA</div>"
# #         "<div class='sub-title'>7,533 empresas colombianas · Años 2022, 2023 y 2024 completos · Metodología CRISP-DM</div>",
# #         unsafe_allow_html=True,
# #     )

# #     col1, col2, col3, col4 = st.columns(4)
# #     tiene_df = "df" in st.session_state and st.session_state["df"] is not None
# #     col1.metric("🏢 Empresas", f"{len(st.session_state['df']):,}" if tiene_df else "7,533")
# #     col2.metric("📅 Años completos", "2022 – 2024")
# #     completas = sum(1 for clave in _SECCIONES if _resolver_funcion(clave)[1])
# #     col3.metric("✅ Fases activas", f"{completas}/{len(_SECCIONES)}")
# #     col4.metric("🏷️ Ranking", "2022-23 · 2023-24")

# #     st.markdown("---")

# #     col_l, col_r = st.columns([1.4, 1])
# #     with col_l:
# #         st.markdown(
# #             """
# #             ### 📖 Punto de partida de este reinicio
# #             El proyecto retoma su desarrollo sobre el dataset **ya consolidado y depurado**:
# #             7.533 empresas con información financiera completa en **2022, 2023 y 2024**, cada
# #             una con su posición en el ranking anual por ingresos operacionales.

# #             ### 🗺️ Cómo avanza este dashboard
# #             Cada fase del CRISP-DM se agrega como un módulo independiente. Las fases marcadas
# #             con 🚧 en el menú lateral todavía no tienen módulo creado — aparecerán activas
# #             automáticamente en cuanto se construyan, sin tocar nada más de este archivo.
# #             """
# #         )
# #     with col_r:
# #         st.markdown("### 📊 Estado de las fases")
# #         for clave, info in _SECCIONES.items():
# #             _, disponible = _resolver_funcion(clave)
# #             icono = "✅" if disponible else "🚧"
# #             st.markdown(f"{icono} {info['label']}")

# # ── Página de inicio ──────────────────────────────────────────────────────────
# def _pagina_inicio():
#     st.markdown(
#         "<div class='main-title'>📈 Modelo Financiero Predictivo con IA</div>"
#         "<div class='sub-title'>Análisis de 7,533 empresas colombianas · Años 2022, 2023 y 2024 · Metodología CRISP-DM</div>",
#         unsafe_allow_html=True,
#     )

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("🏢 Empresas", "7,533")
#     col2.metric("📅 Años", "2022 - 2023 – 2024")
#     col3.metric("🏭 Macrosectores", "6")
#     col4.metric("🗺️ Regiones", "7")

#     st.markdown("---")

#     col_l, col_r = st.columns([1.4, 1])

#     with col_l:
#         st.markdown(
#             """
#             ### 📖 Acerca del Proyecto
#             Este proyecto aplica el marco **CRISP-DM** para desarrollar un modelo de **predicción financiera
#             con Inteligencia Artificial** sobre el universo de 7,533 empresas de Colombia.

#             El dataset incluye variables financieras clave como:
#             - 💵 Ingresos Operacionales
#             - 📊 Ganancia / Pérdida
#             - 🏦 Total Activos y Pasivos
#             - 🏛️ Patrimonio

#             Para los años **2022**, **2023** y **2024**, segmentado por **macrosector**, **región** y **departamento**.

#             ### 🗺️ Flujo de Trabajo
#             Navega por las secciones en la barra lateral siguiendo el orden natural del **CRISP-DM**:
#             1. Carga de datos
#             2. Marco metodológico
#             3. EDA (Análisis Exploratorio)
#             4. Estadística Descriptiva
#             5. Preparación de datos 
#             6. Modelado IA 
#             7. Evaluación 
#             8. Dashboard Predictivo 
#             """,
#         )



# # ── Placeholder para fases aún no construidas ─────────────────────────────────
# def _pagina_pendiente(clave: str):
#     info = _SECCIONES[clave]
#     st.header(info["label"])
#     st.info(
#         f"🚧 El módulo `{info['modulo']}.py` todavía no ha sido creado. "
#         f"En cuanto exista (con una función `{info['funcion']}(df)`), esta sección "
#         f"se activará automáticamente — no requiere cambios en `new_main_streamlit.py`."
#     )


# # ── Orquestador principal ─────────────────────────────────────────────────────
# def main():
#     clave = _sidebar()

#     if clave == "inicio":
#         _pagina_inicio()
#     elif clave == "carga":
#         mostrar_carga_ui()
#     elif clave == "crisp":
#         mostrar_crisp_dm()
#     else:
#         funcion, disponible = _resolver_funcion(clave)
#         if not disponible:
#             _pagina_pendiente(clave)
#         elif "df" not in st.session_state or st.session_state["df"] is None:
#             st.warning("⚠️ Primero carga el dataset en la sección **📂 Carga de Datos**.")
#         else:
#             funcion(st.session_state["df"])


# if __name__ == "__main__":
#     main()

# ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

"""
╔══════════════════════════════════════════════════════════════════════╗
║  ARCHIVO PRINCIPAL: new_main_streamlit.py                            ║
║  PROYECTO: Modelo Financiero Predictivo con IA — REINICIO            ║
║  DATOS: 7,533 Empresas Colombianas — 2022, 2023 y 2024 (completo)    ║
║  ARCHIVO FUENTE: df_empresas_2022_2023_2024_consolidado con          ║
║                  Ranking.xlsx                                        ║
║  METODOLOGÍA: CRISP-DM                                                ║
║  PLATAFORMA: Streamlit                                                ║
╚══════════════════════════════════════════════════════════════════════╝

PUNTO DE PARTIDA
────────────────────────────────────────────────────────────────────────────
Este archivo reinicia el proyecto sobre la base consolidada final:
"df_empresas_2022_2023_2024_consolidado con Ranking.xlsx" (7.533 empresas
con las 5 variables financieras completas en los 3 años, más RANKING
2022-2023 y RANKING 2023-2024 recalculados sobre ese universo). La lectura
de este archivo vive exclusivamente en `data_loader_V.py`.

PATRÓN DE CRECIMIENTO INCREMENTAL — "REGISTRO + IMPORT DEFENSIVO"
────────────────────────────────────────────────────────────────────────────
TODAS las secciones (incluidas "Carga de Datos" y "Marco CRISP-DM", no solo
las fases futuras) se resuelven de forma dinámica vía `importlib`, protegidas
con try/except. Si un módulo no está guardado en la misma carpeta, esa
sección se marca con 🚧 y muestra un aviso — el resto de la app sigue
funcionando con normalidad. Esto evita que un módulo faltante (como ocurrió
con `data_loader_V.py` en una versión anterior) tumbe toda la aplicación.

  1. Cada fase vive en su propio módulo y expone UNA función pública:
       · data_loader_V.py          → mostrar_carga_ui()        (sin argumentos)
       · new_crisp_dm_overview.py  → mostrar_crisp_dm()         (sin argumentos)
       · new_<fase>.py             → mostrar_<fase>(df)         (recibe el DataFrame)

  2. `_SECCIONES` es el ÚNICO lugar que hay que tocar al agregar una fase
     nueva: se agrega una entrada con su módulo y función. El sidebar y el
     router se generan a partir de este diccionario.

  3. `_SECCIONES_SIN_DATAFRAME` lista las secciones cuya función no recibe
     el DataFrame (cargan datos o son puramente informativas).

CÓMO AGREGAR UNA FASE NUEVA (ej. "new_eda.py")
────────────────────────────────────────────────────────────────────────────
  1. Crear `new_eda.py` con una función `mostrar_eda(df)`.
  2. En este archivo, dentro de `_SECCIONES`, agregar:
         "eda": {"label": "🔍 EDA", "modulo": "new_eda", "funcion": "mostrar_eda"},
  3. Nada más cambia: el sidebar, el router y el placeholder se generan solos.
"""

import importlib
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción Financiera IA | Colombia 2022-2024",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);}
    [data-testid="stSidebar"] * {color: #e0e0e0 !important;}
    [data-testid="stSidebar"] .stRadio label {color: #e0e0e0 !important;}
    .main-title {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(90deg, #00b4d8, #0077b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {font-size: 1rem; color: #6c757d; margin-bottom: 1.5rem;}
    [data-testid="stMetric"] {
        background: #1E293B; border-radius: 10px;
        padding: 12px; border-left: 4px solid #0077b6;
    }
    .stTabs [data-baseweb="tab"] {font-size: 0.9rem; font-weight: 600;}
    .stTabs [aria-selected="true"] {border-bottom: 3px solid #0077b6 !important;}
    .stButton > button {
        width: 100%; border-radius: 8px;
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        color: white; font-weight: 700; border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Registro central de secciones ─────────────────────────────────────────────
# Único lugar a editar para agregar una fase nueva.
_SECCIONES = {
    "inicio":      {"label": "🏠 Inicio",                    "modulo": None, "funcion": None},
    "carga":       {"label": "📂 Carga de Datos",             "modulo": "data_loader_V",         "funcion": "mostrar_carga_ui"},
    "crisp":       {"label": "🔄 Marco CRISP-DM",             "modulo": "new_crisp_dm_overview", "funcion": "mostrar_crisp_dm"},
    "eda":         {"label": "🔍 EDA",                        "modulo": "new_eda_VII",           "funcion": "mostrar_eda"},
    "estadistica": {"label": "📊 Estadística Descriptiva",    "modulo": "new_estadistica",       "funcion": "mostrar_estadistica"},
    "preparacion": {"label": "🔧 Preparación de Datos",       "modulo": "new_preparacion",       "funcion": "mostrar_preparacion"},
    "modelado":    {"label": "🤖 Modelado",                   "modulo": "new_modelado",          "funcion": "mostrar_modelado"},
    "evaluacion":  {"label": "📐 Evaluación",                 "modulo": "new_evaluacion",        "funcion": "mostrar_evaluacion"},
    "prediccion":  {"label": "🚀 Despliegue / Predicción",    "modulo": "new_despliegue_prediccion",        "funcion": "mostrar_prediccion"},
}

# Secciones cuya función de render NO recibe el DataFrame como argumento.
_SECCIONES_SIN_DATAFRAME = {"carga", "crisp"}


def _resolver_funcion(clave: str):
    """
    Intenta importar el módulo de la sección y obtener su función de render.
    Retorna (funcion, disponible: bool). Si el módulo no existe o falla al
    importarse, no lanza excepción: retorna disponible=False para mostrar
    el placeholder.
    """
    info = _SECCIONES[clave]
    if info["modulo"] is None:
        return None, True  # sección "inicio": no depende de ningún módulo externo

    try:
        mod = importlib.import_module(info["modulo"])
        mod = importlib.reload(mod)  # fuerza releer el archivo del disco en cada rerun
        fn = getattr(mod, info["funcion"])
        return fn, True
    except Exception:
        return None, False


# ── Sidebar: navegación ───────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 20px 0 10px 0;'>
                <div style='font-size:2.5rem'>📈</div>
                <div style='font-size:1.1rem; font-weight:700; color:#00b4d8'>Predicción Financiera</div>
                <div style='font-size:0.8rem; color:#aaa'>con Inteligencia Artificial</div>
                <div style='font-size:0.75rem; color:#888; margin-top:4px'>Colombia · 2022 – 2024</div>
            </div>
            <hr style='border-color:#333; margin: 10px 0'>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📌 Navegación CRISP-DM")

        opciones = []
        etiquetas_a_clave = {}
        for clave, info in _SECCIONES.items():
            _, disponible = _resolver_funcion(clave)
            marca = "" if disponible else " 🚧"
            etiqueta = f"{info['label']}{marca}"
            opciones.append(etiqueta)
            etiquetas_a_clave[etiqueta] = clave

        seleccion = st.radio("", opciones, label_visibility="collapsed")
        clave_seleccionada = etiquetas_a_clave[seleccion]

        st.markdown("<hr style='border-color:#333'>", unsafe_allow_html=True)

        if "df" in st.session_state and st.session_state["df"] is not None:
            df = st.session_state["df"]
            st.markdown(
                f"""
                <div style='background:#1e3a5f; border-radius:8px; padding:10px; font-size:0.82rem'>
                <b style='color:#00b4d8'>📊 Dataset Activo</b><br>
                <span style='color:#ccc'>📁 {st.session_state.get("fuente_datos","—")}</span><br>
                <span style='color:#ccc'>🏢 {len(df):,} empresas</span><br>
                <span style='color:#ccc'>📋 {df.shape[1]} variables</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#3a1a1a; border-radius:8px; padding:10px; font-size:0.82rem; color:#ffaaaa'>"
                "⚠️ Ningún dataset cargado.<br>Ve a <b>📂 Carga de Datos</b></div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <hr style='border-color:#333'>
            <div style='font-size:0.75rem; color:#666; text-align:center'>
            Metodología: CRISP-DM<br>
            Framework: Streamlit + Plotly<br>
            <b style='color:#00b4d8'>Reinicio · 7,533 empresas · 2022-2024</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return clave_seleccionada


# ── Página de inicio ──────────────────────────────────────────────────────────
def _pagina_inicio():
    st.markdown(
        "<div class='main-title'>📈 Modelo Financiero Predictivo con IA</div>"
        "<div class='sub-title'>7,533 empresas colombianas · Años 2022, 2023 y 2024 completos · Metodología CRISP-DM</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    tiene_df = "df" in st.session_state and st.session_state["df"] is not None
    col1.metric("🏢 Empresas", f"{len(st.session_state['df']):,}" if tiene_df else "7,533")
    col2.metric("📅 Años completos", "2022 – 2024")
    disponibles = sum(1 for clave in _SECCIONES if _resolver_funcion(clave)[1])
    col3.metric("✅ Secciones activas", f"{disponibles}/{len(_SECCIONES)}")
    col4.metric("🏷️ Ranking", "2022-23 · 2023-24")

    st.markdown("---")

    col_l, col_r = st.columns([1.4, 1])
    with col_l:
        st.markdown(
            """
            ### 📖 Punto de partida de este reinicio
            El proyecto retoma su desarrollo sobre el dataset **ya consolidado y depurado**
            (`df_empresas_2022_2023_2024_consolidado con Ranking.xlsx`): 7.533 empresas con
            información financiera completa en **2022, 2023 y 2024**, cada una con su
            posición en el ranking anual por ingresos operacionales.

            ### 🗺️ Cómo avanza este dashboard
            Cada fase del CRISP-DM se agrega como un módulo independiente. Las secciones
            marcadas con 🚧 en el menú lateral todavía no tienen su módulo guardado en esta
            carpeta — se activan automáticamente en cuanto el archivo correspondiente exista,
            sin tocar nada más de este archivo.
            """
        )
    with col_r:
        st.markdown("### 📊 Estado de las secciones")
        for clave, info in _SECCIONES.items():
            _, disponible = _resolver_funcion(clave)
            icono = "✅" if disponible else "🚧"
            st.markdown(f"{icono} {info['label']}")


# ── Placeholder para secciones cuyo módulo no está disponible ────────────────
def _pagina_pendiente(clave: str):
    info = _SECCIONES[clave]
    firma = "()" if clave in _SECCIONES_SIN_DATAFRAME else "(df)"
    st.header(info["label"])
    st.info(
        f"🚧 No se pudo importar el módulo `{info['modulo']}.py`. Verifica que el archivo "
        f"exista en la misma carpeta que `new_main_streamlit.py` y que defina una función "
        f"`{info['funcion']}{firma}`. En cuanto esté disponible, esta sección se activa "
        f"automáticamente — no requiere cambios en `new_main_streamlit.py`."
    )


# ── Orquestador principal ─────────────────────────────────────────────────────
def main():
    clave = _sidebar()

    if clave == "inicio":
        _pagina_inicio()
        return

    funcion, disponible = _resolver_funcion(clave)
    if not disponible:
        _pagina_pendiente(clave)
    elif clave in _SECCIONES_SIN_DATAFRAME:
        funcion()
    elif "df" not in st.session_state or st.session_state["df"] is None:
        st.warning("⚠️ Primero carga el dataset en la sección **📂 Carga de Datos**.")
    else:
        funcion(st.session_state["df"])


if __name__ == "__main__":
    main()