"""
MÓDULO: data_loader_V.py
ETAPA CRISP-DM: Comprensión de los Datos (Data Understanding)
PROPÓSITO: Cargar el dataset consolidado ÚNICO (2022, 2023 y 2024) que ya
           fue fusionado y filtrado por completitud financiera de 3 años
           mediante `generar_dataset_consolidado.py`.

POR QUÉ UN SOLO ARCHIVO EN VEZ DE FUSIONAR EN CADA EJECUCIÓN
────────────────────────────────────────────────────────────────────────────
La fusión de las dos fuentes originales (2023-2024 y 2022-2023) y el filtro
de completitud financiera de 3 años son una transformación de datos FIJA
(Preparación de Datos en CRISP-DM), no una decisión que deba tomar cada
usuario de la app cada vez que la abre. Por eso esa lógica vive en
`generar_dataset_consolidado.py` (se ejecuta una sola vez, o cuando cambien
los archivos fuente) y este módulo simplemente CARGA el resultado ya
consolidado: un único `.xlsx` con exactamente las empresas que tienen las
5 variables financieras reportadas en los 3 años.

CRITERIO DE COMPLETITUD APLICADO EN EL ARCHIVO CONSOLIDADO
────────────────────────────────────────────────────────────────────────────
Cada empresa del dataset tiene valor NO NULO en Ingresos Operacionales,
Ganancia/Pérdida, Total Activos, Total Pasivos y Total Patrimonio para
2022, 2023 Y 2024 simultáneamente (AND estricto, no OR). No hay huecos en
estas 15 columnas financieras.

    Universo tras fusionar ambas fuentes (con huecos): 10.448 empresas
    Universo final tras exigir los 3 años completos  :  7.533 empresas

El usuario avanzado puede seguir subiendo un archivo propio (por ejemplo,
una versión actualizada ya consolidada con el mismo criterio) desde la
interfaz; si no sube nada, se usa el archivo consolidado por defecto del
proyecto.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st

# ─── Constantes de columnas financieras por año ───────────────────────────────

COLUMNAS_FINANCIERAS_2024 = [
    "INGRESOS OPERACIONALES 2024*",
    "GANANCIA (PERDIDA) 2024",
    "TOTAL ACTIVOS 2024",
    "TOTAL PASIVOS 2024",
    "TOTAL PATRIMONIO 2024",
]
COLUMNAS_FINANCIERAS_2023 = [
    "INGRESOS OPERACIONALES 2023*",
    "GANANCIA (PERDIDA) 2023",
    "TOTAL ACTIVOS 2023",
    "TOTAL PASIVOS 2023",
    "TOTAL PATRIMONIO 2023",
]
COLUMNAS_FINANCIERAS_2022 = [
    "INGRESOS OPERACIONALES 2022",
    "GANANCIA (PERDIDA) 2022",
    "TOTAL ACTIVOS 2022",
    "TOTAL PASIVOS 2022",
    "TOTAL PATRIMONIO 2022",
]
TODAS_COLUMNAS_FINANCIERAS = COLUMNAS_FINANCIERAS_2024 + COLUMNAS_FINANCIERAS_2023 + COLUMNAS_FINANCIERAS_2022

COLUMNAS_CATEGORICAS = [
    "MACROSECTOR", "REGION", "DEPARTAMENTO DOMICILIO", "SUPERVISOR",
    "Punto de entrada", "GRUPO EN NIIF",
]

NOMBRE_ARCHIVO_DEFAULT = "df_empresas_2022_2023_2024_consolidado con Ranking.xlsx"


# ─── Carga y validación ───────────────────────────────────────────────────────

def _limpiar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNAS_CATEGORICAS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan})
    return df


def _asegurar_tipos_numericos(df: pd.DataFrame) -> pd.DataFrame:
    """Defensivo: por si el archivo consolidado fue editado manualmente y
    algún valor financiero quedó como texto."""
    for col in TODAS_COLUMNAS_FINANCIERAS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _validar_completitud(df: pd.DataFrame) -> int:
    """Retorna el número de filas que NO cumplen el criterio de completitud
    de 3 años (debería ser 0 en el archivo consolidado oficial)."""
    cols = [c for c in TODAS_COLUMNAS_FINANCIERAS if c in df.columns]
    if not cols:
        return 0
    return int(df[cols].isna().any(axis=1).sum())


def cargar_datos_consolidados(archivo=None) -> pd.DataFrame:
    """
    Carga el dataset consolidado desde `archivo` (ruta, UploadedFile de
    Streamlit, o None para usar el archivo por defecto del proyecto).
    """
    if archivo is None:
        archivo = _resolver_ruta_default()

    if hasattr(archivo, "read"):
        nombre = archivo.name.lower()
        if nombre.endswith(".csv"):
            df = pd.read_csv(archivo, encoding="utf-8-sig")
        else:
            # El archivo consolidado tiene una fila de agrupación por año (fila 1)
            # antes del encabezado real de columnas (fila 2) → header=1 (0-indexed).
            df = pd.read_excel(archivo, engine="openpyxl", header=1)
    else:
        if str(archivo).lower().endswith(".csv"):
            df = pd.read_csv(archivo, encoding="utf-8-sig")
        else:
            df = pd.read_excel(archivo, engine="openpyxl", header=1)

    df = _asegurar_tipos_numericos(df)
    df = _limpiar_categoricas(df)
    return df


def _resolver_ruta_default() -> str:
    """
    Busca el archivo consolidado por defecto en múltiples ubicaciones:
    directorio del módulo, carpeta padre (raíz del proyecto) y cwd de
    Streamlit.
    """
    dir_modulo  = os.path.dirname(os.path.abspath(__file__))
    dir_padre   = os.path.dirname(dir_modulo)
    dir_trabajo = os.getcwd()

    candidatos = [
        os.path.join(dir_modulo,  NOMBRE_ARCHIVO_DEFAULT),
        os.path.join(dir_padre,   NOMBRE_ARCHIVO_DEFAULT),
        os.path.join(dir_trabajo, NOMBRE_ARCHIVO_DEFAULT),
    ]

    for ruta in candidatos:
        if os.path.isfile(ruta):
            return ruta

    rutas_str = "\n  • ".join(candidatos)
    raise FileNotFoundError(
        f"No se encontró '{NOMBRE_ARCHIVO_DEFAULT}' en ninguna de las rutas buscadas:\n  • {rutas_str}\n\n"
        f"Genera este archivo ejecutando: python generar_dataset_consolidado.py"
    )


# ─── UI Streamlit ─────────────────────────────────────────────────────────────

def mostrar_carga_ui():
    """
    Renderiza la interfaz Streamlit para carga de datos. Por defecto usa el
    dataset consolidado único del proyecto (2022, 2023 y 2024, solo empresas
    con los 3 años completos). Opcionalmente permite subir un archivo propio
    que siga el mismo esquema de columnas.
    """
    st.header("📂 Carga de Datos")
    st.markdown(
        """
        La aplicación usa un **dataset único ya consolidado**, que fusiona las
        dos fuentes originales (2023-2024 y 2022-2023) y conserva **solo las
        empresas con información financiera completa en los tres años
        evaluados: 2022, 2023 y 2024**. No se admiten espacios vacíos en estas
        variables — de ahí que el universo de análisis pase de ~10.400 a
        **7.533 empresas**.

        Si no subes ningún archivo, se usa el dataset consolidado por defecto
        del proyecto. También puedes subir tu propio archivo `.xlsx`/`.csv`
        si ya sigue el mismo esquema de columnas.
        """
    )

    archivo = st.file_uploader(
        "Archivo consolidado (opcional)",
        type=["xlsx", "xls", "csv"],
        help="Debe seguir el esquema de columnas del dataset consolidado del proyecto.",
    )

    try:
        if archivo is not None:
            df = cargar_datos_consolidados(archivo)
            fuente_nombre = archivo.name
        else:
            st.info("ℹ️ Usando el dataset consolidado por defecto del proyecto.")
            df = cargar_datos_consolidados(None)
            fuente_nombre = NOMBRE_ARCHIVO_DEFAULT

        st.success(
            f"✅ Datos cargados exitosamente: **{fuente_nombre}** "
            f"— {df.shape[0]:,} empresas, {df.shape[1]} columnas."
        )

        incompletas = _validar_completitud(df)
        if incompletas > 0:
            st.warning(
                f"⚠️ {incompletas:,} empresas no tienen los 3 años financieros completos "
                f"en este archivo (no cumplen el criterio de completitud del proyecto)."
            )
        else:
            st.caption("✅ Verificado: todas las empresas tienen las 5 variables financieras completas en 2022, 2023 y 2024.")

        _mostrar_preview(df)
        st.session_state["df"] = df
        st.session_state["fuente_datos"] = fuente_nombre
        return df

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None


def _mostrar_preview(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Empresas",     f"{df.shape[0]:,}")
    col2.metric("📋 Variables",     f"{df.shape[1]}")
    col3.metric("🏭 Macrosectores", df["MACROSECTOR"].nunique() if "MACROSECTOR" in df.columns else "N/A")
    col4.metric("🗺️ Regiones",      df["REGION"].nunique()      if "REGION"      in df.columns else "N/A")

    with st.expander("👁️ Vista previa del DataFrame (primeras 10 filas)"):
        st.dataframe(df.head(10), use_container_width=True)