"""
MÓDULO: new_preparacion.py
ETAPA CRISP-DM: 3 — PREPARACIÓN DE LOS DATOS
PUNTO DE ENTRADA: mostrar_preparacion(df)

Arquitectura de esta versión
----------------------------
1. Marco y Ruta
2. Diagnóstico
3. Estrategia y Encoding
4. Feature Engineering
5. Partición Train / Test
6. Política de Outliers
7. Escalado y Transformaciones
8. Resultado
9. Base para el LLM
10. Validación y Entregables

Principio metodológico central
------------------------------
Las transformaciones DETERMINISTAS se pueden construir antes de la partición
(ej. ratios, magnitudes, logaritmos ya definidos, flags y tendencias). Las
transformaciones que APRENDEN información del conjunto de datos se ajustan
únicamente con TRAIN y después se aplican a TEST:

    • Frequency Encoding
    • vocabulario One-Hot
    • imputación
    • escalado
    • Feature Selection

Esto evita data leakage y deja a new_modelado.py un conjunto de datos listo
para entrenar/evaluar modelos.

Dependencias de los módulos fuente del proyecto
------------------------------------------------
    feature_engineering.py
    feature_selection.py

El módulo conserva además una segunda salida para el LLM:
    df_consulta + diccionario_datos

Las cifras para el LLM se calculan con Python; el LLM no calcula cifras.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Módulos fuente del Feature Engineering y Feature Selection del proyecto.
# ---------------------------------------------------------------------------
try:
    from feature_engineering import feature_engineering, get_fe_reporte
    _FE_IMPORT_ERROR = None
except Exception as exc_fe:  # intenta el módulo de proyecto entregado junto a esta versión
    try:
        from feature_engineering_proyecto import feature_engineering, get_fe_reporte
        _FE_IMPORT_ERROR = None
    except Exception as exc_fe2:
        feature_engineering = None
        get_fe_reporte = lambda df: {}
        _FE_IMPORT_ERROR = f"{exc_fe}; fallback feature_engineering_proyecto: {exc_fe2}"

try:
    from feature_selection import (
        feature_selection, get_feature_selection_reporte, CORRELATION_THRESHOLD,
    )
    _FS_IMPORT_ERROR = None
except Exception as exc_fs:  # intenta el módulo de proyecto entregado junto a esta versión
    try:
        from feature_selection_proyecto import (
            feature_selection, get_feature_selection_reporte, CORRELATION_THRESHOLD,
        )
        _FS_IMPORT_ERROR = None
    except Exception as exc_fs2:
        feature_selection = None
        get_feature_selection_reporte = lambda df: {}
        CORRELATION_THRESHOLD = 0.92
        _FS_IMPORT_ERROR = f"{exc_fs}; fallback feature_selection_proyecto: {exc_fs2}"


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════

COLS_IDENTIFICADORES = ["NIT", "RAZON SOCIAL"]
COLS_METADATO = ["Punto de entrada", "GRUPO EN NIIF"]
COL_CIIU = "CIIU"
COL_SECCION = "SECCION_CIIU"
SIN_DATO = "SIN DATO"
UMBRAL_OHE = 35
UMBRAL_IDENTIFICADOR = 0.5
ANIOS = (2022, 2023, 2024)

ING = "INGRESOS OPERACIONALES"
GAN = "GANANCIA (PERDIDA)"
ACT = "TOTAL ACTIVOS"
PAS = "TOTAL PASIVOS"
PAT = "TOTAL PATRIMONIO"
PREFIJOS_FIN = (ING, GAN, ACT, PAS, PAT)

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_TOP_N = 30

# Política formal de valores extremos para Machine Learning
OUTLIER_LOWER_Q = 0.01
OUTLIER_UPPER_Q = 0.99
OUTLIER_SKEW_THRESHOLD = 1.50
OUTLIER_MIN_NON_NULL = 20
OUTLIER_RATIO_KEYWORDS = (
    "MARGEN", "ROA", "ROE", "SOLVENCIA", "ENDEUDAMIENTO",
    "ROTACION", "APALANCAMIENTO", "CRECIMIENTO", "VARIACION",
    "DELTA", "CAMBIO", "VOLATILIDAD", "SEVERIDAD", "IMPACTO",
)

SECCIONES_CIIU = {
    "A": "AGRICULTURA, GANADERIA, CAZA, SILVICULTURA Y PESCA",
    "B": "EXPLOTACION DE MINAS Y CANTERAS",
    "C": "INDUSTRIAS MANUFACTURERAS",
    "D": "SUMINISTRO DE ELECTRICIDAD, GAS, VAPOR Y AIRE ACONDICIONADO",
    "E": "DISTRIBUCION DE AGUA, EVACUACION DE AGUAS RESIDUALES Y GESTION DE DESECHOS",
    "F": "CONSTRUCCION",
    "G": "COMERCIO AL POR MAYOR Y AL POR MENOR; REPARACION DE VEHICULOS",
    "H": "TRANSPORTE Y ALMACENAMIENTO",
    "I": "ALOJAMIENTO Y SERVICIOS DE COMIDA",
    "J": "INFORMACION Y COMUNICACIONES",
    "K": "ACTIVIDADES FINANCIERAS Y DE SEGUROS",
    "L": "ACTIVIDADES INMOBILIARIAS",
    "M": "ACTIVIDADES PROFESIONALES, CIENTIFICAS Y TECNICAS",
    "N": "ACTIVIDADES DE SERVICIOS ADMINISTRATIVOS Y DE APOYO",
    "O": "ADMINISTRACION PUBLICA Y DEFENSA",
    "P": "EDUCACION",
    "Q": "SALUD HUMANA Y ASISTENCIA SOCIAL",
    "R": "ARTES, ENTRETENIMIENTO Y RECREACION",
    "S": "OTRAS ACTIVIDADES DE SERVICIOS",
    "T": "ACTIVIDADES DE LOS HOGARES COMO EMPLEADORES",
    "U": "ORGANIZACIONES Y ENTIDADES EXTRATERRITORIALES",
    "Z": "SIN CLASIFICAR",
}

TIPO_DESC = {
    "int8": "Variables dummy 0/1 (One-Hot Encoding)",
    "int16": "Variables dummy/enteros pequeños",
    "int64": "Enteros",
    "float32": "Frecuencias relativas / features numéricas",
    "float64": "Features financieras / transformadas",
    "Int64": "Enteros con soporte de nulos",
}


# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES BÁSICAS
# ═══════════════════════════════════════════════════════════════════════════

def _quitar_tildes(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )


def _norm(valor: Any):
    if pd.isna(valor):
        return np.nan
    t = " ".join(_quitar_tildes(str(valor)).upper().split())
    return t if t else np.nan


def _es_numerica_esperada(col: str) -> bool:
    return str(col).startswith("RANKING") or str(col).startswith(PREFIJOS_FIN)


def _col_fin(d: pd.DataFrame, prefijo: str, anio: int):
    for c in d.columns:
        if str(c).replace("*", "").strip() == f"{prefijo} {anio}":
            return c
    return None


# Catálogo de objetivos de negocio. Nunca se infiere el target desde todas las
# columnas numéricas: las variables técnicas/encoding/IDs quedan fuera.
OBJETIVOS_ORIGINALES = {
    ING: "Ingresos operacionales",
    GAN: "Ganancia o pérdida del ejercicio",
    ACT: "Total activos",
    PAS: "Total pasivos",
    PAT: "Total patrimonio",
}

def catalogo_targets_financieros(df: pd.DataFrame) -> dict[str, str]:
    """Catálogo cerrado de objetivos financieros reales del año 2024.

    El experimento inicial estima el valor de 2024 a partir de antecedentes
    2022-2023 y contexto empresarial estático. Los valores 2023 son predictores,
    no objetivos de entrenamiento. Los indicadores acumulados 2022-2024 se
    mantienen en la base descriptiva, pero no se ofrecen como target.
    """
    targets: dict[str, str] = {}
    for prefijo, descripcion in OBJETIVOS_ORIGINALES.items():
        col = _col_fin(df, prefijo, 2024)
        if col is not None:
            targets[col] = f"{descripcion} — 2024 (variable original)"

    derivados_2024 = {
        "MARGEN_NETO_2024_PCT": "Margen neto = ganancia / ingresos operacionales × 100",
        "ROA_2024_PCT": "Rentabilidad sobre activos = ganancia / activos × 100",
        "ROE_2024_PCT": "Rentabilidad sobre patrimonio = ganancia / patrimonio × 100",
        "ENDEUDAMIENTO_2024_PCT": "Endeudamiento = pasivos / activos × 100",
        "SOLVENCIA_2024_PCT": "Pasivos / patrimonio × 100; interpretar con cautela si patrimonio ≤ 0",
        "ROTACION_ACTIVOS_2024": "Rotación de activos = ingresos operacionales / activos",
        "APALANCAMIENTO_2024": "Apalancamiento = activos / patrimonio",
        "CRECIMIENTO_INGRESOS_2024_PCT": "Crecimiento de ingresos 2023→2024 (%)",
        "CRECIMIENTO_GANANCIA_2024_PCT": "Crecimiento de ganancia 2023→2024 (%)",
        "CRECIMIENTO_ACTIVOS_2024_PCT": "Crecimiento de activos 2023→2024 (%)",
        "CRECIMIENTO_PASIVOS_2024_PCT": "Crecimiento de pasivos 2023→2024 (%)",
        "CRECIMIENTO_PATRIMONIO_2024_PCT": "Crecimiento de patrimonio 2023→2024 (%)",
    }
    for col, desc in derivados_2024.items():
        if col in df.columns:
            targets[col] = f"{desc} — 2024 (indicador derivado)"
    return targets


def crear_indicadores_financieros(df: pd.DataFrame) -> pd.DataFrame:
    """Crea indicadores de negocio interpretables sin parámetros aprendidos.

    Las razones se expresan en porcentaje donde el nombre termina en _PCT.
    Denominadores cero/no positivos se convierten en NaN para no inventar
    razones financieras indefinidas. Las variables originales no se alteran.
    """
    d = df.copy()
    cols = {(p, y): _col_fin(d, p, y) for p in PREFIJOS_FIN for y in ANIOS}
    def num(key, year):
        c = cols.get((key, year))
        return pd.to_numeric(d[c], errors="coerce") if c else pd.Series(np.nan, index=d.index, dtype=float)
    def safe_ratio(numer, denom, positive=True):
        denom_ok = denom.gt(0) if positive else denom.ne(0)
        return (numer / denom.where(denom_ok)) * 100.0

    for y in ANIOS:
        ing, gan, act, pas, pat = (num(p, y) for p in PREFIJOS_FIN)
        definitions = {
            f"MARGEN_NETO_{y}_PCT": safe_ratio(gan, ing, True),
            f"ROA_{y}_PCT": safe_ratio(gan, act, True),
            f"ROE_{y}_PCT": safe_ratio(gan, pat, False),
            f"ENDEUDAMIENTO_{y}_PCT": safe_ratio(pas, act, True),
            f"SOLVENCIA_{y}_PCT": safe_ratio(pas, pat, False),
            f"ROTACION_ACTIVOS_{y}": ing / act.where(act.gt(0)),
            f"APALANCAMIENTO_{y}": act / pat.where(pat.ne(0)),
        }
        for col, values in definitions.items():
            # El indicador canónico se recalcula con fórmula documentada.
            d[col] = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)

    for prefijo, etiqueta in ((ING, "INGRESOS"), (GAN, "GANANCIA"), (ACT, "ACTIVOS"), (PAS, "PASIVOS"), (PAT, "PATRIMONIO")):
        for prev, curr in ((2022, 2023), (2023, 2024)):
            a, b = cols.get((prefijo, prev)), cols.get((prefijo, curr))
            if a is None or b is None:
                continue
            base = pd.to_numeric(d[a], errors="coerce")
            actual = pd.to_numeric(d[b], errors="coerce")
            # Crecimiento porcentual solo si la base previa es positiva.
            d[f"CRECIMIENTO_{etiqueta}_{curr}_PCT"] = ((actual - base) / base.where(base.gt(0)) * 100).replace([np.inf, -np.inf], np.nan)
    a, b = cols.get((ING, 2022)), cols.get((ING, 2024))
    if a is not None and b is not None:
        base = pd.to_numeric(d[a], errors="coerce")
        actual = pd.to_numeric(d[b], errors="coerce")
        d["CRECIMIENTO_INGRESOS_2022_2024_PCT"] = ((actual - base) / base.where(base.gt(0)) * 100).replace([np.inf, -np.inf], np.nan)
    return d


def _num_es(v: float, dec: int = 1) -> str:
    if v is None or pd.isna(v):
        return "n.d."
    s = f"{v:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _mm(v) -> str:
    return "n.d." if v is None or pd.isna(v) else _num_es(v / 1e6, 1)


def _pct(v) -> str:
    return "n.d." if v is None or pd.isna(v) else f"{_num_es(v, 1)} %"


def _txt(v) -> str:
    if v is None:
        return "n.d."
    try:
        if pd.isna(v):
            return "n.d."
    except Exception:
        pass
    return str(v)


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ═══════════════════════════════════════════════════════════════════════════
# 1. ESTANDARIZACIÓN Y DIAGNÓSTICO
# ═══════════════════════════════════════════════════════════════════════════

def estandarizar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normaliza tipos/textos sin aprender parámetros del dataset."""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    incidencias: dict[str, dict] = {}

    for c in list(d.columns):
        if _es_numerica_esperada(c):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        elif c == "NIT":
            d[c] = (
                pd.to_numeric(d[c], errors="coerce")
                .round(0).astype("Int64").astype("string")
            )
        elif c == "RAZON SOCIAL":
            d[c] = d[c].astype("string").str.strip()
        elif not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(_norm)
            vacios = int(d[c].isna().sum())
            atipicos = 0
            if c in COLS_METADATO:
                solo_num = d[c].astype("string").str.fullmatch(r"\d+").fillna(False)
                atipicos = int(solo_num.sum())
                d.loc[solo_num, c] = SIN_DATO
            if c != COL_CIIU:
                d[c] = d[c].fillna(SIN_DATO)
            incidencias[c] = {
                "vacios": vacios,
                "atipicos_numericos": atipicos,
            }

    return d, incidencias


def _plan_encoding_base(d: pd.DataFrame, umbral: int, incluir_meta: bool) -> pd.DataFrame:
    filas = []
    n = len(d)
    for c in d.columns:
        s = d[c]
        if pd.api.types.is_numeric_dtype(s) and c != "NIT":
            continue
        u = int(s.nunique(dropna=True))
        ratio = u / n if n else 0
        ejemplo = _txt(s.dropna().iloc[0]) if s.notna().any() else "—"

        if c in COLS_IDENTIFICADORES or ratio > UMBRAL_IDENTIFICADOR:
            est = "EXCLUIR"
            why = f"Identificador/casi único ({ratio:.0%} de valores distintos)."
        elif c in COLS_METADATO and not incluir_meta:
            est = "EXCLUIR"
            why = "Metadato excluido por configuración."
        elif c == COL_CIIU:
            est = "CIIU → SECCIÓN + ONE-HOT"
            why = "Se reduce el código CIIU a su sección económica."
        elif u <= umbral:
            est = "ONE-HOT"
            why = f"{u} categorías ≤ {umbral}."
        else:
            est = "FREQUENCY"
            why = f"{u} categorías > {umbral}; evita alta dimensionalidad."

        filas.append({
            "Variable": c,
            "Tipo original": str(s.dtype),
            "Categorías únicas": u,
            "% únicos": round(ratio * 100, 2),
            "Ejemplo": ejemplo[:45],
            "Estrategia": est,
            "Justificación": why,
        })
    return pd.DataFrame(filas)


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def _crear_transformaciones_log_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las transformaciones logarítmicas base requeridas por el modelo de
    Feature Engineering heredado del proyecto anterior.

    El módulo fuente espera columnas *_log para ACTIVOS y PASIVOS en 2022/2023.
    En algunos flujos anteriores esas columnas provenían de una etapa de
    limpieza que no siempre estaba presente. Aquí se construyen explícitamente
    para que Feature Engineering sea autocontenido y no omita las variables
    IMPACTO_PERDIDA_CON_PASIVOS y COMPLEJIDAD_ESTRUCTURAL.

    Se usa log1p(abs(x)) para que el transformador sea válido tanto para
    magnitudes positivas como para valores negativos de patrimonio/resultado.
    Estas transformaciones son deterministas: no se ajusta ningún parámetro
    con TRAIN/TEST.
    """
    d = df.copy()
    for prefijo in (ACT, PAS):
        for anio in (2022, 2023):
            col = _col_fin(d, prefijo, anio)
            if col is None:
                continue
            base = pd.to_numeric(d[col], errors="coerce")
            nombre_log = f"{col}_log"
            if nombre_log not in d.columns:
                d[nombre_log] = np.log1p(base.abs())
    return d


@st.cache_data(show_spinner="Construyendo Feature Engineering financiero…")
def ejecutar_feature_engineering(df_est: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta el módulo fuente feature_engineering.py.

    Importante: esta transformación es determinista. No calcula frecuencias,
    medias de imputación, escaladores ni importancias usando el conjunto TEST.
    Por ello puede ejecutarse antes de train/test.
    """
    df_base_fe = _crear_transformaciones_log_base(df_est)
    if feature_engineering is None:
        # Respaldo controlado: la preparación sigue creando indicadores
        # financieros básicos y mantiene la app operativa, dejando advertencia.
        d = df_base_fe.copy()
        _fallback_error = _FE_IMPORT_ERROR
    else:
        d = feature_engineering(df_base_fe)
        _fallback_error = None
    d = crear_indicadores_financieros(d)
    rep = get_fe_reporte(d) if get_fe_reporte and feature_engineering is not None else {}
    if _fallback_error:
        rep["advertencia_modulo_fe"] = _fallback_error
    rep = dict(rep or {})
    rep.setdefault("shape_entrada", df_est.shape)
    rep.setdefault("shape_salida", d.shape)
    rep.setdefault(
        "columnas_nuevas",
        [c for c in d.columns if c not in df_est.columns],
    )
    rep["n_variables_nuevas"] = len(rep["columnas_nuevas"])
    return d, rep


# ═══════════════════════════════════════════════════════════════════════════
# 3. PARTICIÓN + ENCODING SIN DATA LEAKAGE
# ═══════════════════════════════════════════════════════════════════════════

def _infer_year_from_target(target: str) -> int | None:
    m = re.findall(r"20(?:22|23|24)", str(target))
    return int(m[-1]) if m else None


def _predictor_columns_temporalmente_seguras(
    df: pd.DataFrame,
    target: str,
) -> tuple[list[str], list[str]]:
    """
    Deja como predictores solo información disponible ANTES del año objetivo.

    Ejemplo target INGRESOS OPERACIONALES 2024:
      • se permiten variables 2022 y 2023;
      • se excluyen variables que contengan 2024;
      • se excluyen identificadores y rankings derivados.
    """
    year_target = _infer_year_from_target(target)
    if year_target != 2024:
        raise ValueError(
            f"El experimento inicial solo admite objetivos de 2024; se recibió: {target!r}."
        )
    excluidas = []
    permitidas = []

    for c in df.columns:
        if c == target:
            continue
        if c in COLS_IDENTIFICADORES:
            excluidas.append(c)
            continue
        if c.startswith("RANKING"):
            excluidas.append(c)
            continue

        # Solo se admiten antecedentes 2022/2023 y contexto estático. Toda
        # variable con referencia a 2024 o a un horizonte posterior se excluye.
        years = [int(x) for x in re.findall(r"20(?:22|23|24|25|26)", str(c))]
        if years and max(years) >= 2024:
            excluidas.append(c)
            continue
        permitidas.append(c)

    return permitidas, excluidas


def _seccion_ciiu(s: pd.Series) -> pd.Series:
    txt = s.astype("string")
    # Admite formatos A01, A01.1, A-01 y A01 - descripción.
    letra = txt.str.extract(r"^\s*([A-Za-z])", expand=False).str.upper()
    return letra.where(letra.isin(list(SECCIONES_CIIU)), "Z").fillna("Z")


def _fit_transform_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    umbral: int,
    drop_first: bool,
    incluir_meta: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Ajusta el encoding SOLO con TRAIN y aplica el mismo mapa a TEST.

    Este es el punto crítico para evitar leakage:
      - One-Hot aprende categorías en train.
      - Frequency aprende frecuencias en train.
      - categorías desconocidas en test reciben 0 en Frequency y todas las
        dummies 0 en One-Hot.
    """
    tr = train.copy()
    te = test.copy()
    plan = _plan_encoding_base(tr, umbral, incluir_meta)

    excluidas = {}
    for _, row in plan[plan["Estrategia"] == "EXCLUIR"].iterrows():
        excluidas[row["Variable"]] = row["Justificación"]

    # RANKING: no debe entrar como predictor.
    for c in list(tr.columns):
        if str(c).startswith("RANKING"):
            excluidas[c] = "Ranking derivado de ingresos; se excluye para evitar leakage."

    # CIIU → sección; el mapeo es determinista y no aprende parámetros.
    if COL_CIIU in tr.columns:
        tr[COL_SECCION] = _seccion_ciiu(tr[COL_CIIU])
        if COL_CIIU in te.columns:
            te[COL_SECCION] = _seccion_ciiu(te[COL_CIIU])
        tr = tr.drop(columns=[COL_CIIU])
        te = te.drop(columns=[COL_CIIU], errors="ignore")

        if not (plan["Variable"] == COL_SECCION).any():
            plan = pd.concat([
                plan,
                pd.DataFrame([{
                    "Variable": COL_SECCION,
                    "Tipo original": "string",
                    "Categorías únicas": int(tr[COL_SECCION].nunique()),
                    "% únicos": round(tr[COL_SECCION].nunique() / max(len(tr), 1) * 100, 2),
                    "Ejemplo": _txt(tr[COL_SECCION].dropna().iloc[0]) if tr[COL_SECCION].notna().any() else "Z",
                    "Estrategia": "ONE-HOT",
                    "Justificación": "Sección económica derivada de CIIU.",
                }])
            ], ignore_index=True)

    cols_ohe = [c for c in plan.loc[plan["Estrategia"] == "ONE-HOT", "Variable"].tolist() if c in tr.columns]
    cols_freq = [c for c in plan.loc[plan["Estrategia"] == "FREQUENCY", "Variable"].tolist() if c in tr.columns]

    mapa_ohe = {}
    base_ohe = {}
    categorias_ohe = {}
    detalle_ohe = {}

    for c in cols_ohe:
        cats = sorted(tr[c].dropna().astype(str).unique().tolist())
        if not cats:
            continue
        tr_cat = tr[c].astype("string").fillna(SIN_DATO).astype(str)
        te_cat = te[c].astype("string").fillna(SIN_DATO).astype(str)
        cats_full = cats.copy()
        if SIN_DATO not in cats_full and tr_cat.eq(SIN_DATO).any():
            cats_full.append(SIN_DATO)
        if SIN_DATO not in cats_full:
            cats_full = sorted(cats_full)

        tr_cat = pd.Categorical(tr_cat, categories=cats_full)
        te_cat = pd.Categorical(te_cat.where(te_cat.isin(cats_full)), categories=cats_full)

        dtr = pd.get_dummies(tr_cat, prefix=c, drop_first=drop_first, dtype=np.int8)
        dte = pd.get_dummies(te_cat, prefix=c, drop_first=drop_first, dtype=np.int8)
        dte = dte.reindex(columns=dtr.columns, fill_value=0)

        mapa_ohe[c] = dtr.columns.tolist()
        base_ohe[c] = cats_full[0] if drop_first and cats_full else None
        categorias_ohe[c] = cats_full
        vc = tr_cat.value_counts()
        detalle_ohe[c] = pd.DataFrame({
            "Categoría": vc.index.astype(str),
            "Empresas TRAIN": vc.values,
            "% TRAIN": (vc.values / max(len(tr), 1) * 100).round(2),
        })

        tr = tr.drop(columns=[c])
        te = te.drop(columns=[c])
        tr = pd.concat([tr.reset_index(drop=True), dtr.reset_index(drop=True)], axis=1)
        te = pd.concat([te.reset_index(drop=True), dte.reset_index(drop=True)], axis=1)

    mapa_freq = {}
    frecuencias = {}
    detalle_freq = {}
    for c in cols_freq:
        if c not in tr.columns:
            continue
        s_tr = tr[c].astype("string").fillna(SIN_DATO)
        s_te = te[c].astype("string").fillna(SIN_DATO)
        freq = s_tr.value_counts(normalize=True)
        nueva = f"{c}_FREQ"
        tr[nueva] = s_tr.map(freq).fillna(0).astype(np.float32)
        te[nueva] = s_te.map(freq).fillna(0).astype(np.float32)
        mapa_freq[c] = nueva
        frecuencias[c] = {str(k): float(v) for k, v in freq.items()}
        cuentas = s_tr.value_counts()
        detalle_freq[c] = pd.DataFrame({
            "Categoría": cuentas.index.astype(str),
            "Empresas TRAIN": cuentas.values,
            "Frecuencia relativa": [float(freq.get(k, 0.0)) for k in cuentas.index],
        })
        tr = tr.drop(columns=[c])
        te = te.drop(columns=[c])

    # Excluir IDs/rankings/metadatos configurados. Se eliminan de ambos.
    cols_excluir_validas = [c for c in excluidas if c in tr.columns]
    tr = tr.drop(columns=cols_excluir_validas, errors="ignore")
    te = te.drop(columns=cols_excluir_validas, errors="ignore")

    # Asegurar mismas columnas y orden.
    te = te.reindex(columns=tr.columns, fill_value=np.nan)

    # Conversión final de objetos residuales a numérico cuando sea posible.
    for c in tr.columns:
        if not pd.api.types.is_numeric_dtype(tr[c]):
            tr[c] = pd.to_numeric(tr[c], errors="coerce")
        if not pd.api.types.is_numeric_dtype(te[c]):
            te[c] = pd.to_numeric(te[c], errors="coerce")

    tr = tr.replace([np.inf, -np.inf], np.nan)
    te = te.replace([np.inf, -np.inf], np.nan)

    detalle_nuevas = []
    for origen, cols in mapa_ohe.items():
        for col in cols:
            detalle_nuevas.append({
                "Columna nueva": col,
                "Técnica": "One-Hot",
                "Variable origen": origen,
                "Aprendida en": "TRAIN",
            })
    for origen, col in mapa_freq.items():
        detalle_nuevas.append({
            "Columna nueva": col,
            "Técnica": "Frequency",
            "Variable origen": origen,
            "Aprendida en": "TRAIN",
        })

    rep = {
        "plan_train": plan,
        "excluidas": excluidas,
        "cols_ohe": cols_ohe,
        "cols_freq": cols_freq,
        "mapa_ohe": mapa_ohe,
        "base_ohe": base_ohe,
        "categorias_ohe": categorias_ohe,
        "drop_first": bool(drop_first),
        "incluir_meta": bool(incluir_meta),
        "detalle_ohe": detalle_ohe,
        "mapa_freq": mapa_freq,
        "frecuencias": frecuencias,
        "detalle_freq": detalle_freq,
        "detalle_nuevas": pd.DataFrame(detalle_nuevas),
        "shape_train": tr.shape,
        "shape_test": te.shape,
    }
    return tr, te, rep


@st.cache_data(show_spinner="Particionando TRAIN/TEST y ajustando encoding solo con TRAIN…")
def preparar_train_test(
    df_features: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
    umbral_ohe: int,
    drop_first: bool,
    incluir_meta: bool,
) -> dict:
    """Pipeline de partición. Todo lo aprendido se ajusta en TRAIN."""
    if target not in df_features.columns:
        raise ValueError(f"La variable objetivo '{target}' no existe en el dataset.")
    catalogo_autorizado = catalogo_targets_financieros(df_features)
    if target not in catalogo_autorizado:
        raise ValueError(
            "El target seleccionado no pertenece al catálogo financiero autorizado para 2024. "
            "Ejecute nuevamente Preparación y seleccione un objetivo de 2024."
        )

    d = df_features.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d[target] = pd.to_numeric(d[target], errors="coerce")
    d = d.loc[d[target].notna()].copy()

    predictor_cols, excl_temporales = _predictor_columns_temporalmente_seguras(d, target)
    if not predictor_cols:
        raise ValueError(
            "No quedaron variables predictoras después de aplicar las reglas de seguridad temporal."
        )

    # El target se mantiene separado. Así nunca puede ser codificado/excluido.
    X_raw = d[predictor_cols].copy()
    y = d[target].copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    X_train_enc, X_test_enc, rep_enc = _fit_transform_encoding(
        X_train_raw,
        X_test_raw,
        umbral=umbral_ohe,
        drop_first=drop_first,
        incluir_meta=incluir_meta,
    )

    train_model = X_train_enc.copy()
    test_model = X_test_enc.copy()
    train_model[target] = y_train.to_numpy()
    test_model[target] = y_test.to_numpy()

    return {
        "target": target,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_enc": X_train_enc,
        "X_test_enc": X_test_enc,
        "train_model": train_model,
        "test_model": test_model,
        "rep_encoding": rep_enc,
        "predictor_cols": predictor_cols,
        "excl_temporales": excl_temporales,
        "n_original": len(d),
    }


def preparar_matriz_prospectiva(
    df_features: pd.DataFrame,
    pipeline: dict,
    outlier_result: dict,
    fs_result: dict,
    transformaciones: dict,
    desplazamiento_anios: int = 1,
) -> pd.DataFrame:
    """Construye X para inferencia prospectiva aplicando el preprocesamiento aprendido.

    El caso soportado es desplazamiento de un año (p. ej. modelo entrenado para
    2024 con features hasta 2023, aplicado a 2025 con features reales hasta 2024).
    No usa el target futuro ni vuelve a ajustar encoding, límites, imputación o
    escalado. Para 2026 se requiere además construir features recursivas 2025.
    """
    if desplazamiento_anios not in (1,):
        raise ValueError("Por ahora solo se admite horizonte +1 año. Para 2026 debe construirse una inferencia recursiva con variables 2025 estimadas.")
    raw_cols = list(pipeline["predictor_cols"])
    source = df_features.copy()
    source.columns = [str(c).strip() for c in source.columns]
    future_raw = pd.DataFrame(index=source.index)
    for expected in raw_cols:
        years = [int(y) for y in re.findall(r"20(?:22|23|24|25|26)", str(expected))]
        if years:
            shifted = re.sub(r"20(?:22|23|24|25|26)", lambda m: str(int(m.group(0)) + desplazamiento_anios), str(expected))
            # Tolerar asteriscos en las columnas financieras originales.
            found = next((c for c in source.columns if str(c).replace("*", "").strip().upper() == shifted.replace("*", "").strip().upper()), None)
            if found is None:
                future_raw[expected] = np.nan
            else:
                future_raw[expected] = source[found].to_numpy()
        else:
            found = next((c for c in source.columns if str(c).strip().upper() == str(expected).strip().upper()), None)
            future_raw[expected] = source[found].to_numpy() if found is not None else np.nan

    rep = pipeline["rep_encoding"]
    if COL_CIIU in future_raw.columns:
        future_raw[COL_SECCION] = _seccion_ciiu(future_raw[COL_CIIU])
        future_raw = future_raw.drop(columns=[COL_CIIU])

    # Aplicar el mismo vocabulario One-Hot aprendido en TRAIN.
    for col, cats in rep.get("categorias_ohe", {}).items():
        if col not in future_raw.columns:
            continue
        vals = future_raw[col].astype("string").fillna(SIN_DATO).astype(str)
        cat = pd.Categorical(vals.where(vals.isin(cats)), categories=cats)
        dummy = pd.get_dummies(cat, prefix=col, drop_first=rep.get("drop_first", True), dtype=np.int8)
        dummy = dummy.reindex(columns=rep["mapa_ohe"].get(col, []), fill_value=0)
        future_raw = future_raw.drop(columns=[col])
        future_raw = pd.concat([future_raw, dummy.set_axis(future_raw.index)], axis=1)

    for col, nueva in rep.get("mapa_freq", {}).items():
        if col not in future_raw.columns:
            continue
        freq = rep.get("frecuencias", {}).get(col, {})
        vals = future_raw[col].astype("string").fillna(SIN_DATO).astype(str)
        future_raw[nueva] = vals.map(freq).fillna(0.0).astype(np.float32)
        future_raw = future_raw.drop(columns=[col])

    excluidas = list(rep.get("excluidas", {}).keys())
    future_raw = future_raw.drop(columns=[c for c in excluidas if c in future_raw.columns], errors="ignore")
    Xenc = future_raw.reindex(columns=pipeline["X_train_enc"].columns, fill_value=0).copy()
    for c in Xenc.columns:
        Xenc[c] = pd.to_numeric(Xenc[c], errors="coerce")
    Xenc = Xenc.replace([np.inf, -np.inf], np.nan)

    # Reaplicar límites P01/P99 y copias log calculados exclusivamente con TRAIN.
    report = outlier_result.get("reporte", pd.DataFrame())
    if isinstance(report, pd.DataFrame) and not report.empty:
        for _, row in report.iterrows():
            col = row.get("Variable")
            treatment = row.get("Tratamiento")
            if treatment == "Winsorización P01–P99" and col in Xenc.columns:
                lo, hi = row.get("Límite inferior"), row.get("Límite superior")
                if pd.notna(lo) and pd.notna(hi):
                    Xenc[col] = pd.to_numeric(Xenc[col], errors="coerce").clip(float(lo), float(hi))
            elif treatment == "LOG firmado" and col in Xenc.columns:
                Xenc[f"{col}_ML_LOG"] = _signed_log1p(Xenc[col])

    # Las banderas de outlier son descriptivas; el modelo no debe recalcular límites.
    Xout = Xenc.reindex(columns=outlier_result["X_train"].columns, fill_value=0).copy()
    if "N_OUTLIERS_ML" in Xout.columns:
        Xout["N_OUTLIERS_ML"] = 0
    if "FLAG_OUTLIER_ML" in Xout.columns:
        Xout["FLAG_OUTLIER_ML"] = 0

    selected = list(fs_result.get("selected_columns", []))
    Xsel = Xout.reindex(columns=selected, fill_value=np.nan)
    cols = list(transformaciones["columnas"])
    Xnum = Xsel.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    imputer = transformaciones["imputer"]
    Ximp = pd.DataFrame(imputer.transform(Xnum), columns=cols, index=source.index)
    scaler = transformaciones.get("scaler")
    if scaler is not None:
        Xready = pd.DataFrame(scaler.transform(Ximp), columns=cols, index=source.index)
    else:
        Xready = Ximp
    return Xready


# ═══════════════════════════════════════════════════════════════════════════
# 4. POLÍTICA DE OUTLIERS PARA MACHINE LEARNING — SOLO TRAIN APRENDE LÍMITES
# ═══════════════════════════════════════════════════════════════════════════

def _es_variable_ratio_o_crecimiento(col: str) -> bool:
    u = str(col).upper()
    return any(k in u for k in OUTLIER_RATIO_KEYWORDS)


def _es_magnitud_financiera(col: str) -> bool:
    u = str(col).upper()
    if "_LOG" in u or "FLAG_" in u or "N_OUTLIERS" in u:
        return False
    return any(u.startswith(pref) for pref in PREFIJOS_FIN)


def _signed_log1p(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.sign(x) * np.log1p(np.abs(x))


def aplicar_politica_outliers_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    target: str | None = None,
) -> dict:
    """Aplica la política formal de outliers sin eliminar empresas.

    Los límites P01/P99 se aprenden exclusivamente con TRAIN y se aplican
    sin recalcularlos sobre TEST. Las magnitudes financieras originales se
    conservan; para magnitudes muy asimétricas se añade una copia LOG firmada.
    Ratios/tasas se winsorizan. El target ya está separado y queda protegido.
    """
    tr = X_train.copy()
    te = X_test.copy().reindex(columns=X_train.columns)
    reporte = []
    log_cols, winsor_cols = [], []
    total_train = pd.Series(0, index=tr.index, dtype="int64")
    total_test = pd.Series(0, index=te.index, dtype="int64")

    for c in list(X_train.columns):
        if c == target:
            continue
        a = pd.to_numeric(tr[c], errors="coerce")
        b = pd.to_numeric(te[c], errors="coerce")
        n_valid = int(a.notna().sum())
        if n_valid == 0:
            continue
        skew = float(a.skew()) if n_valid >= OUTLIER_MIN_NON_NULL else np.nan

        if (_es_magnitud_financiera(c) and n_valid >= OUTLIER_MIN_NON_NULL
                and pd.notna(skew) and abs(skew) >= OUTLIER_SKEW_THRESHOLD):
            log_name = f"{c}_ML_LOG"
            tr[log_name] = _signed_log1p(a)
            te[log_name] = _signed_log1p(b)
            log_cols.append(log_name)
            reporte.append({
                "Variable": c, "Tratamiento": "LOG firmado",
                "Límite inferior": np.nan, "Límite superior": np.nan,
                "Skew TRAIN": skew, "TRAIN afectados": 0,
                "TEST afectados": 0,
                "Regla": f"|skew| ≥ {OUTLIER_SKEW_THRESHOLD}; se conserva la original.",
            })

        if (_es_variable_ratio_o_crecimiento(c)
                and "FLAG_" not in str(c).upper()
                and "N_OUTLIERS" not in str(c).upper()
                and n_valid >= OUTLIER_MIN_NON_NULL):
            lo = float(a.quantile(OUTLIER_LOWER_Q))
            hi = float(a.quantile(OUTLIER_UPPER_Q))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                mask_tr = a.notna() & ((a < lo) | (a > hi))
                mask_te = b.notna() & ((b < lo) | (b > hi))
                total_train = total_train.add(mask_tr.astype(int), fill_value=0).astype(int)
                total_test = total_test.add(mask_te.astype(int), fill_value=0).astype(int)
                tr[c] = a.clip(lower=lo, upper=hi)
                te[c] = b.clip(lower=lo, upper=hi)
                winsor_cols.append(c)
                reporte.append({
                    "Variable": c, "Tratamiento": "Winsorización P01–P99",
                    "Límite inferior": lo, "Límite superior": hi,
                    "Skew TRAIN": skew,
                    "TRAIN afectados": int(mask_tr.sum()),
                    "TEST afectados": int(mask_te.sum()),
                    "Regla": "Límites calculados exclusivamente con TRAIN.",
                })

    tr["N_OUTLIERS_ML"] = total_train.astype(int)
    te["N_OUTLIERS_ML"] = total_test.astype(int)
    tr["FLAG_OUTLIER_ML"] = (tr["N_OUTLIERS_ML"] > 0).astype("int8")
    te["FLAG_OUTLIER_ML"] = (te["N_OUTLIERS_ML"] > 0).astype("int8")
    rep_df = pd.DataFrame(reporte)
    return {
        "X_train": tr, "X_test": te, "reporte": rep_df,
        "log_columns": log_cols, "winsor_columns": winsor_cols,
        "n_train_empresas": len(tr), "n_test_empresas": len(te),
        "train_empresas_con_outlier": int(tr["FLAG_OUTLIER_ML"].sum()),
        "test_empresas_con_outlier": int(te["FLAG_OUTLIER_ML"].sum()),
        "train_valores_afectados": int(total_train.sum()),
        "test_valores_afectados": int(total_test.sum()),
        "politica": {
            "winsor_lower": OUTLIER_LOWER_Q, "winsor_upper": OUTLIER_UPPER_Q,
            "skew_threshold": OUTLIER_SKEW_THRESHOLD,
            "elimina_filas": False, "calcula_limites_en": "TRAIN",
            "aplica_limites_a": "TRAIN y TEST",
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. FEATURE SELECTION — SOLO TRAIN
# ═══════════════════════════════════════════════════════════════════════════

def ejecutar_feature_selection_train(
    X_train_enc: pd.DataFrame,
    y_train: pd.Series,
    target: str,
    correlation_threshold: float,
    top_n: int,
) -> dict:
    """
    Ejecuta Feature Selection exclusivamente con TRAIN.

    La función fuente recibe un DataFrame con target incluido. Se evita que
    TEST participe en correlaciones, Mutual Information, Random Forest o
    Permutation Importance.
    """
    if feature_selection is None:
        raise RuntimeError(
            "No se pudo importar feature_selection.py. "
            f"Detalle: {_FS_IMPORT_ERROR}"
        )

    tr = X_train_enc.copy()
    tr[target] = pd.to_numeric(y_train, errors="coerce").to_numpy()
    tr = tr.dropna(subset=[target]).replace([np.inf, -np.inf], np.nan)

    try:
        selected = feature_selection(
            tr,
            target=target,
            correlation_threshold=correlation_threshold,
            top_n=top_n,
        )
        rep = get_feature_selection_reporte(selected) if get_feature_selection_reporte else {}
        rep = dict(rep or {})
        rep["metodo"] = "Feature Selection ajustado exclusivamente sobre TRAIN"
        rep["target"] = target
        rep["top_n"] = top_n
        rep["correlation_threshold"] = correlation_threshold
        return {
            "selected_train": selected.drop(columns=[target], errors="ignore"),
            "selected_columns": [c for c in selected.columns if c != target],
            "reporte": rep,
            "error": None,
        }
    except Exception as exc:
        # No bloqueamos la aplicación completa. Se deja trazabilidad y se usa
        # el conjunto completo de features como fallback seguro.
        cols = [c for c in X_train_enc.columns]
        return {
            "selected_train": X_train_enc.copy(),
            "selected_columns": cols,
            "reporte": {
                "metodo": "Fallback por error en Feature Selection",
                "target": target,
                "top_n": top_n,
                "correlation_threshold": correlation_threshold,
            },
            "error": str(exc),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. IMPUTACIÓN / ESCALADO — AJUSTADO SOLO EN TRAIN
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Aplicando imputación y transformaciones solo con TRAIN…")
def preparar_transformaciones(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    escalar: bool,
) -> dict:
    if X_train.empty:
        raise ValueError("No hay variables para transformar después de Feature Selection.")

    cols = list(X_train.columns)
    Xt = X_train.copy().apply(pd.to_numeric, errors="coerce")
    Xv = X_test.copy().reindex(columns=cols).apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    Xt_imp = pd.DataFrame(
        imputer.fit_transform(Xt),
        columns=cols,
        index=Xt.index,
    )
    Xv_imp = pd.DataFrame(
        imputer.transform(Xv),
        columns=cols,
        index=Xv.index,
    )

    scaler = None
    if escalar:
        scaler = StandardScaler()
        Xt_ready = pd.DataFrame(
            scaler.fit_transform(Xt_imp),
            columns=cols,
            index=Xt.index,
        )
        Xv_ready = pd.DataFrame(
            scaler.transform(Xv_imp),
            columns=cols,
            index=Xv.index,
        )
    else:
        Xt_ready = Xt_imp
        Xv_ready = Xv_imp

    return {
        "X_train_imputed": Xt_imp,
        "X_test_imputed": Xv_imp,
        "X_train_ready": Xt_ready,
        "X_test_ready": Xv_ready,
        "imputer": imputer,
        "scaler": scaler,
        "escalar": escalar,
        "columnas": cols,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BASE PARA EL LLM
# ═══════════════════════════════════════════════════════════════════════════

def _razon(d: pd.DataFrame, num, den, factor: float = 100.0):
    if num is None or den is None:
        return pd.Series(np.nan, index=d.index)
    return d[num] / d[den].where(d[den] > 0) * factor


def _ficha(r: dict, cols: dict, n_total: int) -> str:
    """Construye la ficha textual descriptiva para RAG/LLM.

    El Ranking se conserva aquí como dato descriptivo de la empresa.
    No significa que el Ranking sea un predictor de Machine Learning:
    para ML sigue excluido por derivarse de los ingresos.
    """
    def fin(pref, y):
        c = cols.get((pref, y))
        return _mm(r.get(c)) if c else "n.d."

    # La base consolidada utiliza RANKING 2023 2024 para la posición por
    # ingresos 2024. Se deja una segunda opción para bases que sólo tengan
    # ranking de 2023.
    ranking_2024 = r.get("RANKING 2023 2024")
    ranking_2023 = r.get("RANKING 2022 2023")
    if ranking_2024 is not None and not pd.isna(ranking_2024):
        texto_ranking = (
            f"Ocupa el puesto {_txt(ranking_2024)} entre las {_num_es(n_total, 0)} "
            "empresas de la base según sus ingresos operacionales de 2024 "
            "(ranking interno de la base, no nacional)."
        )
    elif ranking_2023 is not None and not pd.isna(ranking_2023):
        texto_ranking = (
            f"Ocupa el puesto {_txt(ranking_2023)} entre las {_num_es(n_total, 0)} "
            "empresas de la base según sus ingresos operacionales de 2023 "
            "(ranking interno de la base, no nacional)."
        )
    else:
        texto_ranking = "No se dispone de ranking de ingresos para esta empresa en la base."

    return (
        f"{r.get('RAZON SOCIAL', 'Empresa')} (NIT {_txt(r.get('NIT'))}) es una empresa "
        f"del macrosector {_txt(r.get('MACROSECTOR'))}, con actividad principal "
        f"CIIU {_txt(r.get('CIIU_CODIGO'))} ({_txt(r.get('CIIU_ACTIVIDAD'))}), "
        f"domiciliada en {_txt(r.get('CIUDAD DOMICILIO'))}, {_txt(r.get('DEPARTAMENTO DOMICILIO'))} "
        f"(región {_txt(r.get('REGION'))}), y vigilada por {_txt(r.get('SUPERVISOR'))}. "
        f"En 2024 registró ingresos operacionales de {fin(ING, 2024)}, "
        f"ganancia (pérdida) de {fin(GAN, 2024)}, activos de {fin(ACT, 2024)}, "
        f"pasivos de {fin(PAS, 2024)} y patrimonio de {fin(PAT, 2024)} "
        f"(miles de millones de COP). Sus ingresos fueron {fin(ING, 2023)} en 2023 "
        f"y {fin(ING, 2022)} en 2022. Crecimiento de ingresos 2023→2024: "
        f"{_pct(r.get('CRECIMIENTO_INGRESOS_2024_PCT'))}; margen neto 2024: "
        f"{_pct(r.get('MARGEN_NETO_2024_PCT'))}; ROA 2024: {_pct(r.get('ROA_2024_PCT'))}; "
        f"endeudamiento 2024: {_pct(r.get('ENDEUDAMIENTO_2024_PCT'))}. "
        f"{texto_ranking}"
    )



_DESC_BASE = {
    "NIT": "Identificador tributario de la empresa (texto)",
    "RAZON SOCIAL": "Nombre legal de la empresa",
    "SUPERVISOR": "Superintendencia que vigila a la empresa",
    "MACROSECTOR": "Macrosector económico",
    COL_SECCION: "Sección económica CIIU Rev.4 A.C.",
    "SECCION_CIIU_DESC": "Nombre de la sección económica CIIU",
    "CIIU_CODIGO": "Código CIIU de la actividad principal",
    "CIIU_ACTIVIDAD": "Descripción de la actividad principal",
    "REGION": "Región económica",
    "DEPARTAMENTO DOMICILIO": "Departamento del domicilio",
    "CIUDAD DOMICILIO": "Ciudad del domicilio",
    "Punto de entrada": "Grupo NIIF / punto de entrada",
    "GRUPO EN NIIF": "Grupo NIIF",
    "RANKING 2022 2023": "Posición por ingresos 2023 dentro de las empresas de la base (1 = mayor ingreso; no es ranking nacional)",
    "RANKING 2023 2024": "Posición por ingresos 2024 dentro de las empresas de la base (1 = mayor ingreso; no es ranking nacional)",
    "FICHA_TEXTO": "Resumen en lenguaje natural de la empresa (insumo para RAG)",
}

_DESC_FIN = {
    ING: "Ingresos operacionales",
    GAN: "Ganancia o pérdida neta del ejercicio",
    ACT: "Total de activos",
    PAS: "Total de pasivos",
    PAT: "Patrimonio total",
}


def _describir(c: str) -> tuple[str, str]:
    if c in _DESC_BASE:
        return _DESC_BASE[c], "posición" if c.startswith("RANKING") else "—"
    base = c.replace("*", "").strip()
    for p, desc in _DESC_FIN.items():
        if base.startswith(p):
            return f"{desc} ({base[-4:]})", "Miles de COP"
    if c.startswith("MARGEN_NETO_"):
        return ("Margen neto = Ganancia / Ingresos operacionales × 100", "%") if c.endswith("_PCT") else ("Margen neto = Ganancia / Ingresos operacionales", "ratio decimal")
    if c.startswith("ROA_"):
        return ("ROA = Ganancia / Activos × 100", "%") if c.endswith("_PCT") else ("ROA = Ganancia / Activos", "ratio decimal")
    if c.startswith("ROE_"):
        return ("ROE = Ganancia / Patrimonio × 100; interpretar con cautela si patrimonio ≤ 0", "%") if c.endswith("_PCT") else ("ROE = Ganancia / Patrimonio; interpretar con cautela si patrimonio ≤ 0", "ratio decimal")
    if c.startswith("ENDEUDAMIENTO_"):
        return ("Endeudamiento = Pasivos / Activos × 100", "%") if c.endswith("_PCT") else ("Endeudamiento = Pasivos / Activos", "ratio decimal")
    if c.startswith("SOLVENCIA_"):
        return ("Pasivos / Patrimonio × 100; interpretar con cautela si patrimonio ≤ 0", "%") if c.endswith("_PCT") else ("Pasivos / Patrimonio; interpretar con cautela si patrimonio ≤ 0", "ratio decimal")
    if c.startswith("CRECIMIENTO_") and c.endswith("_PCT"):
        return "Variación porcentual entre períodos; la base anterior debe ser positiva", "%"
    if c.startswith("ROTACION_ACTIVOS_"):
        return "Rotación de activos = ingresos operacionales / activos", "veces"
    if c.startswith("APALANCAMIENTO_"):
        return "Apalancamiento = activos / patrimonio", "veces"
    if c.startswith("TENDENCIA_"):
        return "Pendiente temporal estimada del indicador nombrado en la variable", "unidad del indicador / año"
    if c.startswith("VOLATILIDAD_"):
        return "Desviación estándar temporal del indicador nombrado en la variable", "unidad del indicador"
    if c.startswith("CRECIMIENTO_"):
        return ("Cambio porcentual entre períodos; base anterior positiva", "%") if c.endswith("_PCT") else ("Cambio relativo entre períodos; revisar sufijo y si está recortado (_CLIP)", "ratio decimal")
    if c.startswith("CAMBIO_"):
        return "Diferencia entre períodos del indicador nombrado en la variable", "unidad del indicador"
    if c.startswith(("FLAG_", "ES_")):
        return "Indicador binario de condición financiera (1 = condición presente; 0 = ausente)", "0/1"
    if c.startswith(("RIESGO_", "IMPACTO_", "COMPLEJIDAD_", "DESBALANCE_")):
        return "Indicador derivado de interacción/riesgo financiera; consultar fórmula del Feature Engineering", "escala derivada"
    if c.startswith(("LOG_", "*_LOG", "*_ML_LOG")) or "_LOG_" in c or c.endswith("_ML_LOG"):
        return "Transformación logarítmica técnica de una magnitud; no es un objetivo financiero de negocio", "escala logarítmica"
    if c.startswith(("ABS_", "MAGNITUD_", "INTENSIDAD_", "SEVERIDAD_")):
        return "Magnitud/intensidad derivada de una variable financiera; consultar sufijo y fórmula", "escala derivada"
    return "Variable original o derivada; revisar el nombre y el reporte de Feature Engineering", "—"


def construir_diccionario(d: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for c in d.columns:
        desc, unidad = _describir(c)
        serie = d[c].dropna()
        ejemplo = str(serie.iloc[0])[:60] if len(serie) else "—"
        filas.append({
            "Variable": c,
            "Tipo": str(d[c].dtype),
            "Unidad": unidad,
            "Descripción": desc,
            "Nulos": int(d[c].isna().sum()),
            "Ejemplo": ejemplo,
        })
    return pd.DataFrame(filas)


@st.cache_data(show_spinner="Construyendo la base legible para el LLM…")
def construir_base_llm(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d, _ = estandarizar(df)

    # Para el análisis por prompt se conserva una vista amplia: columnas
    # originales + todas las features determinísticas (ratios, crecimiento,
    # magnitudes, flags, interacciones, tendencia y volatilidad). No se usa
    # esta vista para entrenar; el LLM debe consultar cifras calculadas en Python.
    try:
        d, _reporte_fe_llm = ejecutar_feature_engineering(d)
    except Exception:
        # La creación de ratios básicos se mantiene como respaldo seguro.
        d = crear_indicadores_financieros(d)

    if COL_CIIU in d.columns:
        ext = d[COL_CIIU].astype("string").str.extract(
            r"^\s*([A-Z])([0-9]+)\s*(?:-\s*)?(.*)$"
        )
        d[COL_SECCION] = ext[0].fillna("Z").astype(str)
        d["SECCION_CIIU_DESC"] = d[COL_SECCION].map(SECCIONES_CIIU).fillna("SIN CLASIFICAR")
        d["CIIU_CODIGO"] = (ext[0].fillna("") + ext[1].fillna("")).replace("", np.nan)
        d["CIIU_ACTIVIDAD"] = ext[2].replace("", np.nan)
        d = d.drop(columns=[COL_CIIU])

    d = crear_indicadores_financieros(d)
    cols = {(p, y): _col_fin(d, p, y) for p in PREFIJOS_FIN for y in ANIOS}

    n_total = len(d)
    d["FICHA_TEXTO"] = [
        _ficha(r, cols, n_total) for r in d.to_dict("records")
    ]

    orden = [
        "NIT", "RAZON SOCIAL", "SUPERVISOR", "MACROSECTOR", COL_SECCION,
        "SECCION_CIIU_DESC", "CIIU_CODIGO", "CIIU_ACTIVIDAD", "REGION",
        "DEPARTAMENTO DOMICILIO", "CIUDAD DOMICILIO", "Punto de entrada",
        "GRUPO EN NIIF", "RANKING 2022 2023", "RANKING 2023 2024",
    ]
    orden = [c for c in orden if c in d.columns]
    resto = [c for c in d.columns if c not in orden and c != "FICHA_TEXTO"]
    d = d[orden + resto + ["FICHA_TEXTO"]]
    return d, construir_diccionario(d)


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENTES DE INTERFAZ
# ═══════════════════════════════════════════════════════════════════════════

def _tab_marco():
    st.subheader("📌 1. Marco y Ruta")
    st.markdown(
        "La Fase 3 de CRISP-DM transforma los datos comprendidos y explorados "
        "en insumos reproducibles para modelado y para el LLM. En esta versión "
        "las transformaciones aprendidas se ajustan exclusivamente con TRAIN."
    )
    a, b = st.columns(2)
    with a:
        st.info(
            "**🅰️ `df_modelado`**\n\n"
            "Dataset numérico final para `new_modelado.py`: target separado, "
            "features seleccionadas, imputadas y opcionalmente escaladas."
        )
    with b:
        st.success(
            "**🅱️ `df_consulta`**\n\n"
            "Dataset legible para LLM/RAG: identificadores, cifras originales, "
            "ratios, crecimiento y ficha textual."
        )

    st.markdown("### 🧭 Ruta CRISP-DM implementada")
    st.dataframe(pd.DataFrame([
        [1, "Diagnóstico y estandarización", "Sin aprendizaje de parámetros"],
        [2, "Encoding", "Plan definido; vocabulario/frecuencia se aprenden en TRAIN"],
        [3, "Feature Engineering", "Ratios, intensidades, magnitudes, log, flags, tendencias"],
        [4, "Train / Test", "Partición antes de transformaciones aprendidas"],
        [5, "Feature Selection", "Correlación + RF + MI + Permutation sobre TRAIN"],
        [6, "Imputación / escalado", "Ajustados sobre TRAIN y aplicados a TEST"],
        [7, "Resultado", "df_modelado listo para Modelado"],
        [8, "LLM", "df_consulta + diccionario + FICHA_TEXTO"],
    ], columns=["Paso", "Actividad", "Regla metodológica"]), use_container_width=True)


def _diagnostico_calidad(df: pd.DataFrame, incidencias: dict) -> dict:
    """Construye el diagnóstico de calidad SOBRE LA BASE ORIGINAL.

    Esta función no transforma los datos ni aprende parámetros para ML.
    Su objetivo es responder, antes de preparar el dataset:
    ¿Cómo están los datos?
    """
    n_empresas = len(df)
    n_columnas = len(df.columns)

    numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categoricas = [c for c in df.columns if c not in numericas]

    # Valores faltantes reales de la fuente, antes de estandarización.
    faltantes = pd.DataFrame([
        {
            "Variable": c,
            "Valores faltantes": int(df[c].isna().sum()),
            "% faltantes": round(df[c].isna().mean() * 100, 2),
        }
        for c in df.columns
        if int(df[c].isna().sum()) > 0
    ]).sort_values(["% faltantes", "Valores faltantes"], ascending=False)

    # Cardinalidad de todas las variables. Para categóricas es especialmente
    # relevante; para numéricas ayuda a detectar campos casi identificadores.
    cardinalidad = pd.DataFrame([
        {
            "Variable": c,
            "Tipo": str(df[c].dtype),
            "Valores únicos": int(df[c].nunique(dropna=True)),
            "% únicos": round(
                df[c].nunique(dropna=True) / max(n_empresas, 1) * 100, 2
            ),
            "Clasificación": (
                "Identificador"
                if c in COLS_IDENTIFICADORES
                else "Casi único"
                if df[c].nunique(dropna=True) / max(n_empresas, 1) > UMBRAL_IDENTIFICADOR
                else "Alta cardinalidad"
                if df[c].nunique(dropna=True) > 35
                else "Baja/media cardinalidad"
            ),
        }
        for c in df.columns
    ]).sort_values("% únicos", ascending=False)

    identificadoras = []
    for c in df.columns:
        ratio = df[c].nunique(dropna=True) / max(n_empresas, 1)
        if c in COLS_IDENTIFICADORES or ratio > UMBRAL_IDENTIFICADOR:
            identificadoras.append({
                "Variable": c,
                "Valores únicos": int(df[c].nunique(dropna=True)),
                "% únicos": round(ratio * 100, 2),
                "Motivo": (
                    "Identificador definido por el proyecto"
                    if c in COLS_IDENTIFICADORES
                    else "Más del 50% de valores distintos; posible identificador/campo casi único"
                ),
            })
    identificadoras = pd.DataFrame(identificadoras).sort_values(
        "% únicos", ascending=False
    ) if identificadoras else pd.DataFrame(
        columns=["Variable", "Valores únicos", "% únicos", "Motivo"]
    )

    # Detección de valores extremos de calidad: IQR sobre columnas numéricas
    # originales. No elimina ni modifica filas y no es la política de outliers
    # de Machine Learning que se ejecuta posteriormente.
    outlier_rows = []
    for c in numericas:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        n_out = int(mask.sum())
        if n_out:
            outlier_rows.append({
                "Variable": c,
                "Valores atípicos IQR": n_out,
                "% sobre válidos": round(n_out / len(s) * 100, 2),
                "Límite inferior": lo,
                "Límite superior": hi,
            })
    outliers = pd.DataFrame(outlier_rows)
    if not outliers.empty:
        outliers = outliers.sort_values("Valores atípicos IQR", ascending=False)

    # Incidencias de estandarización ya detectadas por el proceso existente.
    incidencias_df = pd.DataFrame([
        {
            "Variable": c,
            "Celdas vacías detectadas": int(v.get("vacios", 0)),
            "Valores numéricos atípicos": int(v.get("atipicos_numericos", 0)),
        }
        for c, v in incidencias.items()
        if v.get("vacios", 0) or v.get("atipicos_numericos", 0)
    ])
    if not incidencias_df.empty:
        incidencias_df = incidencias_df.sort_values(
            ["Celdas vacías detectadas", "Valores numéricos atípicos"],
            ascending=False,
        )

    # Problemas detectados: resumen ejecutivo, sin alterar la base.
    problemas = []
    if faltantes.empty:
        problemas.append(("OK", "Valores faltantes", "No se detectaron nulos en la base original."))
    else:
        problemas.append(("ATENCIÓN", "Valores faltantes",
                          f"{len(faltantes)} variables presentan valores faltantes."))

    if outliers.empty:
        problemas.append(("OK", "Valores atípicos de calidad",
                          "No se detectaron valores extremos mediante la regla IQR en variables numéricas."))
    else:
        problemas.append(("REVISAR", "Valores atípicos de calidad",
                          f"{len(outliers)} variables numéricas presentan observaciones extremas por IQR."))

    if identificadoras.empty:
        problemas.append(("OK", "Variables identificadoras",
                          "No se detectaron identificadores/campos casi únicos adicionales."))
    else:
        problemas.append(("ATENCIÓN", "Variables identificadoras",
                          f"{len(identificadoras)} variables deben tratarse como identificadoras o casi únicas."))

    if incidencias_df.empty:
        problemas.append(("OK", "Estandarización",
                          "No se detectaron incidencias de estandarización en las reglas actuales."))
    else:
        problemas.append(("ATENCIÓN", "Estandarización",
                          f"{len(incidencias_df)} variables presentan incidencias que requieren revisión."))

    alta_card = cardinalidad[
        cardinalidad["Clasificación"].isin(["Alta cardinalidad", "Casi único"])
    ]
    if alta_card.empty:
        problemas.append(("OK", "Cardinalidad",
                          "No se detectaron problemas relevantes de cardinalidad."))
    else:
        problemas.append(("REVISAR", "Cardinalidad",
                          f"{len(alta_card)} variables presentan alta cardinalidad o comportamiento casi único."))

    return {
        "n_empresas": n_empresas,
        "n_columnas": n_columnas,
        "numericas": numericas,
        "categoricas": categoricas,
        "faltantes": faltantes,
        "cardinalidad": cardinalidad,
        "identificadoras": identificadoras,
        "outliers": outliers,
        "incidencias": incidencias_df,
        "problemas": pd.DataFrame(problemas, columns=["Estado", "Aspecto", "Hallazgo"]),
    }


def _tab_diagnostico(df_original, df_est, incidencias, fe_rep):
    st.subheader("🔎 2. Diagnóstico")
    st.markdown(
        "### ¿Cómo están los datos antes de prepararlos?"
    )
    st.caption(
        "Este diagnóstico se calcula sobre la **base original**, antes de Feature Engineering, "
        "Encoding, partición Train/Test, imputación, escalado y selección de variables. "
        "Su propósito es identificar la estructura y los principales problemas de calidad "
        "antes de tomar decisiones de preparación."
    )

    diag = _diagnostico_calidad(df_original, incidencias)

    # ── Resumen ejecutivo ──────────────────────────────────────────────────
    a, b, c, d = st.columns(4)
    a.metric("🏢 Número de empresas", f"{diag['n_empresas']:,}")
    b.metric("🧱 Número de columnas", f"{diag['n_columnas']:,}")
    c.metric("🔢 Columnas numéricas", f"{len(diag['numericas']):,}")
    d.metric("🔤 Columnas categóricas", f"{len(diag['categoricas']):,}")

    st.markdown("### 🧭 Lectura rápida de calidad")
    falt_n = len(diag["faltantes"])
    out_n = len(diag["outliers"])
    id_n = len(diag["identificadoras"])
    inc_n = len(diag["incidencias"])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Valores faltantes", f"{falt_n:,} variables")
    k2.metric("Atípicos de calidad", f"{out_n:,} variables")
    k3.metric("Identificadoras / casi únicas", f"{id_n:,}")
    k4.metric("Incidencias de estandarización", f"{inc_n:,}")

    # ── Tipos de variables ─────────────────────────────────────────────────
    st.markdown("### 🧩 Tipos de variables")
    tipos = pd.DataFrame({
        "Tipo": ["Numéricas", "Categóricas / texto"],
        "Cantidad": [len(diag["numericas"]), len(diag["categoricas"])],
        "% del total": [
            round(len(diag["numericas"]) / max(diag["n_columnas"], 1) * 100, 1),
            round(len(diag["categoricas"]) / max(diag["n_columnas"], 1) * 100, 1),
        ],
    })
    col_graf, col_tab = st.columns([1, 1])
    # with col_graf:
    #     fig = px.pie(
    #         tipos,
    #         names="Tipo",
    #         values="Cantidad",
    #         hole=0.45,
    #         title="Composición por tipo de variable",
    #     )
    #     fig.update_traces(textposition="inside", textinfo="percent+label")
    #     fig.update_layout(margin=dict(t=75, l=10, r=10, b=10), height=320)
    #     st.plotly_chart(fig, use_container_width=True)
    
    with col_graf:
        fig = px.pie(
            tipos,
            names="Tipo",
            values="Cantidad",
            hole=0.35,
            title="Composición por tipo de variable",
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            textfont=dict(size=20, color="black", family="Arial, sans-serif"),
            marker=dict(line=dict(color="white", width=2)),
        )
        fig.update_layout(
            margin=dict(t=75, l=10, r=10, b=10),
            height=340,
            title=dict(font=dict(size=17)),
            showlegend=True,
            legend=dict(font=dict(size=14), orientation="v"),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_tab:
        st.dataframe(tipos, use_container_width=True, hide_index=True)

    # ── Valores faltantes ──────────────────────────────────────────────────
    with st.expander("🕳️ Valores faltantes", expanded=not diag["faltantes"].empty):
        if diag["faltantes"].empty:
            st.success("No se detectaron valores faltantes en la base original.")
        else:
            st.warning(
                f"Se detectaron valores faltantes en {len(diag['faltantes'])} variables."
            )
            st.dataframe(diag["faltantes"], use_container_width=True, hide_index=True)

    # ── Valores atípicos de calidad ─────────────────────────────────────────
    with st.expander("📌 Valores atípicos de calidad", expanded=False):
        st.caption(
            "Regla descriptiva IQR (Q1 − 1,5×IQR / Q3 + 1,5×IQR). "
            "Estos valores **no se eliminan** aquí y esta detección no sustituye "
            "la política formal de outliers para Machine Learning de la pestaña 6."
        )
        if diag["outliers"].empty:
            st.success("No se detectaron valores extremos mediante IQR.")
        else:
            st.warning(
                f"{len(diag['outliers'])} variables numéricas presentan valores extremos."
            )
            st.dataframe(diag["outliers"], use_container_width=True, hide_index=True)

    # ── Variables identificadoras ───────────────────────────────────────────
    with st.expander("🪪 Variables identificadoras", expanded=True):
        st.caption(
            "Se consideran identificadores los campos definidos por el proyecto "
            "y los campos con más del 50 % de valores distintos."
        )
        if diag["identificadoras"].empty:
            st.success("No se detectaron variables identificadoras/casi únicas.")
        else:
            st.dataframe(
                diag["identificadoras"],
                use_container_width=True,
                hide_index=True,
            )

    # ── Cardinalidad ────────────────────────────────────────────────────────
    with st.expander("🔢 Cardinalidad", expanded=False):
        st.caption(
            "La cardinalidad muestra cuántos valores distintos tiene cada variable. "
            "Es especialmente importante para decidir posteriormente entre One-Hot, "
            "Frequency Encoding o exclusión."
        )
        st.dataframe(
            diag["cardinalidad"],
            use_container_width=True,
            hide_index=True,
        )

    # ── Incidencias de estandarización ───────────────────────────────────────
    with st.expander("🧹 Incidencias de estandarización", expanded=not diag["incidencias"].empty):
        if diag["incidencias"].empty:
            st.success("No se detectaron incidencias bajo las reglas actuales.")
        else:
            st.warning("Se detectaron incidencias que serán tratadas en la preparación.")
            st.dataframe(
                diag["incidencias"],
                use_container_width=True,
                hide_index=True,
            )

    # ── Problemas detectados ─────────────────────────────────────────────────
    st.markdown("### 🚦 Problemas detectados")
    problemas = diag["problemas"]

    def _estilo_estado(valor):
        if valor == "OK":
            return "background-color: #E8F5E9; color: #1B5E20; font-weight: 700"
        if valor in ("ATENCIÓN", "REVISAR"):
            return "background-color: #FFF3CD; color: #7A4E00; font-weight: 700"
        return ""

    st.dataframe(
        problemas.style.map(_estilo_estado, subset=["Estado"]),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "📍 **Conclusión del diagnóstico:** esta pestaña describe el estado de la fuente "
        "antes de prepararla. Las decisiones de tratamiento —Encoding, Feature Engineering, "
        "Train/Test, outliers, Feature Selection, imputación y escalado— se presentan en "
        "las pestañas siguientes, manteniendo separadas la descripción de la calidad y la "
        "transformación del dataset."
    )


def _tab_encoding(plan, config):
    st.subheader("🏷️ 3. Estrategia y Encoding")
    st.markdown(
        "Esta es la **única tabla de decisión del Encoding**. El plan se determina "
        "sobre TRAIN y **Frequency Encoding y One-Hot se ajustan después de la "
        "partición**, para impedir que TEST influya en los parámetros."
    )
    st.info(
        f"Umbral One-Hot configurado: **{config['umbral_ohe']} categorías**. "
        f"`drop_first={config['drop_first']}`. "
        "Regla: categorías ≤ umbral → One-Hot; categorías > umbral → Frequency."
    )
    st.dataframe(plan.set_index("Variable"), use_container_width=True)

    if "rep_encoding" not in st.session_state:
        st.info("Ejecute la pestaña Partición Train / Test para ver el encoding ajustado.")
        return
    rep = st.session_state["prep_pipeline"]["rep_encoding"]
    a, b, c = st.columns(3)
    a.metric("Variables One-Hot", len(rep["mapa_ohe"]))
    b.metric("Variables Frequency", len(rep["mapa_freq"]))
    c.metric("Columnas nuevas", len(rep["detalle_nuevas"]))

    if rep["mapa_freq"]:
        st.markdown("### Frequency Encoding aprendido en TRAIN")
        for var, col in rep["mapa_freq"].items():
            st.write(f"`{var}` → `{col}`")
    if rep["mapa_ohe"]:
        st.markdown("### One-Hot aprendido en TRAIN")
        st.write(", ".join(rep["mapa_ohe"].keys()))


def _tab_feature_engineering(df_features, fe_rep):
    st.subheader("🧬 4. Feature Engineering")
    st.markdown(
        "Esta pestaña incorpora el modelo desarrollado en `feature_engineering.py`. "
        "Las variables son deterministas y se construyen antes de la partición. Incluyen "
        "ratios financieros, magnitudes, intensidades, logaritmos, severidades, flags, "
        "interacciones y evolución temporal."
    )
    a, b, c = st.columns(3)
    a.metric("Variables de entrada", fe_rep.get("shape_entrada", (0, 0))[1])
    b.metric("Variables de salida", fe_rep.get("shape_salida", df_features.shape)[1])
    c.metric("Variables nuevas", fe_rep.get("n_variables_nuevas", 0))

    nuevas = fe_rep.get("columnas_nuevas", [])
    if nuevas:
        st.markdown("### Variables creadas")
        st.dataframe(pd.DataFrame({"Variable": nuevas}), use_container_width=True, height=420)

    categorias = {
        "Ratios financieros": ["MARGEN", "ROA", "ROE", "SOLVENCIA", "ENDEUDAMIENTO", "ROTACION", "APALANCAMIENTO"],
        "Crecimiento / temporal": ["CRECIMIENTO", "CAMBIO_MARGEN", "ESTABILIDAD"],
        "Log / magnitud / intensidad": ["LOG_", "MAGNITUD", "INTENSIDAD"],
        "Riesgo / severidad": ["SEVERIDAD", "RIESGO", "DETERIORO"],
        "Flags": ["FLAG_", "ES_PERDIDA", "ES_PATRIMONIO_NEGATIVO"],
    }
    resumen = []
    for nombre, prefijos in categorias.items():
        cols = [c for c in nuevas if any(p in c.upper() for p in prefijos)]
        resumen.append({"Familia": nombre, "Cantidad": len(cols)})
    st.dataframe(pd.DataFrame(resumen), use_container_width=True)

    st.caption(
        "Nota: las variables derivadas de 2022–2023 son deterministas. Cuando el objetivo sea 2024, "
        "las variables que incorporen 2024 se excluyen automáticamente como predictores para evitar leakage."
    )


def _tab_particion(pipeline):
    st.subheader("✂️ 5. Partición Train / Test")
    st.markdown(
        "**Esta es la barrera contra data leakage.** Primero se separan TRAIN y TEST; "
        "después se aprende el vocabulario One-Hot y el Frequency Encoding únicamente con TRAIN."
    )
    if pipeline is None:
        st.info("Configure el objetivo y ejecute la preparación para generar la partición.")
        return

    a, b, c, d = st.columns(4)
    a.metric("Target", pipeline["target"])
    b.metric("TRAIN", f"{len(pipeline['y_train']):,}")
    c.metric("TEST", f"{len(pipeline['y_test']):,}")
    d.metric("Features encoded", pipeline["X_train_enc"].shape[1])

    st.markdown("### Regla temporal aplicada")
    if pipeline["excl_temporales"]:
        st.dataframe(pd.DataFrame({"Variables excluidas por seguridad temporal": pipeline["excl_temporales"]}), use_container_width=True)
    else:
        st.success("No se detectaron variables temporales que debieran excluirse.")

    st.markdown("### Dimensiones")
    st.dataframe(pd.DataFrame([
        ["X_train_raw", *pipeline["X_train_raw"].shape],
        ["X_test_raw", *pipeline["X_test_raw"].shape],
        ["X_train_encoded", *pipeline["X_train_enc"].shape],
        ["X_test_encoded", *pipeline["X_test_enc"].shape],
        ["X_train_outlier", *pipeline.get("X_train_outlier", pipeline["X_train_enc"]).shape],
        ["X_test_outlier", *pipeline.get("X_test_outlier", pipeline["X_test_enc"]).shape],
    ], columns=["Objeto", "Filas", "Columnas"]), use_container_width=True)

    st.success(
        "Frequency Encoding y One-Hot fueron aprendidos con TRAIN. "
        "Una categoría que aparezca solamente en TEST no modifica el mapa de TRAIN. "
        "La política de outliers se aplica después sin eliminar empresas."
    )

    st.markdown("### Vista de TRAIN / TEST")
    st.dataframe(pipeline["train_model"].head(5), use_container_width=True)


def _tab_outliers(pipeline):
    st.subheader("🎯 6. Política de Outliers para Machine Learning")
    st.markdown("""
    ### Política formal
    Los valores extremos **no se eliminan automáticamente**: pueden representar empresas grandes,
    pérdidas reales o situaciones financieras excepcionales. La política de ML es distinta de los
    rangos usados en EDA para mejorar una gráfica.

    - **Magnitudes financieras:** se conservan. Si TRAIN presenta asimetría elevada (`|skew| ≥ 1,50`),
      se añade una copia `*_ML_LOG` sin alterar la magnitud original.
    - **Ratios y tasas de crecimiento:** se aplica winsorización P01–P99.
    - **TRAIN aprende los límites**; TEST recibe exactamente esos mismos límites.
    - **No se eliminan empresas** y el target queda fuera de esta transformación.

    El tratamiento ocurre después de Train/Test y antes de Feature Selection, imputación y escalado.
    """)
    if pipeline is None or "outlier_result" not in pipeline:
        st.info("Configure el objetivo y ejecute la preparación para aplicar la política.")
        return
    r = pipeline["outlier_result"]
    a,b,c,d = st.columns(4)
    a.metric("TRAIN con outlier", f"{r['train_empresas_con_outlier']:,}")
    b.metric("TEST con outlier", f"{r['test_empresas_con_outlier']:,}")
    c.metric("Valores afectados TRAIN", f"{r['train_valores_afectados']:,}")
    d.metric("Valores afectados TEST", f"{r['test_valores_afectados']:,}")
    st.markdown("### Reglas aplicadas")
    st.dataframe(pd.DataFrame([
        ["Magnitudes financieras", "Conservar + LOG si |skew| ≥ 1,50", "TRAIN", "No eliminar filas"],
        ["Ratios / crecimientos", "Winsorización P01–P99", "TRAIN", "Mismos límites en TEST"],
        ["Target", "Sin tratamiento de outliers", "—", "Se mantiene separado"],
    ], columns=["Grupo", "Tratamiento", "Parámetro aprendido", "Regla"]), use_container_width=True)
    rep = r.get("reporte")
    if isinstance(rep, pd.DataFrame) and not rep.empty:
        st.markdown("### Trazabilidad por variable")
        st.dataframe(rep, use_container_width=True)
        st.download_button("⬇️ Descargar reporte de outliers", _csv(rep),
                           "reporte_outliers_ml.csv", "text/csv", key="prep_dl_outliers")
    else:
        st.success("No se identificaron variables que requirieran tratamiento bajo esta política.")
    st.info("Los rangos de EDA no se reutilizan como límites de ML; los límites de ML se aprenden únicamente desde TRAIN.")


def _tab_transformaciones(transformaciones, escalar):
    st.subheader("📐 7. Escalado y Transformaciones")
    st.markdown(
        "La imputación y, si se activa, `StandardScaler`, se ajustan exclusivamente con TRAIN. "
        "Los modelos de árboles no necesitan escalado; regresión lineal, redes y otros modelos "
        "sensibles a escala pueden utilizar la salida escalada."
    )
    if transformaciones is None:
        st.info("Aún no se han generado las transformaciones.")
        return

    a, b, c = st.columns(3)
    a.metric("Features", len(transformaciones["columnas"]))
    b.metric("Escalado", "Sí" if escalar else "No")
    c.metric("Imputación", "Mediana TRAIN")

    if transformaciones["scaler"] is not None:
        st.success("StandardScaler ajustado únicamente sobre TRAIN.")
        st.caption("TEST se transforma con los parámetros aprendidos en TRAIN; nunca se recalcula el scaler con TEST.")
    else:
        st.info("Se conserva la escala original después de imputar. Recomendado para Random Forest/XGBoost.")

    st.markdown("### Vista de datos listos para Modelado")
    st.dataframe(transformaciones["X_train_ready"].head(5), use_container_width=True)


def _tab_resultado(pipeline, fs_result, transformaciones, target):
    st.subheader("📊 8. Resultado")
    if pipeline is None or transformaciones is None:
        st.info("Complete la configuración de Train/Test para visualizar el resultado final.")
        return

    selected = fs_result["selected_columns"] if fs_result else list(pipeline["X_train_enc"].columns)
    a, b, c, d = st.columns(4)
    a.metric("Target", target)
    b.metric("Features antes FS", pipeline.get("X_train_outlier", pipeline["X_train_enc"]).shape[1])
    c.metric("Features después FS", len(selected))
    d.metric("Filas TRAIN", f"{len(pipeline['y_train']):,}")

    if fs_result and fs_result.get("error"):
        st.warning("Feature Selection tuvo un error y se utilizó un fallback seguro: " + fs_result["error"])

    st.markdown("### Feature Selection")
    st.caption(
        "Se conserva el método del módulo fuente: eliminación de redundancia por correlación y "
        "ranking mediante Random Forest, Mutual Information y Permutation Importance. Todo se calcula con TRAIN."
    )
    if fs_result and fs_result.get("reporte"):
        rep = fs_result["reporte"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Correlación", rep.get("correlation_threshold", CORRELATION_THRESHOLD))
        m2.metric("Top N", rep.get("top_n", DEFAULT_TOP_N))
        m3.metric("Variables finales", len(selected))

        corr = rep.get("correlaciones_altas")
        if isinstance(corr, pd.DataFrame) and not corr.empty:
            with st.expander("Correlaciones altas detectadas", expanded=False):
                st.dataframe(corr, use_container_width=True)

        ranking = rep.get("ranking_variables")
        if isinstance(ranking, pd.DataFrame) and not ranking.empty:
            with st.expander("Ranking de importancia de variables", expanded=True):
                st.dataframe(ranking.head(50), use_container_width=True)

    st.markdown("### Dataset final para `new_modelado.py`")
    st.dataframe(transformaciones["X_train_ready"].head(5), use_container_width=True)

    st.session_state["df_modelado"] = transformaciones["X_train_ready"]
    st.session_state["X_train"] = transformaciones["X_train_ready"]
    st.session_state["X_test"] = transformaciones["X_test_ready"]
    st.session_state["y_train"] = pipeline["y_train"]
    st.session_state["y_test"] = pipeline["y_test"]

    st.download_button(
        "⬇️ Descargar X_train listo para Modelado",
        _csv(transformaciones["X_train_ready"]),
        "X_train_modelado.csv",
        "text/csv",
        key="prep_dl_xtrain",
    )
    st.download_button(
        "⬇️ Descargar X_test listo para Modelado",
        _csv(transformaciones["X_test_ready"]),
        "X_test_modelado.csv",
        "text/csv",
        key="prep_dl_xtest",
    )


def _tab_llm(df_llm, dicc):
    st.subheader("🤖 9. Base para el LLM")
    a, b, c = st.columns(3)
    a.metric("Empresas", f"{len(df_llm):,}")
    b.metric("Columnas", len(df_llm.columns))
    c.metric("Fichas", f"{df_llm['FICHA_TEXTO'].notna().sum():,}")

    st.dataframe(df_llm.drop(columns=["FICHA_TEXTO"], errors="ignore").head(5), use_container_width=True)
    st.markdown("### 📝 Ficha textual")
    consulta = st.text_input("Buscar empresa por razón social o NIT:", key="prep_buscar_ficha")
    if consulta.strip():
        q = consulta.strip().upper()
        mask = (
            df_llm["RAZON SOCIAL"].astype(str).str.upper().str.contains(q, regex=False, na=False)
            | df_llm["NIT"].astype(str).str.contains(q, regex=False, na=False)
        )
        resultado = df_llm[mask].head(3)
    else:
        resultado = df_llm.head(3)

    if resultado.empty:
        st.warning("No se encontraron empresas.")
    for _, fila in resultado.iterrows():
        with st.container(border=True):
            st.markdown(f"**{fila.get('RAZON SOCIAL', 'Empresa')}** · NIT {fila.get('NIT', 'n.d.')}")
            st.write(fila["FICHA_TEXTO"])

    st.markdown("### 📖 Diccionario de datos")
    st.dataframe(dicc.set_index("Variable"), use_container_width=True, height=420)
    st.info("Regla: las cifras y agregaciones se calculan con Python; el LLM interpreta y redacta, no inventa cifras.")


def _tab_validacion(df_llm, dicc, pipeline, fs_result, transformaciones):
    st.subheader("✅ 10. Validación y Entregables")

    checks = []
    if pipeline is not None:
        checks.extend([
            ("TRAIN y TEST tienen filas", len(pipeline["y_train"]) > 0 and len(pipeline["y_test"]) > 0),
            ("Mismas columnas en X_train y X_test", list(pipeline["X_train_enc"].columns) == list(pipeline["X_test_enc"].columns)),
            ("Target separado de los predictores", pipeline["target"] not in pipeline["X_train_enc"].columns),
            ("Sin identificadores en X_train", not any(c in pipeline["X_train_enc"].columns for c in COLS_IDENTIFICADORES)),
        ])
    if pipeline is not None and "outlier_result" in pipeline:
        checks.extend([
            ("La política de outliers no eliminó filas", pipeline["outlier_result"]["n_train_empresas"] == len(pipeline["X_train_outlier"])),
            ("Los límites se aprendieron en TRAIN", pipeline["outlier_result"]["politica"]["calcula_limites_en"] == "TRAIN"),
            ("TRAIN y TEST conservan las mismas columnas tras outliers", list(pipeline["X_train_outlier"].columns) == list(pipeline["X_test_outlier"].columns)),
        ])
    if transformaciones is not None:
        checks.extend([
            ("X_train final sin nulos", int(transformaciones["X_train_ready"].isna().sum().sum()) == 0),
            ("X_test final sin nulos", int(transformaciones["X_test_ready"].isna().sum().sum()) == 0),
            ("X_train final sin infinitos", int(np.isinf(transformaciones["X_train_ready"].to_numpy()).sum()) == 0),
            ("X_test final sin infinitos", int(np.isinf(transformaciones["X_test_ready"].to_numpy()).sum()) == 0),
        ])
    checks.append(("df_consulta conserva NIT y Razón Social", all(c in df_llm.columns for c in COLS_IDENTIFICADORES)))
    checks.append(("Todas las empresas tienen FICHA_TEXTO", bool(df_llm["FICHA_TEXTO"].notna().all())))

    for texto, ok in checks:
        (st.success if ok else st.error)(f"{'✅' if ok else '❌'} {texto}")

    st.markdown("### 📦 Entregables")
    entregables = [
        ["df_consulta", "LLM / RAG", f"{df_llm.shape[0]:,} × {df_llm.shape[1]}"],
        ["diccionario_datos", "Contexto LLM", f"{dicc.shape[0]:,} × {dicc.shape[1]}"],
    ]
    if pipeline is not None:
        out_rep = pipeline.get("outlier_result", {}).get("reporte", pd.DataFrame())
        xtr_out = pipeline.get("X_train_outlier", pipeline["X_train_enc"])
        xte_out = pipeline.get("X_test_outlier", pipeline["X_test_enc"])
        entregables += [
            ["Reporte de outliers", "Trazabilidad", f"{len(out_rep):,} variables"],
            ["X_train / y_train", "new_modelado.py", f"{xtr_out.shape[0]:,} × {xtr_out.shape[1]}"],
            ["X_test / y_test", "Evaluación", f"{xte_out.shape[0]:,} × {xte_out.shape[1]}"],
        ]
    if transformaciones is not None:
        entregables.append(["df_modelado", "Modelado / SHAP", f"{transformaciones['X_train_ready'].shape[0]:,} × {transformaciones['X_train_ready'].shape[1]} + target separado"])

    st.dataframe(pd.DataFrame(entregables, columns=["Objeto", "Consumidor", "Dimensión"]), use_container_width=True)

    st.markdown("### ➡️ Siguiente fase")
    st.markdown(
        "`new_modelado.py` debe recibir **X_train, X_test, y_train, y_test** ya preparados. "
        "No debe volver a ajustar Frequency Encoding, imputación o StandardScaler sobre todo el dataset."
    )


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

def mostrar_preparacion(df: pd.DataFrame):
    """Contrato de integración con new_main_streamlit.py."""
    st.header("🔧 Preparación de los Datos")
    st.caption(
        "Fase 3 · CRISP-DM — preparación reproducible para Machine Learning y LLM, "
        "con control explícito de data leakage."
    )

    # ----------------------------------------------------------------------
    # Configuración principal. Se mantiene fuera de las pestañas para que el
    # pipeline sea estable y no se creen widgets duplicados.
    # ----------------------------------------------------------------------
    df_est, incidencias = estandarizar(df)

    # FE se ejecuta una sola vez y es determinista.
    try:
        df_features, fe_rep = ejecutar_feature_engineering(df_est)
    except Exception as exc:
        st.error(f"Error en Feature Engineering: {exc}")
        df_features = df_est.copy()
        fe_rep = {
            "shape_entrada": df_est.shape,
            "shape_salida": df_features.shape,
            "columnas_nuevas": [],
            "n_variables_nuevas": 0,
        }

    # Catálogo explícito: únicamente variables financieras originales o
    # indicadores derivados de negocio; nunca IDs, ranking, logs ni encoding.
    catalogo_targets = catalogo_targets_financieros(df_features)
    targets_disponibles = list(catalogo_targets.keys())
    preferido = next((c for c in targets_disponibles if c.replace("*", "").strip() == f"{ING} 2024"), None)

    with st.expander("⚙️ Configuración de la preparación", expanded=True):
        # El año del objetivo queda fijado en 2024; no existe selector de año.
        c_tipo, c_target = st.columns([1, 3])
        tipo_opciones = ["Todos", "Variable financiera original", "Indicador financiero derivado"]
        with c_tipo:
            tipo_target = st.selectbox(
                "Tipo de indicador",
                tipo_opciones,
                index=0,
                key="prep_tipo_target_2024_v1",
                help="El experimento está fijado en objetivos reales de 2024; se excluyen IDs, rankings y transformaciones técnicas.",
            )
        filtrados = []
        for col in targets_disponibles:
            desc = catalogo_targets[col]
            es_original = "variable original" in desc.lower()
            if tipo_target == "Variable financiera original" and not es_original:
                continue
            if tipo_target == "Indicador financiero derivado" and es_original:
                continue
            filtrados.append(col)
        if not filtrados:
            st.warning("No se encontraron objetivos financieros válidos de 2024 en el dataset.")
        default_target = preferido if preferido in filtrados else (filtrados[0] if filtrados else None)
        with c_target:
            target = st.selectbox(
                "Variable financiera objetivo (target 2024)",
                filtrados,
                index=filtrados.index(default_target) if default_target is not None else 0,
                format_func=lambda c: f"{c}  |  {catalogo_targets[c]}",
                key="prep_target_2024_v1",
                help="Se predice el valor real de 2024 usando predictores de 2022/2023 y contexto estático. 2025/2026 quedan para inferencia prospectiva.",
            ) if filtrados else None
        if target is not None:
            st.info(f"**Objetivo seleccionado:** {target}. El experimento usa antecedentes de 2022 y 2023, evalúa contra el valor real de 2024 en TEST y excluye todas las variables que revelen resultados de 2024.")

        c1, c2, c3 = st.columns(3)
        with c1:
            test_size = st.slider("Proporción TEST", 0.10, 0.40, DEFAULT_TEST_SIZE, 0.05, key="prep_test_size_v2")
            random_state = st.number_input("Random state", 0, 9999, DEFAULT_RANDOM_STATE, 1, key="prep_random_state_v2")
        with c2:
            umbral_ohe = st.slider("Umbral One-Hot", 5, 100, UMBRAL_OHE, key="prep_umbral_v2")
            drop_first = st.checkbox("drop_first en One-Hot", True, key="prep_drop_first_v2")
        with c3:
            incluir_meta = st.checkbox("Incluir metadatos NIIF", True, key="prep_incluir_meta_v2")

    c4, c5 = st.columns(2)
    with c4:
        top_n = st.number_input("Top N variables para Feature Selection", 5, 100, DEFAULT_TOP_N, 5, key="prep_top_n_v2")
    with c5:
        escalar = st.checkbox(
            "Aplicar StandardScaler para la salida de modelos sensibles a escala",
            value=False,
            key="prep_escalar_v2",
        )

    pipeline = None
    fs_result = None
    transformaciones = None

    # Plan preliminar de respaldo para la interfaz. IMPORTANTE:
    # utiliza el umbral seleccionado por el usuario, no la constante UMBRAL_OHE.
    plan_encoding = _plan_encoding_base(
        df_features,
        int(umbral_ohe),
        incluir_meta=bool(incluir_meta),
    )

    if target is not None:
        try:
            pipeline = preparar_train_test(
                df_features,
                target=target,
                test_size=float(test_size),
                random_state=int(random_state),
                umbral_ohe=int(umbral_ohe),
                drop_first=bool(drop_first),
                incluir_meta=bool(incluir_meta),
            )

            outlier_result = aplicar_politica_outliers_train_test(
                pipeline["X_train_enc"], pipeline["X_test_enc"], target=target
            )
            pipeline["X_train_outlier"] = outlier_result["X_train"]
            pipeline["X_test_outlier"] = outlier_result["X_test"]
            pipeline["outlier_result"] = outlier_result

            # La tabla visible de Estrategia y Encoding debe representar el
            # mismo plan que realmente se aplicó al TRAIN.
            plan_encoding = pipeline["rep_encoding"]["plan_train"].copy()

            fs_result = ejecutar_feature_selection_train(
                pipeline["X_train_outlier"],
                pipeline["y_train"],
                target=target,
                correlation_threshold=float(CORRELATION_THRESHOLD),
                top_n=int(top_n),
            )

            selected_cols = fs_result["selected_columns"]
            X_train_sel = pipeline["X_train_outlier"].reindex(columns=selected_cols)
            X_test_sel = pipeline["X_test_outlier"].reindex(columns=selected_cols)

            transformaciones = preparar_transformaciones(
                X_train_sel,
                X_test_sel,
                escalar=bool(escalar),
            )

            st.session_state["prep_pipeline"] = pipeline
            st.session_state["outlier_reporte"] = outlier_result
            st.session_state["prep_feature_selection"] = fs_result
            st.session_state["prep_transformaciones"] = transformaciones
            st.session_state["df_modelado"] = transformaciones["X_train_ready"]
            st.session_state["X_train"] = transformaciones["X_train_ready"]
            st.session_state["X_test"] = transformaciones["X_test_ready"]
            st.session_state["y_train"] = pipeline["y_train"]
            st.session_state["y_test"] = pipeline["y_test"]
            st.session_state["target_modelado"] = target
            st.session_state["target_modelado_descripcion"] = catalogo_targets.get(target, "Indicador financiero")
            st.session_state["catalogo_targets_financieros"] = catalogo_targets
            st.session_state["feature_selection_reporte"] = fs_result["reporte"]
            st.session_state["feature_engineering_reporte"] = fe_rep

        except Exception as exc:
            st.error(f"No fue posible construir el pipeline Train/Test: {exc}")
            pipeline = None
            fs_result = None
            transformaciones = None

    # Base LLM independiente del pipeline predictivo.
    try:
        df_llm, dicc = construir_base_llm(df)
    except Exception as exc:
        st.error(f"Error al construir df_consulta: {exc}")
        df_llm = estandarizar(df)[0]
        df_llm["FICHA_TEXTO"] = "Información no disponible por error de construcción."
        dicc = construir_diccionario(df_llm)

    st.session_state["df_consulta"] = df_llm
    st.session_state["df_contexto_llm"] = df_llm
    st.session_state["diccionario_datos"] = dicc
    st.session_state["df_features"] = df_features
    st.session_state["df_features_financieras"] = df_features.copy()
    st.session_state["catalogo_targets_financieros"] = catalogo_targets
    st.session_state["resumen_disponibilidad_datos"] = {
        "filas_empresas": int(len(df)),
        "columnas_originales": int(len(df.columns)),
        "columnas_feature_engineering": int(len(df_features.columns)),
        "targets_financieros_disponibles": list(catalogo_targets.keys()),
        "periodos_reales_en_dataset": [y for y in ANIOS if any(_col_fin(df_features, p, y) for p in PREFIJOS_FIN)],
        "nota_prospectiva": "El dataset contiene valores reales 2022-2024. Las estimaciones 2025/2026 requieren desplazar temporalmente las features y no pueden evaluarse sin valores reales del año objetivo."
    }

    tabs = st.tabs([
        "📌 1. Marco y Ruta",
        "🔎 2. Diagnóstico",
        "🏷️ 3. Estrategia y Encoding",
        "🧬 4. Feature Engineering",
        "✂️ 5. Partición Train / Test",
        "🎯 6. Política de Outliers",
        "📐 7. Escalado y Transformaciones",
        "📊 8. Resultado",
        "🤖 9. Base para el LLM",
        "✅ 10. Validación y Entregables",
    ])

    with tabs[0]:
        _tab_marco()
    with tabs[1]:
        _tab_diagnostico(df, df_features, incidencias, fe_rep)
    with tabs[2]:
        _tab_encoding(plan_encoding, {
            "umbral_ohe": umbral_ohe,
            "drop_first": drop_first,
        })
    with tabs[3]:
        _tab_feature_engineering(df_features, fe_rep)
    with tabs[4]:
        _tab_particion(pipeline)
    with tabs[5]:
        _tab_outliers(pipeline)
    with tabs[6]:
        _tab_transformaciones(transformaciones, escalar)
    with tabs[7]:
        _tab_resultado(pipeline, fs_result, transformaciones, target)
    with tabs[8]:
        _tab_llm(df_llm, dicc)
    with tabs[9]:
        _tab_validacion(df_llm, dicc, pipeline, fs_result, transformaciones)
