"""
MÓDULO: new_modelado.py
ETAPA CRISP-DM: 4 — MODELADO DE LOS DATOS

Contrato de entrada
-------------------
Recibe exclusivamente los artefactos producidos por new_preparacion.py:
    X_train, X_test, y_train, y_test y target.

El módulo NO vuelve a calcular encoding, imputación, escalado, outliers ni
feature selection sobre todo el dataset. Esto protege contra data leakage.

Objetivo
--------
Entrenar y comparar modelos de regresión para indicadores financieros
originales y derivados autorizados por new_preparacion.py. El año del target
forma parte de la definición del experimento. Este módulo no acepta IDs,
ranking, dummies ni transformaciones técnicas como objetivos de negocio.

La predicción final será consumida posteriormente por Evaluación y Despliegue.
El LLM/API no sustituye al modelo numérico: Python calcula la predicción y la
API se utilizará posteriormente para interpretar, explicar y responder el
prompt del usuario.
"""
from __future__ import annotations

import time
import re
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    XGBOOST_ERROR = None
except Exception as exc:  # pragma: no cover
    XGBRegressor = None
    XGBOOST_AVAILABLE = False
    XGBOOST_ERROR = str(exc)

try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_ERROR = None
except Exception as exc:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False
    SHAP_ERROR = str(exc)
    

# --- Generación de informes PDF legibles para humanos (complemento del .joblib) ---
try:
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
    REPORTLAB_ERROR = None
except Exception as exc:  # pragma: no cover
    REPORTLAB_AVAILABLE = False
    REPORTLAB_ERROR = str(exc)


def generar_informe_pdf_candidato(
    ruta_pdf,
    objetivo,
    nombre_modelo,
    paquete,
):
    """Genera y valida un informe PDF independiente del modelo .joblib.

    El .joblib es el artefacto técnico (VS Code / Python / scikit-learn);
    este PDF es la versión legible para humanos de ese mismo candidato.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(f"reportlab no está disponible: {REPORTLAB_ERROR}")

    ruta_pdf = Path(ruta_pdf)
    ruta_pdf.parent.mkdir(parents=True, exist_ok=True)

    estilos = getSampleStyleSheet()
    estilos.add(
        ParagraphStyle(
            name="DetallePequeno",
            parent=estilos["BodyText"],
            fontSize=8,
            leading=11,
        )
    )

    def texto_seguro(valor):
        return escape(str(valor))

    def fmt_metrica(valor):
        try:
            return f"{float(valor):,.4f}"
        except (TypeError, ValueError):
            return str(valor)

    metricas_train = paquete.get("metricas_train", {}) or {}
    metricas_test = paquete.get("metricas_test", {}) or {}

    filas = [["Métrica", "TRAIN", "TEST"]]

    for metrica in ["MAE", "RMSE", "R²", "Varianza Explicada", "MAPE (%)", "MAE / Media (%)"]:
        filas.append([
            metrica,
            fmt_metrica(metricas_train.get(metrica, "No disponible")),
            fmt_metrica(metricas_test.get(metrica, "No disponible")),
        ])

    tabla = Table(
        filas,
        colWidths=[5 * cm, 4.5 * cm, 4.5 * cm],
        repeatRows=1,
    )
    tabla.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#174A6E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    variables = paquete.get("features", [])
    parametros = paquete.get("best_params", {})

    contenido = [
        Paragraph(
            "Informe del candidato para Despliegue",
            estilos["Title"],
        ),
        Spacer(1, 0.4 * cm),
        Paragraph(
            f"<b>Objetivo:</b> {texto_seguro(objetivo)}",
            estilos["BodyText"],
        ),
        Paragraph(
            f"<b>Modelo:</b> {texto_seguro(nombre_modelo)}",
            estilos["BodyText"],
        ),
        Paragraph(
            f"<b>Fecha:</b> {texto_seguro(paquete.get('fecha_guardado', 'No disponible'))}",
            estilos["BodyText"],
        ),
        Paragraph(
            f"<b>Metodología:</b> {texto_seguro(paquete.get('metodologia', 'CRISP-DM Fase 4 — Modelado'))}",
            estilos["BodyText"],
        ),
        Spacer(1, 0.4 * cm),
        Paragraph("Métricas registradas", estilos["Heading2"]),
        tabla,
        Spacer(1, 0.4 * cm),
        Paragraph("Variables de entrada", estilos["Heading2"]),
        Paragraph(
            texto_seguro(", ".join(map(str, variables))
                         or "No disponibles"),
            estilos["DetallePequeno"],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("Parámetros del modelo", estilos["Heading2"]),
        Paragraph(
            texto_seguro(parametros),
            estilos["DetallePequeno"],
        ),
        Spacer(1, 0.4 * cm),
        Paragraph("Limitaciones de interpretación", estilos["Heading2"]),
        Paragraph(
            "Este informe resume la información registrada del modelo. "
            "Las métricas históricas de TEST no garantizan por sí solas "
            "la precisión de predicciones para 2025 o 2026. Antes de "
            "utilizar el modelo operativamente, debe verificarse la "
            "compatibilidad de las variables de entrada y realizarse "
            "la validación temporal correspondiente.",
            estilos["BodyText"],
        ),
    ]

    documento = SimpleDocTemplate(
        str(ruta_pdf),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
    )
    documento.build(contenido)

    # No dar por válido el PDF hasta verificar su firma.
    with ruta_pdf.open("rb") as archivo:
        firma = archivo.read(5)

    if firma != b"%PDF-":
        raise ValueError(
            f"El informe generado no tiene una firma PDF válida: {ruta_pdf}"
        )

    if ruta_pdf.stat().st_size == 0:
        raise ValueError("El informe PDF está vacío.")

    return ruta_pdf



# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

RANDOM_STATE_DEFAULT = 42
CV_DEFAULT = 5
N_ITER_RANDOM_DEFAULT = 12
MODELOS_DISPONIBLES = [
    "Regresión Lineal",
    "Ridge",
    "Lasso",
    "Random Forest",
    "XGBoost",
    "MLP",
]

OBJETIVOS_2024 = [
    "INGRESOS OPERACIONALES 2024",
    "GANANCIA (PERDIDA) 2024",
    "TOTAL ACTIVOS 2024",
    "TOTAL PASIVOS 2024",
    "TOTAL PATRIMONIO 2024",
]

OBJETIVO_DESCRIPCION = {
    "INGRESOS OPERACIONALES 2024": "Ingresos generados por la operación de la empresa.",
    "GANANCIA (PERDIDA) 2024": "Resultado neto: utilidad o pérdida del ejercicio.",
    "TOTAL ACTIVOS 2024": "Recursos económicos controlados por la empresa.",
    "TOTAL PASIVOS 2024": "Obligaciones y deudas de la empresa.",
    "TOTAL PATRIMONIO 2024": "Valor residual de los activos después de los pasivos.",
}

def _catalogo_targets_session() -> dict[str, str]:
    """Catálogo autorizado generado por Preparación; fallback para sesiones antiguas."""
    cat = st.session_state.get("catalogo_targets_financieros")
    if isinstance(cat, dict) and cat:
        return {str(k): str(v) for k, v in cat.items()}
    return {k: OBJETIVO_DESCRIPCION.get(k, "Indicador financiero") for k in OBJETIVOS_2024}

MODELO_EXPLICACION = {
    "Regresión Lineal": "Línea base: busca una relación aproximadamente lineal entre las variables y el objetivo.",
    "Ridge": "Regresión lineal con regularización L2; estabiliza coeficientes cuando existen relaciones fuertes entre variables.",
    "Lasso": "Regresión lineal con regularización L1; puede llevar algunos coeficientes a cero y actuar como selección implícita.",
    "Random Forest": "Muchos árboles independientes cuyas predicciones se promedian; captura relaciones no lineales e interacciones.",
    "XGBoost": "Árboles construidos secuencialmente para corregir errores anteriores; candidato principal para datos tabulares.",
    "MLP": "Red neuronal pequeña que aprende relaciones no lineales mediante capas de neuronas.",
}


# ═══════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════

def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [str(c).replace("*", "").strip() for c in d.columns]
    return d


def _encontrar_columna(df: pd.DataFrame, nombre: str) -> str | None:
    objetivo = nombre.replace("*", "").strip().upper()
    for c in df.columns:
        if str(c).replace("*", "").strip().upper() == objetivo:
            return c
    return None


def _cop(df: pd.DataFrame) -> pd.DataFrame:
    """Copia numérica profunda, sin vistas ni buffers de solo lectura.

    Algunas combinaciones de pandas/NumPy/scikit-learn entregan arrays con
    WRITEABLE=False (por ejemplo, tras transformaciones o cachés). Se crea un
    buffer NumPy C-contiguo y escribible por columna para evitar que el fallo
    se propague a los estimadores.
    """
    d = df.copy(deep=True)
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    # Reconstruir el DataFrame a partir de un array nuevo, contiguo y escribible.
    arr = np.array(d.to_numpy(dtype=np.float64, copy=True), dtype=np.float64, order="C", copy=True)
    # El array se crea con ownership propio; no se intenta reactivar el flag de
    # un buffer externo o compartido.
    return pd.DataFrame(arr, columns=list(d.columns), index=d.index.copy(), copy=True)


def _matriz_sklearn(df: pd.DataFrame) -> np.ndarray:
    """Entrega una matriz float C-contigua y con escritura habilitada."""
    return np.array(df.to_numpy(dtype=np.float64, copy=True), dtype=np.float64, order="C", copy=True)


def _metricas(y_true, y_pred) -> dict[str, float]:
    """Métricas robustas sin depender de buffers de solo lectura.

    MAPE se calcula solo para valores reales alejados de cero y se acompaña
    de cobertura; evita cifras engañosas cuando el target tiene ceros/casi ceros.
    """
    yt = np.array(y_true, dtype=np.float64, order="C", copy=True).reshape(-1)
    yp = np.array(y_pred, dtype=np.float64, order="C", copy=True).reshape(-1)
    if yt.size != yp.size:
        raise ValueError(f"Longitudes incompatibles: y_true={yt.size}, y_pred={yp.size}.")
    if yt.size == 0:
        raise ValueError("No se pueden calcular métricas con un conjunto vacío.")
    if not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Las métricas recibieron NaN o infinitos.")
    errores = yt - yp
    mae = float(np.mean(np.abs(errores)))
    rmse = float(np.sqrt(np.mean(np.square(errores))))
    media_y = float(np.mean(yt))
    var_y = float(np.var(yt))
    var_error = float(np.var(errores))
    ss_res = float(np.sum(np.square(errores)))
    ss_tot = float(np.sum(np.square(yt - media_y)))
    eps = np.finfo(np.float64).eps
    r2 = (1.0 if ss_res <= eps else 0.0) if ss_tot <= eps else float(1.0 - ss_res / ss_tot)
    evs = (1.0 if var_error <= eps else 0.0) if var_y <= eps else float(1.0 - var_error / var_y)
    # Excluir denominadores diminutos para que MAPE no explote artificialmente.
    umbral = max(float(np.nanpercentile(np.abs(yt), 10)) * 0.1, float(np.median(np.abs(yt))) * 0.01, 1e-12)
    mask_mape = np.abs(yt) > umbral
    if mask_mape.any():
        mape = float(np.mean(np.abs(errores[mask_mape]) / np.abs(yt[mask_mape])) * 100.0)
    else:
        mape = np.nan
    media_abs_y = float(np.mean(np.abs(yt)))
    return {
        "MAE": mae, "RMSE": rmse, "Varianza Explicada": evs, "R²": r2,
        "MAPE (%)": mape, "MAPE cobertura (%)": float(mask_mape.mean() * 100.0),
        "Umbral MAPE": float(umbral), "Media |y|": media_abs_y,
        "MAE / Media (%)": float(mae / media_abs_y * 100.0) if media_abs_y > 0 else np.nan,
    }


def _diagnostico_target(y, target: str) -> dict[str, Any]:
    """Diagnóstico de la variable objetivo para los 12 targets derivados y originales."""
    arr = np.array(pd.to_numeric(pd.Series(y), errors="coerce"), dtype=float, copy=True)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "alertas": ["El objetivo no contiene valores numéricos finitos."], "apto_modelado": False}
    q01, q25, q50, q75, q99 = np.percentile(arr, [1, 25, 50, 75, 99])
    abs_arr = np.abs(arr)
    med_abs = float(np.median(abs_arr))
    zero_pct = float(np.mean(np.isclose(arr, 0.0)) * 100)
    neg_pct = float(np.mean(arr < 0) * 100)
    alerts = []
    name = str(target).upper()
    if zero_pct >= 5:
        alerts.append(f"{zero_pct:.1f}% de los valores del objetivo son cero; MAPE puede no ser apropiado.")
    if neg_pct > 0 and any(t in name for t in ("CRECIMIENTO", "ROA", "ROE", "GANANCIA", "MARGEN")):
        alerts.append(f"{neg_pct:.1f}% de valores son negativos; revisar interpretación financiera y métricas porcentuales.")
    if "SOLVENCIA" in name or "APALANCAMIENTO" in name:
        alerts.append("Razón patrimonial: verificar patrimonio <= 0 y denominadores muy pequeños antes de interpretar.")
    if "CRECIMIENTO" in name:
        alerts.append("Crecimiento: revisar denominadores del año anterior cercanos a cero y cambios de signo.")
    if med_abs > 0 and float(q99 - q01) > 1000 * med_abs:
        alerts.append("Distribución muy dispersa: la amplitud P99-P1 supera 1.000 veces la mediana absoluta; inspeccionar extremos.")
    if arr.size < 100:
        alerts.append("Menos de 100 observaciones válidas: las métricas pueden ser inestables.")
    return {
        "n": int(arr.size), "porcentaje_ceros": zero_pct, "porcentaje_negativos": neg_pct,
        "min": float(np.min(arr)), "p01": float(q01), "p25": float(q25), "mediana": float(q50),
        "p75": float(q75), "p99": float(q99), "max": float(np.max(arr)),
        "alertas": alerts, "apto_modelado": bool(arr.size >= 30),
    }


def _calificacion(r2: float, evs: float, mejora_mae_baseline: float | None = None,
                  mae_media_pct: float | None = None, target_alertas: list[str] | None = None) -> str:
    """Etiqueta conservadora: nunca confunde terminar el fit con ser fiable."""
    mejora_ok = mejora_mae_baseline is not None and np.isfinite(mejora_mae_baseline) and mejora_mae_baseline >= 10.0
    error_rel_ok = mae_media_pct is not None and np.isfinite(mae_media_pct) and mae_media_pct <= 50.0
    if mejora_mae_baseline is None or not np.isfinite(mejora_mae_baseline) or mejora_mae_baseline <= 0:
        return "⚠️ No supera baseline"
    if r2 >= 0.50 and evs >= 0.50 and mejora_ok and error_rel_ok:
        return "💎 Buen desempeño en TEST"
    if r2 >= 0.20 and mejora_mae_baseline > 0:
        return "✅ Desempeño prometedor; revisar alertas"
    return "⚠️ Evidencia predictiva débil"


def _modelo_y_busqueda(nombre: str, random_state: int, cv: int, n_iter: int):
    """Devuelve estimador, tipo de búsqueda y espacio de hiperparámetros."""
    if nombre == "Regresión Lineal":
        return LinearRegression(), None, None

    if nombre == "Ridge":
        return (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=random_state))]),
            "grid",
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        )

    if nombre == "Lasso":
        return (
            Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=10_000, random_state=random_state))]),
            "grid",
            {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]},
        )

    if nombre == "Random Forest":
        return (
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            "grid",
            {
                "n_estimators": [200, 400, 600],
                "max_depth": [None, 8, 16],
                "min_samples_split": [2, 10, 20],
                "max_features": ["sqrt", 0.5],
            },
        )

    if nombre == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError(f"XGBoost no está disponible: {XGBOOST_ERROR}")
        return (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist",
            ),
            "random",
            {
                "n_estimators": [300, 500, 800],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 6, 9],
                "subsample": [0.7, 0.9],
                "colsample_bytree": [0.7, 0.9],
                "reg_lambda": [1, 5],
            },
        )

    if nombre == "MLP":
        return (
            Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(random_state=random_state, early_stopping=True, max_iter=500))]),
            "random",
            {
                "model__hidden_layer_sizes": [(64,), (128, 64)],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [0.0001, 0.001],
                "model__learning_rate_init": [0.001, 0.01],
            },
        )
    raise ValueError(f"Modelo no soportado: {nombre}")


def _preparar_entrada_desde_session() -> dict[str, Any] | None:
    """Obtiene el contrato generado por new_preparacion.py."""
    keys = ["X_train", "X_test", "y_train", "y_test", "target_modelado"]
    if not all(k in st.session_state for k in keys):
        return None
    return {
        "X_train": st.session_state["X_train"],
        "X_test": st.session_state["X_test"],
        "y_train": st.session_state["y_train"],
        "y_test": st.session_state["y_test"],
        "target": st.session_state["target_modelado"],
    }


def calcular_estimacion_prospectiva_2025(nombre_modelo: str) -> pd.DataFrame:
    """Predice el target 2025 desplazando +1 año las features históricas.

    Requiere un modelo entrenado con target 2024 y los artefactos de preparación
    almacenados por new_preparacion.py. Es una estimación no verificada: el
    dataset fuente llega hasta 2024 y no contiene el valor real de 2025.
    """
    resultados = st.session_state.get("modelado_resultados", {})
    if nombre_modelo not in resultados:
        raise ValueError("Seleccione un modelo ya entrenado en esta sesión.")
    target = str(st.session_state.get("target_modelado", ""))
    years = re.findall(r"20(?:22|23|24|25|26)", target)
    if not years or int(years[-1]) != 2024:
        raise ValueError("La estimación 2025 requiere un modelo entrenado con un target histórico de 2024.")
    df_features = st.session_state.get("df_features_financieras")
    pipeline = st.session_state.get("prep_pipeline")
    outlier_result = st.session_state.get("outlier_reporte")
    fs_result = st.session_state.get("prep_feature_selection")
    transformaciones = st.session_state.get("prep_transformaciones")
    if any(v is None for v in (df_features, pipeline, outlier_result, fs_result, transformaciones)):
        raise ValueError("Faltan artefactos de Preparación. Vuelva a ejecutar new_preparacion.py.")
    from new_preparacion import preparar_matriz_prospectiva
    X_future = preparar_matriz_prospectiva(
        df_features, pipeline, outlier_result, fs_result, transformaciones, desplazamiento_anios=1
    )
    estimator = resultados[nombre_modelo]["estimador"]
    pred = np.asarray(estimator.predict(X_future), dtype=float)
    target_2025 = re.sub(r"2024", "2025", target)
    cols_id = [c for c in ["NIT", "RAZON SOCIAL", "MACROSECTOR", "REGION", "DEPARTAMENTO DOMICILIO", "CIUDAD DOMICILIO"] if c in df_features.columns]
    out = df_features[cols_id].copy() if cols_id else pd.DataFrame(index=df_features.index)
    out[f"ESTIMACION_NO_VERIFICADA_{target_2025}"] = pred
    out["ANIO_OBJETIVO"] = 2025
    out["ESTADO"] = "Estimación prospectiva; valor real 2025 no disponible para verificación"
    st.session_state["predicciones_prospectivas_2025"] = out
    st.session_state["prediccion_prospectiva_metadata"] = {
        "target_historico": target, "target_prospectivo": target_2025,
        "modelo": nombre_modelo, "horizonte": 1,
        "estado": "no_verificada", "n_empresas": len(out),
        "limitacion": "La estimación usa un modelo entrenado para 2024 aplicado a features desplazadas de 2023-2024. No equivale a validación con valores reales 2025."
    }
    return out


def _validar_entrada(entrada: dict[str, Any]) -> tuple[bool, str]:
    if entrada is None:
        return False, "No existe una preparación disponible. Ejecute primero new_preparacion.py."
    for k in ["X_train", "X_test", "y_train", "y_test"]:
        if entrada.get(k) is None:
            return False, f"Falta el artefacto {k}."
    Xtr, Xte = entrada["X_train"], entrada["X_test"]
    if not isinstance(Xtr, pd.DataFrame) or not isinstance(Xte, pd.DataFrame):
        return False, "X_train y X_test deben ser DataFrames."
    if Xtr.empty or Xte.empty:
        return False, "TRAIN o TEST están vacíos."
    if list(Xtr.columns) != list(Xte.columns):
        return False, "TRAIN y TEST no tienen exactamente las mismas columnas."
    if len(entrada["y_train"]) != len(Xtr) or len(entrada["y_test"]) != len(Xte):
        return False, "Las longitudes de X e y no coinciden."
    target = str(entrada["target"])
    years = re.findall(r"20(?:22|23|24|25|26)", target)
    if not years or int(years[-1]) != 2024:
        return False, "El experimento vigente solo admite un objetivo financiero de 2024. Vuelva a ejecutar Preparación."
    upper = target.upper()
    prohibidas = ("NIT", "RAZON SOCIAL", "RANKING", "_LOG", "LOG_", "_FREQ", "FLAG_", "ONEHOT", "ONE-HOT")
    if any(x in upper for x in prohibidas):
        return False, "El target es un identificador, ranking o variable técnica y no es un objetivo financiero de negocio."
    catalogo = _catalogo_targets_session()
    if target not in catalogo:
        return False, "El target no pertenece al catálogo financiero autorizado por new_preparacion.py. Vuelva a ejecutar Preparación."
    return True, "OK"


def _estado_modelo(nombre: str) -> str:
    if nombre == "XGBoost" and not XGBOOST_AVAILABLE:
        return "No disponible"
    return "Disponible"


# ═══════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════

def entrenar_modelo(
    nombre: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = RANDOM_STATE_DEFAULT,
    cv: int = CV_DEFAULT,
    optimizar: bool = True,
    n_iter: int = N_ITER_RANDOM_DEFAULT,
) -> dict[str, Any]:
    Xtr = _cop(X_train)
    Xte = _cop(X_test).reindex(columns=Xtr.columns)
    ytr = pd.to_numeric(y_train, errors="coerce")
    yte = pd.to_numeric(y_test, errors="coerce")

    # La imputación ya pertenece a Preparación. Se verifica que no queden nulos.
    if Xtr.isna().any().any() or Xte.isna().any().any():
        raise ValueError("X_train/X_test contienen nulos. Revise la imputación en new_preparacion.py.")
    if ytr.isna().any() or yte.isna().any():
        raise ValueError("El target contiene nulos. Revise la preparación del target.")

    # Garantía explícita de arrays escribibles antes de entrar en sklearn.
    # El modelo no requiere nombres de columnas para entrenar; se conservan
    # aparte en Xtr/Xte para reportes e interpretabilidad.
    Xtr_fit = _matriz_sklearn(Xtr)
    Xte_fit = _matriz_sklearn(Xte)
    ytr_fit = np.array(ytr.to_numpy(dtype=np.float64, copy=True), dtype=np.float64, order="C", copy=True)
    yte_fit = np.array(yte.to_numpy(dtype=np.float64, copy=True), dtype=np.float64, order="C", copy=True)

    if len(Xtr) < 10:
        raise ValueError("Se requieren al menos 10 observaciones TRAIN para entrenar y evaluar con estabilidad mínima.")
    cv_efectivo = min(max(2, int(cv)), len(Xtr))
    estimador, tipo_busqueda, espacio = _modelo_y_busqueda(nombre, random_state, cv_efectivo, n_iter)
    inicio = time.perf_counter()
    mejor = estimador
    best_params = {}
    best_cv_score = np.nan
    busqueda = "Sin optimización"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if optimizar and tipo_busqueda == "grid":
            search = GridSearchCV(
                estimator=estimador,
                param_grid=espacio,
                scoring="neg_mean_absolute_error",
                cv=cv_efectivo,
                n_jobs=-1,
                refit=True,
            )
            search.fit(Xtr_fit, ytr_fit)
            mejor = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = float(search.best_score_)
            busqueda = f"GridSearchCV · {cv_efectivo}-fold · criterio MAE"
        elif optimizar and tipo_busqueda == "random":
            search = RandomizedSearchCV(
                estimator=estimador,
                param_distributions=espacio,
                n_iter=min(n_iter, 12),
                scoring="neg_mean_absolute_error",
                cv=cv_efectivo,
                random_state=random_state,
                n_jobs=-1,
                refit=True,
            )
            search.fit(Xtr_fit, ytr_fit)
            mejor = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = float(search.best_score_)
            busqueda = f"RandomizedSearchCV · {cv_efectivo}-fold · criterio MAE · {min(n_iter, 12)} iteraciones"
        else:
            mejor.fit(Xtr_fit, ytr_fit)

    ypred_train = mejor.predict(Xtr_fit)
    ypred_test = mejor.predict(Xte_fit)
    m_train = _metricas(ytr, ypred_train)
    m_test = _metricas(yte, ypred_test)
    # Referencia ingenua: predecir en TEST la mediana observada en TRAIN.
    # Un modelo complejo no debe considerarse útil si no mejora esta base.
    baseline_valor = float(np.median(ytr_fit))
    baseline_pred = np.full(shape=len(yte_fit), fill_value=baseline_valor, dtype=np.float64)
    baseline_metricas = _metricas(yte_fit, baseline_pred)
    mejora_mae_baseline = (
        float((baseline_metricas["MAE"] - m_test["MAE"]) / baseline_metricas["MAE"] * 100)
        if baseline_metricas["MAE"] > 0 else np.nan
    )
    duracion = time.perf_counter() - inicio
    diagnostico_target = _diagnostico_target(ytr, str(st.session_state.get("target_modelado", "")))

    return {
        "modelo": nombre,
        "estimador": mejor,
        "y_pred_train": np.asarray(ypred_train),
        "y_pred_test": np.asarray(ypred_test),
        "metricas_train": m_train,
        "metricas_test": m_test,
        "metricas_baseline_test": baseline_metricas,
        "baseline_valor_mediana_train": baseline_valor,
        "mejora_mae_vs_baseline_pct": mejora_mae_baseline,
        "best_params": best_params,
        "best_cv_neg_mae": best_cv_score,
        "busqueda": busqueda,
        "duracion_s": float(duracion),
        "n_features": int(Xtr.shape[1]),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "diagnostico_target": diagnostico_target,
    }


def _fila_comparacion(resultado: dict[str, Any]) -> dict[str, Any]:
    mt = resultado["metricas_train"]
    mv = resultado["metricas_test"]
    return {
        "Modelo": resultado["modelo"],
        "R² TRAIN": mt["R²"],
        "R² TEST": mv["R²"],
        "Brecha R²": mt["R²"] - mv["R²"],
        "MAE TEST": mv["MAE"],
        "RMSE TEST": mv["RMSE"],
        "MAPE TEST (%)": mv["MAPE (%)"],
        "Cobertura MAPE (%)": mv.get("MAPE cobertura (%)", np.nan),
        "Varianza Explicada TEST": mv["Varianza Explicada"],
        "MAE / Media TEST (%)": mv["MAE / Media (%)"],
        "MAE baseline (mediana TRAIN)": resultado.get("metricas_baseline_test", {}).get("MAE", np.nan),
        "Mejora MAE vs baseline (%)": resultado.get("mejora_mae_vs_baseline_pct", np.nan),
        "Score CV (−MAE)": resultado["best_cv_neg_mae"],
        "Calificación": _calificacion(
            mv["R²"], mv["Varianza Explicada"],
            resultado.get("mejora_mae_vs_baseline_pct", np.nan),
            mv.get("MAE / Media (%)", np.nan),
            resultado.get("diagnostico_target", {}).get("alertas", []),
        ),
        "Alertas objetivo": " | ".join(resultado.get("diagnostico_target", {}).get("alertas", [])) or "Sin alertas automáticas",
        "Tiempo (s)": resultado["duracion_s"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# SHAP Y EXPLICABILIDAD
# ═══════════════════════════════════════════════════════════════════════════

def _obtener_importancias(resultado: dict[str, Any], X_test: pd.DataFrame) -> pd.DataFrame:
    modelo = resultado["estimador"]
    if hasattr(modelo, "feature_importances_"):
        vals = modelo.feature_importances_
    elif hasattr(modelo, "named_steps") and hasattr(modelo.named_steps.get("model"), "coef_"):
        vals = np.abs(modelo.named_steps["model"].coef_)
    elif hasattr(modelo, "coef_"):
        vals = np.abs(modelo.coef_)
    else:
        return pd.DataFrame(columns=["Variable", "Importancia"])
    return (
        pd.DataFrame({"Variable": X_test.columns, "Importancia": np.asarray(vals).ravel()})
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )


def _explicacion_shap(resultado: dict[str, Any], X_test: pd.DataFrame, max_muestra: int = 250):
    if not SHAP_AVAILABLE:
        return None, f"SHAP no está disponible: {SHAP_ERROR}"
    modelo = resultado["estimador"]
    X = X_test.head(max_muestra).copy()
    try:
        if hasattr(modelo, "feature_importances_"):
            explainer = shap.TreeExplainer(modelo)
            values = explainer.shap_values(X)
        else:
            # Para modelos lineales/pipelines se utiliza el explainer genérico.
            background = X.sample(min(50, len(X)), random_state=42)
            explainer = shap.Explainer(modelo.predict, background)
            values = explainer(X)
            values = values.values
        return np.asarray(values), None
    except Exception as exc:
        return None, str(exc)


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑAS STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════

def _tab_marco(entrada):
    st.subheader("📌 1. Marco y objetivos")
    st.markdown("### Fase 4 de CRISP-DM: convertir datos preparados en un modelo que pueda aprender")
    st.write(
        "En esta etapa la computadora recibe ejemplos del pasado y aprende una relación entre "
        "las variables financieras y un resultado que queremos estimar. La meta no es que la "
        "computadora memorice empresas: es que aprenda un patrón que pueda usar con datos que "
        "no vio durante el entrenamiento."
    )
    st.info(
        "🧒 **Como para un niño:** imagina que queremos enseñar a una máquina a adivinar cuánto "
        "puede valer el patrimonio de una empresa. Le mostramos muchos ejemplos del pasado. "
        "La máquina busca pistas —ingresos, activos, pasivos, ratios y características del sector— "
        "y aprende a combinarlas para hacer una estimación."
    )

    st.markdown("### ¿Qué debe entregar esta fase?")
    pasos = pd.DataFrame([
        ["1", "Elegir target", "Definir qué cifra financiera queremos predecir."],
        ["2", "Elegir algoritmos", "Probar modelos sencillos y modelos capaces de aprender relaciones no lineales."],
        ["3", "Entrenar", "El modelo aprende usando únicamente TRAIN."],
        ["4", "Optimizar", "Buscar hiperparámetros que mejoren el aprendizaje mediante validación cruzada."],
        ["5", "Comparar", "Medir errores y detectar sobreajuste."],
        ["6", "Explicar", "Mostrar qué variables contribuyeron a las predicciones."],
        ["7", "Entregar", "Guardar el mejor modelo y sus metadatos para Evaluación/Despliegue."],
    ], columns=["Paso", "Actividad", "Propósito"])
    st.dataframe(pasos, use_container_width=True, hide_index=True)

    if entrada:
        st.success(f"Entrada recibida desde Preparación: target = **{entrada['target']}**")
    else:
        st.warning("Aún no se ha recibido el contrato de datos de new_preparacion.py.")


def _tab_objetivos(entrada):
    st.subheader("🎯 2. Objetivos predictivos")
    st.write(
        "Cada ejecución entrena un objetivo financiero definido en Preparación: puede ser una cifra original o un indicador derivado con fórmula explícita. "
        "Los identificadores, rankings, variables de encoding y transformaciones técnicas no son objetivos de negocio."
    )
    catalogo = _catalogo_targets_session()
    tabla = pd.DataFrame([
        {"Objetivo autorizado": t, "Definición": desc, "Tipo": "Regresión supervisada"}
        for t, desc in catalogo.items()
    ])
    st.dataframe(tabla, use_container_width=True, hide_index=True)
    st.info(
        "🧒 **Target = la respuesta que queremos aprender.** Es como una pregunta de examen. "
        "Las demás columnas son las pistas. Si preguntamos por los activos de 2024, no podemos "
        "darle al modelo los activos de 2024 como pista: eso sería mirar la respuesta antes del examen."
    )
    if entrada:
        valido, mensaje = _validar_entrada(entrada)
        if valido:
            desc = _catalogo_targets_session().get(str(entrada["target"]), "Indicador financiero")
            st.success(f"Target autorizado: **{entrada['target']}** — {desc}")
        else:
            st.error(mensaje)
        anio_match = re.findall(r"20(?:22|23|24|25|26)", str(entrada["target"]))
        anio_target = int(anio_match[-1]) if anio_match else None
        if anio_target == 2024:
            st.info("La prueba con target 2024 usa variables disponibles antes de 2024 y evalúa empresas reservadas en TEST. Es una validación histórica entre empresas; no reemplaza una validación temporal con años adicionales.")
        elif anio_target and anio_target >= 2025:
            st.warning("El dataset fuente llega hasta 2024. No se puede entrenar ni verificar un target 2025/2026 real si esos valores no están en el dataset. Para inferencia prospectiva debe aplicarse el desplazamiento temporal de features y etiquetar la salida como estimación no verificada.")


def _tab_entrada(entrada):
    st.subheader("📥 3. Datos de entrada al Modelado")
    if not entrada:
        st.warning("Ejecute primero la Preparación de Datos.")
        return
    ok, msg = _validar_entrada(entrada)
    if not ok:
        st.error(msg)
        return
    Xtr, Xte = entrada["X_train"], entrada["X_test"]
    ytr, yte = entrada["y_train"], entrada["y_test"]
    a, b, c, d = st.columns(4)
    a.metric("Target", str(entrada["target"]))
    b.metric("Empresas TRAIN", f"{len(Xtr):,}")
    c.metric("Empresas TEST", f"{len(Xte):,}")
    d.metric("Features", f"{Xtr.shape[1]:,}")
    st.markdown("### Contrato recibido")
    st.dataframe(pd.DataFrame([
        ["X_train", *Xtr.shape, "Predictores para aprender"],
        ["X_test", *Xte.shape, "Predictores reservados para prueba final"],
        ["y_train", len(ytr), 1, "Respuesta conocida para aprender"],
        ["y_test", len(yte), 1, "Respuesta reservada para medir"],
    ], columns=["Objeto", "Filas", "Columnas", "Uso"]), use_container_width=True, hide_index=True)
    st.caption("TEST debe permanecer apartado hasta la evaluación final del modelo.")


def _tab_configuracion():
    st.subheader("⚙️ 4. Configuración experimental")
    c1, c2, c3 = st.columns(3)
    with c1:
        modelos = st.multiselect("Modelos a entrenar", MODELOS_DISPONIBLES, default=MODELOS_DISPONIBLES)
    with c2:
        cv = st.number_input("Pliegues de validación cruzada", 3, 10, CV_DEFAULT, 1)
        random_state = st.number_input("Random state", 0, 9999, RANDOM_STATE_DEFAULT, 1)
    with c3:
        optimizar = st.checkbox("Optimizar hiperparámetros", True)
        n_iter = st.slider("Iteraciones RandomizedSearch", 4, 30, N_ITER_RANDOM_DEFAULT)
    st.session_state["modelado_config"] = {
        "modelos": modelos,
        "cv": int(cv),
        "random_state": int(random_state),
        "optimizar": bool(optimizar),
        "n_iter": int(n_iter),
    }
    if "XGBoost" in modelos and not XGBOOST_AVAILABLE:
        st.warning(f"XGBoost fue seleccionado pero no está disponible. {XGBOOST_ERROR}")
    st.markdown("### ¿Qué es un hiperparámetro?")
    st.write(
        "Es una decisión que configuramos antes o durante la búsqueda del entrenamiento: por ejemplo, "
        "cuántos árboles tendrá un Random Forest o qué tan fuerte será la regularización. La búsqueda "
        "automática prueba varias opciones y conserva la que obtiene mejor R² en validación cruzada."
    )


def _tab_entrenamiento(entrada):
    st.subheader("🤖 5. Entrenamiento de modelos")
    if not entrada:
        st.warning("Ejecute primero la Preparación de Datos.")
        return
    cfg = st.session_state.get("modelado_config", {})
    modelos = cfg.get("modelos", MODELOS_DISPONIBLES)
    if not modelos:
        st.info("Seleccione al menos un modelo en Configuración experimental.")
        return
    if st.button("🚀 Entrenar modelos seleccionados", type="primary", key="btn_entrenar_modelado"):
        resultados = {}
        progreso = st.progress(0)
        estado = st.empty()
        for i, nombre in enumerate(modelos, start=1):
            if nombre == "XGBoost" and not XGBOOST_AVAILABLE:
                st.warning("Se omite XGBoost porque no está instalado.")
                continue
            try:
                estado.write(f"Entrenando **{nombre}**…")
                resultados[nombre] = entrenar_modelo(
                    nombre,
                    entrada["X_train"], entrada["X_test"],
                    entrada["y_train"], entrada["y_test"],
                    random_state=cfg.get("random_state", RANDOM_STATE_DEFAULT),
                    cv=cfg.get("cv", CV_DEFAULT),
                    optimizar=cfg.get("optimizar", True),
                    n_iter=cfg.get("n_iter", N_ITER_RANDOM_DEFAULT),
                )
            except Exception as exc:
                detalle = traceback.format_exc()
                st.error(f"{nombre}: {exc}")
                with st.expander(f"Detalle técnico del error — {nombre}"):
                    st.code(detalle, language="text")
            progreso.progress(i / max(len(modelos), 1))
        estado.success("Entrenamiento terminado.")
        st.session_state["modelado_resultados"] = resultados
        st.session_state["modelado_target"] = entrada["target"]

    resultados = st.session_state.get("modelado_resultados", {})
    if resultados:
        st.success(f"Modelos disponibles: {len(resultados)}")
        tabla_resultados = pd.DataFrame([_fila_comparacion(r) for r in resultados.values()])
        st.dataframe(tabla_resultados, use_container_width=True, hide_index=True)
        primer_resultado = next(iter(resultados.values()))
        diag = primer_resultado.get("diagnostico_target", {})
        if diag:
            with st.expander("🔎 Diagnóstico de calidad del objetivo y limitaciones"):
                st.write({k: v for k, v in diag.items() if k != "alertas"})
                if diag.get("alertas"):
                    for alerta in diag["alertas"]:
                        st.warning(alerta)
                else:
                    st.success("No se detectaron alertas automáticas en la distribución del objetivo. Esto no garantiza por sí solo un modelo confiable.")
                st.caption("La selección final debe considerar el baseline, MAE/RMSE, R², estabilidad y revisión financiera; ninguna etiqueta automática sustituye la evaluación de negocio.")


def _tab_optimizacion():
    st.subheader("🔍 6. Optimización y validación cruzada")
    resultados = st.session_state.get("modelado_resultados", {})
    if not resultados:
        st.info("Entrene primero los modelos.")
        return
    filas = []
    for r in resultados.values():
        filas.append({
            "Modelo": r["modelo"],
            "Método": r["busqueda"],
            "Score CV (−MAE)": r["best_cv_neg_mae"],
            "Hiperparámetros": str(r["best_params"]),
            "Tiempo (s)": r["duracion_s"],
        })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
    st.info(
        "🧒 **Validación cruzada:** en vez de confiar en una sola división de los datos de entrenamiento, "
        "se hacen varias pruebas internas. Es como estudiar con cinco exámenes diferentes y escoger la "
        "configuración que funciona bien de forma consistente. TEST sigue guardado para el examen final."
    )


def _tab_comparacion(entrada):
    st.subheader("📊 7. Comparación de desempeño")
    resultados = st.session_state.get("modelado_resultados", {})
    if not resultados or not entrada:
        st.info("Entrene al menos un modelo para visualizar la comparación.")
        return
    comp = pd.DataFrame([_fila_comparacion(r) for r in resultados.values()])
    st.dataframe(comp.sort_values("R² TEST", ascending=False), use_container_width=True, hide_index=True)

    fig = px.bar(
        comp.sort_values("R² TEST", ascending=True),
        x="R² TEST", y="Modelo", orientation="h",
        title="R² sobre TEST — mayor es mejor",
        text="R² TEST",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    mejor_nombre = comp.sort_values(["Mejora MAE vs baseline (%)", "R² TEST", "RMSE TEST"], ascending=[False, False, True]).iloc[0]["Modelo"]
    mejor = resultados[mejor_nombre]
    y = np.asarray(entrada["y_test"])
    yp = mejor["y_pred_test"]
    fig2 = px.scatter(
        x=y, y=yp,
        labels={"x": "Real", "y": "Predicho"},
        title=f"Real vs. Predicho — {mejor_nombre}",
    )
    mn, mx = float(np.nanmin(y)), float(np.nanmax(y))
    fig2.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Predicción perfecta"))
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

    residuos = y - yp
    fig3 = px.scatter(
        x=yp, y=residuos,
        labels={"x": "Predicción", "y": "Residuo (real - predicho)"},
        title=f"Residuos — {mejor_nombre}",
    )
    fig3.add_hline(y=0, line_dash="dash")
    fig3.update_layout(height=450)
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "La elección del ganador no debe hacerse solo mirando R². También se revisan MAE, RMSE, "
        "MAPE, brecha TRAIN/TEST y la estabilidad observada en validación cruzada. La Fase 5 de CRISP-DM "
        "hará la evaluación formal de negocio."
    )


def _tab_explicabilidad(entrada):
    st.subheader("🔬 8. Explicabilidad: importancias y SHAP")
    resultados = st.session_state.get("modelado_resultados", {})
    if not resultados or not entrada:
        st.info("Entrene primero los modelos.")
        return
    nombre = st.selectbox("Modelo que desea explicar", list(resultados.keys()), key="modelado_explicar_modelo")
    r = resultados[nombre]
    Xte = _cop(entrada["X_test"])

    imp = _obtener_importancias(r, Xte)
    if not imp.empty:
        fig = px.bar(imp.head(20).sort_values("Importancia"), x="Importancia", y="Variable", orientation="h", title="Principales variables por importancia")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(imp.head(30), use_container_width=True, hide_index=True)

    if st.button("🧠 Calcular explicación SHAP", key="btn_shap"):
        values, error = _explicacion_shap(r, Xte)
        if error:
            st.warning(f"No fue posible calcular SHAP para este modelo: {error}")
        else:
            mean_abs = np.mean(np.abs(values), axis=0)
            shap_df = pd.DataFrame({"Variable": Xte.columns, "|SHAP| medio": mean_abs}).sort_values("|SHAP| medio", ascending=False).head(20)
            fig = px.bar(shap_df.sort_values("|SHAP| medio"), x="|SHAP| medio", y="Variable", orientation="h", title="SHAP global: contribución media absoluta")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.info(
                "🧒 **SHAP es como repartir el mérito de una predicción entre los jugadores de un equipo.** "
                "Muestra cuánto empujó cada variable la respuesta del modelo. Así la IA no queda como una caja negra."
            )


def _tab_robustez(entrada):
    st.subheader("🛡️ 9. Robustez, Stress Test y entregables")
    resultados = st.session_state.get("modelado_resultados", {})
    if not resultados or not entrada:
        st.info("Entrene primero los modelos.")
        return
    comp = pd.DataFrame([_fila_comparacion(r) for r in resultados.values()])
    comp["Brecha ≤ 0,10"] = comp["Brecha R²"] <= 0.10
    comp["R² ≥ 0,50"] = comp["R² TEST"] >= 0.50
    comp["MAPE ≤ 25 %"] = comp["MAPE TEST (%)"] <= 25
    comp["Supera baseline"] = comp["Mejora MAE vs baseline (%)"] > 0
    comp["Listo para producción"] = comp[["Brecha ≤ 0,10", "R² ≥ 0,50", "Supera baseline"]].all(axis=1)
    st.dataframe(comp[["Modelo", "R² TEST", "Brecha R²", "MAPE TEST (%)", "R² ≥ 0,50", "Brecha ≤ 0,10", "Supera baseline", "Listo para producción", "Calificación"]], use_container_width=True, hide_index=True)
    st.info(
        "El Stress Test traduce las métricas a una pregunta práctica: **¿este modelo es suficientemente "
        "confiable para este objetivo?**. La regla de producción de referencia del proyecto es R² ≥ 0,50, "
        "brecha TRAIN/TEST ≤ 0,10 y MAPE ≤ 25 %. Los umbrales son criterios de gobierno y pueden ajustarse "
        "en Evaluación."
    )

    mejor = comp.sort_values(["Listo para producción", "R² TEST", "RMSE TEST"], ascending=[False, False, True]).iloc[0]
    mejor_nombre = mejor["Modelo"]
    st.markdown(f"### 🏆 Candidato actual: **{mejor_nombre}**")
    st.caption("La selección definitiva corresponde a la Fase 5 de Evaluación; aquí se entrega el candidato técnico.")

    if st.button("💾 Guardar candidato para Despliegue", key="btn_guardar_modelo"):
        try:
            import joblib
            carpeta = Path("artifacts_modelado")
            carpeta.mkdir(exist_ok=True)
            r = resultados[mejor_nombre]
            # ruta = carpeta / f"modelo_{str(entrada['target']).replace(' ', '_').replace('/', '_')}.joblib"
            
            
            # Construir un nombre de archivo seguro para Windows y otros sistemas.
            target_original = str(entrada["target"]).strip()

            # Reemplazar caracteres no permitidos en nombres de archivo.
            target_seguro = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", target_original)

            # Evitar nombres vacíos y puntos o espacios al final.
            target_seguro = target_seguro.strip(" ._")

            if not target_seguro:
                target_seguro = "objetivo_financiero"

            # Limitar la longitud para evitar rutas excesivamente largas.
            target_seguro = target_seguro[:120]

            # Crear y resolver la ruta del artefacto.
            carpeta = Path("artifacts_modelado")
            carpeta.mkdir(parents=True, exist_ok=True)

            ruta = (carpeta / f"modelo_{target_seguro}.joblib").resolve()

            # Comprobación defensiva: no permitir que la ruta salga de la carpeta.
            carpeta_resuelta = carpeta.resolve()
            if ruta.parent != carpeta_resuelta:
                raise ValueError("La ruta del modelo no es válida.")

            
            paquete = {
                "modelo": r["estimador"],
                "target": entrada["target"],
                "target_descripcion": st.session_state.get("target_modelado_descripcion", _catalogo_targets_session().get(str(entrada["target"]), "Indicador financiero")),
                "target_tipo": "indicador_financiero_original_o_derivado",
                "target_anio": (int(re.findall(r"20(?:22|23|24|25|26)", str(entrada["target"]))[-1]) if re.findall(r"20(?:22|23|24|25|26)", str(entrada["target"])) else None),
                "features": list(entrada["X_train"].columns),
                "feature_engineering_reporte": st.session_state.get("feature_engineering_reporte", {}),
                "nota_temporal": "Métricas TEST son holdout entre empresas para el período histórico seleccionado. No equivalen por sí solas a validación prospectiva 2025/2026.",
                "metricas_train": r["metricas_train"],
                "metricas_test": r["metricas_test"],
                "best_params": r["best_params"],
                "random_state": st.session_state.get("modelado_config", {}).get("random_state", RANDOM_STATE_DEFAULT),
                "metodologia": "CRISP-DM Fase 4 — Modelado",
                "fecha_guardado": datetime.now().astimezone().isoformat(timespec="seconds"),
            }

            # 1. Artefacto técnico: modelo serializado para Python / VS Code.
            joblib.dump(paquete, ruta)

            # 2. Artefacto legible: informe PDF para lectura humana directa.
            ruta_pdf = (
                Path("artifacts_modelado")
                / f"informe_{target_seguro}.pdf"
            )
            ruta_pdf = generar_informe_pdf_candidato(
                ruta_pdf=ruta_pdf,
                objetivo=entrada["target"],
                nombre_modelo=mejor_nombre,
                paquete=paquete,
            )

            st.session_state["modelo_ganador"] = paquete
            st.session_state["modelo_ganador_rutas"] = {"joblib": str(ruta), "pdf": str(ruta_pdf)}
            st.success(f"Modelo guardado: {ruta}")
            st.success(f"Informe PDF generado y validado: {ruta_pdf}")
            with open(ruta_pdf, "rb") as f_pdf:
                st.download_button(
                    "Descargar informe PDF del candidato",
                    data=f_pdf.read(),
                    file_name=ruta_pdf.name,
                    mime="application/pdf",
                    key="download_pdf_candidato_modelado",
                )
            st.info(
                "La carpeta `artifacts_modelado` ahora contiene dos versiones del mismo candidato: "
                "el `.joblib` (ábralo con VS Code / Python — no es un PDF, por eso Acrobat Reader falla) "
                "y el `.pdf` (ábralo con cualquier lector de PDF)."
            )
        except Exception as exc:
            st.error(f"No fue posible guardar el modelo: {exc}")


def _tab_api_contrato(entrada):
    st.subheader("🔌 10. Contrato para Evaluación, Despliegue y API/LLM")
    st.write(
        "El prompt del usuario final se atenderá posteriormente en Despliegue. Este módulo no deja que "
        "la API invente cifras ni entrene modelos. La arquitectura correcta es: prompt → identificación de "
        "intención/objetivo → modelo numérico → predicción calculada por Python → contexto financiero calculado "
        "por Python → LLM/API para redactar una respuesta comprensible."
    )
    st.code(
        "Prompt del usuario\n"
        "      ↓\n"
        "API / LLM: entiende qué pregunta quiere resolver\n"
        "      ↓\n"
        "Modelo financiero entrenado (joblib)\n"
        "      ↓\n"
        "Predicción numérica calculada por Python\n"
        "      ↓\n"
        "Métricas + SHAP + contexto empresarial\n"
        "      ↓\n"
        "API / LLM: explica el resultado en lenguaje natural\n"
        "      ↓\n"
        "Respuesta en Streamlit",
        language="text",
    )
    if entrada:
        st.json({
            "target_preparado": str(entrada["target"]),
            "target_descripcion": st.session_state.get("target_modelado_descripcion", "Indicador financiero"),
            "train_rows": len(entrada["X_train"]),
            "test_rows": len(entrada["X_test"]),
            "features": list(entrada["X_train"].columns),
            "fuentes_contexto_llm": ["df_consulta", "diccionario_datos", "df_features_financieras", "feature_engineering_reporte"],
            "regla_cifras": "Calcular cifras con pandas/modelo Python; API/LLM solo interpreta y redacta.",
            "limite_temporal": "Dataset real 2022-2024; previsiones 2025/2026 no verificadas sin valores reales posteriores.",
        })
        target_years = re.findall(r"20(?:22|23|24|25|26)", str(entrada["target"]))
        resultados = st.session_state.get("modelado_resultados", {})
        if target_years and int(target_years[-1]) == 2024 and resultados:
            st.markdown("#### Estimación prospectiva para 2025")
            st.warning("La cifra 2025 será una estimación no verificada. Se desplazan las variables de entrada un año (2023–2024) y se aplica el modelo entrenado para el objetivo 2024. No hay datos reales 2025 para medir su error.")
            modelo_futuro = st.selectbox("Modelo entrenado para estimar 2025", list(resultados.keys()), key="modelado_modelo_prospectivo_2025")
            if st.button("🔮 Calcular estimación 2025 para las empresas", key="btn_estimacion_2025"):
                try:
                    predicciones = calcular_estimacion_prospectiva_2025(modelo_futuro)
                    st.success(f"Estimación calculada para {len(predicciones):,} registros empresariales.")
                    st.dataframe(predicciones.head(30), use_container_width=True, hide_index=True)
                    col_pred = next(c for c in predicciones.columns if c.startswith("ESTIMACION_NO_VERIFICADA_"))
                    resumen = predicciones[col_pred].describe(percentiles=[.1, .5, .9]).to_frame("Estadístico")
                    st.dataframe(resumen, use_container_width=True)
                    st.download_button("Descargar predicciones 2025 (CSV)", predicciones.to_csv(index=False).encode("utf-8-sig"), file_name="estimaciones_prospectivas_2025.csv", mime="text/csv", key="download_estimaciones_2025")
                except Exception as exc:
                    st.error(f"No fue posible calcular la estimación prospectiva: {exc}")
        else:
            st.info("Para calcular una estimación 2025, entrene un modelo con un target histórico de 2024. Los modelos 2026 recursivos no se habilitan automáticamente porque requieren estimar 2025 y reconstruir con esas cifras las variables derivadas para el siguiente horizonte.")
        if "predicciones_prospectivas_2025" in st.session_state:
            st.caption("Hay una estimación prospectiva 2025 calculada en esta sesión; su valor real todavía no está disponible para validación.")


# ═══════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

def mostrar_modelado(df: pd.DataFrame | None = None):
    """Punto de entrada para new_main_streamlit.py."""
    st.header("🤖 Modelado de los Datos")
    st.caption("Fase 4 · CRISP-DM — entrenamiento, comparación y explicabilidad de modelos predictivos.")

    entrada = _preparar_entrada_desde_session()
    if entrada is not None:
        _entrada_valida, _mensaje_entrada = _validar_entrada(entrada)
        if not _entrada_valida:
            st.error(_mensaje_entrada)
            entrada = None
    if entrada is None and df is not None:
        st.warning(
            "Se recibió un DataFrame, pero el contrato oficial de este módulo es X_train/X_test/y_train/y_test "
            "producido por new_preparacion.py. Ejecute primero Preparación de Datos."
        )

    tabs = st.tabs([
        "📌 1. Marco y objetivos",
        "🎯 2. Objetivos predictivos",
        "📥 3. Datos de entrada",
        "⚙️ 4. Configuración",
        "🤖 5. Entrenamiento",
        "🔍 6. Optimización",
        "📊 7. Comparación",
        "🔬 8. Explicabilidad",
        "🛡️ 9. Robustez",
        "🔌 10. Contrato API/LLM",
    ])

    with tabs[0]: _tab_marco(entrada)
    with tabs[1]: _tab_objetivos(entrada)
    with tabs[2]: _tab_entrada(entrada)
    with tabs[3]: _tab_configuracion()
    with tabs[4]: _tab_entrenamiento(entrada)
    with tabs[5]: _tab_optimizacion()
    with tabs[6]: _tab_comparacion(entrada)
    with tabs[7]: _tab_explicabilidad(entrada)
    with tabs[8]: _tab_robustez(entrada)
    with tabs[9]: _tab_api_contrato(entrada)


if __name__ == "__main__":
    st.set_page_config(page_title="Modelado · Plataforma Predictiva Financiera", layout="wide")
    st.title("🤖 Fase 4 — Modelado de los Datos")
    st.info("Este archivo está diseñado para ser invocado desde new_main_streamlit.py mediante mostrar_modelado().")
