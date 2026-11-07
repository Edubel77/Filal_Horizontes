"""
MÓDULO: new_evaluacion.py
FASE CRISP-DM: 5 — EVALUACIÓN

Interfaz para evaluar los modelos producidos por new_modelado.py y decidir,
con evidencia documentada, si un candidato puede pasar a Despliegue/Predicción.

Contrato con Modelado (new_modelado.py):
- st.session_state["modelado_resultados"]: dict de resultados por modelo.
- Cada resultado contiene, cuando está disponible:
    modelo, estimador, y_pred_test, y_pred_train, metricas_train, metricas_test,
    metricas_baseline_test, mejora_mae_vs_baseline_pct, best_params,
    best_cv_neg_mae, busqueda, diagnostico_target, n_train, n_test.
- st.session_state["y_test"], ["y_train"], ["X_test"], ["X_train"],
  ["target_modelado"] se originan en Preparación de Datos.
- st.session_state["modelado_target"] se actualiza al finalizar el entrenamiento.

El módulo NO reentrena modelos ni ajusta transformaciones. TEST se usa solo para
diagnóstico y evaluación. Las proyecciones 2025/2026 se identifican por separado
y no se presentan como resultados reales verificados.

Función pública compatible con el registro de new_main_streamlit.py:
    mostrar_evaluacion(df=None)
"""
from __future__ import annotations

import json
import math
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Generación de informe PDF legible para humanos (complemento del .json) ---
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


# ---------------------------------------------------------------------------
# Helpers de lectura segura
# ---------------------------------------------------------------------------

def _finite_array(values: Any) -> np.ndarray:
    """Crea un array propio, C-contiguo y finito cuando es posible."""
    arr = np.array(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=np.float64,
                   order="C", copy=True).reshape(-1)
    return arr


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
        if not np.isfinite(v):
            return "No disponible"
        return f"{v:,.{digits}f}"
    except (TypeError, ValueError):
        return "No disponible"


def _pct(value: Any, digits: int = 2) -> str:
    try:
        v = float(value)
        return f"{v:,.{digits}f}%" if np.isfinite(v) else "No disponible"
    except (TypeError, ValueError):
        return "No disponible"


def _target_actual() -> str:
    return str(
        st.session_state.get("modelado_target")
        or st.session_state.get("target_modelado")
        or "Objetivo no identificado"
    )


def _resultados() -> dict[str, dict[str, Any]]:
    r = st.session_state.get("modelado_resultados", {})
    return r if isinstance(r, dict) else {}


def _alinear_real_pred(resultado: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Obtiene y_test y y_pred_test, verificando que sean comparables."""
    y_real = st.session_state.get("y_test")
    y_pred = resultado.get("y_pred_test")
    if y_real is None:
        return np.array([], dtype=float), np.array([], dtype=float), (
            "No se encuentra st.session_state['y_test']. Vuelva a Preparación de Datos "
            "y entrene los modelos en esta sesión."
        )
    if y_pred is None:
        return np.array([], dtype=float), np.array([], dtype=float), (
            "El resultado del modelo no contiene y_pred_test. Vuelva a entrenar con la versión actual de new_modelado.py."
        )
    real = _finite_array(y_real)
    pred = _finite_array(y_pred)
    if len(real) != len(pred):
        return real, pred, (
            f"Las longitudes no coinciden: y_test={len(real):,}, predicciones={len(pred):,}. "
            "No se calcularán gráficos ni métricas adicionales."
        )
    valid = np.isfinite(real) & np.isfinite(pred)
    real, pred = real[valid], pred[valid]
    if not len(real):
        return real, pred, "No hay pares reales/predichos finitos para evaluar."
    return real, pred, None


def _baseline(resultado: dict[str, Any]) -> dict[str, Any]:
    val = resultado.get("metricas_baseline_test", {})
    return val if isinstance(val, dict) else {}


def _metrics_row(nombre: str, r: dict[str, Any]) -> dict[str, Any]:
    train = r.get("metricas_train", {}) or {}
    test = r.get("metricas_test", {}) or {}
    base = _baseline(r)
    r2_train = _num(train.get("R²"))
    r2_test = _num(test.get("R²"))
    mejora = _num(r.get("mejora_mae_vs_baseline_pct"))
    mae_test = _num(test.get("MAE"))
    mae_base = _num(base.get("MAE"))
    mae_media = _num(test.get("MAE / Media (%)"))
    brecha = r2_train - r2_test if np.isfinite(r2_train) and np.isfinite(r2_test) else np.nan
    calif = str(_calificacion_existente(r, r2_test, mejora, mae_media))
    return {
        "Modelo": r.get("modelo", nombre),
        "R² TRAIN": r2_train,
        "R² TEST": r2_test,
        "Brecha R²": brecha,
        "MAE TEST": mae_test,
        "RMSE TEST": _num(test.get("RMSE")),
        "Varianza explicada TEST": _num(test.get("Varianza Explicada")),
        "MAE baseline TEST": mae_base,
        "Mejora MAE vs baseline (%)": mejora,
        "MAPE TEST (%)": _num(test.get("MAPE (%)")),
        "Cobertura MAPE (%)": _num(test.get("MAPE cobertura (%)")),
        "MAE / Media TEST (%)": mae_media,
        "CV (−MAE)": _num(r.get("best_cv_neg_mae")),
        "Calificación Modelado": calif,
        "Tiempo (s)": _num(r.get("duracion_s")),
    }


def _num(x: Any) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _calificacion_existente(r: dict[str, Any], r2_test: float,
                            mejora: float, mae_media: float) -> str:
    """Reutiliza la calificación oficial de Modelado; fallback conservador."""
    for key in ("Calificación", "calificacion", "calificación"):
        if key in r:
            return r[key]
    if not np.isfinite(mejora) or mejora <= 0:
        return "No supera baseline"
    if np.isfinite(r2_test) and r2_test >= .5 and mejora >= 10 and np.isfinite(mae_media) and mae_media <= 50:
        return "Buen desempeño en TEST"
    if np.isfinite(r2_test) and r2_test >= .2:
        return "Desempeño prometedor; revisar alertas"
    return "Evidencia predictiva débil"


def _diagnostico_financiero(target: str, y: np.ndarray) -> list[str]:
    """Alertas interpretativas sin eliminar ni alterar observaciones."""
    alerts: list[str] = []
    if not len(y):
        return ["No hay valores reales válidos para diagnosticar el objetivo."]
    upper = target.upper()
    zero_pct = float(np.mean(np.isclose(y, 0.0)) * 100)
    neg_pct = float(np.mean(y < 0) * 100)
    q01, q25, q50, q75, q99 = np.percentile(y, [1, 25, 50, 75, 99])
    scale = max(float(np.median(np.abs(y))), 1e-12)
    if zero_pct > 0:
        alerts.append(f"{zero_pct:.1f}% de los valores de TEST son cero; MAPE puede ser inadecuado o tener cobertura limitada.")
    if neg_pct > 0 and any(k in upper for k in ("ROA", "ROE", "MARGEN", "CRECIMIENTO", "GANANCIA", "PATRIMONIO")):
        alerts.append(f"{neg_pct:.1f}% de los valores de TEST son negativos. Verifique su interpretación económica; no todos los negativos son errores.")
    if "CRECIMIENTO" in upper:
        alerts.append("Indicador de crecimiento: revisar valores del año base cercanos a cero y cambios de signo.")
    if any(k in upper for k in ("SOLVENCIA", "APALANCAMIENTO", "ENDEUDAMIENTO")):
        alerts.append("Razón financiera patrimonial: confirmar patrimonio cero/negativo y denominadores pequeños antes de interpretar.")
    if (q99 - q01) > 1000 * scale:
        alerts.append("La amplitud P99–P1 es muy grande respecto a la mediana absoluta; inspeccione la cola y posibles empresas influyentes.")
    if len(y) < 100:
        alerts.append(f"TEST contiene solo {len(y):,} observaciones válidas; la estimación de desempeño puede ser inestable.")
    if np.std(y) <= 1e-12:
        alerts.append("El objetivo tiene variabilidad prácticamente nula en TEST; R² y comparaciones relativas pueden ser poco informativos.")
    return alerts


def _decision_fila(row: pd.Series, alerts: list[str], n: int) -> tuple[str, list[str]]:
    """Decisión de gobernanza, separada de la etiqueta automática de Modelado."""
    reasons: list[str] = []
    r2 = _num(row.get("R² TEST"))
    mejora = _num(row.get("Mejora MAE vs baseline (%)"))
    mae_media = _num(row.get("MAE / Media TEST (%)"))
    brecha = _num(row.get("Brecha R²"))
    if not np.isfinite(mejora) or mejora <= 0:
        reasons.append("No demuestra mejora frente al baseline.")
    if not np.isfinite(r2) or r2 < 0.20:
        reasons.append("R² TEST es insuficiente o no está disponible para sustentar uso predictivo.")
    if np.isfinite(brecha) and brecha > 0.20:
        reasons.append("Brecha TRAIN–TEST elevada; posible sobreajuste.")
    if np.isfinite(mae_media) and mae_media > 50:
        reasons.append("Error MAE relativo elevado frente a la magnitud media del objetivo.")
    if n < 100:
        reasons.append("Muestra de TEST menor de 100 observaciones.")
    if alerts:
        reasons.append("Existen alertas de calidad/interpretación del objetivo que deben revisarse.")
    if not reasons:
        return "Aprobado con restricciones", ["Las métricas son favorables bajo los umbrales técnicos iniciales; falta confirmar utilidad financiera y validación temporal."]
    if (np.isfinite(mejora) and mejora >= 10 and np.isfinite(r2) and r2 >= .50
            and (not np.isfinite(mae_media) or mae_media <= 50) and not alerts
            and (not np.isfinite(brecha) or brecha <= .10) and n >= 100):
        return "Aprobado para uso validado", ["Supera el baseline con margen material, R² y error relativo dentro de los umbrales iniciales, sin alertas automáticas."]
    if np.isfinite(mejora) and mejora > 0:
        return "Solo experimental", reasons
    return "No aprobado", reasons


def _crear_residuos(real: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "Real": real,
        "Predicho": pred,
        "Residuo (real - predicho)": real - pred,
        "Error absoluto": np.abs(real - pred),
    })


def _safe_json(value: Any) -> Any:
    """Convierte tipos NumPy y valores no finitos a JSON serializable."""
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# UI — siete pestañas del diseño funcional
# ---------------------------------------------------------------------------

def mostrar_evaluacion(df: pd.DataFrame | None = None) -> None:
    """Renderiza el módulo de Evaluación (Fase 5 CRISP-DM)."""
    st.title("📐 Evaluación de modelos — Fase 5 de CRISP-DM")
    st.caption(
        "La evaluación comprueba si los modelos aportan valor frente a una referencia, "
        "si sus errores son aceptables y si existe evidencia suficiente para pasar a Despliegue."
    )

    resultados = _resultados()
    target = _target_actual()
    if not resultados:
        st.warning("Todavía no hay resultados de Modelado en esta sesión.")
        st.markdown(
            "**Para continuar:** carga los datos, completa Preparación de Datos, selecciona un objetivo y entrena al menos un modelo en **🤖 Modelado**."
        )
        st.info("Evaluación no entrena modelos ni inventa métricas. Cuando existan predicciones TEST, aparecerán aquí.")
        return

    # Se valida que el target mostrado coincida con el que produjo los resultados.
    target_modelado = str(st.session_state.get("modelado_target") or target)
    if target_modelado != target and target != "Objetivo no identificado":
        st.warning(f"El objetivo activo ({target}) no coincide con el último entrenamiento ({target_modelado}). Se evaluarán los resultados del último entrenamiento: **{target_modelado}**.")
        target = target_modelado

    nombres = list(resultados.keys())
    labels = {
        nombre: f"{nombre} — {_calificacion_existente(resultados[nombre], _num((resultados[nombre].get('metricas_test') or {}).get('R²')), _num(resultados[nombre].get('mejora_mae_vs_baseline_pct')), _num((resultados[nombre].get('metricas_test') or {}).get('MAE / Media (%)')))}"
        for nombre in nombres
    }
    elegido = st.selectbox(
        "Modelo candidato para inspección detallada",
        nombres,
        format_func=lambda x: labels.get(x, x),
        key="evaluacion_modelo_seleccionado",
    )
    resultado = resultados[elegido]
    real, pred, error_alineacion = _alinear_real_pred(resultado)
    metricas = resultado.get("metricas_test", {}) or {}
    train = resultado.get("metricas_train", {}) or {}
    baseline = _baseline(resultado)
    rows = [_metrics_row(n, r) for n, r in resultados.items()]
    comparacion = pd.DataFrame(rows)
    alerts = _diagnostico_financiero(target, real) if len(real) else ["No se pudo calcular el diagnóstico del objetivo."]
    alerts.extend((resultado.get("diagnostico_target") or {}).get("alertas", []) or [])
    # Deduplicar conservando orden.
    alerts = list(dict.fromkeys(str(a) for a in alerts))
    row_elegida = comparacion[comparacion["Modelo"] == resultado.get("modelo", elegido)]
    row = row_elegida.iloc[0] if not row_elegida.empty else comparacion.iloc[0]
    decision, motivos = _decision_fila(row, alerts, len(real))

    tabs = st.tabs([
        "1. Resumen",
        "2. Modelos y baseline",
        "3. Errores y residuos",
        "4. Estabilidad",
        "5. Calidad del objetivo",
        "6. Explicabilidad",
        "7. Aprobación y Despliegue",
    ])

    # 1 — Resumen
    with tabs[0]:
        st.subheader("Resumen ejecutivo")
        st.write(f"**Objetivo evaluado:** `{target}`")
        st.write(f"**Modelo inspeccionado:** {elegido}")
        if error_alineacion:
            st.error(error_alineacion)
        if len(real):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE TEST", _fmt(metricas.get("MAE")))
            c2.metric("RMSE TEST", _fmt(metricas.get("RMSE")))
            c3.metric("R² TEST", _fmt(metricas.get("R²")))
            c4.metric("Mejora MAE vs baseline", _pct(resultado.get("mejora_mae_vs_baseline_pct")))
            st.metric("Estado de evaluación", decision)
            st.caption("La decisión es un control técnico inicial; no sustituye la validación financiera ni demuestra validez temporal 2025/2026.")
        else:
            st.warning("No hay datos suficientes para una síntesis cuantitativa.")
        st.markdown("#### Limitaciones y alertas")
        if alerts:
            for a in alerts:
                st.warning(a)
        else:
            st.success("No se detectaron alertas automáticas. Esto no garantiza por sí solo que la predicción sea confiable.")
        st.markdown("#### Motivos de la decisión")
        for m in motivos:
            st.write(f"- {m}")
        st.markdown("#### Datos del experimento")
        st.write({
            "target": target,
            "n_train": resultado.get("n_train", len(st.session_state.get("y_train", []))),
            "n_test_validado": len(real),
            "features": len(st.session_state.get("X_train", pd.DataFrame()).columns)
                if isinstance(st.session_state.get("X_train"), pd.DataFrame) else "No disponible",
            "búsqueda": resultado.get("busqueda", "No disponible"),
            "parámetros": resultado.get("best_params", {}),
            "seed": (st.session_state.get("modelado_config") or {}).get("random_state", 42),
        })

    # 2 — comparación
    with tabs[1]:
        st.subheader("Comparación de modelos frente al baseline")
        display_cols = [
            "Modelo", "R² TRAIN", "R² TEST", "Brecha R²", "MAE TEST", "RMSE TEST",
            "MAE baseline TEST", "Mejora MAE vs baseline (%)", "MAPE TEST (%)",
            "Cobertura MAPE (%)", "CV (−MAE)", "Calificación Modelado",
        ]
        st.dataframe(comparacion[display_cols], use_container_width=True, hide_index=True)
        st.caption("Las métricas proceden de Modelado. La pestaña no reentrena ni cambia el TEST.")
        graf = comparacion.dropna(subset=["MAE TEST", "MAE baseline TEST"])
        if not graf.empty:
            graf_long = graf.melt(
                id_vars="Modelo",
                value_vars=["MAE TEST", "MAE baseline TEST"],
                var_name="Referencia",
                value_name="MAE",
            )
            fig = px.bar(graf_long, x="Modelo", y="MAE", color="Referencia", barmode="group",
                         title="MAE de cada modelo frente al baseline (menor es mejor)")
            fig.update_layout(height=440, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        if not comparacion.empty:
            st.markdown("#### Ranking orientativo por MAE TEST")
            st.dataframe(
                comparacion.sort_values("MAE TEST", ascending=True)[
                    ["Modelo", "MAE TEST", "Mejora MAE vs baseline (%)", "R² TEST", "Calificación Modelado"]
                ],
                use_container_width=True, hide_index=True,
            )

    # 3 — errores y residuos
    with tabs[2]:
        st.subheader("Diagnóstico de errores y residuos")
        if error_alineacion:
            st.warning(error_alineacion)
        elif len(real):
            resid = _crear_residuos(real, pred)
            left, right = st.columns(2)
            with left:
                fig = px.scatter(resid, x="Predicho", y="Real", title="Real frente a predicho",
                                 labels={"Predicho": "Predicción", "Real": "Valor real"})
                mn = float(min(resid["Real"].min(), resid["Predicho"].min()))
                mx = float(max(resid["Real"].max(), resid["Predicho"].max()))
                if np.isfinite(mn) and np.isfinite(mx):
                    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", name="Predicción ideal"))
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True)
            with right:
                fig = px.scatter(resid, x="Predicho", y="Residuo (real - predicho)",
                                 title="Residuos frente a predicción")
                fig.add_hline(y=0, line_dash="dash")
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True)
            fig_hist = px.histogram(resid, x="Residuo (real - predicho)", nbins=40,
                                    title="Distribución de residuos")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.write({
                "sesgo medio del residuo": _fmt(resid["Residuo (real - predicho)"].mean()),
                "residuo mediano": _fmt(resid["Residuo (real - predicho)"].median()),
                "percentil 90 del error absoluto": _fmt(resid["Error absoluto"].quantile(.90)),
                "percentil 95 del error absoluto": _fmt(resid["Error absoluto"].quantile(.95)),
                "porcentaje con error absoluto superior a 2×MAE": _pct(
                    (resid["Error absoluto"] > 2 * resid["Error absoluto"].mean()).mean() * 100
                ),
            })
            st.dataframe(resid.head(100), use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar errores TEST (CSV)",
                resid.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"residuos_{_slug(target)}_{_slug(elegido)}.csv",
                mime="text/csv",
            )
        else:
            st.info("Los gráficos se activan cuando hay valores reales y predicciones TEST alineados.")

    # 4 — estabilidad
    with tabs[3]:
        st.subheader("Validación, estabilidad y generalización")
        n_train = resultado.get("n_train", len(st.session_state.get("y_train", [])))
        n_test = len(real)
        c1, c2, c3 = st.columns(3)
        c1.metric("Observaciones TRAIN", f"{int(n_train):,}" if n_train is not None else "No disponible")
        c2.metric("Observaciones TEST válidas", f"{n_test:,}")
        c3.metric("Score CV (−MAE)", _fmt(resultado.get("best_cv_neg_mae")))
        r2tr = _num(train.get("R²"))
        r2te = _num(metricas.get("R²"))
        if np.isfinite(r2tr) and np.isfinite(r2te):
            brecha = r2tr - r2te
            st.metric("Brecha R² TRAIN − TEST", _fmt(brecha))
            if brecha > .20:
                st.warning("Brecha alta: posible sobreajuste o cambio de distribución.")
            elif r2te < .20:
                st.warning("R² TEST bajo: el modelo explica poco de la variabilidad de empresas no vistas.")
            else:
                st.info("La brecha por sí sola no determina la calidad; revise MAE, baseline y diagnóstico de residuos.")
        st.warning(
            "El CV reportado por Modelado es un score de selección calculado dentro de TRAIN cuando hubo optimización. "
            "No equivale a validación temporal 2025/2026 ni reemplaza el TEST final."
        )
        if isinstance(st.session_state.get("y_train"), (pd.Series, pd.DataFrame)):
            yt = _finite_array(st.session_state["y_train"])
            yt = yt[np.isfinite(yt)]
            if len(yt) and len(real):
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=yt, name="TRAIN", opacity=.65))
                fig.add_trace(go.Histogram(x=real, name="TEST", opacity=.65))
                fig.update_layout(barmode="overlay", title="Distribución del objetivo: TRAIN frente a TEST", height=400)
                st.plotly_chart(fig, use_container_width=True)
        st.caption("Una partición aleatoria entre empresas del año 2024 no demuestra por sí sola que el modelo prediga correctamente años posteriores.")

    # 5 — calidad del objetivo
    with tabs[4]:
        st.subheader("Calidad y diagnóstico financiero del objetivo")
        st.write(f"**Target:** `{target}`")
        if len(real):
            q = np.percentile(real, [0, 1, 5, 25, 50, 75, 95, 99, 100])
            diag = pd.DataFrame({
                "Percentil": ["Mínimo", "P01", "P05", "P25", "P50/mediana", "P75", "P95", "P99", "Máximo"],
                "Valor": q,
            })
            st.dataframe(diag, use_container_width=True, hide_index=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ceros", _pct(np.isclose(real, 0).mean() * 100))
            c2.metric("Negativos", _pct((real < 0).mean() * 100))
            c3.metric("Media", _fmt(np.mean(real)))
            c4.metric("Desviación estándar", _fmt(np.std(real)))
            fig = px.histogram(x=real, nbins=50, title=f"Distribución de TEST — {target}")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("#### Alertas automáticas")
            if alerts:
                for a in alerts:
                    st.warning(a)
            else:
                st.success("No se identificaron alertas automáticas; se requiere revisión de negocio.")
        else:
            st.info("No hay valores reales suficientes para diagnosticar la distribución.")
        st.markdown("#### Nota financiera")
        up = target.upper()
        if "CRECIMIENTO" in up:
            st.info("Los porcentajes de crecimiento pueden ser extremos cuando el valor del año base es cero, cercano a cero o cambia de signo.")
        elif any(k in up for k in ("ROA", "ROE", "MARGEN")):
            st.info("En indicadores de rentabilidad, los negativos pueden ser económicamente válidos. Deben distinguirse de errores de cálculo.")
        elif any(k in up for k in ("SOLVENCIA", "APALANCAMIENTO", "ENDEUDAMIENTO")):
            st.info("Revise patrimonio nulo/negativo y denominadores pequeños. Una razón matemáticamente calculable no siempre tiene interpretación financiera convencional.")

    # 6 — explicabilidad
    with tabs[5]:
        st.subheader("Explicabilidad y utilidad financiera")
        features = st.session_state.get("X_test")
        importancias = None
        modelo = resultado.get("estimador")
        if modelo is not None and isinstance(features, pd.DataFrame):
            try:
                vals = None
                if hasattr(modelo, "feature_importances_"):
                    vals = np.asarray(modelo.feature_importances_, dtype=float)
                elif hasattr(modelo, "named_steps") and hasattr(modelo.named_steps.get("model"), "coef_"):
                    vals = np.abs(np.asarray(modelo.named_steps["model"].coef_, dtype=float)).reshape(-1)
                elif hasattr(modelo, "coef_"):
                    vals = np.abs(np.asarray(modelo.coef_, dtype=float)).reshape(-1)
                if vals is not None and len(vals) == features.shape[1]:
                    importancias = pd.DataFrame({"Variable": list(features.columns), "Importancia": vals})
                    importancias = importancias.sort_values("Importancia", ascending=False)
            except Exception as exc:
                st.warning(f"No fue posible obtener importancias: {exc}")
        if importancias is not None and not importancias.empty:
            fig = px.bar(importancias.head(20).sort_values("Importancia"), x="Importancia", y="Variable",
                         orientation="h", title="Importancia disponible del modelo")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(importancias.head(30), use_container_width=True, hide_index=True)
            st.caption("En modelos lineales se muestran coeficientes absolutos; no son comparables directamente con importancias de árboles.")
        else:
            st.info("El estimador no expone importancias compatibles o faltan features guardadas. SHAP no se inventa ni se calcula automáticamente en esta pestaña.")
        st.warning("Importancia predictiva no equivale a causalidad. Esta pantalla describe asociaciones utilizadas por el modelo, no demuestra causas financieras.")

    # 7 — aprobación y despliegue
    with tabs[6]:
        st.subheader("Decisión de aprobación y preparación para Despliegue")
        st.write(f"**Modelo:** {elegido} · **Objetivo:** `{target}`")
        st.metric("Estado técnico propuesto", decision)
        st.markdown("#### Checklist de control")
        checks = {
            "Predicciones TEST y valores reales alineados": error_alineacion is None and len(real) > 0,
            "Mejora frente al baseline": np.isfinite(_num(row.get("Mejora MAE vs baseline (%)"))) and _num(row.get("Mejora MAE vs baseline (%)")) > 0,
            "R² TEST ≥ 0,20": np.isfinite(_num(row.get("R² TEST"))) and _num(row.get("R² TEST")) >= .20,
            "Brecha R² ≤ 0,10": np.isfinite(_num(row.get("Brecha R²"))) and _num(row.get("Brecha R²")) <= .10,
            "No hay alertas automáticas de objetivo": len(alerts) == 0,
            "Estimador disponible": resultado.get("estimador") is not None,
            "Features de entrenamiento disponibles": isinstance(st.session_state.get("X_train"), pd.DataFrame),
        }
        for label, passed in checks.items():
            (st.success if passed else st.warning)(f"{'✓' if passed else '⚠'} {label}")
        st.caption("Los umbrales son criterios iniciales de gobierno del modelo, no leyes universales para todos los objetivos financieros.")
        if decision == "Aprobado para uso validado":
            st.success("El candidato cumple los criterios técnicos iniciales. Antes de uso operativo, documente el rango de aplicación y realice validación temporal cuando haya datos posteriores.")
        elif decision == "Aprobado con restricciones":
            st.info("Uso restringido al contexto documentado; mantener advertencias y no asumir validez temporal.")
        elif decision == "Solo experimental":
            st.warning("Mantener el modelo para exploración. No presentar sus estimaciones como predicciones confiables.")
        else:
            st.error("No autorizar este candidato como predictor fiable hasta corregir las limitaciones identificadas.")
        st.markdown("#### Registrar evaluación")
        notas = st.text_area(
            "Notas del analista (opcional)",
            key=f"evaluacion_notas_{_slug(target)}_{_slug(elegido)}",
            placeholder="Uso previsto, límites conocidos, decisión de negocio, acciones pendientes...",
        )
        if st.button("💾 Guardar informe y manifiesto", type="primary", key="btn_guardar_evaluacion"):
            try:
                informe = {
                    "fecha_evaluacion": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "metodologia": "CRISP-DM Fase 5 — Evaluación",
                    "target": target,
                    "modelo": elegido,
                    "estado": decision,
                    "motivos": motivos,
                    "alertas": alerts,
                    "metricas": _safe_json(row.to_dict()),
                    "parametros": _safe_json(resultado.get("best_params", {})),
                    "n_train": resultado.get("n_train"),
                    "n_test": len(real),
                    "notas_analista": notas,
                    "nota_temporal": (
                        "La evaluación es histórica sobre TEST del período objetivo. "
                        "No acredita precisión para 2025/2026 sin valores reales de esos años."
                    ),
                    "features": list(st.session_state["X_train"].columns)
                        if isinstance(st.session_state.get("X_train"), pd.DataFrame) else None,
                }
                carpeta = Path("artifacts_evaluacion")
                carpeta.mkdir(parents=True, exist_ok=True)
                base = f"evaluacion_{_slug(target)}_{_slug(elegido)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                informe_json = json.dumps(_safe_json(informe), ensure_ascii=False, indent=2)

                # 1. Manifiesto técnico: JSON para VS Code / Python.
                archivo = carpeta / f"{base}.json"
                archivo.write_text(informe_json, encoding="utf-8")

                # 2. Informe legible: PDF para lectura humana directa.
                ruta_pdf = generar_informe_pdf_evaluacion(carpeta / f"{base}.pdf", informe)

                st.session_state["evaluacion_ultimo_informe"] = informe
                st.session_state["evaluacion_ultimas_rutas"] = {"json": str(archivo), "pdf": str(ruta_pdf)}
                st.success(f"Informe guardado: {archivo}")
                st.success(f"Informe PDF generado y validado: {ruta_pdf}")
                st.download_button(
                    "Descargar manifiesto de evaluación (JSON — VS Code)",
                    data=informe_json.encode("utf-8"),
                    file_name=archivo.name,
                    mime="application/json",
                    key="download_manifest_evaluacion",
                )
                with open(ruta_pdf, "rb") as f_pdf:
                    st.download_button(
                        "Descargar informe de evaluación (PDF)",
                        data=f_pdf.read(),
                        file_name=ruta_pdf.name,
                        mime="application/pdf",
                        key="download_pdf_evaluacion",
                    )
                st.info(
                    "La carpeta `artifacts_evaluacion` ahora contiene dos versiones del mismo informe: "
                    "el `.json` (ábralo con VS Code / Python) y el `.pdf` (ábralo con cualquier lector de PDF)."
                )
            except Exception as exc:
                st.error(f"No fue posible guardar el informe: {exc}")
                st.code(traceback.format_exc())


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_")
    return slug[:100] or "objetivo"


def generar_informe_pdf_evaluacion(ruta_pdf, informe: dict[str, Any]) -> Path:
    """Genera y valida el informe PDF legible de la evaluación (Fase 5).

    El `.json` es el manifiesto técnico (VS Code / Python); este PDF es la
    versión legible para humanos del mismo informe.
    """
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError(f"reportlab no está disponible: {REPORTLAB_ERROR}")

    ruta_pdf = Path(ruta_pdf)
    ruta_pdf.parent.mkdir(parents=True, exist_ok=True)

    estilos = getSampleStyleSheet()
    estilos.add(ParagraphStyle(name="DetallePequeno", parent=estilos["BodyText"], fontSize=8, leading=11))
    estilos.add(ParagraphStyle(
        name="AvisoTemporal",
        parent=estilos["BodyText"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#8A5A00"),
    ))

    def t(valor) -> str:
        return escape(str(valor))

    def fmt(valor) -> str:
        try:
            return f"{float(valor):,.4f}"
        except (TypeError, ValueError):
            return str(valor)

    metricas = informe.get("metricas") or {}
    claves_metricas = [
        "MAE TEST", "RMSE TEST", "R² TEST", "Brecha R²",
        "Mejora MAE vs baseline (%)", "MAE / Media TEST (%)",
        "MAPE TEST (%)", "CV (−MAE)", "Calificación Modelado",
    ]
    filas = [["Métrica", "Valor"]]
    for clave in claves_metricas:
        if clave in metricas:
            filas.append([clave, fmt(metricas.get(clave))])
    tabla = Table(filas, colWidths=[8 * cm, 7 * cm], repeatRows=1)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174A6E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    motivos = informe.get("motivos") or ["Sin motivos registrados."]
    alertas = informe.get("alertas") or ["Sin alertas automáticas."]
    items_motivos = [Paragraph(f"• {t(m)}", estilos["BodyText"]) for m in motivos]
    items_alertas = [Paragraph(f"• {t(a)}", estilos["BodyText"]) for a in alertas]

    contenido = [
        Paragraph("Informe de Evaluación — CRISP-DM Fase 5", estilos["Title"]),
        Spacer(1, 0.4 * cm),
        Paragraph(f"<b>Objetivo:</b> {t(informe.get('target', 'No disponible'))}", estilos["BodyText"]),
        Paragraph(f"<b>Modelo:</b> {t(informe.get('modelo', 'No disponible'))}", estilos["BodyText"]),
        Paragraph(f"<b>Estado:</b> {t(informe.get('estado', 'No disponible'))}", estilos["BodyText"]),
        Paragraph(f"<b>Fecha de evaluación:</b> {t(informe.get('fecha_evaluacion', 'No disponible'))}", estilos["BodyText"]),
        Paragraph(f"<b>Muestra:</b> TRAIN n={t(informe.get('n_train', 'N/D'))} · TEST n={t(informe.get('n_test', 'N/D'))}", estilos["BodyText"]),
        Spacer(1, 0.4 * cm),
        Paragraph("Métricas de evaluación", estilos["Heading2"]),
        tabla,
        Spacer(1, 0.4 * cm),
        Paragraph("Motivos de la decisión", estilos["Heading2"]),
        *items_motivos,
        Spacer(1, 0.3 * cm),
        Paragraph("Alertas y limitaciones", estilos["Heading2"]),
        *items_alertas,
        Spacer(1, 0.3 * cm),
        Paragraph("Notas del analista", estilos["Heading2"]),
        Paragraph(t(informe.get("notas_analista") or "Sin notas registradas."), estilos["BodyText"]),
        Spacer(1, 0.4 * cm),
        Paragraph("Nota temporal", estilos["Heading2"]),
        Paragraph(
            t(informe.get(
                "nota_temporal",
                "La evaluación es histórica sobre TEST del período objetivo. "
                "No acredita precisión para 2025/2026 sin valores reales de esos años.",
            )),
            estilos["AvisoTemporal"],
        ),
        Spacer(1, 0.4 * cm),
        Paragraph("Parámetros del modelo", estilos["Heading2"]),
        Paragraph(t(informe.get("parametros", {})), estilos["DetallePequeno"]),
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

    # No dar por válido el PDF hasta verificar su firma y contenido.
    with ruta_pdf.open("rb") as archivo:
        firma = archivo.read(5)
    if firma != b"%PDF-":
        raise ValueError(f"El informe generado no tiene una firma PDF válida: {ruta_pdf}")
    if ruta_pdf.stat().st_size == 0:
        raise ValueError("El informe PDF está vacío.")

    return ruta_pdf
