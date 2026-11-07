"""Fase 6 CRISP-DM: Despliegue / Predicción para la Plataforma IA Financiera.

El modelo numérico calcula cifras; el proveedor LLM únicamente las interpreta y redacta.
Las proyecciones 2025/2026 se etiquetan como estimaciones no verificadas porque el dataset
base cubre 2022-2024. Las claves API se solicitan en la barra lateral y no se guardan en disco.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROVEEDORES = {
    "OpenAI": {"modelos": ["gpt-4o-mini", "gpt-4o"], "default": "gpt-4o-mini"},
    "Gemini": {"modelos": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"], "default": "gemini-3.8-flash"},
    "Claude": {"modelos": ["claude-3-5-haiku-latest", "claude-3-7-sonnet-latest"], "default": "claude-3-5-haiku-latest"},
    "Local (Ollama)": {"modelos": ["qwen2.5:7b", "llama3.1:8b"], "default": "qwen2.5:7b"},
}


def _slug(texto: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(texto)).strip("._")[:100] or "objetivo_financiero"


def _serializar(valor: Any) -> Any:
    if isinstance(valor, (np.integer,)): return int(valor)
    if isinstance(valor, (np.floating,)): return float(valor) if np.isfinite(valor) else None
    if isinstance(valor, (np.bool_,)): return bool(valor)
    if isinstance(valor, (pd.Timestamp, datetime)): return valor.isoformat()
    if isinstance(valor, (np.ndarray,)): return valor.tolist()
    return str(valor)


def _llamar_api(proveedor: str, api_key: str, modelo: str, system: str, prompt: str) -> str:
    """Invoca la API seleccionada mediante HTTP; errores de proveedor se muestran al usuario."""
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Falta requests. Instálelo en el entorno de Streamlit: pip install requests") from exc
    if proveedor == "OpenAI":
        r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": modelo, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "temperature": 0.2}, timeout=180)
        if not r.ok: raise RuntimeError(f"OpenAI ({r.status_code}): {r.text[:800]}")
        return r.json()["choices"][0]["message"]["content"]
    if proveedor == "Gemini":
        # Gemini REST generateContent: modelo estable y errores explicados sin exponer la clave.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"
        r = requests.post(
            url,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 6000},
            },
            timeout=180,
        )
        if not r.ok:
            detalle = r.text[:1000]
            if r.status_code == 404:
                raise RuntimeError(
                    f"Gemini (404): el modelo '{modelo}' no está disponible para esta API o proyecto. "
                    "Seleccione gemini-3.8-flash o gemini-3.6-flash en la barra lateral y vuelva a ejecutar. "
                    f"Detalle del proveedor: {detalle}"
                )
            if r.status_code == 400:
                raise RuntimeError(f"Gemini (400): solicitud rechazada. Verifique el modelo y la configuración de la API. Detalle: {detalle}")
            if r.status_code in (401, 403):
                raise RuntimeError(f"Gemini ({r.status_code}): la API Key no es válida, no tiene permisos o la API no está habilitada. Revise Google AI Studio. Detalle: {detalle}")
            raise RuntimeError(f"Gemini ({r.status_code}): {detalle}")
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback", {})
            raise RuntimeError(f"Gemini no devolvió texto. Revisión de seguridad/respuesta: {json.dumps(feedback, ensure_ascii=False)[:700]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        texto = "\n".join(p.get("text", "") for p in parts if p.get("text"))
        if not texto.strip():
            raise RuntimeError("Gemini respondió sin contenido de texto. Intente de nuevo o seleccione otro modelo disponible.")
        return texto
    if proveedor == "Claude":
        r = requests.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, json={"model": modelo, "max_tokens": 6000, "system": system, "messages": [{"role": "user", "content": prompt}]}, timeout=180)
        if not r.ok: raise RuntimeError(f"Claude ({r.status_code}): {r.text[:800]}")
        return "\n".join(p.get("text", "") for p in r.json().get("content", []) if p.get("type") == "text")
    if proveedor == "Local (Ollama)":
        base = st.session_state.get("ollama_url", "http://localhost:11434").strip().rstrip("/")
        # Diagnostica primero el servicio y confirma que el modelo esté descargado.
        try:
            health = requests.get(f"{base}/api/tags", timeout=(5, 15))
        except requests.RequestException as exc:
            raise RuntimeError(
                f"No se pudo conectar con Ollama en {base}. Abra Ollama en Windows y verifique la URL. "
                f"Detalle: {exc}"
            ) from exc
        if not health.ok:
            raise RuntimeError(f"Ollama no responde correctamente ({health.status_code}): {health.text[:500]}")
        installed = [m.get("name", "") for m in health.json().get("models", [])]
        if modelo not in installed and not any(name.split(":")[0] == modelo.split(":")[0] for name in installed):
            disponibles = ", ".join(installed) if installed else "ninguno detectado"
            raise RuntimeError(
                f"El modelo '{modelo}' no aparece instalado en Ollama. Modelos detectados: {disponibles}. "
                f"En PowerShell ejecute: ollama pull {modelo}"
            )
        payload = {
            "model": modelo,
            "stream": False,
            "keep_alive": "10m",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # Limita el contexto/generación para reducir tiempos en equipos sin GPU potente.
            "options": {"num_ctx": 4096, "num_predict": 900, "temperature": 0.2},
        }
        try:
            r = requests.post(f"{base}/api/chat", json=payload, timeout=(10, 600))
        except requests.ReadTimeout as exc:
            raise RuntimeError(
                "Ollama tardó más de 10 minutos en generar la respuesta. Pruebe primero con un prompt más corto, "
                "cierre aplicaciones que consuman RAM y compruebe el modelo con `ollama run " + modelo + "`."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Falló la conexión con Ollama en {base}: {exc}") from exc
        if not r.ok:
            raise RuntimeError(f"Ollama ({r.status_code}): {r.text[:800]}")
        texto = r.json().get("message", {}).get("content", "")
        if not texto.strip():
            raise RuntimeError("Ollama respondió sin texto. Pruebe otro modelo o reduzca la longitud del prompt.")
        return texto
    raise ValueError("Proveedor de IA no reconocido.")


def _configurar_api() -> tuple[str, str, str]:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔌 API de Inteligencia Artificial")
    proveedor = st.sidebar.selectbox("Proveedor API", list(PROVEEDORES), key="despliegue_proveedor")
    info = PROVEEDORES[proveedor]
    modelo = st.sidebar.selectbox("Modelo LLM", info["modelos"], index=0, key="despliegue_modelo_llm")
    if proveedor == "Local (Ollama)":
        st.sidebar.caption("Modo local: requiere Ollama activo en el equipo.")
        st.session_state["ollama_url"] = st.sidebar.text_input("URL de Ollama", "http://localhost:11434", key="despliegue_ollama_url")
        clave = ""
    else:
        env_key = {"OpenAI": "OPENAI_API_KEY", "Gemini": "GEMINI_API_KEY", "Claude": "ANTHROPIC_API_KEY"}[proveedor]
        clave = st.sidebar.text_input(f"API Key · {proveedor}", value=os.getenv(env_key, ""), type="password", key=f"despliegue_key_{proveedor}", help="Se usa solo en esta sesión; no se escribe en archivos.")
        st.sidebar.caption("La API puede generar costos según proveedor, modelo y volumen de tokens.")
    return proveedor, clave, modelo


def _obtener_target() -> str:
    return str(st.session_state.get("target_modelado", st.session_state.get("modelado_target", "")) or "")


def _obtener_resultados() -> dict:
    r = st.session_state.get("modelado_resultados", {})
    return r if isinstance(r, dict) else {}


def _resumen_datos(df: pd.DataFrame | None, prompt: str, max_rows: int = 12) -> dict:
    """Crea evidencia numérica compacta para el LLM desde columnas reales disponibles."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"estado": "No hay DataFrame de contexto cargado."}
    num = df.select_dtypes(include=[np.number])
    resumen: dict[str, Any] = {"filas": int(len(df)), "columnas": int(len(df.columns)), "columnas_disponibles": list(map(str, df.columns[:60]))}
    if not num.empty:
        # Un prompt con cientos de columnas derivadas puede ser demasiado lento para LLM locales.
        keywords_financieras = ("INGRES", "GANANC", "PERDIDA", "PÉRDIDA", "ACTIVO", "PASIVO", "PATRIMON", "MARGEN", "ROA", "ROE", "UTILIDAD", "VENTAS")
        columnas_prioritarias = [c for c in num.columns if any(k in str(c).upper() for k in keywords_financieras)]
        columnas_resumen = list(dict.fromkeys(columnas_prioritarias + list(num.columns[:20])))[:30]
        num_resumen = num[columnas_resumen].replace([np.inf, -np.inf], np.nan)
        resumen["columnas_usadas_en_estadisticos"] = list(map(str, columnas_resumen))
        resumen["estadisticos_numericos"] = num_resumen.describe(percentiles=[.25, .5, .75]).round(4).to_dict()
    prompt_upper = prompt.upper()
    posibles = [c for c in df.columns if any(k in str(c).upper() for k in ["RAZON SOCIAL", "NIT", "MACROSECTOR", "REGION", "DEPARTAMENTO", "CIUDAD"]) ]
    for col in posibles:
        if col in df.columns and any(token in prompt_upper for token in [str(v).upper() for v in df[col].dropna().astype(str).unique()[:500] if len(str(v)) >= 4]):
            vals = df[col].dropna().astype(str).unique()
            matches = [v for v in vals if v.upper() in prompt_upper]
            if matches:
                resumen.setdefault("entidades_detectadas", {})[str(col)] = matches[:10]
    if not num.empty:
        resumen["medianas"] = num_resumen.median(numeric_only=True).round(4).to_dict()
    return resumen


def _grafico_predicciones(df_pred: pd.DataFrame, col_pred: str, anio: int):
    vals = pd.to_numeric(df_pred[col_pred], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        st.info(f"No hay valores numéricos válidos para graficar {anio}.")
        return
    g1, g2 = st.columns(2)
    with g1:
        fig = px.histogram(pd.DataFrame({"Estimación": vals}), x="Estimación", nbins=35, title=f"Distribución de estimaciones {anio}", marginal="box")
        fig.update_layout(height=420, xaxis_title="Valor estimado (unidades del dataset)", yaxis_title="Número de empresas", title_x=0.03)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        qs = vals.quantile([.1, .25, .5, .75, .9]).rename_axis("Percentil").reset_index(name="Valor estimado")
        qs["Percentil"] = qs["Percentil"].map(lambda x: f"P{int(x*100)}")
        fig = px.bar(qs, x="Percentil", y="Valor estimado", title=f"Percentiles de la estimación {anio}", text_auto=".3s")
        fig.update_layout(height=420, xaxis_title="Distribución de empresas", yaxis_title="Valor estimado", title_x=0.03)
        st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empresas estimadas", f"{len(vals):,}")
    c2.metric("Mediana", f"{vals.median():,.3g}")
    c3.metric("P10", f"{vals.quantile(.1):,.3g}")
    c4.metric("P90", f"{vals.quantile(.9):,.3g}")


PREFIJOS_MAGNITUD = ("INGRESOS OPERACIONALES", "GANANCIA (PERDIDA)", "TOTAL ACTIVOS", "TOTAL PASIVOS", "TOTAL PATRIMONIO")
ANIOS_HISTORICOS = (2022, 2023, 2024)
COLS_ID_PROSPECTIVO = ["NIT", "RAZON SOCIAL", "MACROSECTOR", "REGION", "REGIÓN", "SUPERVISOR",
                       "DEPARTAMENTO DOMICILIO", "CIUDAD DOMICILIO"]
COLOR_HISTORICO = "#4C9BE8"
COLOR_ESTIMADO = "#FF9F1C"


def _normalizar(texto: Any) -> str:
    """MAYÚSCULAS sin tildes ni espacios repetidos (para comparar nombres)."""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.upper().split())


def _perfil_target(target: str) -> dict:
    """Unidad, escala y forma de resumir el objetivo seleccionado en Preparación."""
    limpio = str(target).replace("*", "").strip().upper()
    if any(limpio.startswith(p) for p in PREFIJOS_MAGNITUD):
        return {"monetario": True, "aditivo": True, "unidad": "miles de millones de COP", "factor": 1e-6, "dif": "pct"}
    if limpio.endswith("_PCT"):
        return {"monetario": False, "aditivo": False, "unidad": "%", "factor": 1.0, "dif": "pp"}
    return {"monetario": False, "aditivo": False, "unidad": "veces", "factor": 1.0, "dif": "pct"}


def _fmt_valor(v: Any, perfil: dict) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n.d."
    if not np.isfinite(v):
        return "n.d."
    if perfil["monetario"]:
        dec = 2 if abs(v) >= 10 else (3 if abs(v) >= 1 else 4)
    else:
        dec = 2 if perfil["unidad"] == "%" else 3
    return f"{v:,.{dec}f}" + (" %" if perfil["unidad"] == "%" else "")


def _columna_historica(df: pd.DataFrame, target: str, anio: int) -> str | None:
    """Columna del mismo indicador para otro año (p. ej. ROA_2024_PCT → ROA_2023_PCT)."""
    objetivo = str(target).replace("*", "").strip().upper().replace(str(2024), str(anio))
    return next((c for c in df.columns if str(c).replace("*", "").strip().upper() == objetivo), None)


def _prediccion_vigente(anio: int, modelo: str, target: str):
    """Resultados guardados SOLO si corresponden al modelo y objetivo actuales."""
    pred = st.session_state.get(f"predicciones_prospectivas_{anio}")
    meta = st.session_state.get(f"prediccion_prospectiva_metadata_{anio}", {})
    if not isinstance(pred, pd.DataFrame) or pred.empty or "VALOR_REAL_2024" not in pred.columns:
        return None, None, False
    if meta.get("modelo") != modelo or meta.get("target_historico") != target:
        return None, meta, True   # existen, pero son de otro modelo/objetivo
    return pred, meta, False


def _calcular_futuro(anio: int, nombre_modelo: str) -> pd.DataFrame:
    """Calcula 2025 (ventana real 2023-2024) y, si se pide 2026, también lo encadena.

    2026 se estima con la ventana (2024 real, 2025 estimado), por lo que necesita
    la estimación 2025: ambas se calculan y se guardan juntas con el mismo modelo.
    Cada tabla guarda además el histórico real 2022-2024 del objetivo, para poder
    graficar la evolución completa (2022-2026).
    """
    if anio not in (2025, 2026):
        raise ValueError("El horizonte admitido es 2025 o 2026.")
    target = _obtener_target()
    years = re.findall(r"20(?:22|23|24|25|26)", target)
    if not years or int(years[-1]) != 2024:
        raise ValueError("La proyección 2025/2026 requiere un target entrenado para 2024. Seleccione un target 2024 en Preparación de Datos y vuelva a entrenar/evaluar.")
    resultados = _obtener_resultados()
    if nombre_modelo not in resultados or resultados[nombre_modelo].get("estimador") is None:
        raise ValueError("No se encuentra el estimador entrenado en esta sesión. Ejecute Modelado primero.")
    needed = ["df_features_financieras", "prep_pipeline", "outlier_reporte", "prep_feature_selection", "prep_transformaciones"]
    missing = [k for k in needed if st.session_state.get(k) is None]
    if missing:
        raise ValueError("Faltan artefactos de Preparación: " + ", ".join(missing))

    # Integración con el contrato REAL de new_preparacion.py. En ese módulo
    # la función pública se llama preparar_matriz_prospectiva; no existen
    # obtener_df_estandarizado ni proyectar_prospectivo.
    try:
        from new_preparacion import preparar_matriz_prospectiva
    except ImportError as exc:
        raise RuntimeError(
            "No se pudo importar preparar_matriz_prospectiva desde new_preparacion.py. "
            "Verifique que Streamlit esté ejecutando el archivo new_preparacion.py del proyecto actualizado."
        ) from exc

    features = st.session_state["df_features_financieras"].copy()
    pipeline = st.session_state["prep_pipeline"]
    outlier_result = st.session_state["outlier_reporte"]
    fs_result = st.session_state["prep_feature_selection"]
    transformaciones = st.session_state["prep_transformaciones"]
    estimador = resultados[nombre_modelo]["estimador"]

    # 2025: desplazar las variables temporales un año y reutilizar exactamente
    # encoding, selección, imputación y escalado aprendidos en Preparación.
    X_2025 = preparar_matriz_prospectiva(
        features, pipeline, outlier_result, fs_result, transformaciones,
        desplazamiento_anios=1,
    )
    pred_2025 = np.asarray(estimador.predict(X_2025), dtype=float).reshape(-1)
    if len(pred_2025) != len(features):
        raise ValueError("El modelo devolvió una cantidad de predicciones distinta al número de empresas.")
    predicciones = {2025: pred_2025}
    diagnosticos = {2025: {"variables_de_entrada": int(X_2025.shape[1]), "variables_con_huecos_imputados": 0, "pct_celdas_imputadas": 0.0}}

    if anio == 2026:
        # Construir un DataFrame prospectivo 2025 a partir de la tendencia 2022-24.
        # El método es explícito y conservador: 60% del cambio reciente + 40% del
        # cambio anterior, amortiguado al 50% y limitado a +/-50% frente a 2024.
        features_2025 = features.copy()
        cols = list(features.columns)
        for c24 in cols:
            m = re.search(r"20(?:22|23|24)(?:\*)?$", str(c24).strip())
            if not m or int(m.group(0)) != 2024:
                continue
            c22 = next((c for c in cols if str(c).replace("*", "").strip().upper() == str(c24).replace("2024", "2022").replace("*", "").strip().upper()), None)
            c23 = next((c for c in cols if str(c).replace("*", "").strip().upper() == str(c24).replace("2024", "2023").replace("*", "").strip().upper()), None)
            c25 = str(c24)[:-4] + "2025"
            v24 = pd.to_numeric(features[c24], errors="coerce")
            if c22 is not None and c23 is not None:
                v22 = pd.to_numeric(features[c22], errors="coerce")
                v23 = pd.to_numeric(features[c23], errors="coerce")
                delta_recent = v24 - v23
                delta_previous = v23 - v22
                delta = (0.60 * delta_recent + 0.40 * delta_previous) * 0.50
                cap = v24.abs() * 0.50
                delta = delta.clip(lower=-cap, upper=cap)
                estimate = v24 + delta
                # Si la historia no está completa, usar 2024 como escenario plano.
                estimate = estimate.where(v24.notna() & v23.notna() & v22.notna(), v24)
            else:
                estimate = v24
            features_2025[c25] = estimate

        # El target 2025 también se conserva como estimación del modelo para que
        # esté disponible si la matriz recursiva lo necesita como predictor.
        target_2025 = re.sub(r"2024", "2025", target)
        features_2025[target_2025] = pred_2025
        X_2026 = preparar_matriz_prospectiva(
            features_2025, pipeline, outlier_result, fs_result, transformaciones,
            desplazamiento_anios=1,
        )
        pred_2026 = np.asarray(estimador.predict(X_2026), dtype=float).reshape(-1)
        if len(pred_2026) != len(features):
            raise ValueError("El modelo devolvió una cantidad de predicciones 2026 distinta al número de empresas.")
        predicciones[2026] = pred_2026
        diagnosticos[2026] = {"variables_de_entrada": int(X_2026.shape[1]), "variables_con_huecos_imputados": 0, "pct_celdas_imputadas": 0.0}

    salida = {
        "predicciones": predicciones,
        "diagnostico": diagnosticos,
        "supuestos": [
            "2025: inferencia con preparar_matriz_prospectiva de new_preparacion.py; se reutilizan los artefactos aprendidos en TRAIN.",
            "2026: inferencia recursiva con variables 2025 estimadas a partir de la tendencia 2022-2024; cambio ponderado 60/40, amortiguado 50% y limitado a +/-50% del valor 2024.",
            "Estimaciones no verificadas; no hay datos reales posteriores a 2024 para validar los resultados.",
        ],
        "parametros": {"peso_cambio_reciente": 0.60, "amortiguacion_tendencia": 0.50, "limite_cambio_relativo": 0.50},
    }

    # Identificación + histórico real del MISMO indicador (2022, 2023, 2024)
    id_cols = [c for c in COLS_ID_PROSPECTIVO if c in features.columns]
    base = features[id_cols].copy().reset_index(drop=True) if id_cols else pd.DataFrame(index=range(len(features)))
    idx = features.index
    base["CONJUNTO_MODELO"] = np.where(idx.isin(pipeline["y_train"].index), "TRAIN",
                                       np.where(idx.isin(pipeline["y_test"].index), "TEST", "SIN OBJETIVO"))
    for a in ANIOS_HISTORICOS:
        col = _columna_historica(features, target, a)
        base[f"VALOR_REAL_{a}"] = (pd.to_numeric(features[col], errors="coerce").to_numpy(dtype=float)
                                   if col else np.nan)

    m = resultados[nombre_modelo].get("metricas_test", {}) or {}
    calidad = {k: m.get(k) for k in ("MAE", "RMSE", "R²", "MAE / Media (%)")}
    calidad["mejora_mae_vs_baseline_pct"] = resultados[nombre_modelo].get("mejora_mae_vs_baseline_pct")
    evaluacion = st.session_state.get("evaluacion_ultimo_informe") or {}
    estado_eval = (evaluacion.get("estado") if evaluacion.get("target") == target
                   and evaluacion.get("modelo") == resultados[nombre_modelo].get("modelo", nombre_modelo) else None)

    # Iterar únicamente los horizontes calculados en esta ejecución.
    # `anios` no estaba definido y provocaba NameError al construir las tablas.
    anios = sorted(salida["predicciones"].keys())
    for a in anios:
        out = base.copy()
        if a == 2026:
            out["REFERENCIA_ESTIMACION_2025"] = salida["predicciones"][2025]
        target_future = re.sub(r"2024", str(a), target)
        out[f"ESTIMACION_NO_VERIFICADA_{target_future}"] = salida["predicciones"][a]
        out["ANIO_OBJETIVO"] = a
        out["ESTADO"] = ("Estimación encadenada no verificada (2024 real + 2025 estimado)" if a == 2026
                         else "Estimación prospectiva no verificada (ventana real 2023-2024)")
        st.session_state[f"predicciones_prospectivas_{a}"] = out
        st.session_state[f"prediccion_prospectiva_metadata_{a}"] = {
            "target_historico": target, "target_prospectivo": target_future, "modelo": nombre_modelo,
            "anio": a, "horizonte_anios": a - 2024, "estado": "no_verificada", "n_empresas": len(out),
            "fecha_calculo": datetime.now().astimezone().isoformat(timespec="seconds"),
            "metodo": ("Ventana móvil de dos años: 2025 con (2023, 2024) reales; 2026 con (2024 real, 2025 estimado)."),
            "supuestos": salida["supuestos"], "parametros": salida["parametros"],
            "diagnostico": salida["diagnostico"][a], "calidad_modelo": calidad,
            "estado_evaluacion": estado_eval,
        }
    return st.session_state[f"predicciones_prospectivas_{anio}"]


def _limitaciones(anio: int, meta: dict, perfil: dict) -> list[str]:
    """Limitaciones de la estimación, ajustadas al método y a las cifras de este cálculo."""
    cal = meta.get("calidad_modelo", {}) or {}
    diag = meta.get("diagnostico", {}) or {}
    modelo = meta.get("modelo", "el modelo")
    try:
        r2 = f"{float(cal.get('R²')):.3f}"
    except (TypeError, ValueError):
        r2 = "n.d."
    try:
        mae = _fmt_valor(float(cal.get("MAE")) * perfil["factor"], perfil) + ("" if perfil["unidad"] == "%" else f" ({perfil['unidad']})")
    except (TypeError, ValueError):
        mae = "n.d."

    items = [
        f"**Son estimaciones, no datos observados.** El dataset llega hasta 2024; no existen valores reales de {anio} con los cuales verificarlas.",
        "**El modelo aprendió una sola transición histórica** (años 2022-2023 → 2024) y supone que esa relación se mantiene. No incorpora el entorno macroeconómico (inflación, tasa de cambio, tasas de interés), shocks sectoriales, cambios regulatorios ni decisiones o eventos particulares de cada empresa.",
        f"**La calidad se midió solo sobre el pasado.** En el conjunto TEST, {modelo} obtuvo R² = {r2} y un error absoluto medio (MAE) de {mae} por empresa. La validación fue aleatoria entre empresas, no temporal; no demuestra capacidad de acertar años futuros. Las cifras de una empresa individual son menos fiables que las de grupos grandes, aunque estos pueden arrastrar un sesgo sistemático.",
        "**Cifras nominales.** Los datos históricos y las estimaciones están en pesos corrientes, sin ajuste por inflación: las variaciones mezclan crecimiento real e inflación.",
        "**Universo limitado.** La base contiene solo las 7.533 empresas con información financiera completa en 2022, 2023 y 2024 (sesgo de supervivencia). Las sumas y estadísticos de un macrosector, región o supervisor describen a esas empresas, no al sector, la región o la entidad completos.",
        ("**Forma de resumir.** La suma solo se ofrece para magnitudes aditivas (ingresos, ganancia, activos, pasivos, patrimonio). Para indicadores (%, veces) se usa la mediana o el promedio de las empresas, que no equivalen al indicador consolidado del grupo."
         if not perfil["aditivo"] else
         "**Forma de resumir.** La suma, el promedio y la mediana se calculan sobre las empresas filtradas; el promedio es sensible a empresas muy grandes, por lo que la mediana suele ser más representativa."),
        "**Empresas del conjunto TRAIN.** Para las empresas con las que se entrenó el modelo (columna CONJUNTO_MODELO), este ya conoce su valor 2024; su estimación es menos independiente que la de las empresas TEST.",
    ]
    if anio == 2025:
        items.append("**Cálculo de 2025.** Usa las cifras reales de 2023 y 2024 (la misma estructura de dos años con la que se entrenó el modelo). El año 2022 no entra directamente en este cálculo: interviene en el entrenamiento y se muestra como contexto histórico.")
    else:
        pm = meta.get("parametros", {}) or {}
        items += [
            "**Cálculo de 2026 (estimación encadenada).** Usa las cifras reales de 2024 y la estimación de 2025. Como la estimación de 2025 ya es incierta, **el error se acumula**: 2026 hereda el error de 2025 y agrega el suyo. Debe leerse como un escenario orientativo, no como un pronóstico confiable.",
            ("**Supuesto sobre las demás magnitudes.** El modelo estima solo la variable objetivo. Las otras magnitudes de 2025 que necesita el cálculo de 2026 se proyectan con la tendencia 2022-2024 de cada empresa "
             f"(cambio ponderado {float(pm.get('peso_cambio_reciente', 0.6)):.0%} reciente / {1 - float(pm.get('peso_cambio_reciente', 0.6)):.0%} anterior, amortiguado al {float(pm.get('amortiguacion_tendencia', 0.5)):.0%} y limitado a ±{float(pm.get('limite_cambio_relativo', 0.5)):.0%} del valor de 2024). "
             "Es un supuesto del analista, no un modelo validado."),
        ]
        if "indicador derivado" in " ".join(meta.get("supuestos", [])):
            items.append("**Objetivo derivado.** Al ser un indicador (ratio o crecimiento), las cinco magnitudes base de 2025 se proyectan por tendencia y la estimación de 2025 del indicador no realimenta directamente a 2026: la confiabilidad es aún menor.")
    huecos = diag.get("variables_con_huecos_imputados")
    if huecos:
        items.append(f"**Datos imputados.** {int(huecos)} de {int(diag.get('variables_de_entrada', 0))} variables de entrada tuvieron valores faltantes en este cálculo y se rellenaron con la mediana de entrenamiento ({diag.get('pct_celdas_imputadas', 0)} % de las celdas).")
    neg = diag.get("estimaciones_negativas_ajustadas_a_cero")
    if neg:
        items.append(f"**Ajuste a cero.** {int(neg):,} estimaciones resultaron negativas para una magnitud que no puede serlo y se ajustaron a 0.")
    estado_eval = meta.get("estado_evaluacion")
    items.append(f"**Estado de evaluación (Fase 5):** {estado_eval}. Úselo como criterio de gobierno del modelo." if estado_eval
                 else "**Estado de evaluación (Fase 5):** no hay una evaluación formal guardada para este modelo y objetivo en esta sesión; trate las cifras como exploratorias.")
    items.append("**Uso.** Información exploratoria y de apoyo analítico; no constituye asesoría financiera, contable, tributaria ni de inversión.")
    return items


def _filtros_consulta(features: pd.DataFrame) -> dict:
    """Controles de la consulta: se pueden usar por separado o combinados."""
    st.markdown("#### 🔎 Consulta específica")
    st.caption("Escriba una empresa o un NIT, o elija un macrosector, una región o un supervisor. "
               "Puede combinar varios filtros; sin filtros se consideran todas las empresas del dataset.")

    def _opciones(nombre):
        col = next((c for c in features.columns if str(c).replace("*", "").strip().upper() == nombre), None)
        if col is None:
            return None, ["Todos"]
        vals = sorted(features[col].dropna().astype(str).str.strip().unique().tolist(), key=str.casefold)
        return col, ["Todos"] + vals

    c1, c2 = st.columns(2)
    with c1:
        empresa = st.text_input("Empresa (nombre completo o parcial)", key="futuro_empresa_q", placeholder="Ej.: CONSUMER ELECTRONICS")
    with c2:
        nit = st.text_input("NIT (con o sin puntos y dígito de verificación)", key="futuro_nit_q", placeholder="Ej.: 900579611")
    c3, c4, c5 = st.columns(3)
    col_macro, op_macro = _opciones("MACROSECTOR")
    col_region, op_region = _opciones("REGION")
    if col_region is None:
        col_region, op_region = _opciones("REGIÓN")
    col_sup, op_sup = _opciones("SUPERVISOR")
    with c3:
        macro = st.selectbox("Macrosector", op_macro, key="futuro_macro")
    with c4:
        region = st.selectbox("Región", op_region, key="futuro_region")
    with c5:
        sup = st.selectbox("Supervisor", op_sup, key="futuro_supervisor")
    return {"empresa": empresa.strip(), "nit": nit.strip(), "macro": macro, "region": region, "supervisor": sup,
            "cols": {"macro": col_macro, "region": col_region, "supervisor": col_sup}}


def _aplicar_filtros(df: pd.DataFrame, f: dict) -> tuple[pd.DataFrame, list[str]]:
    """Aplica los filtros y devuelve (filas filtradas, descripción de los filtros activos)."""
    mask = pd.Series(True, index=df.index)
    activos: list[str] = []
    col_razon = next((c for c in df.columns if str(c).upper() == "RAZON SOCIAL"), None)
    col_nit = next((c for c in df.columns if str(c).upper() == "NIT"), None)
    if f["empresa"] and col_razon:
        tokens = _normalizar(f["empresa"]).split()
        nombres = df[col_razon].astype(str).map(_normalizar)
        for t in tokens:
            mask &= nombres.str.contains(t, regex=False, na=False)
        activos.append(f"empresa «{f['empresa']}»")
    if f["nit"] and col_nit:
        dig = re.sub(r"\D", "", f["nit"])
        nits = df[col_nit].astype(str).map(lambda x: re.sub(r"\D", "", x))
        m_nit = nits.str.startswith(dig, na=False) if dig else pd.Series(False, index=df.index)
        if dig and not m_nit.any() and len(dig) > 9:   # tolera el dígito de verificación
            m_nit = nits == dig[:-1]
        mask &= m_nit
        activos.append(f"NIT {f['nit']}")
    for clave, etiqueta in (("macro", "macrosector"), ("region", "región"), ("supervisor", "supervisor")):
        col = f["cols"].get(clave)
        if f[clave] != "Todos" and col and col in df.columns:
            mask &= df[col].astype(str).str.strip() == f[clave]
            activos.append(f"{etiqueta} {f[clave]}")
    return df[mask], activos


def _tabla_evolucion(df_f: pd.DataFrame, perfil: dict, estadistico: str, col_est: dict) -> pd.DataFrame:
    """Una fila por año: histórico real (2022-2024) + estimaciones disponibles."""
    factor = perfil["factor"]

    def _stats(serie):
        s = pd.to_numeric(serie, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna() * factor
        if s.empty:
            return {"Empresas con dato": 0, "Suma": np.nan, "Promedio": np.nan, "Mediana": np.nan, "P10": np.nan, "P90": np.nan}
        return {"Empresas con dato": int(len(s)), "Suma": float(s.sum()), "Promedio": float(s.mean()),
                "Mediana": float(s.median()), "P10": float(s.quantile(0.10)), "P90": float(s.quantile(0.90))}

    filas = [{"Año": a, "Tipo": "Histórico (real)", **_stats(df_f[f"VALOR_REAL_{a}"])} for a in ANIOS_HISTORICOS]
    filas += [{"Año": a, "Tipo": "Estimado (no verificado)", **_stats(df_f[c])} for a, c in sorted(col_est.items())]
    evo = pd.DataFrame(filas)
    evo["Valor"] = evo[estadistico]
    previo = evo["Valor"].shift(1)
    if perfil["dif"] == "pp":
        evo["Variación"] = evo["Valor"] - previo
    else:
        evo["Variación"] = (evo["Valor"] - previo) / previo.abs().where(previo.abs() > 0) * 100
    return evo


def _grafico_evolucion(evo: pd.DataFrame, perfil: dict, titulo: str, etiqueta_y: str, clave: str):
    hist = evo[evo["Tipo"].str.startswith("Hist")].dropna(subset=["Valor"])
    est = evo[evo["Tipo"].str.startswith("Estim")].dropna(subset=["Valor"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Año"], y=hist["Valor"], mode="lines+markers+text", name="Histórico (real)",
        line=dict(color=COLOR_HISTORICO, width=3), marker=dict(size=10),
        text=[_fmt_valor(v, perfil) for v in hist["Valor"]], textposition="top center",
        hovertemplate="%{x}: %{y:,.4f}<extra>Histórico real</extra>",
    ))
    if not est.empty and not hist.empty:
        xs = [int(hist["Año"].iloc[-1])] + [int(a) for a in est["Año"]]
        ys = [float(hist["Valor"].iloc[-1])] + [float(v) for v in est["Valor"]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers+text", name="Estimado (no verificado)",
            line=dict(color=COLOR_ESTIMADO, width=3, dash="dash"), marker=dict(size=11, symbol="diamond"),
            text=[""] + [_fmt_valor(v, perfil) for v in est["Valor"]], textposition="top center",
            hovertemplate="%{x}: %{y:,.4f}<extra>Estimación no verificada</extra>",
        ))
        fig.add_vrect(x0=2024.5, x1=max(xs) + 0.4, fillcolor="rgba(255,159,28,0.10)", line_width=0,
                      annotation_text="Proyección", annotation_position="top left")
    todos = sorted(set(int(a) for a in evo["Año"]))
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=16)), height=430, margin=dict(t=70, l=20, r=20, b=40),
        xaxis=dict(tickmode="array", tickvals=todos, title="Año"),
        yaxis=dict(title=etiqueta_y, tickformat=",.2~f"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, key=clave)


def _panel_anio(anio: int, modelo: str, target: str, perfil: dict, estadistico: str, filtros: dict):
    """Botón de cálculo + tabla + gráfica + detalle + limitaciones de UN año."""
    st.markdown(f"### Año {anio}")
    if st.button(f"Calcular estimación {anio}", key=f"despliegue_calcular_{anio}"):
        try:
            with st.spinner(f"Calculando estimaciones {anio} con Python" + (" (encadena la estimación 2025)…" if anio == 2026 else "…")):
                _calcular_futuro(anio, modelo)
            st.success("Estimaciones calculadas" + (" para 2025 y 2026 (2026 se encadena sobre 2025)." if anio == 2026 else "."))
        except Exception as exc:
            st.error(f"No fue posible calcular {anio}: {exc}")
    pred, meta, obsoleta = _prediccion_vigente(anio, modelo, target)
    if pred is None:
        st.info("Los resultados guardados corresponden a otro modelo u objetivo. Pulse «Calcular» para actualizarlos." if obsoleta
                else f"Pulse «Calcular estimación {anio}» para ver la tabla y la gráfica de evolución.")
        return

    col_pred = next((c for c in pred.columns if c.startswith("ESTIMACION_NO_VERIFICADA_")), None)
    col_est = {anio: col_pred}
    if anio == 2026:
        col_est[2025] = "REFERENCIA_ESTIMACION_2025"
    filtrada, activos = _aplicar_filtros(pred, filtros)
    st.caption(f"Empresas que cumplen la consulta: **{len(filtrada):,}** de {len(pred):,}"
               + (f"  ·  Filtros: {'; '.join(activos)}" if activos else "  ·  Sin filtros (todas las empresas)"))
    if filtrada.empty:
        st.warning("Ninguna empresa coincide con los filtros. Cambie o limpie uno o varios criterios.")
        return

    n = len(filtrada)
    etiqueta = "Valor" if n == 1 else estadistico
    evo = _tabla_evolucion(filtrada, perfil, estadistico, col_est)

    # Resumen de la consulta
    if n == 1:
        f0 = filtrada.iloc[0]
        st.markdown(f"**{f0.get('RAZON SOCIAL', 'Empresa')}**  ·  NIT {f0.get('NIT', 'n.d.')}")
        st.caption("  ·  ".join(str(f0[c]) for c in ("MACROSECTOR", "REGION", "SUPERVISOR", "CIUDAD DOMICILIO") if c in filtrada.columns)
                   + f"  ·  Conjunto del modelo: {f0.get('CONJUNTO_MODELO', 'n.d.')}")
    kpi_anios = [2024] + sorted(col_est)
    cols_kpi = st.columns(len(kpi_anios))
    for col, a in zip(cols_kpi, kpi_anios):
        fila = evo[evo["Año"] == a].iloc[0]
        var = fila["Variación"]
        unidad_var = "p.p." if perfil["dif"] == "pp" else "%"
        delta = f"{var:+,.2f} {unidad_var} vs. {a - 1}" if pd.notna(var) else None
        col.metric(f"{'Real' if a == 2024 else 'Estimado'} {a} · {etiqueta}", _fmt_valor(fila["Valor"], perfil), delta=delta, delta_color="off")

    # Tabla de evolución 2022 → año estimado
    st.markdown(f"##### 📋 Evolución {ANIOS_HISTORICOS[0]}–{max(col_est)} · {target.replace('2024', '').replace('*', '').strip()}")
    st.caption(f"Cifras en **{perfil['unidad']}**. Las filas sombreadas son estimaciones no verificadas.")
    cab_var = f"Δ {etiqueta} vs. año anterior ({'p.p.' if perfil['dif'] == 'pp' else '%'})"
    if n == 1:
        mostrar = evo[["Año", "Tipo", "Valor", "Variación"]].rename(columns={"Valor": f"Valor ({perfil['unidad']})", "Variación": cab_var})
        cols_valor = [f"Valor ({perfil['unidad']})"]
    else:
        base_cols = ["Año", "Tipo", "Empresas con dato"] + (["Suma"] if perfil["aditivo"] else []) + ["Promedio", "Mediana", "P10", "P90", "Variación"]
        mostrar = evo[base_cols].rename(columns={"Variación": cab_var})
        cols_valor = [c for c in ("Suma", "Promedio", "Mediana", "P10", "P90") if c in mostrar.columns]
    formatos = {c: (lambda v, p=perfil: _fmt_valor(v, p)) for c in cols_valor}
    formatos[cab_var] = lambda v: "—" if pd.isna(v) else f"{v:+,.2f}"
    if "Empresas con dato" in mostrar.columns:
        formatos["Empresas con dato"] = "{:,.0f}"
    formatos["Año"] = "{:d}"
    sombrear = lambda fila: (["background-color: rgba(255,159,28,0.22)"] * len(fila)) if str(fila["Tipo"]).startswith("Estim") else [""] * len(fila)
    st.dataframe(mostrar.style.format(formatos).apply(sombrear, axis=1), use_container_width=True, hide_index=True)

    # Gráfica de evolución
    _grafico_evolucion(
        evo, perfil,
        f"Evolución {ANIOS_HISTORICOS[0]}–{max(col_est)}: {target.replace('2024', '').replace('*', '').strip()} · {etiqueta}",
        f"{etiqueta} ({perfil['unidad']})", f"despliegue_evolucion_{anio}",
    )

    # Detalle por empresa y distribución (solo cuando hay varias empresas)
    detalle_cols = [c for c in ("RAZON SOCIAL", "NIT", "MACROSECTOR", "REGION", "SUPERVISOR", "CONJUNTO_MODELO") if c in filtrada.columns]
    valor_cols = [f"VALOR_REAL_{a}" for a in ANIOS_HISTORICOS] + ([ "REFERENCIA_ESTIMACION_2025"] if anio == 2026 else []) + [col_pred]
    detalle = filtrada[detalle_cols + valor_cols].copy()
    detalle[valor_cols] = detalle[valor_cols] * perfil["factor"]
    detalle = detalle.rename(columns={c: f"{c} ({perfil['unidad']})" for c in valor_cols})
    if n > 1:
        with st.expander(f"Detalle por empresa — 50 mayores estimaciones {anio}"):
            st.dataframe(detalle.sort_values(f"{col_pred} ({perfil['unidad']})", ascending=False).head(50), use_container_width=True, hide_index=True)
        with st.expander(f"Distribución de las estimaciones {anio} de las empresas filtradas"):
            _grafico_predicciones(filtrada, col_pred, anio)
    st.download_button(f"Descargar detalle de la consulta {anio} (CSV)", filtrada.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"estimaciones_no_verificadas_{anio}_{_slug(target)}.csv", mime="text/csv",
                       key=f"despliegue_download_{anio}")

    with st.expander(f"⚠️ Limitaciones de la estimación {anio}", expanded=True):
        for item in _limitaciones(anio, meta, perfil):
            st.markdown(f"- {item}")
    with st.expander("Metadatos técnicos del cálculo"):
        st.json(meta)


def _tab_prompt(proveedor: str, clave: str, modelo_llm: str):
    st.subheader("💬 Consulta financiera en lenguaje natural")
    st.write("Redacte una pregunta sobre el dataset. Python calcula los indicadores y las predicciones; la API interpreta esa evidencia y redacta la respuesta.")
    df_ctx = st.session_state.get("df_contexto_llm", st.session_state.get("df_consulta", st.session_state.get("df_features_financieras", st.session_state.get("df"))))
    target = _obtener_target()
    resultados = _obtener_resultados()
    modelo_elegido = st.selectbox("Modelo predictivo entrenado", list(resultados.keys()), key="despliegue_modelo_predictivo") if resultados else "Sin modelo entrenado"
    prompt = st.text_area("Escriba su prompt", height=130, placeholder="Ejemplo: Analiza los riesgos financieros de las empresas del macrosector Comercio y compara su distribución de pasivos.", key="despliegue_prompt")
    ejecutar = st.button("Generar análisis financiero", type="primary", key="despliegue_ejecutar_prompt", disabled=not prompt.strip())
    if ejecutar:
        if df_ctx is None or not isinstance(df_ctx, pd.DataFrame):
            st.error("No se encontró un DataFrame de contexto. Cargue los datos y ejecute Preparación de Datos.")
            return
        if proveedor != "Local (Ollama)" and not clave.strip():
            st.error(f"Ingrese la API Key de {proveedor} en la barra lateral para continuar.")
            return
        resumen = _resumen_datos(df_ctx, prompt)
        metricas = st.session_state.get("evaluacion_metricas", st.session_state.get("metricas_evaluacion", {}))
        if not metricas and resultados and modelo_elegido in resultados:
            metricas = {"modelo": modelo_elegido, "target": target, "nota": "Métricas de evaluación no encontradas en session_state."}
        pred_ctx = {}
        if modelo_elegido in resultados:
            pred_ctx = {"modelo_predictivo": modelo_elegido, "target": target, "metricas_disponibles": metricas, "n_features": len(resultados[modelo_elegido].get("features", []))}
        contexto = {"fecha": datetime.now().astimezone().isoformat(timespec="seconds"), "dataset_periodo": "2022-2024", "resumen_dataset": resumen, "modelo_y_metricas": pred_ctx, "restriccion": "No inventar cifras. Si una métrica no aparece en el contexto, declarar que no está disponible. Los valores 2025/2026 son estimaciones, no resultados observados ni verificados."}
        system = (
            "Eres un analista financiero y empresarial sénior especializado en Colombia. "
            "Responde en español profesional, con títulos claros: Resumen ejecutivo; Evidencia cuantitativa; "
            "Interpretación financiera/empresarial; Riesgos y limitaciones; Recomendaciones accionables; Conclusión. "
            "Basa toda cifra en el contexto provisto; no inventes datos, benchmarks, causas ni métricas. "
            "Distingue asociación de causalidad y estimación de dato real. "
            "Explica unidades (ten especial cuidado al interpretar cifras suministradas por el usuario en el prompt, "
            "ya que las cifras del dataset están dadas en miles de pesos COP, es decir, que por ejemplo una cifra que "
            "aparece en el dataset como $100.000.000 ó cien millones COP, en realidad se refiere a $100.000.000.000 "
            "ó cien mil millones COP; DE AHÍ QUE SI por ejemplo un usuario cita en el prompt una cifra de mil millones "
            "COP ó $1.000.000.000, en realidad esta cifra se interpreta para el análisis y a la luz del dataset como "
            "$1.000.000 ó un millón COP) y señala limitaciones del dataset 2022-2024. "
            "Sé específico y comprensible para directivos."
        )
        user_msg = f"PREGUNTA DEL USUARIO:\n{prompt}\n\nCONTEXTO CALCULADO POR PYTHON (JSON):\n{json.dumps(contexto, ensure_ascii=False, default=_serializar)}"
        with st.spinner(f"Consultando {proveedor} y redactando el análisis..."):
            try:
                respuesta = _llamar_api(proveedor, clave.strip(), modelo_llm, system, user_msg)
                st.session_state["despliegue_respuesta_actual"] = respuesta
                st.session_state["despliegue_contexto_actual"] = contexto
                st.session_state["despliegue_prompt_actual"] = prompt
            except Exception as exc:
                st.error(f"No fue posible generar la respuesta de la API: {exc}")
    respuesta = st.session_state.get("despliegue_respuesta_actual")
    if respuesta:
        st.markdown("### Informe generado")
        st.markdown(respuesta)
        st.download_button("Descargar análisis (.md)", respuesta.encode("utf-8"), file_name=f"analisis_financiero_{datetime.now():%Y%m%d_%H%M%S}.md", mime="text/markdown")
        st.caption("Los gráficos siguientes se calculan con los datos disponibles en Python; no son ilustraciones inventadas por el LLM.")
        if isinstance(df_ctx, pd.DataFrame):
            num = df_ctx.select_dtypes(include=[np.number])
            prompt_upper = str(st.session_state.get("despliegue_prompt_actual", "")).upper()
            preferred = [c for c in num.columns if any(w in str(c).upper() for w in ["INGRESOS", "GANANCIA", "PASIVOS", "ACTIVOS", "PATRIMONIO", "MARGEN", "ROA", "ROE"])]
            if preferred:
                col = st.selectbox("Indicador para gráfico de distribución", preferred, key="despliegue_grafico_variable")
                chart_df = pd.DataFrame({"Valor": pd.to_numeric(df_ctx[col], errors="coerce")}).replace([np.inf, -np.inf], np.nan).dropna()
                if not chart_df.empty:
                    fig = px.histogram(chart_df, x="Valor", nbins=35, marginal="box", title=f"Distribución observada: {col}")
                    fig.update_layout(height=430, xaxis_title=f"{col} (unidad original del dataset)", yaxis_title="Número de registros")
                    st.plotly_chart(fig, use_container_width=True)
            cats = [c for c in df_ctx.columns if any(w in str(c).upper() for w in ["MACROSECTOR", "REGION", "REGIÓN", "DEPTO", "DEPARTAMENTO"]) and df_ctx[c].nunique(dropna=True) <= 30]
            if cats and not num.empty:
                cat = st.selectbox("Dimensión de comparación", cats, key="despliegue_grafico_dimension")
                val = st.selectbox("Indicador a comparar", preferred or list(num.columns), key="despliegue_grafico_indicador")
                temp = df_ctx[[cat, val]].copy(); temp[val] = pd.to_numeric(temp[val], errors="coerce"); temp = temp.replace([np.inf, -np.inf], np.nan).dropna()
                if not temp.empty:
                    agg = temp.groupby(cat, dropna=False)[val].median().sort_values(ascending=False).head(15).reset_index()
                    fig = px.bar(agg.sort_values(val), x=val, y=cat, orientation="h", title=f"Mediana de {val} por {cat} (15 categorías principales)", text_auto=".3s")
                    fig.update_layout(height=max(420, min(750, 28*len(agg)+150)), xaxis_title=f"Mediana de {val} (unidad original)", yaxis_title=cat)
                    st.plotly_chart(fig, use_container_width=True)


def _tab_futuro():
    st.subheader("🔮 Proyecciones prospectivas 2025 y 2026")
    st.warning(
        "El dataset observado llega hasta 2024. Las cifras 2025 y 2026 son **estimaciones de modelo, no datos reales ni "
        "predicciones verificadas**. 2025 se calcula con las cifras reales de 2023-2024. 2026 es **encadenado**: usa 2024 real "
        "y la estimación de 2025 (con la tendencia histórica 2022-2024 para las magnitudes que el modelo no estima), "
        "por lo que su incertidumbre es mayor. Consulte las limitaciones de cada año."
    )
    target = _obtener_target()
    if not target:
        st.info("Ejecute Preparación de Datos y seleccione un target antes de calcular proyecciones.")
        return
    st.write(f"**Target seleccionado en Preparación de Datos:** `{target}`")
    resultados = _obtener_resultados()
    if not resultados:
        st.info("Primero ejecute el módulo Modelado y entrene al menos un estimador.")
        return
    features = st.session_state.get("df_features_financieras")
    if not isinstance(features, pd.DataFrame) or features.empty:
        st.error("No se encontró el DataFrame financiero de Preparación de Datos. Vuelva a ejecutar Preparación.")
        return
    perfil = _perfil_target(target)
    modelo = st.selectbox("Estimador para proyección", list(resultados.keys()), key="despliegue_modelo_futuro")

    filtros = _filtros_consulta(features)
    opciones_est = ["Suma", "Mediana", "Promedio"] if perfil["aditivo"] else ["Mediana", "Promedio"]
    estadistico = st.radio(
        "Estadístico para resumir las empresas de la consulta (si es una sola empresa se muestra su valor)",
        opciones_est, horizontal=True, key=f"futuro_estadistico_{_slug(target)}",
    )
    st.divider()
    _panel_anio(2025, modelo, target, perfil, estadistico, filtros)
    st.divider()
    _panel_anio(2026, modelo, target, perfil, estadistico, filtros)


def _tab_artifactos():
    st.subheader("📦 Artefactos de modelo y estado de despliegue")
    target = _obtener_target()
    paquete = st.session_state.get("modelo_ganador")
    resultados = _obtener_resultados()
    c1, c2, c3 = st.columns(3)
    c1.metric("Target preparado", target or "No definido")
    c2.metric("Modelos entrenados en sesión", len(resultados))
    c3.metric("Modelo ganador cargado", "Sí" if paquete else "No")
    carpeta = Path("artifacts_modelado")
    archivos = sorted(carpeta.glob("*.joblib")) if carpeta.exists() else []
    if archivos:
        st.write("Modelos serializados encontrados:")
        st.dataframe(pd.DataFrame([{"Archivo": p.name, "Tamaño (KB)": round(p.stat().st_size/1024, 1), "Modificado": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes")} for p in archivos]), use_container_width=True, hide_index=True)
        seleccion = st.selectbox("Cargar artefacto .joblib", [p.name for p in archivos], key="despliegue_artifacto_joblib")
        if st.button("Cargar modelo seleccionado", key="despliegue_cargar_joblib"):
            try:
                import joblib
                obj = joblib.load(carpeta / seleccion)
                if not isinstance(obj, dict) or obj.get("modelo") is None:
                    raise ValueError("El archivo no contiene el paquete esperado con clave 'modelo'.")
                st.session_state["modelo_ganador"] = obj
                st.success(f"Artefacto cargado: {seleccion}")
                st.json({k: v for k, v in obj.items() if k not in ["modelo"]})
            except Exception as exc:
                st.error(f"No se pudo cargar el artefacto: {exc}")
    else:
        st.info("No hay artefactos .joblib en artifacts_modelado. Entrene y guarde un modelo en el módulo Modelado.")
    st.caption("La API de lenguaje no sustituye el estimador financiero. El estimador debe producir las cifras y la API explica el significado y las limitaciones.")


def mostrar_prediccion(df: pd.DataFrame | None = None):
    """Punto de entrada usado por new_main_streamlit.py."""
    st.header("🚀 Despliegue / Predicción")
    st.caption("CRISP-DM · Fase 6: consumo del modelo, consulta con API/LLM, gráficos y estimaciones prospectivas.")
    proveedor, clave, modelo_llm = _configurar_api()
    tabs = st.tabs(["💬 Prompt financiero + gráficos", "🔮 Predicciones 2025/2026", "📦 Artefactos y gobierno"])
    with tabs[0]: _tab_prompt(proveedor, clave, modelo_llm)
    with tabs[1]: _tab_futuro()
    with tabs[2]: _tab_artifactos()


if __name__ == "__main__":
    st.set_page_config(page_title="Despliegue / Predicción · IA Financiera", layout="wide")
    mostrar_prediccion()
