"""
feature_selection_proyecto.py
==============================
Feature Selection para el proyecto financiero actual (7.533 empresas,
2022-2024).

CRISP-DM: Preparación de los Datos -> Feature Selection.

IMPORTANTE
----------
Este módulo NO realiza Train/Test. new_preparacion.py le entrega únicamente
TRAIN. Por tanto correlación, Mutual Information, Random Forest y Permutation
Importance se calculan exclusivamente con TRAIN y no contaminan TEST.

Metodología heredada/adaptada del proyecto anterior:
    1. detección de multicolinealidad
    2. Random Forest Importance
    3. Mutual Information
    4. Permutation Importance
    5. Score Final ponderado
    6. preservación de variables categóricas encoded

Adaptación al dataset actual:
    - 2022, 2023 y 2024
    - columnas reales del consolidado actual
    - prefijos One-Hot de la estructura actual
    - exclusión de identificadores/ranking
    - manejo robusto de NaN, infinitos, columnas constantes y datasets
      pequeños
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance

CORRELATION_THRESHOLD = 0.92
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10
PERMUTATION_REPEATS = 3


def _array_escribible(datos) -> np.ndarray:
    """Crea una matriz NumPy float, contigua y escribible para scikit-learn."""
    # Copia propietaria, contigua y escribible sin mutar flags del array fuente.
    return np.array(datos, dtype=np.float64, order="C", copy=True)

PREFIJOS_CATEGORICOS_PROTEGIDOS = (
    "SECCION_CIIU_",
    "MACROSECTOR_",
    "SUPERVISOR_",
    "REGION_",
    "REGIÓN_",
    "DEPARTAMENTO DOMICILIO_",
    "CIUDAD DOMICILIO_",
    "Punto de entrada_",
    "GRUPO EN NIIF_",
)

VARIABLES_FINANCIERAS_ESTRATEGICAS = (
    "ROA",
    "ROE",
    "MARGEN",
    "SOLVENCIA",
    "ENDEUDAMIENTO",
    "ROTACION",
    "APALANCAMIENTO",
    "ACTIVOS",
    "PASIVOS",
    "PATRIMONIO",
    "INGRESOS",
    "GANANCIA",
    "PERDIDA",
    "CRECIMIENTO",
    "RIESGO",
    "INTENSIDAD",
    "MAGNITUD",
    "TENDENCIA",
    "VOLATILIDAD",
)


def limpiar_inf(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


def obtener_columnas_numericas(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def es_variable_categorica_protegida(col: str) -> bool:
    s = str(col).upper()
    return any(s.startswith(str(p).upper()) for p in PREFIJOS_CATEGORICOS_PROTEGIDOS)


def es_variable_financiera_estrategica(col: str) -> bool:
    s = str(col).upper()
    return any(str(k).upper() in s for k in VARIABLES_FINANCIERAS_ESTRATEGICAS)


def obtener_variables_numericas_analizables(df: pd.DataFrame, target: str) -> list[str]:
    cols = []
    for c in obtener_columnas_numericas(df):
        if c == target:
            cols.append(c)
            continue
        if es_variable_categorica_protegida(c):
            continue
        if str(c).upper().startswith("RANKING"):
            continue
        cols.append(c)
    return cols


def detectar_multicolinealidad(
    df: pd.DataFrame,
    target: str,
    threshold: float = CORRELATION_THRESHOLD,
) -> tuple[list[str], pd.DataFrame]:
    d = limpiar_inf(df)
    cols = obtener_variables_numericas_analizables(d, target)
    cols = [c for c in cols if c in d.columns and d[c].nunique(dropna=True) > 1]
    if len(cols) < 2:
        return [], pd.DataFrame(columns=["Variable_1", "Variable_2", "Correlacion"])

    corr = d[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Conservamos la variable más directamente estratégica; si ninguna lo es,
    # conservamos la primera y eliminamos la segunda.
    eliminar: set[str] = set()
    pares: list[dict] = []
    for col in upper.columns:
        for fila, valor in upper[col].dropna().items():
            if valor <= threshold:
                continue
            pares.append({
                "Variable_1": fila,
                "Variable_2": col,
                "Correlacion": round(float(valor), 6),
            })
            if fila == target or col == target:
                continue
            f_strat = es_variable_financiera_estrategica(fila)
            c_strat = es_variable_financiera_estrategica(col)
            if f_strat and not c_strat:
                eliminar.add(col)
            elif c_strat and not f_strat:
                eliminar.add(fila)
            else:
                # Mantener la variable con mayor variabilidad.
                std_f = d[fila].std(skipna=True)
                std_c = d[col].std(skipna=True)
                if pd.notna(std_f) and pd.notna(std_c) and std_f >= std_c:
                    eliminar.add(col)
                else:
                    eliminar.add(fila)

    return sorted(eliminar), pd.DataFrame(pares)


def _safe_minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def calcular_importancia_variables(
    df: pd.DataFrame,
    target: str,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    d = limpiar_inf(df.copy())
    if target not in d.columns:
        raise ValueError(f"La variable objetivo '{target}' no existe.")

    cols = obtener_variables_numericas_analizables(d, target)
    cols = [c for c in cols if c != target and d[c].nunique(dropna=True) > 1]
    if not cols:
        return pd.DataFrame(columns=["Variable", "RandomForest_Importance", "MutualInformation", "PermutationImportance", "Score_Final"])

    y = pd.to_numeric(d[target], errors="coerce")
    valid_y = y.notna()
    X = d.loc[valid_y, cols].copy()
    y = y.loc[valid_y]

    # Para mantener la interfaz ágil con cientos de features, la estimación
    # de importancias puede trabajar sobre una muestra reproducible de TRAIN.
    # Nunca se utiliza TEST y el límite no cambia la definición de TRAIN.
    IMPORTANCE_SAMPLE_SIZE = 4000
    if len(X) > IMPORTANCE_SAMPLE_SIZE:
        idx = X.sample(IMPORTANCE_SAMPLE_SIZE, random_state=random_state).index
        X = X.loc[idx]
        y = y.loc[idx]

    if len(X) < 10:
        # Para conjuntos extremadamente pequeños, devolvemos un ranking por
        # correlación absoluta como fallback explícito.
        rows = []
        for c in cols:
            corr = abs(pd.concat([X[c], y], axis=1).corr().iloc[0, 1])
            rows.append({
                "Variable": c,
                "RandomForest_Importance": 0.0,
                "MutualInformation": 0.0,
                "PermutationImportance": 0.0,
                "Correlacion_Target": 0.0 if pd.isna(corr) else float(corr),
            })
        out = pd.DataFrame(rows)
        out["Score_Final"] = out["Correlacion_Target"]
        return out.sort_values("Score_Final", ascending=False).reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    X_imp = _array_escribible(imputer.fit_transform(X))
    y_fit = _array_escribible(np.asarray(y).reshape(-1)).reshape(-1)

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(_array_escribible(X_imp), y_fit)

    rf = pd.Series(model.feature_importances_, index=cols, name="RandomForest_Importance")

    try:
        mi_values = mutual_info_regression(_array_escribible(X_imp), _array_escribible(y_fit), random_state=random_state)
    except Exception:
        mi_values = np.zeros(len(cols))
    mi = pd.Series(mi_values, index=cols, name="MutualInformation")

    try:
        perm = permutation_importance(
            model,
            _array_escribible(X_imp),
            _array_escribible(y_fit),
            n_repeats=PERMUTATION_REPEATS,
            random_state=random_state,
            n_jobs=-1,
        )
        perm_values = perm.importances_mean
    except Exception:
        perm_values = np.zeros(len(cols))
    perm_s = pd.Series(perm_values, index=cols, name="PermutationImportance")

    out = pd.concat([rf, mi, perm_s], axis=1).reset_index().rename(columns={"index": "Variable"})
    # Normalización antes de combinar: MI y permutation están en escalas
    # distintas de RF.
    out["RF_Normalizado"] = _safe_minmax(out["RandomForest_Importance"])
    out["MI_Normalizado"] = _safe_minmax(out["MutualInformation"])
    out["Permutation_Normalizado"] = _safe_minmax(out["PermutationImportance"].clip(lower=0))
    out["Score_Final"] = (
        0.50 * out["RF_Normalizado"]
        + 0.20 * out["MI_Normalizado"]
        + 0.30 * out["Permutation_Normalizado"]
    )
    return out.sort_values("Score_Final", ascending=False).reset_index(drop=True)


def feature_selection(
    df: pd.DataFrame,
    target: str,
    correlation_threshold: float = CORRELATION_THRESHOLD,
    top_n: int = 30,
) -> pd.DataFrame:
    """Selecciona variables sobre el DataFrame que debe corresponder a TRAIN."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("feature_selection espera un pandas.DataFrame")
    if target not in df.columns:
        raise ValueError(f"La variable objetivo '{target}' no existe.")
    if top_n < 1:
        raise ValueError("top_n debe ser >= 1")

    original = limpiar_inf(df.copy())
    protected = [c for c in original.columns if es_variable_categorica_protegida(c)]

    to_drop, corr_report = detectar_multicolinealidad(original, target, correlation_threshold)
    reduced = original.drop(columns=[c for c in to_drop if c != target], errors="ignore")

    importance = calcular_importancia_variables(reduced, target)
    top_numeric = importance.head(top_n)["Variable"].tolist() if not importance.empty else []

    # Las categóricas encoded se preservan porque representan información
    # estructural útil para segmentación, explicación y predicción.
    final_cols = []
    for c in top_numeric + protected + [target]:
        if c in reduced.columns and c not in final_cols:
            final_cols.append(c)

    result = reduced[final_cols].copy()
    result.attrs["feature_selection_reporte"] = {
        "shape_original": original.shape,
        "shape_final": result.shape,
        "variables_eliminadas_correlacion": [c for c in to_drop if c in original.columns and c != target],
        "cantidad_eliminadas_correlacion": len([c for c in to_drop if c in original.columns and c != target]),
        "variables_categoricas_preservadas": protected,
        "cantidad_categoricas_preservadas": len(protected),
        "ranking_variables": importance,
        "correlaciones_altas": corr_report,
        "top_n": top_n,
        "correlation_threshold": correlation_threshold,
        "metodologia": "Correlación 0.92 + Random Forest 0.50 + Mutual Information 0.20 + Permutation Importance 0.30",
        "dataset_entrenamiento": "TRAIN",
    }
    return result


def get_feature_selection_reporte(df: pd.DataFrame) -> dict:
    return df.attrs.get("feature_selection_reporte", {})
