"""
feature_engineering_proyecto.py
================================
Proyecto: Plataforma Predictiva Financiera Empresarial con IA
Dataset actual: 7.533 empresas colombianas, años 2022, 2023 y 2024.

Este módulo NO depende de los dataframes del proyecto anterior. Trabaja con
las columnas financieras reales del dataset consolidado actual y detecta las
columnas por prefijo + año, tolerando el asterisco de 2023/2024.

CRISP-DM: Preparación de los Datos -> Construct Data / Feature Engineering.

Principio metodológico
----------------------
Todas las transformaciones de este archivo son determinísticas: no calculan
frecuencias, medias, escaladores, importancias ni parámetros aprendidos del
conjunto completo. Por tanto pueden ejecutarse antes de Train/Test.
Las transformaciones que APRENDEN de los datos (Frequency Encoding, OHE,
imputación, escalado y Feature Selection) deben ejecutarse posteriormente y
solo con TRAIN; eso se controla desde new_preparacion.py.

Familias de variables generadas
-------------------------------
1. Ratios financieros: margen, ROA, ROE, endeudamiento, solvencia,
   rotación de activos y apalancamiento.
2. Magnitudes: valores absolutos y log1p de magnitudes financieras.
3. Intensidades: log1p de magnitudes positivas, pérdidas y cambios.
4. Crecimiento: 2022->2023, 2023->2024 y 2022->2024. Los indicadores
   2024 y acumulados son descriptivos; los predictores de modelado se filtran
   después para evitar utilizar resultados contemporáneos de 2024.
5. Cambios de margen y estructura patrimonial.
6. Flags de pérdidas, patrimonio negativo, crecimiento/decrecimiento y ceros.
7. Interacciones financieras y señales de riesgo.
8. Tendencias lineales simples 2022-2024 y volatilidad transversal temporal.

No se agregan variables que requieran información externa ni modelos
entrenados dentro de este módulo.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

YEARS = (2022, 2023, 2024)
EPS = 1e-9

FINANCIAL_PREFIXES = {
    "ING": "INGRESOS OPERACIONALES",
    "GAN": "GANANCIA (PERDIDA)",
    "ACT": "TOTAL ACTIVOS",
    "PAS": "TOTAL PASIVOS",
    "PAT": "TOTAL PATRIMONIO",
}

# Crecimientos extremos son poco informativos para entrenamiento si el
# denominador es casi cero. El crecimiento bruto se conserva en una variable
# separada; las versiones *_CLIP se limitan para las técnicas estadísticas.
GROWTH_CLIP = (-5.0, 5.0)       # -500% a +500%
MARGIN_CLIP = (-10.0, 10.0)     # -1000% a +1000%


def _clean_col(c: object) -> str:
    return str(c).replace("*", "").strip().upper()


def _find_col(df: pd.DataFrame, prefix: str, year: int) -> str | None:
    target = f"{prefix} {year}".upper()
    for c in df.columns:
        if _clean_col(c) == target:
            return c
    return None


def _to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    # Permite números con separadores de miles si aparecen como texto.
    return pd.to_numeric(
        s.astype("string").str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def _safe_div(num: pd.Series, den: pd.Series, positive_den: bool = False) -> pd.Series:
    den2 = den.copy()
    if positive_den:
        den2 = den2.where(den2 > 0)
    else:
        den2 = den2.where(den2 != 0)
    return num / den2


def _add_if_absent(df: pd.DataFrame, name: str, values) -> None:
    if name not in df.columns:
        df[name] = values


def _financial_map(df: pd.DataFrame) -> dict[tuple[str, int], str | None]:
    return {
        (key, year): _find_col(df, prefix, year)
        for key, prefix in FINANCIAL_PREFIXES.items()
        for year in YEARS
    }


def _numeric_financial_columns(df: pd.DataFrame, fmap: dict) -> None:
    for col in fmap.values():
        if col is not None:
            df[col] = _to_num(df[col])


def _growth(df: pd.DataFrame, current: pd.Series, previous: pd.Series) -> tuple[pd.Series, pd.Series]:
    valid = previous.notna() & (previous.abs() > EPS)
    raw = (current - previous) / previous.abs().where(valid)
    clipped = raw.replace([np.inf, -np.inf], np.nan).clip(*GROWTH_CLIP)
    return raw, clipped


def _log_abs(s: pd.Series) -> pd.Series:
    return np.log1p(s.abs())


def _log_positive(s: pd.Series) -> pd.Series:
    return np.log1p(s.clip(lower=0))


def _add_financial_ratios(df: pd.DataFrame, fmap: dict) -> None:
    for year in YEARS:
        ing = df[fmap["ING", year]] if fmap["ING", year] else None
        gan = df[fmap["GAN", year]] if fmap["GAN", year] else None
        act = df[fmap["ACT", year]] if fmap["ACT", year] else None
        pas = df[fmap["PAS", year]] if fmap["PAS", year] else None
        pat = df[fmap["PAT", year]] if fmap["PAT", year] else None
        # No abortar el resto de ratios si falta ingresos: cada indicador depende
        # únicamente de sus propias magnitudes.
        if gan is not None and ing is not None:
            _add_if_absent(df, f"MARGEN_NETO_{year}", _safe_div(gan, ing, True))
            _add_if_absent(df, f"MARGEN_NETO_{year}_PCT", _safe_div(gan, ing, True) * 100.0)
        if act is not None and gan is not None:
            _add_if_absent(df, f"ROA_{year}", _safe_div(gan, act, True))
            _add_if_absent(df, f"ROA_{year}_PCT", _safe_div(gan, act, True) * 100.0)
        if pat is not None and gan is not None:
            _add_if_absent(df, f"ROE_{year}", _safe_div(gan, pat, False))
            _add_if_absent(df, f"ROE_{year}_PCT", _safe_div(gan, pat, True) * 100.0)
        if pas is not None and act is not None:
            _add_if_absent(df, f"ENDEUDAMIENTO_{year}", _safe_div(pas, act, True))
            _add_if_absent(df, f"ENDEUDAMIENTO_{year}_PCT", _safe_div(pas, act, True) * 100.0)
        if pas is not None and pat is not None:
            # Pasivo/patrimonio solo es interpretable como razón convencional con patrimonio positivo.
            # Para patrimonio <= 0 se conserva NaN y se deja constancia en diagnóstico.
            _add_if_absent(df, f"SOLVENCIA_{year}", _safe_div(pas, pat, True))
            _add_if_absent(df, f"SOLVENCIA_{year}_PCT", _safe_div(pas, pat, True) * 100.0)
        if ing is not None and act is not None:
            _add_if_absent(df, f"ROTACION_ACTIVOS_{year}", _safe_div(ing, act, True))
        if act is not None and pat is not None:
            _add_if_absent(df, f"APALANCAMIENTO_{year}", _safe_div(act, pat, True))

        # Intensidades y escalas logarítmicas de magnitudes financieras.
        for key, series, label in (
            ("ING", ing, "INGRESOS"),
            ("GAN", gan, "GANANCIA"),
            ("ACT", act, "ACTIVOS"),
            ("PAS", pas, "PASIVOS"),
            ("PAT", pat, "PATRIMONIO"),
        ):
            if series is None:
                continue
            _add_if_absent(df, f"{label}_ABS_{year}", series.abs())
            _add_if_absent(df, f"{label}_LOG_ABS_{year}", _log_abs(series))
            _add_if_absent(df, f"{label}_LOG_POS_{year}", _log_positive(series))


def _add_profitability_and_flags(df: pd.DataFrame, fmap: dict) -> None:
    for year in YEARS:
        gan_col = fmap["GAN", year]
        ing_col = fmap["ING", year]
        pat_col = fmap["PAT", year]
        act_col = fmap["ACT", year]
        if gan_col:
            gan = df[gan_col]
            _add_if_absent(df, f"ES_PERDIDA_{year}", (gan < 0).astype(np.int8))
            _add_if_absent(df, f"GANANCIA_POSITIVA_{year}", gan.clip(lower=0))
            _add_if_absent(df, f"PERDIDA_NEGATIVA_{year}", gan.clip(upper=0))
            _add_if_absent(df, f"INTENSIDAD_GANANCIA_{year}", _log_positive(gan))
            _add_if_absent(df, f"INTENSIDAD_PERDIDA_{year}", np.log1p(gan.clip(upper=0).abs()))
        if ing_col:
            ing = df[ing_col]
            _add_if_absent(df, f"FLAG_INGRESOS_CERO_{year}", (ing.fillna(0) == 0).astype(np.int8))
            _add_if_absent(df, f"LOG_ABS_INGRESOS_{year}", _log_abs(ing))
        if pat_col:
            pat = df[pat_col]
            _add_if_absent(df, f"ES_PATRIMONIO_NEGATIVO_{year}", (pat < 0).astype(np.int8))
            _add_if_absent(df, f"ABS_PATRIMONIO_{year}", pat.abs())
            _add_if_absent(df, f"LOG_ABS_PATRIMONIO_{year}", _log_abs(pat))
        if act_col:
            act = df[act_col]
            _add_if_absent(df, f"FLAG_ACTIVOS_CERO_{year}", (act.fillna(0) == 0).astype(np.int8))

        # Severidad patrimonial: magnitud del patrimonio negativo en escala log.
        if pat_col:
            neg = (df[pat_col] < 0).astype(float)
            _add_if_absent(df, f"SEVERIDAD_PATRIMONIO_NEGATIVO_{year}", neg * _log_abs(df[pat_col]))
        if act_col and pat_col:
            _add_if_absent(
                df,
                f"INTENSIDAD_PATRIMONIO_SOBRE_ACTIVOS_{year}",
                _safe_div(df[pat_col].abs(), df[act_col], True),
            )


def _add_growth_features(df: pd.DataFrame, fmap: dict) -> None:
    for key, label in (("ING", "INGRESOS"), ("GAN", "GANANCIA"), ("ACT", "ACTIVOS"), ("PAS", "PASIVOS"), ("PAT", "PATRIMONIO")):
        for prev, curr in ((2022, 2023), (2023, 2024), (2022, 2024)):
            c_prev, c_curr = fmap[key, prev], fmap[key, curr]
            if not c_prev or not c_curr:
                continue
            raw, clipped = _growth(df, df[c_curr], df[c_prev])
            suffix = f"{prev}_{curr}"
            _add_if_absent(df, f"CRECIMIENTO_{label}_{suffix}", raw)
            _add_if_absent(df, f"CRECIMIENTO_{label}_{suffix}_CLIP", clipped)
            _add_if_absent(df, f"MAGNITUD_CRECIMIENTO_{label}_{suffix}", np.log1p(clipped.abs()))
            _add_if_absent(df, f"FLAG_CRECIMIENTO_POSITIVO_{label}_{suffix}", (clipped > 0).astype(np.int8))
            _add_if_absent(df, f"FLAG_DECRECIMIENTO_{label}_{suffix}", (clipped < 0).astype(np.int8))

    # Evolución de ratios entre años.
    for ratio in ("MARGEN_NETO", "ROA", "ROE", "ENDEUDAMIENTO", "SOLVENCIA", "ROTACION_ACTIVOS", "APALANCAMIENTO"):
        for prev, curr in ((2022, 2023), (2023, 2024), (2022, 2024)):
            a, b = f"{ratio}_{prev}", f"{ratio}_{curr}"
            if a in df.columns and b in df.columns:
                delta = df[b] - df[a]
                suffix = f"{prev}_{curr}"
                _add_if_absent(df, f"CAMBIO_{ratio}_{suffix}", delta)
                _add_if_absent(df, f"MAGNITUD_CAMBIO_{ratio}_{suffix}", np.log1p(delta.abs()))
                _add_if_absent(df, f"FLAG_MEJORA_{ratio}_{suffix}", (delta > 0).astype(np.int8))
                _add_if_absent(df, f"FLAG_DETERIORO_{ratio}_{suffix}", (delta < 0).astype(np.int8))


def _add_temporal_interactions(df: pd.DataFrame, fmap: dict) -> None:
    # Pérdida consecutiva y cambio de estado.
    if "ES_PERDIDA_2022" in df and "ES_PERDIDA_2023" in df:
        _add_if_absent(df, "PERDIDA_CONSECUTIVA_2022_2023", df["ES_PERDIDA_2022"] * df["ES_PERDIDA_2023"])
        _add_if_absent(df, "CAMBIO_ESTADO_PERDIDA_2022_2023", df["ES_PERDIDA_2023"] - df["ES_PERDIDA_2022"])
    if "ES_PERDIDA_2023" in df and "ES_PERDIDA_2024" in df:
        _add_if_absent(df, "PERDIDA_CONSECUTIVA_2023_2024", df["ES_PERDIDA_2023"] * df["ES_PERDIDA_2024"])
        _add_if_absent(df, "CAMBIO_ESTADO_PERDIDA_2023_2024", df["ES_PERDIDA_2024"] - df["ES_PERDIDA_2023"])

    # Desbalance entre crecimiento de ingresos y activos.
    for a, b in ((2022, 2023), (2023, 2024), (2022, 2024)):
        gi = f"CRECIMIENTO_INGRESOS_{a}_{b}_CLIP"
        ga = f"CRECIMIENTO_ACTIVOS_{a}_{b}_CLIP"
        if gi in df.columns and ga in df.columns:
            x = df[gi] - df[ga]
            _add_if_absent(df, f"DESBALANCE_CRECIMIENTO_{a}_{b}", x)
            _add_if_absent(df, f"MAGNITUD_DESBALANCE_CRECIMIENTO_{a}_{b}", np.log1p(x.abs()))

    # Riesgo patrimonial combinado con endeudamiento.
    for year in YEARS:
        neg = f"ES_PATRIMONIO_NEGATIVO_{year}"
        end = f"ENDEUDAMIENTO_{year}"
        if neg in df.columns and end in df.columns:
            _add_if_absent(df, f"RIESGO_ESTRUCTURAL_{year}", df[neg] * df[end])
        sev = f"SEVERIDAD_PATRIMONIO_NEGATIVO_{year}"
        if sev in df.columns and end in df.columns:
            _add_if_absent(df, f"RIESGO_PATRIMONIAL_AJUSTADO_{year}", df[sev] * (1 + df[end].abs()))

    # Impacto de pérdidas sobre pasivos y complejidad estructural.
    for year in YEARS:
        loss = f"INTENSIDAD_PERDIDA_{year}"
        pas = f"TOTAL PASIVOS {year}"
        act = f"TOTAL ACTIVOS {year}"
        if loss in df.columns and pas in fmap.get(("PAS", year), ""):
            pass
        if loss in df.columns and fmap["PAS", year]:
            pas_log = df[f"TOTAL PASIVOS {year}_LOG_ABS"] if f"TOTAL PASIVOS {year}_LOG_ABS" in df.columns else _log_abs(df[fmap["PAS", year]])
            _add_if_absent(df, f"IMPACTO_PERDIDA_PASIVOS_{year}", df[loss] * pas_log)
        if fmap["ACT", year] and fmap["PAS", year]:
            act_log = df[f"TOTAL ACTIVOS {year}_LOG_ABS"] if f"TOTAL ACTIVOS {year}_LOG_ABS" in df.columns else _log_abs(df[fmap["ACT", year]])
            pas_log = df[f"TOTAL PASIVOS {year}_LOG_ABS"] if f"TOTAL PASIVOS {year}_LOG_ABS" in df.columns else _log_abs(df[fmap["PAS", year]])
            comp = act_log * pas_log
            _add_if_absent(df, f"COMPLEJIDAD_ESTRUCTURAL_ACTIVOS_PASIVOS_{year}", comp)

    # Tendencia lineal simple de cada ratio/magnitud en 2022-2024.
    for base in ("MARGEN_NETO", "ROA", "ROE", "ENDEUDAMIENTO", "SOLVENCIA", "ROTACION_ACTIVOS", "APALANCAMIENTO"):
        cols = [f"{base}_{y}" for y in YEARS]
        if all(c in df.columns for c in cols):
            y = df[cols].to_numpy(dtype=float)
            slope = np.full(len(df), np.nan)
            valid = np.isfinite(y).sum(axis=1) >= 2
            x = np.array([0.0, 1.0, 2.0])
            for i in np.where(valid)[0]:
                yy = y[i]
                mask = np.isfinite(yy)
                slope[i] = np.polyfit(x[mask], yy[mask], 1)[0]
            _add_if_absent(df, f"TENDENCIA_{base}_2022_2024", pd.Series(slope, index=df.index))

    # Volatilidad temporal de ratios.
    for base in ("MARGEN_NETO", "ROA", "ROE", "ENDEUDAMIENTO", "SOLVENCIA"):
        cols = [f"{base}_{y}" for y in YEARS if f"{base}_{y}" in df.columns]
        if len(cols) >= 2:
            _add_if_absent(df, f"VOLATILIDAD_{base}_2022_2024", df[cols].std(axis=1, ddof=0))


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta Feature Engineering descriptivo para la base 2022-2024.

    Este módulo conserva ratios, crecimientos y tendencias de los tres años
    para EDA/fichas financieras. No define el conjunto X del experimento: la
    selección temporal en new_preparacion.py limita X a antecedentes 2022/2023
    y contexto estático cuando el target observado es 2024.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("feature_engineering espera un pandas.DataFrame")

    d = df.copy()
    original_columns = list(d.columns)
    fmap = _financial_map(d)
    _numeric_financial_columns(d, fmap)

    _add_financial_ratios(d, fmap)
    _add_profitability_and_flags(d, fmap)
    _add_growth_features(d, fmap)
    _add_temporal_interactions(d, fmap)

    # Alias explícitos usados por el catálogo de targets del módulo de preparación.
    # Los indicadores en porcentaje se expresan en puntos porcentuales, no como fracción.
    for year in YEARS:
        for base in ("ROA", "ROE", "MARGEN_NETO", "ENDEUDAMIENTO", "SOLVENCIA"):
            source = f"{base}_{year}"
            target_alias = f"{base}_{year}_PCT"
            if source in d.columns and target_alias not in d.columns:
                d[target_alias] = pd.to_numeric(d[source], errors="coerce") * 100.0
        # Crecimiento anual convencional con denominador absoluto; no se recorta el target.
        if year == 2024:
            for key, label in (("ING", "INGRESOS"), ("GAN", "GANANCIA"), ("ACT", "ACTIVOS"), ("PAS", "PASIVOS"), ("PAT", "PATRIMONIO")):
                cprev, ccur = fmap.get((key, 2023)), fmap.get((key, 2024))
                alias = f"CRECIMIENTO_{label}_{year}_PCT"
                if cprev and ccur and alias not in d.columns:
                    prev = pd.to_numeric(d[cprev], errors="coerce")
                    curr = pd.to_numeric(d[ccur], errors="coerce")
                    valid = prev.notna() & (prev.abs() > EPS)
                    d[alias] = ((curr - prev) / prev.abs().where(valid)) * 100.0

    d.replace([np.inf, -np.inf], np.nan, inplace=True)

    new_columns = [c for c in d.columns if c not in original_columns]
    families = {
        "ratios_financieros": [c for c in new_columns if any(x in c for x in ("MARGEN_NETO", "ROA_", "ROE_", "ENDEUDAMIENTO_", "SOLVENCIA_", "ROTACION_ACTIVOS_", "APALANCAMIENTO_"))],
        "magnitudes_log_intensidades": [c for c in new_columns if any(x in c for x in ("LOG_", "ABS_", "INTENSIDAD_", "MAGNITUD_"))],
        "crecimiento": [c for c in new_columns if "CRECIMIENTO_" in c],
        "flags": [c for c in new_columns if c.startswith("FLAG_") or c.startswith("ES_")],
        "riesgo_interacciones": [c for c in new_columns if any(x in c for x in ("RIESGO_", "IMPACTO_", "COMPLEJIDAD_", "DESBALANCE_"))],
        "temporal": [c for c in new_columns if c.startswith("CAMBIO_") or c.startswith("TENDENCIA_") or c.startswith("VOLATILIDAD_") or c.startswith("PERDIDA_CONSECUTIVA")],
    }
    d.attrs["feature_engineering_reporte"] = {
        "shape_entrada": df.shape,
        "shape_salida": d.shape,
        "n_variables_nuevas": len(new_columns),
        "columnas_nuevas": new_columns,
        "familias": {k: len(v) for k, v in families.items()},
        "anios": list(YEARS),
        "fuente_dataset": "df_empresas_2022_2023_2024_consolidado con Ranking.xlsx",
        "deterministico": True,
        "reglas_indicadores_derivados": {
            "porcentajes": "Los nombres *_PCT se expresan multiplicando la razón por 100.",
            "solvencia_y_apalancamiento": "Se devuelven como NaN si el patrimonio es <= 0; no interpretar razones convencionales en esos casos.",
            "crecimientos": "Los objetivos brutos no se recortan; las versiones *_CLIP son solo auxiliares y no deben sustituir silenciosamente al objetivo.",
            "denominadores_pequenos": f"Se consideran no válidos para crecimiento cuando |denominador| <= {EPS}.",
        },
        "limite_temporal": (
            "Las variables derivadas de 2025/2026 no se crean como valores reales. "
            "Para pronosticar esos años se deben proyectar primero las magnitudes "
            "financieras base y recalcular ratios, crecimientos, tendencias e "
            "interacciones usando únicamente valores históricos o estimados disponibles."
        ),
    }
    return d


def get_fe_reporte(df_features: pd.DataFrame) -> dict:
    return df_features.attrs.get("feature_engineering_reporte", {})
