"""
MÓDULO: new_crisp_dm_overview.py
ETAPA CRISP-DM: Transversal
PROPÓSITO: Descripción corta y concisa de las seis fases CRISP-DM aplicadas
           a este proyecto, sin referencias al estado de avance de cada
           módulo. Es una presentación metodológica, no un tablero de
           seguimiento.
"""

import streamlit as st


def mostrar_crisp_dm():
    st.header("🔄 Marco CRISP-DM — Metodología del Proyecto")

    st.markdown(
        """
        El proyecto se desarrolla sobre el dataset consolidado
        `df_empresas_2022_2023_2024_consolidado con Ranking.xlsx`: **7.533 empresas colombianas**
        con información financiera completa —Ingresos Operacionales, Ganancia/Pérdida,
        Total Activos, Total Pasivos y Total Patrimonio— en **2022, 2023 y 2024**, junto
        con su posición en el `RANKING 2022 2023` y `RANKING 2023 2024`.

        El **CRISP-DM** (*Cross Industry Standard Process for Data Mining*) organiza el
        desarrollo del modelo predictivo en seis fases iterativas:
        """
    )

    st.markdown("### Fases del Proyecto")

    fases = [
        {
            "numero": "01",
            "emoji": "🏢",
            "titulo": "Comprensión del Negocio",
            "color": "#2a6049",
            "descripcion": (
                "Predecir los indicadores financieros clave (ingresos, ganancia, activos, "
                "pasivos y patrimonio) de las 7.533 empresas colombianas con trayectoria "
                "completa 2022-2024, segmentadas por macrosector y región, y contrastadas "
                "con su posición en el ranking anual por ingresos operacionales."
            ),
        },
        {
            "numero": "02",
            "emoji": "🔍",
            "titulo": "Comprensión de los Datos",
            "color": "#1a4a7a",
            "descripcion": (
                "Exploración del dataset consolidado: estructura, distribución sectorial y "
                "geográfica, comportamiento de las variables financieras a lo largo de los "
                "tres años, y relación con el ranking."
            ),
        },
        {
            "numero": "03",
            "emoji": "🔧",
            "titulo": "Preparación de los Datos",
            "color": "#7a5a1a",
            "descripcion": (
                "Fusión de las fuentes originales por NIT, filtro de completitud financiera "
                "de 3 años, recálculo del ranking sobre el universo final, ingeniería de "
                "variables (ratios financieros, variación interanual), encoding de "
                "categóricas y partición train/test."
            ),
        },
        {
            "numero": "04",
            "emoji": "🤖",
            "titulo": "Modelado",
            "color": "#5a1a6a",
            "descripcion": (
                "Entrenamiento de modelos predictivos (Regresión Lineal, Ridge, Lasso, "
                "Random Forest, XGBoost, Redes Neuronales) sobre el dataset de 7.533 "
                "empresas, con optimización de hiperparámetros."
            ),
        },
        {
            "numero": "05",
            "emoji": "📐",
            "titulo": "Evaluación",
            "color": "#8a1a2a",
            "descripcion": (
                "Medición del desempeño con métricas estándar (R², MAE, RMSE, MAPE), "
                "comparación entre modelos y validación frente a los objetivos definidos "
                "en la Fase 1."
            ),
        },
        {
            "numero": "06",
            "emoji": "🚀",
            "titulo": "Despliegue",
            "color": "#1a6a5a",
            "descripcion": (
                "Predicción interactiva por empresa, sector o región dentro del dashboard "
                "Streamlit, con documentación técnica final del proyecto."
            ),
        },
    ]

    for fase in fases:
        st.markdown(
            f"""
            <div style="background:#f4f6f8; border-left:6px solid {fase['color']};
                        border-radius:6px; padding:16px 20px; margin-bottom:14px;
                        box-shadow:0 1px 4px rgba(0,0,0,0.07);">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:6px;">
                    <span style="font-size:1.4rem; font-weight:800; color:{fase['color']}; min-width:36px;">
                        {fase['numero']}
                    </span>
                    <span style="font-size:1.05rem; font-weight:700; color:#1a1a2e;">
                        {fase['emoji']} {fase['titulo']}
                    </span>
                </div>
                <p style="margin:0 0 0 48px; font-size:0.92rem; color:#2c2c3a; line-height:1.55;">
                    {fase['descripcion']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📊 Dataset del Proyecto")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏢 Empresas", "7,533")
    col2.metric("📅 Años", "2022 - 2023 – 2024")
    col3.metric("📈 Variables financieras", "15 (5 × 3 años)")
    col4.metric("🏷️ Rankings", "2022-2023 · 2023-2024")

