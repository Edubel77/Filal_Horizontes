#?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
#?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

# """
# MÓDULO: new_eda_VII.py
# ETAPA CRISP-DM: Comprensión de los Datos (Data Understanding)
# PROPÓSITO: Pestañas del Análisis Exploratorio de Datos (EDA):
#              · "🧹 Transformación y Limpieza de Datos"
#              · "🏭 Distribución Sectorial"
#              · "🗺️ Distribución Geográfica"   ← nueva en esta versión

#            Respecto a "new_eda_VI.py", este archivo agrega la pestaña de
#            Distribución Geográfica: número de empresas por REGIÓN mediante
#            barras verticales, un Treemap, y un mapa real de Colombia
#            (choropleth por departamento, coloreado según la REGIÓN a la
#            que pertenece cada departamento).

# MAPA DE COLOMBIA — CÓMO SE CONSTRUYE
# ────────────────────────────────────────────────────────────────────────────
# El dataset agrupa los 31 departamentos en 7 REGIONES ya definidas por la
# fuente original (columna "REGION"). No existe un GeoJSON público con las
# fronteras de esas 7 regiones tal cual, pero sí de los 32 departamentos de
# Colombia. Por eso el mapa se construye así:

#   1. Se toma un GeoJSON público de los departamentos de Colombia
#      (`colombia_departamentos.geojson`, debe estar en la misma carpeta).
#   2. Se deriva la relación Departamento → Región directamente del propio
#      dataset (agrupando "DEPARTAMENTO DOMICILIO" por "REGION"): mapeo 1 a 1
#      verificado sobre los 31 departamentos presentes, sin ambigüedades.
#   3. A cada polígono de departamento se le asigna el número total de
#      empresas de SU REGIÓN (no el de su departamento individual) — así,
#      todos los departamentos de una misma región se pintan con el mismo
#      color/valor, mostrando visualmente el número de empresas POR REGIÓN
#      sobre un mapa real de Colombia.
#   4. Los nombres de departamento del GeoJSON no coinciden 100% en texto con
#      los del dataset (p. ej. "VALLE DEL CAUCA" vs "VALLE", o "BOGOTÁ, D.C."
#      vs "BOGOTA D.C."); se normalizan (mayúsculas, sin tildes, sin comas) y
#      se resuelven 2 casos especiales con un diccionario de equivalencias.
#      Guainía y Vaupés no tienen empresas en el dataset y quedan sin color
#      (gris claro) en el mapa — no es un error, es la ausencia real de datos
#      para esos 2 departamentos.

# CONTENIDO DE ESTA PESTAÑA (en orden)
# ────────────────────────────────────────────────────────────────────────────
#   1. Tabla "Tipo de datos originales"        → como llegan desde data_loader_V.
#   2. Tabla "Nuevos tipos de datos asignados"  → tipo correcto por variable.
#   3. Reestructuración de variables:
#        · Texto (RAZÓN SOCIAL, SUPERVISOR, REGIÓN, DEPARTAMENTO DOMICILIO,
#          CIUDAD DOMICILIO, CIIU, MACROSECTOR) → MAYÚSCULAS SIN TILDES.
#        · Variables financieras (15 columnas, 2022-2023-2024) → se deja
#          explícito que son cifras en MILES DE PESOS COLOMBIANOS (COP) y que
#          el signo "-" al inicio de una cifra indica un valor NEGATIVO
#          (pérdida, patrimonio negativo, etc.), no un separador visual.
#   4. Tabla de calidad del dataset ya transformado: Nulos, Campos vacíos y
#      Valores únicos por variable.
#   5. Dos gráficas de barras (% de nulos y % de campos vacíos por variable).
#   6. Vista con formato de moneda (solo para presentación) de las 15
#      variables financieras: signo "$" al frente y separador de miles.

# El DataFrame ya transformado (financieras en int64) se guarda en
# `st.session_state["df_limpio"]` para que las siguientes pestañas del EDA
# y las fases posteriores lo reutilicen sin repetir esta limpieza. La vista
# con formato de moneda es SOLO PARA VISUALIZACIÓN — no reemplaza a
# `df_limpio`, porque convertir las cifras a texto ("$1,234,567") rompería
# cualquier cálculo aritmético posterior (estadística descriptiva, ratios,
# modelado). Esa vista formateada queda disponible en
# `st.session_state["df_formateado"]` por si otra pestaña quiere mostrarla.

# ORIGEN DE LOS VALORES VACÍOS EN "GRUPO EN NIIF" Y "Punto de entrada"
# ────────────────────────────────────────────────────────────────────────────
# Se verificó directamente contra los dos archivos Excel ORIGINALES (antes de
# cualquier fusión o filtro de completitud):

#   · "Punto de entrada" (columna exclusiva del archivo 2023-2024): de las
#     9.000 filas originales, 7 ya llegaban en blanco. De esas, 6 sobreviven
#     en el dataset final de 7.533 empresas (una de las 7 no cumplía el
#     criterio de completitud financiera de 3 años y quedó excluida por esa
#     razón, no por este campo).
#   · "GRUPO EN NIIF" (columna exclusiva del archivo 2022-2023): de las 9.000
#     filas originales, 152 ya llegaban en blanco. De esas, 113 sobreviven en
#     el dataset final.

# Es decir: NO son vacíos introducidos por la fusión de fuentes ni por el
# filtro de completitud de los 3 años (ese filtro solo exige datos completos
# en las 15 variables financieras). Son huecos de captura que YA VENÍAN en el
# archivo original — la entidad que reportó el dato no diligenció esa
# clasificación para esas empresas. Por tratarse de un metadato categórico no
# reportado (no de un valor numérico ausente), se registran en este módulo
# como CAMPOS VACÍOS, no como valores nulos.

# Además, dentro de "Punto de entrada" se detectó un valor atípico: el
# registro numérico "414" en lugar de una de las categorías esperadas
# (GRUPO 1/2/3, "10 NIIF Plenas...", etc.) — probablemente una referencia
# truncada a la Resolución 414 de 2014 mal transcrita en el archivo fuente.
# No se corrige automáticamente (no hay forma de inferir con certeza la
# categoría correcta); queda documentado aquí para que se tenga en cuenta si
# esta columna se usa más adelante para segmentar o modelar.
# """

# import json
# import os
# import unicodedata

# import numpy as np
# import pandas as pd
# import plotly.colors as pc
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st

# # ─── Constantes de columnas ───────────────────────────────────────────────────

# COLUMNAS_RANKING = ["RANKING 2022 2023", "RANKING 2023 2024"]

# COLUMNAS_FINANCIERAS_2024 = [
#     "INGRESOS OPERACIONALES 2024*", "GANANCIA (PERDIDA) 2024",
#     "TOTAL ACTIVOS 2024", "TOTAL PASIVOS 2024", "TOTAL PATRIMONIO 2024",
# ]
# COLUMNAS_FINANCIERAS_2023 = [
#     "INGRESOS OPERACIONALES 2023*", "GANANCIA (PERDIDA) 2023",
#     "TOTAL ACTIVOS 2023", "TOTAL PASIVOS 2023", "TOTAL PATRIMONIO 2023",
# ]
# COLUMNAS_FINANCIERAS_2022 = [
#     "INGRESOS OPERACIONALES 2022", "GANANCIA (PERDIDA) 2022",
#     "TOTAL ACTIVOS 2022", "TOTAL PASIVOS 2022", "TOTAL PATRIMONIO 2022",
# ]
# COLUMNAS_FINANCIERAS = COLUMNAS_FINANCIERAS_2024 + COLUMNAS_FINANCIERAS_2023 + COLUMNAS_FINANCIERAS_2022

# # Texto libre: quedan como object, pero SIN normalizar mayúsculas/tildes
# # (NIT es un identificador numérico convertido a texto; "Punto de entrada"
# # y "GRUPO EN NIIF" son metadatos de la fuente, no descriptores geográficos
# # ni sectoriales).
# # Metadatos de fuente: object, sin normalizar mayúsculas/tildes. Sus NaN de
# # origen (ver docstring del módulo) se registran como CAMPOS VACÍOS, no como
# # nulos, porque representan una categoría no diligenciada en la fuente.
# COLUMNAS_METADATO_FUENTE = ["Punto de entrada", "GRUPO EN NIIF"]

# COLUMNAS_OBJECT_SIN_NORMALIZAR = ["NIT"] + COLUMNAS_METADATO_FUENTE

# # Categóricas descriptivas: object + MAYÚSCULAS SIN TILDES.
# COLUMNAS_MAYUSCULA_SIN_TILDE = [
#     "RAZON SOCIAL", "SUPERVISOR", "REGION", "DEPARTAMENTO DOMICILIO",
#     "CIUDAD DOMICILIO", "CIIU", "MACROSECTOR",
# ]

# COLUMNAS_OBJECT_TOTAL = COLUMNAS_OBJECT_SIN_NORMALIZAR + COLUMNAS_MAYUSCULA_SIN_TILDE

# # Tamaño de fuente mínimo garantizado en todas las gráficas (consistencia
# # tipográfica con el resto del proyecto).
# FONT_SIZE = 14

# # Paleta sobria y de alto contraste para los 6 MACROSECTORES (basada en la
# # paleta Okabe-Ito, diseñada para ser nítida y distinguible incluso para
# # personas con daltonismo). El mismo color identifica siempre al mismo
# # macrosector en todas las gráficas de esta pestaña.
# COLORES_MACROSECTOR = {
#     "COMERCIO":              "#0072B2",  # azul
#     "SERVICIOS":             "#56B4E9",  # azul cielo
#     "MANUFACTURA":           "#CC79A7",  # púrpura
#     "CONSTRUCCION":          "#E69F00",  # naranja
#     "AGROPECUARIO":          "#009E73",  # verde
#     "MINERO-HIDROCARBUROS":  "#D55E00",  # rojo teja
# }

# # Colores fijos para distinguir Promedio vs. Mediana en las gráficas
# # financieras (independientes del color de macrosector).
# COLOR_PROMEDIO = "#1D3557"   # azul marino
# COLOR_MEDIANA = "#9CA3AF"    # gris sobrio

# # Los valores financieros del dataset están en MILES de pesos colombianos.
# # Para expresarlos en MILES DE MILLONES de COP se divide entre 1.000.000.
# DIVISOR_MILES_DE_MILLONES = 1_000_000

# ANIOS_DISPONIBLES = [2022, 2023, 2024]

# # Paleta sobria y distintiva para las 7 REGIONES (colores medios/oscuros
# # para garantizar buen contraste con texto blanco encima).
# COLORES_REGION = {
#     "BOGOTA - CUNDINAMARCA": "#1D3557",  # azul marino
#     "ANTIOQUIA":             "#457B9D",  # azul acero
#     "COSTA PACIFICA":        "#2A9D8F",  # verde azulado
#     "COSTA ATLANTICA":       "#E63946",  # rojo
#     "CENTRO - ORIENTE":      "#6A4C93",  # púrpura
#     "SUR Y ORIENTE":         "#BC6C25",  # ocre/marrón
#     "EJE CAFETERO":          "#606C38",  # oliva oscuro
# }

# # ─── Constantes para el mapa de Colombia (pestaña Distribución Geográfica) ──

# NOMBRE_GEOJSON_DEPARTAMENTOS = "colombia_departamentos.geojson"

# # Equivalencias entre el nombre de departamento del GeoJSON (normalizado:
# # mayúsculas, sin tildes, sin comas) y el nombre usado en el dataset. La
# # gran mayoría de los 31 departamentos coincide directamente tras
# # normalizar; solo estos 2 requieren un ajuste manual.
# EQUIVALENCIAS_DEPARTAMENTO_GEOJSON = {
#     "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "SAN ANDRES Y PROVIDENCIA",
#     "VALLE DEL CAUCA": "VALLE",
# }


# def _layout_base(**kwargs) -> dict:
#     """Layout Plotly con tipografía unificada (mín. 14 pt)."""
#     base = dict(
#         font=dict(size=FONT_SIZE),
#         title_font=dict(size=FONT_SIZE + 2),
#         legend=dict(font=dict(size=FONT_SIZE)),
#         xaxis=dict(title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE)),
#         yaxis=dict(title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE)),
#     )
#     base.update(kwargs)
#     return base


# # ─── Utilidad de normalización de texto ───────────────────────────────────────

# def _quitar_tildes(texto: str) -> str:
#     """Elimina tildes/diacríticos conservando el resto del texto tal cual."""
#     return "".join(
#         c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
#     )


# def _normalizar_mayuscula_sin_tilde(serie: pd.Series) -> pd.Series:
#     """Convierte una serie de texto a MAYÚSCULAS y sin tildes, preservando NaN."""
#     return serie.apply(lambda x: _quitar_tildes(str(x)).upper().strip() if pd.notna(x) else x)


# # ─── 1. Tipos de datos originales ────────────────────────────────────────────

# def _tabla_tipos_originales(df: pd.DataFrame) -> pd.DataFrame:
#     return pd.DataFrame({
#         "Variable": df.columns,
#         "Tipo de dato original": df.dtypes.astype(str).values,
#     }).set_index("Variable")


# # ─── 2. Reasignación de tipos + 3. Reestructuración de variables ────────────

# def _clasificar_tipo_nuevo(col: str) -> str:
#     if col in COLUMNAS_RANKING:
#         return "int64"
#     if col in COLUMNAS_FINANCIERAS:
#         return "int64"
#     if col in COLUMNAS_OBJECT_TOTAL:
#         return "object"
#     return "object"  # cualquier columna no contemplada explícitamente


# def _justificacion_tipo(col: str) -> str:
#     if col in COLUMNAS_RANKING:
#         return "Posición entera dentro del ranking (1..7.533)"
#     if col == "NIT":
#         return "Identificador de empresa: no es una cantidad, no se opera aritméticamente"
#     if col in COLUMNAS_MAYUSCULA_SIN_TILDE:
#         return "Variable categórica/descriptiva de texto (normalizada en mayúsculas sin tildes)"
#     if col in ("Punto de entrada", "GRUPO EN NIIF"):
#         return "Metadato categórico propio de la fuente de origen"
#     if col in COLUMNAS_FINANCIERAS:
#         return "Cifra financiera en miles de COP, sin decimales"
#     return "—"


# def _tabla_tipos_nuevos(df: pd.DataFrame) -> pd.DataFrame:
#     return pd.DataFrame({
#         "Variable": df.columns,
#         "Tipo de dato asignado": [_clasificar_tipo_nuevo(c) for c in df.columns],
#         "Justificación": [_justificacion_tipo(c) for c in df.columns],
#     }).set_index("Variable")


# def _reasignar_y_transformar(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Aplica sobre una copia del DataFrame original:
#       · Ranking y variables financieras → int64.
#       · NIT, "Punto de entrada", "GRUPO EN NIIF" → object (sin normalizar).
#       · Variables categóricas descriptivas → object, MAYÚSCULAS SIN TILDES.
#     """
#     df2 = df.copy()

#     # Ranking → int64
#     for col in COLUMNAS_RANKING:
#         if col in df2.columns:
#             df2[col] = pd.to_numeric(df2[col], errors="coerce").round(0).astype("int64")

#     # Variables financieras → int64 (miles de COP, sin decimales)
#     for col in COLUMNAS_FINANCIERAS:
#         if col in df2.columns:
#             df2[col] = pd.to_numeric(df2[col], errors="coerce").round(0).astype("int64")

#     # NIT → object (identificador, no cantidad)
#     if "NIT" in df2.columns:
#         df2["NIT"] = pd.to_numeric(df2["NIT"], errors="coerce").round(0).astype("Int64").astype(str)

#     # Metadatos de fuente → object, sin normalizar mayúsculas/tildes.
#     # Los NaN de origen se registran como CAMPO VACÍO ("") en vez de nulo:
#     # ver "ORIGEN DE LOS VALORES VACÍOS" en el docstring del módulo.
#     for col in COLUMNAS_METADATO_FUENTE:
#         if col in df2.columns:
#             df2[col] = df2[col].astype(object).where(df2[col].notna(), "")

#     # Categóricas descriptivas → object, MAYÚSCULAS SIN TILDES
#     for col in COLUMNAS_MAYUSCULA_SIN_TILDE:
#         if col in df2.columns:
#             df2[col] = _normalizar_mayuscula_sin_tilde(df2[col].astype(object))

#     return df2


# # ─── 4. Calidad del dataset transformado ─────────────────────────────────────

# def _campos_vacios(serie: pd.Series) -> int:
#     """
#     Cuenta celdas de texto en blanco ('' tras strip) DISTINTAS de NaN.
#     Para columnas numéricas (int64) no aplica el concepto → 0.
#     """
#     if pd.api.types.is_numeric_dtype(serie):
#         return 0
#     return int(serie.dropna().astype(str).str.strip().eq("").sum())


# def _tabla_calidad(df: pd.DataFrame) -> pd.DataFrame:
#     filas = []
#     for col in df.columns:
#         filas.append({
#             "Variable": col,
#             "Nulos": int(df[col].isna().sum()),
#             "Campos vacíos": _campos_vacios(df[col]),
#             "Valores únicos": int(df[col].nunique(dropna=True)),
#         })
#     tabla = pd.DataFrame(filas)
#     tabla["% Nulos"] = (tabla["Nulos"] / len(df) * 100).round(2)
#     tabla["% Campos vacíos"] = (tabla["Campos vacíos"] / len(df) * 100).round(2)
#     return tabla.set_index("Variable")


# # ─── 5. Gráficas de barras (% nulos, % campos vacíos) ────────────────────────

# def _grafico_porcentaje(tabla_calidad: pd.DataFrame, columna_pct: str, titulo: str, color: str):
#     datos = tabla_calidad[tabla_calidad[columna_pct] > 0][columna_pct].sort_values(ascending=False)
#     if datos.empty:
#         st.success(f"✅ No se detectaron {titulo.lower()} en ninguna variable.")
#         return

#     fig = px.bar(
#         x=datos.index, y=datos.values,
#         labels={"x": "Variable", "y": "% del total de registros"},
#         title=titulo,
#         text=datos.values.round(2),
#         color_discrete_sequence=[color],
#     )
#     fig.update_traces(texttemplate="%{text}%", textposition="outside")
#     fig.update_layout(**_layout_base(
#         height=420, showlegend=False,
#         xaxis=dict(tickangle=-35, title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE)),
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# # ─── 6. Vista con formato de moneda (solo presentación) ──────────────────────

# def _formatear_moneda_cop(valor) -> str:
#     """
#     '$' al frente + separador de miles. El signo negativo se deja delante
#     del '$' (p. ej. -$1,234,567) para que sea visualmente inconfundible,
#     consistente con la convención ya explicada de que el "-" indica un
#     valor negativo.
#     """
#     if pd.isna(valor):
#         return ""
#     signo = "-" if valor < 0 else ""
#     return f"{signo}${abs(valor):,.0f}"


# def _construir_vista_formateada(df_limpio: pd.DataFrame) -> pd.DataFrame:
#     """
#     Copia de `df_limpio` en la que las 15 variables financieras se muestran
#     como texto con formato de moneda ($ + separador de miles). Es SOLO PARA
#     VISUALIZACIÓN: al convertir la cifra a texto deja de ser apta para
#     cálculos aritméticos, por eso no reemplaza a `df_limpio`.
#     """
#     df_fmt = df_limpio.copy()
#     for col in COLUMNAS_FINANCIERAS:
#         if col in df_fmt.columns:
#             df_fmt[col] = df_fmt[col].apply(_formatear_moneda_cop)
#     return df_fmt


# # ─── Función principal de la pestaña ──────────────────────────────────────────

# def mostrar_transformacion_limpieza(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Renderiza el contenido completo de la pestaña "Transformación y Limpieza
#     de Datos" y retorna el DataFrame ya transformado (para que otras
#     pestañas/fases lo reutilicen).
#     """
#     st.subheader("🧹 Transformación y Limpieza de Datos")
#     st.markdown(
#         """
#         Antes de explorar el dataset, se documentan los tipos de dato con los que llega
#         el archivo consolidado, se reasigna el tipo correcto a cada variable y se
#         normaliza el texto de las variables categóricas.
#         """
#     )

#     # 1) Tipos originales
#     st.markdown("#### 1️⃣ Tipo de datos originales")
#     st.dataframe(_tabla_tipos_originales(df), use_container_width=True)

#     # 2) Nuevos tipos asignados
#     st.markdown("#### 2️⃣ Nuevos tipos de datos asignados")
#     st.dataframe(_tabla_tipos_nuevos(df), use_container_width=True)

#     # 3) Reestructuración de variables
#     df_limpio = _reasignar_y_transformar(df)

#     st.markdown("#### 3️⃣ Reestructuración de variables")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.markdown("**Variables categóricas → MAYÚSCULAS sin tildes**")
#         ejemplo_cat = df_limpio[COLUMNAS_MAYUSCULA_SIN_TILDE].head(5)
#         st.dataframe(ejemplo_cat, use_container_width=True)
#     with col2:
#         st.markdown("**Variables financieras (2022, 2023 y 2024) → `int64`**")
#         st.info(
#             "💰 Las 15 variables financieras (Ingresos Operacionales, Ganancia/Pérdida, "
#             "Total Activos, Total Pasivos y Total Patrimonio, para cada uno de los 3 años) "
#             "están expresadas en **miles de pesos colombianos (COP)**.\n\n"
#             "El signo **\"-\"** al inicio de una cifra indica un valor **negativo** "
#             "(p. ej. una pérdida en Ganancia/Pérdida, o un patrimonio negativo), "
#             "no es un separador visual ni un dato faltante."
#         )
#         ejemplo_fin = df_limpio[["GANANCIA (PERDIDA) 2024", "TOTAL PATRIMONIO 2024"]].head(5)
#         st.dataframe(ejemplo_fin, use_container_width=True)

#     # 4) Tabla de calidad del dataset transformado
#     st.markdown("---")
#     st.markdown("#### 4️⃣ Calidad del dataset transformado: Nulos, Campos vacíos y Valores únicos")
#     tabla_calidad = _tabla_calidad(df_limpio)
#     st.dataframe(
#         tabla_calidad[["Nulos", "% Nulos", "Campos vacíos", "% Campos vacíos", "Valores únicos"]],
#         use_container_width=True,
#     )

#     total_nulos = int(tabla_calidad["Nulos"].sum())
#     total_vacios = int(tabla_calidad["Campos vacíos"].sum())
#     col_a, col_b, col_c = st.columns(3)
#     col_a.metric("🔴 Total valores nulos", f"{total_nulos:,}")
#     col_b.metric("🟠 Total campos vacíos", f"{total_vacios:,}")
#     col_c.metric("🔵 Registros del dataset", f"{len(df_limpio):,}")

#     with st.expander("ℹ️ ¿De dónde salen los campos vacíos de 'Punto de entrada' y 'GRUPO EN NIIF'?"):
#         st.markdown(
#             """
#             Se verificó directamente contra los dos archivos Excel **originales** (antes de
#             fusionar fuentes o filtrar por completitud financiera):

#             - **`Punto de entrada`** (solo existe en el archivo 2023-2024): de las 9.000 filas
#               originales, **7** ya llegaban en blanco. De esas, **6** sobreviven en este dataset
#               de 7.533 empresas.
#             - **`GRUPO EN NIIF`** (solo existe en el archivo 2022-2023): de las 9.000 filas
#               originales, **152** ya llegaban en blanco. De esas, **113** sobreviven aquí.

#             Es decir: **no** son huecos que se hayan generado al fusionar las dos fuentes ni al
#             aplicar el filtro de completitud de los 3 años (ese filtro solo exige datos completos
#             en las 15 variables financieras). Ya venían vacíos en el archivo original — la entidad
#             que reportó el dato no diligenció esa clasificación para esas empresas. Por tratarse de
#             una categoría no reportada (no de una cifra ausente), aquí se registran como
#             **campos vacíos**, no como valores nulos.

#             **Dato atípico detectado:** dentro de `Punto de entrada` aparece el registro numérico
#             **"414"** en lugar de una de las categorías esperadas (GRUPO 1/2/3, "10 NIIF Plenas...",
#             etc.) — probablemente una referencia truncada a la *Resolución 414 de 2014*, mal
#             transcrita en el archivo fuente. No se corrige automáticamente (no hay forma de inferir
#             con certeza la categoría correcta); queda documentado para tenerlo en cuenta si esta
#             columna se usa más adelante para segmentar o modelar.
#             """
#         )

#     # 5) Gráficas de % nulos y % campos vacíos
#     st.markdown("#### 5️⃣ Porcentaje de Nulos y de Campos Vacíos por Variable")
#     col_g1, col_g2 = st.columns(2)
#     with col_g1:
#         _grafico_porcentaje(
#             tabla_calidad, "% Nulos",
#             "Porcentaje de Valores Nulos por Variable", "#c0392b",
#         )
#     with col_g2:
#         _grafico_porcentaje(
#             tabla_calidad, "% Campos vacíos",
#             "Porcentaje de Campos Vacíos por Variable", "#e08e0b",
#         )

#     # 6) Vista con formato de moneda (solo presentación)
#     st.markdown("---")
#     st.markdown("#### 6️⃣ Vista con Formato de Moneda (Miles de COP)")
#     st.caption(
#         "Únicamente para presentación: los valores se muestran como texto "
#         "(\"$1,234,567\") con el signo negativo delante del \"$\" cuando aplica. "
#         "El dataset que usan las siguientes fases (`df_limpio`) conserva las "
#         "15 variables en `int64` para poder seguir operando con ellas."
#     )
#     df_formateado = _construir_vista_formateada(df_limpio)
#     columnas_vista = ["NIT", "RAZON SOCIAL"] + COLUMNAS_FINANCIERAS_2024
#     st.dataframe(df_formateado[columnas_vista].head(10), use_container_width=True)
#     with st.expander("Ver las 15 variables financieras con formato de moneda completas"):
#         st.dataframe(
#             df_formateado[["NIT", "RAZON SOCIAL"] + COLUMNAS_FINANCIERAS].head(20),
#             use_container_width=True,
#         )

#     st.session_state["df_limpio"] = df_limpio
#     st.session_state["df_formateado"] = df_formateado
#     return df_limpio


# # ═══════════════════════════════════════════════════════════════════════════
# # PESTAÑA: 🏭 Distribución Sectorial
# # ═══════════════════════════════════════════════════════════════════════════

# def _orden_macrosectores(df: pd.DataFrame) -> list[str]:
#     """
#     Orden fijo de los macrosectores (de mayor a menor número de empresas),
#     reutilizado en TODAS las gráficas de esta pestaña para que la lectura
#     sea consistente de una gráfica a otra.
#     """
#     return df["MACROSECTOR"].value_counts().index.tolist()


# # ─── 1. Conteo de empresas por macrosector (barras horizontales) ────────────

# def _grafico_conteo_macrosector(df: pd.DataFrame, orden: list[str]):
#     conteo = df["MACROSECTOR"].value_counts().reindex(orden)

#     fig = go.Figure(
#         go.Bar(
#             x=conteo.values,
#             y=conteo.index,
#             orientation="h",
#             marker_color=[COLORES_MACROSECTOR.get(m, "#888888") for m in conteo.index],
#             text=[f"{v:,}" for v in conteo.values],
#             textposition="inside",
#             insidetextanchor="end",
#             textfont=dict(size=20, color="white", weight="bold"),
#         )
#     )
#     fig.update_layout(**_layout_base(
#         title="Número de Empresas por Macrosector",
#         xaxis_title="Número de empresas",
#         yaxis_title=None,
#         height=430,
#         margin=dict(l=10, r=60, t=60, b=40),
#         yaxis=dict(autorange="reversed", tickfont=dict(size=FONT_SIZE + 1)),
#         xaxis=dict(tickfont=dict(size=FONT_SIZE)),
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# # ─── 2. Porcentaje de empresas por macrosector (circular) ────────────────────

# def _grafico_pie_macrosector(df: pd.DataFrame, orden: list[str]):
#     conteo = df["MACROSECTOR"].value_counts().reindex(orden)

#     fig = go.Figure(
#         go.Pie(
#             labels=conteo.index,
#             values=conteo.values,
#             hole=0.42,
#             marker=dict(
#                 colors=[COLORES_MACROSECTOR.get(m, "#888888") for m in conteo.index],
#                 line=dict(color="white", width=2),
#             ),
#             textinfo="percent",
#             textfont=dict(size=20, color="white", weight="bold"),
#             insidetextorientation="horizontal",
#             hovertemplate="%{label}<br>%{value:,} empresas (%{percent})<extra></extra>",
#         )
#     )
#     fig.update_layout(**_layout_base(
#         title="Participación (%) de Empresas por Macrosector",
#         height=430,
#         margin=dict(l=10, r=10, t=60, b=10),
#         legend=dict(font=dict(size=FONT_SIZE), orientation="v", yanchor="middle", y=0.5),
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# # ─── 3/4. Promedio y mediana por macrosector, por año (barras agrupadas) ────

# def _resumen_promedio_mediana(df: pd.DataFrame, columna: str, orden: list[str]) -> pd.DataFrame:
#     resumen = (
#         df.groupby("MACROSECTOR")[columna]
#         .agg(Promedio="mean", Mediana="median")
#         .reindex(orden)
#         / DIVISOR_MILES_DE_MILLONES
#     )
#     return resumen.reset_index()


# def _grafico_promedio_mediana(df: pd.DataFrame, columna: str, anio: int, variable_label: str, orden: list[str]):
#     resumen = _resumen_promedio_mediana(df, columna, orden)
#     resumen_largo = resumen.melt(
#         id_vars="MACROSECTOR", value_vars=["Promedio", "Mediana"],
#         var_name="Métrica", value_name="Valor",
#     )

#     fig = px.bar(
#         resumen_largo, x="MACROSECTOR", y="Valor", color="Métrica",
#         barmode="group",
#         category_orders={"MACROSECTOR": orden},
#         color_discrete_map={"Promedio": COLOR_PROMEDIO, "Mediana": COLOR_MEDIANA},
#         text=resumen_largo["Valor"].map(lambda v: f"{v:,.1f}"),
#         title=f"{variable_label} por Macrosector — {anio}",
#     )
#     fig.update_traces(textposition="outside", textfont=dict(size=13))
#     fig.update_layout(**_layout_base(
#         height=440,
#         xaxis_title="Macrosector",
#         yaxis_title="Miles de Millones COP",
#         xaxis=dict(tickangle=-15, tickfont=dict(size=FONT_SIZE)),
#         legend=dict(title=None, font=dict(size=FONT_SIZE), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#         margin=dict(t=80),
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# # ─── Gráfica de evolución (comparación entre los 3 años) ────────────────────

# def _preparar_evolucion(df: pd.DataFrame, columnas_por_anio: dict[int, str], orden: list[str]) -> pd.DataFrame:
#     filas = []
#     for anio, columna in columnas_por_anio.items():
#         resumen = _resumen_promedio_mediana(df, columna, orden)
#         resumen["Año"] = anio
#         filas.append(resumen)
#     return pd.concat(filas, ignore_index=True)


# def _grafico_evolucion(df_evolucion: pd.DataFrame, metrica: str, variable_label: str, orden: list[str]):
#     fig = go.Figure()
#     for macrosector in orden:
#         datos = df_evolucion[df_evolucion["MACROSECTOR"] == macrosector].sort_values("Año")
#         fig.add_trace(go.Scatter(
#             x=datos["Año"], y=datos[metrica],
#             mode="lines+markers+text",
#             name=macrosector,
#             line=dict(color=COLORES_MACROSECTOR.get(macrosector, "#888888"), width=3),
#             marker=dict(size=9),
#             text=[f"{v:,.1f}" for v in datos[metrica]],
#             textposition="top center",
#             textfont=dict(size=11),
#         ))

#     fig.update_layout(**_layout_base(
#         title=f"Evolución del {metrica} de {variable_label} por Macrosector (2022–2024)",
#         xaxis_title="Año",
#         yaxis_title="Miles de Millones COP",
#         height=460,
#         xaxis=dict(tickmode="array", tickvals=[2022, 2023, 2024], tickfont=dict(size=FONT_SIZE)),
#         legend=dict(font=dict(size=FONT_SIZE), orientation="v", yanchor="middle", y=0.5, x=1.02),
#         margin=dict(r=160),
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# # ─── Bloque reutilizable: 3 pestañas por año + gráfica de evolución ─────────

# def _bloque_variable_financiera(
#     df: pd.DataFrame,
#     columnas_por_anio: dict[int, str],
#     variable_label: str,
#     orden: list[str],
#     key_prefix: str,
# ):
#     st.markdown(f"##### {variable_label} por Macrosector, año a año")
#     tabs_anio = st.tabs([str(a) for a in ANIOS_DISPONIBLES])
#     for tab, anio in zip(tabs_anio, ANIOS_DISPONIBLES):
#         with tab:
#             _grafico_promedio_mediana(df, columnas_por_anio[anio], anio, variable_label, orden)

#     st.markdown(f"##### Evolución de {variable_label} — comparación 2022-2023-2024")
#     metrica = st.radio(
#         "Métrica a comparar en la evolución:",
#         ["Promedio", "Mediana"],
#         horizontal=True,
#         key=f"{key_prefix}_metrica_evolucion",
#     )
#     df_evolucion = _preparar_evolucion(df, columnas_por_anio, orden)
#     _grafico_evolucion(df_evolucion, metrica, variable_label, orden)


# # ─── Función principal de la pestaña ──────────────────────────────────────────

# def mostrar_distribucion_sectorial(df: pd.DataFrame):
#     """
#     Renderiza la pestaña "🏭 Distribución Sectorial" del EDA: composición del
#     dataset por macrosector (conteo y participación %), y el comportamiento
#     de Ingresos Operacionales y Ganancia/Pérdida por macrosector a lo largo
#     de los tres años (2022, 2023, 2024), con una gráfica de evolución que
#     permite comparar los tres años entre sí.
#     """
#     st.subheader("🏭 Distribución Sectorial")
#     st.markdown(
#         "Composición del dataset de **7.533 empresas** por macrosector, y el comportamiento "
#         "de sus principales variables financieras —Ingresos Operacionales y Ganancia/Pérdida— "
#         "en miles de millones de COP, a lo largo de 2022, 2023 y 2024."
#     )

#     orden = _orden_macrosectores(df)

#     st.markdown("#### 1️⃣ ¿Cuántas empresas hay en cada Macrosector?")
#     col1, col2 = st.columns(2)
#     with col1:
#         _grafico_conteo_macrosector(df, orden)
#     with col2:
#         _grafico_pie_macrosector(df, orden)

#     st.markdown("---")
#     st.markdown("#### 2️⃣ Ingresos Operacionales por Macrosector")
#     _bloque_variable_financiera(
#         df,
#         columnas_por_anio={
#             2022: "INGRESOS OPERACIONALES 2022",
#             2023: "INGRESOS OPERACIONALES 2023*",
#             2024: "INGRESOS OPERACIONALES 2024*",
#         },
#         variable_label="Ingresos Operacionales",
#         orden=orden,
#         key_prefix="ingresos",
#     )

#     st.markdown("---")
#     st.markdown("#### 3️⃣ Ganancia (Pérdida) por Macrosector")
#     _bloque_variable_financiera(
#         df,
#         columnas_por_anio={
#             2022: "GANANCIA (PERDIDA) 2022",
#             2023: "GANANCIA (PERDIDA) 2023",
#             2024: "GANANCIA (PERDIDA) 2024",
#         },
#         variable_label="Ganancia (Pérdida)",
#         orden=orden,
#         key_prefix="ganancia",
#     )


# # ═══════════════════════════════════════════════════════════════════════════
# # PESTAÑA: 🗺️ Distribución Geográfica
# # ═══════════════════════════════════════════════════════════════════════════

# def _conteo_por_region(df: pd.DataFrame) -> pd.Series:
#     """Número de empresas por región, ordenado de mayor a menor."""
#     return df["REGION"].value_counts()


# # ─── 1. Barras verticales: empresas por región ──────────────────────────────

# def _grafico_barras_region(conteo: pd.Series):
#     fig = go.Figure(
#         go.Bar(
#             x=conteo.index,
#             y=conteo.values,
#             marker_color=[COLORES_REGION.get(r, "#888888") for r in conteo.index],
#             text=[f"{v:,}" for v in conteo.values],
#             textposition="inside",
#             insidetextanchor="end",
#             textfont=dict(size=20, color="white", weight="bold"),
#         )
#     )
#     fig.update_layout(**_layout_base(
#         title="Número de Empresas por Región",
#         xaxis_title="Región",
#         yaxis_title="Número de empresas",
#         height=520,
#         xaxis=dict(tickangle=-20, tickfont=dict(size=FONT_SIZE + 2)),
#         yaxis=dict(tickfont=dict(size=FONT_SIZE + 1)),
#         title_font=dict(size=FONT_SIZE + 6),
#         margin=dict(t=70, b=90),
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# # ─── 2. Treemap: empresas por región ─────────────────────────────────────────

# def _grafico_treemap_region(conteo: pd.Series):
#     colores = [COLORES_REGION.get(r, "#888888") for r in conteo.index]
#     fig = go.Figure(
#         go.Treemap(
#             labels=conteo.index,
#             parents=[""] * len(conteo),
#             values=conteo.values,
#             marker=dict(
#                 colors=colores,
#                 line=dict(color="white", width=3),
#                 pad=dict(t=6, l=6, r=6, b=6),
#             ),
#             text=[f"{v:,} empresas" for v in conteo.values],
#             texttemplate="<b>%{label}</b><br>%{text}",
#             textfont=dict(size=22, color="white", weight="bold"),
#             textposition="middle center",
#             hovertemplate="%{label}<br>%{value:,} empresas<extra></extra>",
#         )
#     )
#     fig.update_layout(**_layout_base(
#         title="Participación de Empresas por Región (Treemap)",
#         title_font=dict(size=FONT_SIZE + 6),
#         height=520,
#         margin=dict(t=70, l=10, r=10, b=10),
#         uniformtext=dict(minsize=15, mode="show"),
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# # ─── 3. Heatmap: empresas por Región × Macrosector ──────────────────────────

# def _grafico_heatmap_region_macrosector(df: pd.DataFrame, orden_region: list[str], orden_macrosector: list[str]):
#     tabla = pd.crosstab(df["REGION"], df["MACROSECTOR"]).reindex(index=orden_region, columns=orden_macrosector).fillna(0).astype(int)

#     fig = go.Figure(
#         go.Heatmap(
#             z=tabla.values,
#             x=tabla.columns,
#             y=tabla.index,
#             colorscale="Blues",
#             colorbar=dict(title="Empresas", tickfont=dict(size=FONT_SIZE)),
#             xgap=3, ygap=3,
#             hovertemplate="Región: %{y}<br>Macrosector: %{x}<br>Empresas: %{z:,}<extra></extra>",
#         )
#     )

#     # Color del texto dinámico: blanco sobre celdas oscuras, oscuro sobre
#     # celdas claras, para que el número SIEMPRE sea legible.
#     umbral = tabla.values.max() * 0.5
#     anotaciones = []
#     for i, region in enumerate(tabla.index):
#         for j, sector in enumerate(tabla.columns):
#             valor = tabla.values[i, j]
#             color_texto = "white" if valor > umbral else "#1a1a1a"
#             anotaciones.append(dict(
#                 x=sector, y=region, text=f"{valor:,}",
#                 showarrow=False,
#                 font=dict(size=17, color=color_texto, weight="bold"),
#             ))

#     fig.update_layout(**_layout_base(
#         title="Número de Empresas por Región y Macrosector",
#         title_font=dict(size=FONT_SIZE + 6),
#         height=520,
#         xaxis=dict(title="Macrosector", tickangle=-15, tickfont=dict(size=FONT_SIZE + 1), side="bottom"),
#         yaxis=dict(title="Región", tickfont=dict(size=FONT_SIZE + 1)),
#         margin=dict(t=70, l=10, r=10, b=90),
#         annotations=anotaciones,
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# # ─── 4. Mapas de Colombia: por Departamento y por Región ────────────────────

# def _quitar_tildes_geo(texto: str) -> str:
#     return "".join(
#         c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
#     )


# def _normalizar_nombre_departamento(nombre: str) -> str:
#     limpio = _quitar_tildes_geo(nombre).upper().strip().replace(",", "")
#     limpio = " ".join(limpio.split())
#     return EQUIVALENCIAS_DEPARTAMENTO_GEOJSON.get(limpio, limpio)


# def _resolver_ruta_geojson() -> str:
#     """Busca el GeoJSON de departamentos en las mismas ubicaciones donde
#     `data_loader_V.py` busca el archivo consolidado."""
#     dir_modulo  = os.path.dirname(os.path.abspath(__file__))
#     dir_padre   = os.path.dirname(dir_modulo)
#     dir_trabajo = os.getcwd()

#     candidatos = [
#         os.path.join(dir_modulo,  NOMBRE_GEOJSON_DEPARTAMENTOS),
#         os.path.join(dir_padre,   NOMBRE_GEOJSON_DEPARTAMENTOS),
#         os.path.join(dir_trabajo, NOMBRE_GEOJSON_DEPARTAMENTOS),
#     ]
#     for ruta in candidatos:
#         if os.path.isfile(ruta):
#             return ruta

#     rutas_str = "\n  • ".join(candidatos)
#     raise FileNotFoundError(
#         f"No se encontró '{NOMBRE_GEOJSON_DEPARTAMENTOS}' en ninguna de las rutas buscadas:\n  • {rutas_str}"
#     )


# @st.cache_data
# def _cargar_geojson_departamentos() -> dict:
#     with open(_resolver_ruta_geojson(), encoding="utf-8") as f:
#         return json.load(f)


# def _construir_tabla_departamentos(df: pd.DataFrame, geojson: dict) -> pd.DataFrame:
#     """
#     Una fila por CADA departamento del GeoJSON (los 33, incluidos Guainía,
#     Vaupés y el archipiélago de San Andrés), con su número de empresas
#     individual y la región del dataset a la que pertenece. Si el
#     departamento no tiene ninguna empresa registrada, no hay región que
#     inferir y se marca explícitamente como sin registro.
#     """
#     conteo_depto = df["DEPARTAMENTO DOMICILIO"].value_counts().to_dict()
#     mapa_depto_region = (
#         df.groupby("DEPARTAMENTO DOMICILIO")["REGION"]
#         .agg(lambda s: s.value_counts().index[0])
#         .to_dict()
#     )

#     filas = []
#     for feature in geojson["features"]:
#         nombre_geojson = feature["properties"]["DPTO_CNMBR"]
#         nombre_normalizado = _normalizar_nombre_departamento(nombre_geojson)
#         empresas = conteo_depto.get(nombre_normalizado)
#         region = mapa_depto_region.get(nombre_normalizado)

#         if empresas is None:
#             hover = f"<b>{nombre_geojson}</b><br>No hay registro de empresas para este departamento."
#         else:
#             hover = f"<b>{nombre_geojson}</b><br>Región: {region}<br>{empresas:,} empresas"

#         filas.append({
#             "Departamento_GeoJSON": nombre_geojson,
#             "REGION": region,
#             "Empresas": empresas if empresas is not None else 0,
#             "Hover": hover,
#         })
#     return pd.DataFrame(filas)


# def _construir_tabla_regiones(df: pd.DataFrame, geojson: dict) -> pd.DataFrame:
#     """
#     Una fila por CADA departamento del GeoJSON, pero coloreada según el
#     total de empresas de SU REGIÓN (no del departamento individual), para
#     visualizar la distribución por región sobre el mapa real de Colombia.
#     """
#     mapa_depto_region = (
#         df.groupby("DEPARTAMENTO DOMICILIO")["REGION"]
#         .agg(lambda s: s.value_counts().index[0])
#         .to_dict()
#     )
#     conteo_region = _conteo_por_region(df).to_dict()

#     filas = []
#     for feature in geojson["features"]:
#         nombre_geojson = feature["properties"]["DPTO_CNMBR"]
#         nombre_normalizado = _normalizar_nombre_departamento(nombre_geojson)
#         region = mapa_depto_region.get(nombre_normalizado)
#         empresas = conteo_region.get(region) if region else None

#         if empresas is None:
#             hover = f"<b>{nombre_geojson}</b><br>No hay registro de empresas para este departamento."
#         else:
#             hover = f"<b>{nombre_geojson}</b><br>Región: {region}<br>{empresas:,} empresas en la región"

#         filas.append({
#             "Departamento_GeoJSON": nombre_geojson,
#             "REGION": region,
#             "Empresas": empresas,
#             "Hover": hover,
#         })
#     return pd.DataFrame(filas)


# # Color "divergente" (contrastante frente a la escala secuencial de datos)
# # para señalar departamentos SIN empresas registradas.
# COLOR_SIN_REGISTRO = "#4B5563"

# # Coordenadas aproximadas del archipiélago de San Andrés y Providencia,
# # usadas para superponer un círculo grande y legible (la isla es minúscula
# # frente al resto del país y, a escala nacional, su polígono real es casi
# # invisible sin este refuerzo visual).
# COORD_SAN_ANDRES = (12.5847, -81.7006)


# def _color_para_valor(colorscale, valor: float, zmin: float, zmax: float) -> str:
#     """
#     Devuelve el color exacto que le correspondería a `valor` dentro de la
#     misma escala de colores (colorscale) y el mismo rango (zmin/zmax) que
#     usa el mapa — para que el círculo de San Andrés se pinte con el color
#     que de verdad representa su cifra, no un color arbitrario.
#     """
#     frac = 0.0 if zmax == zmin else (valor - zmin) / (zmax - zmin)
#     frac = min(max(frac, 0.0), 1.0)
#     return pc.sample_colorscale(colorscale, [frac])[0]


# def _agregar_marcador_san_andres(fig: go.Figure, tabla: pd.DataFrame,
#                                   colorscale, zmin: float, zmax: float, valor_color: float):
#     """
#     Superpone un círculo grande sobre el archipiélago de San Andrés y
#     Providencia. El relleno se colorea según `valor_color` (el número de
#     empresas de la Región Sur y Oriente, a la que pertenece San Andrés),
#     evaluado en la MISMA escala de colores y rango que usa el mapa — así el
#     círculo se ve en el mismo tono "bajo" en ambos mapas. El texto emergente
#     (hover) sigue mostrando la cifra real y específica de ese mapa. No
#     modifica el tamaño ni el encuadre del mapa principal.
#     """
#     fila = tabla[tabla["Departamento_GeoJSON"].str.contains("SAN ANDR", case=False, na=False)]
#     if fila.empty:
#         return
#     hover = fila.iloc[0]["Hover"]
#     color_relleno = _color_para_valor(colorscale, valor_color, zmin, zmax)
#     lat, lon = COORD_SAN_ANDRES

#     fig.add_trace(go.Scattergeo(
#         lat=[lat], lon=[lon],
#         mode="markers+text",
#         marker=dict(size=34, color=color_relleno, line=dict(color="#1a1a1a", width=2.5)),
#         text=["<b>San Andrés y<br>Providencia</b>"],
#         textposition="top center",
#         textfont=dict(size=14, color="#1a1a1a", weight="bold"),
#         hovertext=[hover],
#         hovertemplate="%{hovertext}<extra></extra>",
#         showlegend=False,
#     ))


# def _grafico_mapa_departamento(df: pd.DataFrame, geojson: dict):
#     tabla = _construir_tabla_departamentos(df, geojson)
#     colores_borde = [COLORES_REGION.get(r, "#BBBBBB") for r in tabla["REGION"]]

#     fig = go.Figure(
#         go.Choropleth(
#             geojson=geojson,
#             locations=tabla["Departamento_GeoJSON"],
#             z=tabla["Empresas"],
#             featureidkey="properties.DPTO_CNMBR",
#             colorscale="YlOrRd",
#             marker=dict(line=dict(color=colores_borde, width=1.8)),
#             colorbar_title="Empresas",
#             hovertext=tabla["Hover"],
#             hovertemplate="%{hovertext}<extra></extra>",
#         )
#     )
#     fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
#     conteo_sur_oriente = df["REGION"].value_counts().get("SUR Y ORIENTE", 0)
#     _agregar_marcador_san_andres(
#         fig, tabla, colorscale="YlOrRd",
#         zmin=tabla["Empresas"].min(), zmax=tabla["Empresas"].max(),
#         valor_color=conteo_sur_oriente,
#     )
#     fig.update_layout(**_layout_base(
#         title="Empresas por Departamento",
#         title_font=dict(size=FONT_SIZE + 4),
#         height=600,
#         margin=dict(t=60, l=10, r=10, b=10),
#     ))
#     st.plotly_chart(fig, use_container_width=True)
#     st.caption(
#         "El color de relleno representa el número de empresas de cada departamento. "
#         "El borde de color indica, de forma tenue, la REGIÓN a la que pertenece — al pasar "
#         "el cursor (o tocar) un departamento se muestra su región y su cifra exacta; los "
#         "departamentos sin empresas registradas lo indican explícitamente. San Andrés y "
#         "Providencia se muestra con un círculo (su polígono real es casi invisible a esta "
#         "escala), coloreado según su propia cifra de empresas."
#     )


# def _grafico_mapa_region(df: pd.DataFrame, geojson: dict):
#     tabla = _construir_tabla_regiones(df, geojson)
#     tiene_datos = tabla["Empresas"].notna()
#     tabla_con_datos = tabla[tiene_datos]
#     tabla_sin_datos = tabla[~tiene_datos]

#     fig = go.Figure(
#         go.Choropleth(
#             geojson=geojson,
#             locations=tabla_con_datos["Departamento_GeoJSON"],
#             z=tabla_con_datos["Empresas"],
#             featureidkey="properties.DPTO_CNMBR",
#             colorscale="Blues",
#             marker_line_color="white",
#             marker_line_width=1.2,
#             colorbar_title="Empresas",
#             hovertext=tabla_con_datos["Hover"],
#             hovertemplate="%{hovertext}<extra></extra>",
#         )
#     )

#     # Guainía y Vaupés no pertenecen a ninguna región del dataset (no tienen
#     # empresas registradas): se pintan en un color divergente y sólido para
#     # que sigan siendo visibles en el mapa, distinguiéndolos claramente de
#     # la escala de azules usada para las regiones con datos.
#     if len(tabla_sin_datos) > 0:
#         fig.add_trace(go.Choropleth(
#             geojson=geojson,
#             locations=tabla_sin_datos["Departamento_GeoJSON"],
#             z=[1] * len(tabla_sin_datos),
#             featureidkey="properties.DPTO_CNMBR",
#             colorscale=[[0, COLOR_SIN_REGISTRO], [1, COLOR_SIN_REGISTRO]],
#             showscale=False,
#             marker_line_color="white",
#             marker_line_width=1.2,
#             hovertext=tabla_sin_datos["Hover"],
#             hovertemplate="%{hovertext}<extra></extra>",
#             name="Sin registro de empresas",
#             showlegend=True,
#         ))

#     fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
#     conteo_sur_oriente = df["REGION"].value_counts().get("SUR Y ORIENTE", 0)
#     _agregar_marcador_san_andres(
#         fig, tabla, colorscale="Blues",
#         zmin=tabla_con_datos["Empresas"].min(), zmax=tabla_con_datos["Empresas"].max(),
#         valor_color=conteo_sur_oriente,
#     )
#     fig.update_layout(**_layout_base(
#         title="Empresas por Región",
#         title_font=dict(size=FONT_SIZE + 4),
#         height=600,
#         margin=dict(t=60, l=10, r=10, b=10),
#         legend=dict(font=dict(size=FONT_SIZE, color="#1a1a1a"), yanchor="bottom", y=0.02, xanchor="left", x=0.02,
#                     bgcolor="rgba(255,255,255,0.92)", bordercolor="#9CA3AF", borderwidth=1),
#     ))
#     st.plotly_chart(fig, use_container_width=True)
#     st.caption(
#         "Cada departamento se colorea según el total de empresas de la REGIÓN a la que "
#         "pertenece (no de su departamento individual) — todos los departamentos de una "
#         "misma región comparten color. Guainía y Vaupés se muestran en gris porque no "
#         "tienen empresas registradas en el dataset (no pertenecen a ninguna región). San "
#         "Andrés y Providencia se muestra con un círculo, coloreado según la cifra de su "
#         "región (Sur y Oriente)."
#     )


# def _nota_departamentos_sur_oriente():
#     """Nota grande y legible, requerida explícitamente por el usuario,
#     debajo de los dos mapas de Colombia."""
#     st.markdown(
#         """
#         <div style="background:#fff4e5; border-left:6px solid #BC6C25; border-radius:6px;
#                     padding:18px 22px; margin-top:20px; font-size:1.08rem; line-height:1.65;
#                     color:#3a2a12;">
#         📌 <b>Nota metodológica:</b> en el dataset original, los departamentos de
#         <b>Chocó</b>, <b>La Guajira</b>, <b>Cesar</b>, <b>Tolima</b> y <b>San Andrés y Providencia</b> aparecen clasificados
#         dentro del listado de empresas de la región <b>"Sur y Oriente"</b>, además de los departamantos propios de esta región.
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


# def _grafico_mapas_colombia(df: pd.DataFrame):
#     try:
#         geojson = _cargar_geojson_departamentos()
#     except FileNotFoundError as e:
#         st.warning(
#             f"⚠️ No se pudo cargar el mapa: {e}\n\n"
#             f"Copia el archivo `{NOMBRE_GEOJSON_DEPARTAMENTOS}` a la misma carpeta del proyecto "
#             f"para ver los mapas de Colombia."
#         )
#         return

#     col_mapa1, col_mapa2 = st.columns(2)
#     with col_mapa1:
#         _grafico_mapa_departamento(df, geojson)
#     with col_mapa2:
#         _grafico_mapa_region(df, geojson)

#     _nota_departamentos_sur_oriente()


# # ─── Función principal de la pestaña ──────────────────────────────────────────

# def mostrar_distribucion_geografica(df: pd.DataFrame):
#     """
#     Renderiza la pestaña "🗺️ Distribución Geográfica" del EDA: número de
#     empresas por región mediante barras verticales, un Treemap, un heatmap
#     cruzando Región × Macrosector, y dos mapas reales de Colombia (por
#     departamento y por región).
#     """
#     st.subheader("🗺️ Distribución Geográfica")
#     st.markdown(
#         "Distribución de las **7.533 empresas** del dataset entre las **7 regiones** "
#         "en las que se agrupan los 31 departamentos de domicilio."
#     )

#     conteo = _conteo_por_region(df)
#     orden_region = conteo.index.tolist()
#     orden_macrosector = _orden_macrosectores(df)

#     st.markdown("#### 1️⃣ Empresas por Región")
#     _grafico_barras_region(conteo)

#     st.markdown("#### 2️⃣ Participación por Región (Treemap)")
#     _grafico_treemap_region(conteo)

#     st.markdown("#### 3️⃣ Empresas por Región y Macrosector")
#     _grafico_heatmap_region_macrosector(df, orden_region, orden_macrosector)

#     st.markdown("#### 4️⃣ Mapas de Colombia")
#     _grafico_mapas_colombia(df)


# # ═══════════════════════════════════════════════════════════════════════════
# # PESTAÑA: 💰 Variables Financieras
# # ═══════════════════════════════════════════════════════════════════════════

# # Variable financiera (etiqueta) → columna correspondiente en cada año.
# VARIABLES_FINANCIERAS = {
#     "Ingresos Operacionales": {
#         2022: "INGRESOS OPERACIONALES 2022",
#         2023: "INGRESOS OPERACIONALES 2023*",
#         2024: "INGRESOS OPERACIONALES 2024*",
#     },
#     "Ganancia (Pérdida)": {
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

# # Color sobrio y distinto por año, reutilizado en el histograma y en la
# # gráfica de comparación entre años.
# COLOR_ANIO = {2022: "#264653", 2023: "#2A9D8F", 2024: "#E76F51"}

# # Rangos FIJOS de cada barra del histograma, en Miles de Millones COP,
# # iguales para los 3 años (así los histogramas y la gráfica de comparación
# # son directamente comparables entre 2022, 2023 y 2024). El primer borde de
# # Ingresos Operacionales y de Total Activos coincide con el valor mínimo
# # real de esa variable en el dataset; el resto de bordes son múltiplos
# # exactos del ancho de barra indicado.
# VARIABLE_BINS = {
#     "Ingresos Operacionales": [0.004] + list(range(10, 510, 10)),
#     # "Ganancia (Pérdida)": list(range(-20, 51, 5)),
#     "Ganancia (Pérdida)": [-250] + list(range(-225, 400, 5)),
#     # "Total Activos": [1] + list(range(10, 701, 10)),
#     "Total Activos": [0.06] + list(range(5, 1000, 5)),
#     "Total Pasivos": list(range(0, 1000, 5)),
#     # "Total Patrimonio": list(range(-50, 1001, 50)),
#     "Total Patrimonio": [-250] + list(range(-225, 400, 5))
# }


# def _serie_en_miles_de_millones(df: pd.DataFrame, columna: str) -> pd.Series:
#     return (df[columna] / DIVISOR_MILES_DE_MILLONES).dropna()


# def _contar_por_bins(serie: pd.Series, bordes: list[float]) -> pd.DataFrame:
#     """
#     Cuenta cuántas empresas caen en cada rango fijo (`bordes`, límites de
#     barra en Miles de Millones COP). Retorna una fila por rango, con su
#     etiqueta ("19 - 30"), su punto medio (para graficar) y el conteo.
#     """
#     conteos, _ = np.histogram(serie, bins=bordes)
#     etiquetas = [f"{bordes[i]:,} - {bordes[i + 1]:,}" for i in range(len(bordes) - 1)]
#     medios = [(bordes[i] + bordes[i + 1]) / 2 for i in range(len(bordes) - 1)]
#     return pd.DataFrame({"Rango": etiquetas, "Medio": medios, "Empresas": conteos})


# # ─── 1. Histograma: distribución de una variable en un año ──────────────────

# def _grafico_histograma(df: pd.DataFrame, variable_label: str, anio: int):
#     columna = VARIABLES_FINANCIERAS[variable_label][anio]
#     serie = _serie_en_miles_de_millones(df, columna)
#     bordes = VARIABLE_BINS[variable_label]
#     tabla = _contar_por_bins(serie, bordes)

#     fig = go.Figure(
#         go.Bar(
#             x=tabla["Medio"],
#             y=tabla["Empresas"],
#             width=[bordes[i + 1] - bordes[i] for i in range(len(bordes) - 1)],
#             marker_color=COLOR_ANIO[anio],
#             marker_line=dict(color="white", width=0.6),
#             customdata=tabla["Rango"],
#             hovertemplate=(
#                 f"<b>{variable_label} — {anio}</b><br>"
#                 "Rango: %{customdata} Miles de Millones COP<br>"
#                 "Empresas: %{y:,}<extra></extra>"
#             ),
#         )
#     )
#     fig.update_layout(**_layout_base(
#         title=f"Distribución de {variable_label} — {anio}",
#         title_font=dict(size=FONT_SIZE + 4),
#         xaxis_title="Miles de Millones COP",
#         yaxis_title="Número de empresas",
#         height=460,
#         xaxis=dict(tickfont=dict(size=FONT_SIZE)),
#         yaxis=dict(tickfont=dict(size=FONT_SIZE)),
#         bargap=0.02,
#         showlegend=False,
#     ))
#     st.plotly_chart(fig, use_container_width=True)

#     incluidas = int(tabla["Empresas"].sum())
#     total = len(serie)
#     if incluidas < total:
#         st.caption(
#             f"⚠️ Rango fijo de {variable_label} entre {bordes[0]:,} y {bordes[-1]:,} Miles de "
#             f"Millones COP: incluye {incluidas:,} de {total:,} empresas — las {total - incluidas:,} "
#             f"restantes tienen valores fuera de este rango (por lo general, empresas atípicamente "
#             f"grandes o pequeñas) y no aparecen en esta gráfica."
#         )
#     else:
#         st.caption(
#             f"Rango fijo de {variable_label} entre {bordes[0]:,} y {bordes[-1]:,} Miles de Millones "
#             f"COP, igual en los tres años, para que las gráficas sean directamente comparables."
#         )


# # ─── 2. Comparación entre años (misma escala del histograma) ────────────────

# def _grafico_comparacion_anios(df: pd.DataFrame, variable_label: str):
#     bordes = VARIABLE_BINS[variable_label]

#     fig = go.Figure()
#     for anio in ANIOS_DISPONIBLES:
#         serie = _serie_en_miles_de_millones(df, VARIABLES_FINANCIERAS[variable_label][anio])
#         tabla = _contar_por_bins(serie, bordes)
#         fig.add_trace(go.Scatter(
#             x=tabla["Medio"], y=tabla["Empresas"],
#             mode="lines+markers",
#             name=str(anio),
#             line=dict(color=COLOR_ANIO[anio], width=3.5, shape="spline"),
#             marker=dict(size=6),
#             customdata=tabla["Rango"],
#             hovertemplate=(
#                 f"<b>{anio}</b><br>Rango: %{{customdata}} Miles de Millones COP<br>"
#                 "Empresas: %{y:,}<extra></extra>"
#             ),
#         ))

#     fig.update_layout(**_layout_base(
#         title=f"Comparación entre Años — {variable_label}",
#         title_font=dict(size=FONT_SIZE + 4),
#         xaxis_title="Miles de Millones COP",
#         yaxis_title="Número de empresas",  # misma escala/unidad que el histograma
#         height=460,
#         legend=dict(title="Año", font=dict(size=FONT_SIZE)),
#         xaxis=dict(tickfont=dict(size=FONT_SIZE)),
#         yaxis=dict(tickfont=dict(size=FONT_SIZE)),
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# # ─── 3/4. Box plots por Macrosector y por Región ─────────────────────────────

# def _rango_y_ajustado_por_bigotes(df_plot: pd.DataFrame, columna_grupo: str, orden: list[str]) -> tuple[float, float] | None:
#     """
#     Calcula un rango de eje Y ajustado a los bigotes (fences) reales de cada
#     grupo -en vez del mínimo/máximo global, que queda dominado por unos
#     pocos outliers dispersos-, para que las cajas (Q1-Q3, mediana) se vean
#     grandes y legibles. Los puntos que caigan fuera de este rango simplemente
#     no se muestran, pero siguen formando parte del cálculo de la caja.
#     """
#     limites_inf, limites_sup = [], []
#     for grupo in orden:
#         datos = df_plot.loc[df_plot[columna_grupo] == grupo, "Valor"].dropna()
#         if len(datos) < 4:
#             continue
#         q1, q3 = datos.quantile(0.25), datos.quantile(0.75)
#         iqr = q3 - q1
#         limites_inf.append(max(datos.min(), q1 - 1.5 * iqr))
#         limites_sup.append(min(datos.max(), q3 + 1.5 * iqr))

#     if not limites_inf:
#         return None

#     minimo, maximo = min(limites_inf), max(limites_sup)
#     margen = (maximo - minimo) * 0.15
#     return minimo - margen, maximo + margen


# def _grafico_boxplot(df: pd.DataFrame, variable_label: str, anio: int,
#                       columna_grupo: str, orden: list[str], paleta: dict, titulo_eje_x: str):
#     columna_valor = VARIABLES_FINANCIERAS[variable_label][anio]
#     bordes = VARIABLE_BINS[variable_label]
#     limite_inferior, limite_superior = bordes[0], bordes[-1]

#     df_plot = df[[columna_grupo, columna_valor]].copy()
#     df_plot["Valor"] = df_plot[columna_valor] / DIVISOR_MILES_DE_MILLONES

#     total = len(df_plot)
#     # Mismo rango fijo que el histograma de esta variable: los valores fuera
#     # de este rango (outliers extremos) se EXCLUYEN por completo del cálculo
#     # de la caja — no solo de la vista — para que Q1, Q3, mediana y los
#     # bigotes (fences) se dibujen a una escala legible en todos los casos.
#     df_plot = df_plot[(df_plot["Valor"] >= limite_inferior) & (df_plot["Valor"] <= limite_superior)]
#     incluidas = len(df_plot)

#     fig = go.Figure()
#     for grupo in orden:
#         datos_grupo = df_plot.loc[df_plot[columna_grupo] == grupo, "Valor"]
#         color = paleta.get(grupo, "#888888")
#         fig.add_trace(go.Box(
#             y=datos_grupo,
#             name=grupo,
#             boxpoints="outliers",          # solo los outliers DENTRO del rango fijo, no todos los puntos
#             marker=dict(color=color, size=5, opacity=0.65),
#             line=dict(color=color, width=2),
#             fillcolor=color,
#             opacity=0.55,
#         ))

#     # Para "Ganancia (Pérdida)" el mínimo/máximo global queda dominado por
#     # unos pocos outliers dispersos (una empresa con pérdida o ganancia
#     # atípicamente grande), lo que aplasta visualmente las cajas de todos
#     # los demás grupos. Se acota el eje Y a los bigotes reales para que las
#     # cajas se vean más amplias y legibles; los outliers que caigan fuera
#     # de este rango ajustado simplemente no se dibujan (pero siguen
#     # formando parte del cálculo de la caja).
#     yaxis_config = dict(tickfont=dict(size=FONT_SIZE))
#     if variable_label == "Ganancia (Pérdida)":
#         rango_ajustado = _rango_y_ajustado_por_bigotes(df_plot, columna_grupo, orden)
#         if rango_ajustado is not None:
#             yaxis_config["range"] = list(rango_ajustado)

#     fig.update_layout(**_layout_base(
#         title=f"{variable_label} por {titulo_eje_x} — {anio}",
#         title_font=dict(size=FONT_SIZE + 4),
#         xaxis_title=titulo_eje_x,
#         yaxis_title="Miles de Millones COP",
#         height=1000,
#         showlegend=False,
#         xaxis=dict(tickangle=-15, tickfont=dict(size=FONT_SIZE + 1)),
#         yaxis=yaxis_config,
#     ))
#     st.plotly_chart(fig, use_container_width=True)

#     if variable_label == "Ganancia (Pérdida)":
#         st.caption(
#             "🔎 El eje Y de esta gráfica se acota a los bigotes (fences) reales de cada caja "
#             "para que se vean grandes y legibles — algunos outliers puntuales de ganancia o "
#             "pérdida extrema quedan fuera de la vista, pero siguen incluidos en el cálculo de "
#             "cada caja (Q1, Q3, mediana)."
#         )

#     if incluidas < total:
#         st.caption(
#             f"⚠️ Se aplica el mismo rango fijo del histograma ({limite_inferior:,} a "
#             f"{limite_superior:,} Miles de Millones COP): incluye {incluidas:,} de {total:,} "
#             f"empresas — las {total - incluidas:,} restantes son outliers extremos fuera de "
#             f"ese rango y no se tienen en cuenta en esta gráfica."
#         )


# # ─── Función principal de la pestaña ──────────────────────────────────────────

# def mostrar_variables_financieras(df: pd.DataFrame):
#     """
#     Renderiza la pestaña "💰 Variables Financieras" del EDA: histograma y
#     comparación entre años para la variable seleccionada, y dos box plots
#     (por Macrosector y por Región) para el año y la variable seleccionados.
#     """
#     st.subheader("💰 Variables Financieras")
#     st.markdown(
#         "Distribución de las 5 variables financieras clave —Ingresos Operacionales, "
#         "Ganancia/Pérdida, Total Activos, Total Pasivos y Total Patrimonio— en miles de "
#         "millones de COP, para 2022, 2023 y 2024."
#     )

#     col_sel1, col_sel2 = st.columns([2, 1])
#     with col_sel1:
#         variable_label = st.selectbox(
#             "Variable financiera:", list(VARIABLES_FINANCIERAS.keys()), key="fin_variable",
#         )
#     with col_sel2:
#         anio = st.radio("Año:", ANIOS_DISPONIBLES, horizontal=True, key="fin_anio")

#     orden_macrosector = _orden_macrosectores(df)
#     orden_region = _conteo_por_region(df).index.tolist()

#     st.markdown("#### 1️⃣ Distribución del Número de Empresas")
#     col_h1, col_h2 = st.columns(2)
#     with col_h1:
#         _grafico_histograma(df, variable_label, anio)
#     with col_h2:
#         _grafico_comparacion_anios(df, variable_label)

#     st.markdown("#### 2️⃣ Dispersión por Macrosector")
#     _grafico_boxplot(df, variable_label, anio, "MACROSECTOR", orden_macrosector, COLORES_MACROSECTOR, "Macrosector")

#     st.markdown("#### 3️⃣ Dispersión por Región")
#     _grafico_boxplot(df, variable_label, anio, "REGION", orden_region, COLORES_REGION, "Región")


# # ═══════════════════════════════════════════════════════════════════════════
# # PESTAÑA: 📈 Comparativo 2022 vs 2023 vs 2024
# # ═══════════════════════════════════════════════════════════════════════════

# # Etiqueta de fila solicitada para esta tabla (distinta de la etiqueta usada
# # en la pestaña "Variables Financieras": aquí "Ganancia/Pérdida" en vez de
# # "Ganancia (Pérdida)", y "Total activos" en minúscula, tal como se pidió).
# _ETIQUETA_FILA_COMPARATIVO = {
#     "Ingresos Operacionales": "Ingresos Operacionales",
#     "Ganancia (Pérdida)": "Ganancia/Pérdida",
#     "Total Activos": "Total activos",
#     "Total Pasivos": "Total Pasivos",
#     "Total Patrimonio": "Total Patrimonio",
# }


# def _variacion_porcentual(valor_anterior: float, valor_actual: float) -> float:
#     """(actual - anterior) / |anterior| * 100. Si el valor anterior es 0,
#     no hay una variación porcentual bien definida → se retorna NaN."""
#     if valor_anterior == 0 or pd.isna(valor_anterior) or pd.isna(valor_actual):
#         return float("nan")
#     return (valor_actual - valor_anterior) / abs(valor_anterior) * 100


# def _promedio_sin_outliers(df: pd.DataFrame, variable_label: str, anio: int) -> float:
#     """
#     Promedio de una variable financiera para un año, EXCLUYENDO los outliers
#     extremos fuera del rango fijo ya establecido para el histograma y el box
#     plot de esa misma variable (`VARIABLE_BINS`) — mismo criterio, para que
#     esta tabla sea consistente con el resto de la pestaña EDA. Así, por
#     ejemplo, las 3 empresas con Ingresos Operacionales 2022 del orden de
#     decenas de billones de COP quedan fuera del cálculo del promedio.
#     """
#     columna = VARIABLES_FINANCIERAS[variable_label][anio]
#     bordes = VARIABLE_BINS[variable_label]
#     limite_inferior, limite_superior = bordes[0], bordes[-1]

#     serie = df[columna] / DIVISOR_MILES_DE_MILLONES
#     serie_filtrada = serie[(serie >= limite_inferior) & (serie <= limite_superior)]
#     return serie_filtrada.mean()


# def _tabla_comparativo_anios(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Construye la tabla comparativa: promedio de cada variable financiera en
#     cada uno de los 3 años (en Miles de Millones COP) y su variación
#     porcentual año a año (2022→2023 y 2023→2024). Los outliers extremos
#     (fuera del rango fijo de `VARIABLE_BINS` de cada variable) se excluyen
#     del cálculo del promedio.
#     """
#     filas = []
#     for variable_label in VARIABLES_FINANCIERAS:
#         promedio_2022 = _promedio_sin_outliers(df, variable_label, 2022)
#         promedio_2023 = _promedio_sin_outliers(df, variable_label, 2023)
#         promedio_2024 = _promedio_sin_outliers(df, variable_label, 2024)

#         filas.append({
#             "Variable": _ETIQUETA_FILA_COMPARATIVO[variable_label],
#             "Promedio 2022 (MM COP)": promedio_2022,
#             "Promedio 2023 (MM COP)": promedio_2023,
#             "Promedio 2024 (MM COP)": promedio_2024,
#             "Variación (%) 2022-2023": _variacion_porcentual(promedio_2022, promedio_2023),
#             "Variación (%) 2023-2024": _variacion_porcentual(promedio_2023, promedio_2024),
#         })

#     return pd.DataFrame(filas).set_index("Variable")


# def _colorear_variacion(valor: float) -> str:
#     """Verde para variación positiva, rojo para negativa, gris para 0/NaN
#     — consistente con el color de fuente de las tablas, sin distraer con
#     fondos llamativos."""
#     if pd.isna(valor):
#         return "color: #888888;"
#     if valor > 0:
#         return "color: #1B7A43; font-weight: 700;"
#     if valor < 0:
#         return "color: #C0392B; font-weight: 700;"
#     return "color: #555555;"


# def _tabla_comparativo_estilizada(tabla: pd.DataFrame):
#     columnas_variacion = ["Variación (%) 2022-2023", "Variación (%) 2023-2024"]
#     columnas_promedio = ["Promedio 2022 (MM COP)", "Promedio 2023 (MM COP)", "Promedio 2024 (MM COP)"]

#     estilo = (
#         tabla.style
#         .format({c: "{:,.1f}" for c in columnas_promedio})
#         .format({c: "{:+.2f}%" for c in columnas_variacion}, na_rep="N/D")
#         .map(_colorear_variacion, subset=columnas_variacion)
#         .set_table_styles([
#             {"selector": "th", "props": [("font-size", "14px"), ("text-align", "center")]},
#             {"selector": "td", "props": [("font-size", "14px"), ("text-align", "center")]},
#         ])
#     )
#     st.dataframe(estilo, use_container_width=True)


# def _grafico_variacion(tabla: pd.DataFrame, columna: str, titulo: str):
#     """
#     Barras verticales con la variación porcentual de cada variable
#     financiera para el periodo indicado. Color por signo (verde = creció,
#     rojo = cayó) para que se lea de un vistazo; el nombre de cada variable
#     queda justo debajo de su barra, en el eje X.
#     """
#     datos = tabla[columna]

#     fig = go.Figure(
#         go.Bar(
#             x=datos.index,
#             y=datos.values,
#             marker_color=["#1B7A43" if v >= 0 else "#C0392B" for v in datos.values],
#             text=[f"{v:+.2f}%" for v in datos.values],
#             textposition="outside",
#             textfont=dict(size=16, color="#1a1a1a", weight="bold"),
#         )
#     )
#     fig.update_layout(**_layout_base(
#         title=titulo,
#         title_font=dict(size=FONT_SIZE + 4),
#         xaxis_title="Variable",
#         yaxis_title="Variación (%)",
#         height=460,
#         showlegend=False,
#         xaxis=dict(tickangle=-15, tickfont=dict(size=FONT_SIZE)),
#         yaxis=dict(tickfont=dict(size=FONT_SIZE), zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
#         margin=dict(t=70, b=90),
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# def _resumen_grupo_variacion(df: pd.DataFrame, variable_label: str, columna_grupo: str, orden: list[str]) -> pd.DataFrame:
#     """
#     Para cada grupo (macrosector o región), calcula el promedio de la
#     variable en cada uno de los 3 años (Miles de Millones COP) —excluyendo
#     los mismos outliers extremos fuera del rango fijo (`VARIABLE_BINS`)
#     usado en el resto de la pestaña— y su variación porcentual año a año.
#     """
#     bordes = VARIABLE_BINS[variable_label]
#     limite_inferior, limite_superior = bordes[0], bordes[-1]

#     filas = []
#     for grupo in orden:
#         fila = {columna_grupo: grupo}
#         for anio in ANIOS_DISPONIBLES:
#             columna = VARIABLES_FINANCIERAS[variable_label][anio]
#             serie = df.loc[df[columna_grupo] == grupo, columna] / DIVISOR_MILES_DE_MILLONES
#             serie_filtrada = serie[(serie >= limite_inferior) & (serie <= limite_superior)]
#             fila[anio] = serie_filtrada.mean()
#         fila["Variación 2022-2023"] = _variacion_porcentual(fila[2022], fila[2023])
#         fila["Variación 2023-2024"] = _variacion_porcentual(fila[2023], fila[2024])
#         filas.append(fila)

#     return pd.DataFrame(filas)


# def _mezclar_color(color_hex: str, color_objetivo_hex: str, fraccion: float) -> str:
#     """Mezcla `color_hex` hacia `color_objetivo_hex` en la proporción `fraccion`
#     (0 = color original, 1 = color objetivo). Usado para generar tonos claro/
#     oscuro de un mismo color base, uno por año."""
#     r1, g1, b1 = pc.hex_to_rgb(color_hex)
#     r2, g2, b2 = pc.hex_to_rgb(color_objetivo_hex)
#     r = round(r1 + (r2 - r1) * fraccion)
#     g = round(g1 + (g2 - g1) * fraccion)
#     b = round(b1 + (b2 - b1) * fraccion)
#     return f"rgb({r},{g},{b})"


# def _tonos_por_anio(color_base: str) -> dict[int, str]:
#     """3 tonos de un mismo color base -uno por año-: 2022 más claro (mezclado
#     con blanco), 2023 el color base tal cual, 2024 más oscuro (mezclado con
#     negro). Así cada región conserva su color identificador de 'Distribución
#     Geográfica', diferenciando el año por la intensidad del tono."""
#     return {
#         2022: _mezclar_color(color_base, "#FFFFFF", 0.45),
#         2023: color_base,
#         2024: _mezclar_color(color_base, "#000000", 0.30),
#     }


# def _formato_max4_decimales(valor: float) -> str:
#     """Formatea un número con HASTA 4 decimales, recortando ceros sobrantes
#     (p. ej. 64.1000 → '64.1', 64.0 → '64')."""
#     if pd.isna(valor):
#         return "N/D"
#     texto = f"{valor:.4f}".rstrip("0").rstrip(".")
#     return texto if texto not in ("", "-") else "0"


# def _color_variacion_texto(valor: float) -> str:
#     """Verde si la variación es positiva, rojo si es negativa (mismo criterio
#     de color que el resto de la pestaña 'Comparativo')."""
#     if pd.isna(valor):
#         return "#888888"
#     return "#1B7A43" if valor >= 0 else "#C0392B"


# def _grafico_variacion_por_grupo(
#     df: pd.DataFrame, variable_label: str, columna_grupo: str, orden: list[str],
#     obtener_color_anio,
# ):
#     """
#     Para una variable financiera: 3 barras (2022, 2023, 2024) por grupo
#     (macrosector o región) en el eje Y izquierdo (Miles de Millones COP, de
#     10 en 10). La variación porcentual entre años ya NO se dibuja con
#     líneas: aparece como una etiqueta de texto (verde si creció, rojo si
#     cayó) DENTRO de las barras, justo en el límite entre cada par de barras
#     contiguas. Al pasar el cursor sobre cualquier barra, la cifra se
#     muestra con un máximo de 4 decimales.

#     `obtener_color_anio(grupo, anio)` retorna el color de la barra de ese
#     grupo y ese año — así el mismo esqueleto de gráfica sirve tanto para
#     macrosector (color fijo por año) como para región (tono del color de la
#     región, distinto por año).
#     """
#     resumen = _resumen_grupo_variacion(df, variable_label, columna_grupo, orden)

#     # Posiciones numéricas explícitas (en vez de categóricas) para poder
#     # calcular con precisión el punto medio ("límite") entre cada par de
#     # barras contiguas del mismo grupo.
#     centros = list(range(len(orden)))
#     ancho_barra = 0.26
#     desplazamiento = 0.27
#     x_por_anio = {
#         2022: [c - desplazamiento for c in centros],
#         2023: [float(c) for c in centros],
#         2024: [c + desplazamiento for c in centros],
#     }

#     fig = go.Figure()

#     for anio in ANIOS_DISPONIBLES:
#         valores = resumen[anio].tolist()
#         colores = [obtener_color_anio(grupo, anio) for grupo in orden]
#         fig.add_trace(go.Bar(
#             name=str(anio),
#             x=x_por_anio[anio], y=valores,
#             width=ancho_barra,
#             marker_color=colores,
#             text=[f"{v:,.1f}" for v in valores],
#             textposition="outside",
#             textfont=dict(size=11, color="#1a1a1a"),
#             hovertext=[
#                 f"<b>{grupo}</b><br>{anio}: {_formato_max4_decimales(v)} Miles de Millones COP"
#                 for grupo, v in zip(orden, valores)
#             ],
#             hovertemplate="%{hovertext}<extra></extra>",
#         ))

#     # Etiquetas de variación DENTRO de las barras, en el límite entre cada
#     # par contiguo (2022↔2023 y 2023↔2024).
#     anotaciones = []
#     for i, grupo in enumerate(orden):
#         v2022, v2023, v2024 = resumen.loc[i, 2022], resumen.loc[i, 2023], resumen.loc[i, 2024]
#         var_22_23 = resumen.loc[i, "Variación 2022-2023"]
#         var_23_24 = resumen.loc[i, "Variación 2023-2024"]

#         x_limite_1 = (x_por_anio[2022][i] + x_por_anio[2023][i]) / 2
#         y_limite_1 = min(v2022, v2023) * 0.5 if pd.notna(v2022) and pd.notna(v2023) else 0
#         anotaciones.append(dict(
#             x=x_limite_1, y=y_limite_1, text=f"{var_22_23:+.2f}%", showarrow=False,
#             font=dict(size=12, color=_color_variacion_texto(var_22_23), weight="bold"),
#             bgcolor="rgba(255,255,255,0.55)",
#         ))

#         x_limite_2 = (x_por_anio[2023][i] + x_por_anio[2024][i]) / 2
#         y_limite_2 = min(v2023, v2024) * 0.5 if pd.notna(v2023) and pd.notna(v2024) else 0
#         anotaciones.append(dict(
#             x=x_limite_2, y=y_limite_2, text=f"{var_23_24:+.2f}%", showarrow=False,
#             font=dict(size=12, color=_color_variacion_texto(var_23_24), weight="bold"),
#             bgcolor="rgba(255,255,255,0.55)",
#         ))

#     fig.update_layout(**_layout_base(
#         title=f"{variable_label} por {'Macrosector' if columna_grupo == 'MACROSECTOR' else 'Región'} — 2022, 2023 y 2024",
#         title_font=dict(size=FONT_SIZE + 4),
#         barmode="group",
#         height=500,
#         showlegend=True,
#         legend=dict(font=dict(size=FONT_SIZE), orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
#         annotations=anotaciones,
#         xaxis=dict(
#             title="Macrosector" if columna_grupo == "MACROSECTOR" else "Región",
#             tickmode="array", tickvals=centros, ticktext=orden,
#             tickangle=-15, tickfont=dict(size=FONT_SIZE),
#         ),
#         yaxis=dict(title="Miles de Millones COP", dtick=10, tickfont=dict(size=FONT_SIZE)),
#         margin=dict(t=70),
#     ))
#     st.plotly_chart(fig, use_container_width=True)


# def mostrar_comparativo(df: pd.DataFrame):
#     """
#     Renderiza la pestaña "📈 Comparativo 2022 vs 2023 vs 2024" del EDA.
#     Primer bloque: tabla de promedios por variable financiera y su
#     variación porcentual año a año.
#     """
#     st.subheader("📈 Comparativo 2022 vs 2023 vs 2024")
#     st.markdown(
#         "Promedio de cada variable financiera clave —en Miles de Millones COP— para las "
#         "7.533 empresas del dataset, en cada uno de los tres años, junto con su variación "
#         "porcentual año a año."
#     )

#     st.markdown("#### 1️⃣ Promedios y Variación Porcentual por Variable")
#     tabla = _tabla_comparativo_anios(df)
#     _tabla_comparativo_estilizada(tabla)
#     st.caption(
#         "Variación (%) = (valor del año más reciente − valor del año anterior) / "
#         "|valor del año anterior| × 100. En verde: crecimiento; en rojo: caída.\n\n"
#         "Los promedios excluyen los outliers extremos fuera del rango fijo ya establecido "
#         "para cada variable en el histograma y el box plot de la pestaña anterior (p. ej., "
#         "las 3 empresas con Ingresos Operacionales 2022 del orden de decenas de billones de "
#         "COP no se tienen en cuenta en este cálculo)."
#     )

#     st.markdown("#### 2️⃣ Variación Porcentual por Variable, Año a Año")
#     col_izq, col_der = st.columns(2)
#     with col_izq:
#         _grafico_variacion(tabla, "Variación (%) 2022-2023", "Variación 2022 → 2023")
#     with col_der:
#         _grafico_variacion(tabla, "Variación (%) 2023-2024", "Variación 2023 → 2024")

#     st.markdown("#### 3️⃣ Variación por Macrosector")
#     st.caption(
#         "Las cifras corresponden al promedio de cada variable por año para cada Macrosector. "
#         "Cada gráfica excluye los mismos outliers extremos que el histograma y el box plot "
#         "de 'Variables Financieras'."
#     )
#     orden_macrosector = _orden_macrosectores(df)
#     tabs_macrosector = st.tabs(list(VARIABLES_FINANCIERAS.keys()))
#     for tab, variable_label in zip(tabs_macrosector, VARIABLES_FINANCIERAS.keys()):
#         with tab:
#             _grafico_variacion_por_grupo(
#                 df, variable_label, "MACROSECTOR", orden_macrosector,
#                 obtener_color_anio=lambda grupo, anio: COLOR_ANIO[anio],
#             )

#     st.markdown("#### 4️⃣ Variación por Región")
#     st.caption(
#         "Las cifras corresponden al promedio de cada variable por año para cada Región. "
#         "Cada barra conserva el color identificador de esa región (el mismo de la pestaña "
#         "'Distribución Geográfica'), en un tono más claro para 2022 y más oscuro para 2024. "
#         "Cada gráfica excluye los mismos outliers extremos que el histograma y el box plot "
#         "de 'Variables Financieras'."
#     )
#     orden_region = _conteo_por_region(df).index.tolist()
#     tabs_region = st.tabs(list(VARIABLES_FINANCIERAS.keys()))
#     for tab, variable_label in zip(tabs_region, VARIABLES_FINANCIERAS.keys()):
#         with tab:
#             _grafico_variacion_por_grupo(
#                 df, variable_label, "REGION", orden_region,
#                 obtener_color_anio=lambda grupo, anio: _tonos_por_anio(
#                     COLORES_REGION.get(grupo, "#888888")
#                 )[anio],
#             )


# # ─── Entrada pública de la fase EDA (envuelve la pestaña en st.tabs) ─────────

# def mostrar_eda(df: pd.DataFrame):
#     """
#     Punto de entrada de la fase EDA para el registro `_SECCIONES` de
#     `new_main_streamlit.py`. Contiene las pestañas construidas hasta el
#     momento; las siguientes se irán agregando a este mismo `st.tabs(...)`.
#     """
#     st.header("🔍 Análisis Exploratorio de Datos (EDA)")

#     tabs = st.tabs([
#         "🧹 Transformación y Limpieza de Datos",
#         "🏭 Distribución Sectorial",
#         "🗺️ Distribución Geográfica",
#         "💰 Variables Financieras",
#         "📈 Comparativo 2022 vs 2023 vs 2024",
#     ])
#     with tabs[0]:
#         df_limpio = mostrar_transformacion_limpieza(df)
#     with tabs[1]:
#         mostrar_distribucion_sectorial(df_limpio)
#     with tabs[2]:
#         mostrar_distribucion_geografica(df_limpio)
#     with tabs[3]:
#         mostrar_variables_financieras(df_limpio)
#     with tabs[4]:
#         mostrar_comparativo(df_limpio)

#??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
#??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????

"""
MÓDULO: new_eda_VII.py
ETAPA CRISP-DM: Comprensión de los Datos (Data Understanding)
PROPÓSITO: Pestañas del Análisis Exploratorio de Datos (EDA):
             · "🧹 Transformación y Limpieza de Datos"
             · "🏭 Distribución Sectorial"
             · "🗺️ Distribución Geográfica"   ← nueva en esta versión

           Respecto a "new_eda_VI.py", este archivo agrega la pestaña de
           Distribución Geográfica: número de empresas por REGIÓN mediante
           barras verticales, un Treemap, y un mapa real de Colombia
           (choropleth por departamento, coloreado según la REGIÓN a la
           que pertenece cada departamento).

MAPA DE COLOMBIA — CÓMO SE CONSTRUYE
────────────────────────────────────────────────────────────────────────────
El dataset agrupa los 31 departamentos en 7 REGIONES ya definidas por la
fuente original (columna "REGION"). No existe un GeoJSON público con las
fronteras de esas 7 regiones tal cual, pero sí de los 32 departamentos de
Colombia. Por eso el mapa se construye así:

  1. Se toma un GeoJSON público de los departamentos de Colombia
     (`colombia_departamentos.geojson`, debe estar en la misma carpeta).
  2. Se deriva la relación Departamento → Región directamente del propio
     dataset (agrupando "DEPARTAMENTO DOMICILIO" por "REGION"): mapeo 1 a 1
     verificado sobre los 31 departamentos presentes, sin ambigüedades.
  3. A cada polígono de departamento se le asigna el número total de
     empresas de SU REGIÓN (no el de su departamento individual) — así,
     todos los departamentos de una misma región se pintan con el mismo
     color/valor, mostrando visualmente el número de empresas POR REGIÓN
     sobre un mapa real de Colombia.
  4. Los nombres de departamento del GeoJSON no coinciden 100% en texto con
     los del dataset (p. ej. "VALLE DEL CAUCA" vs "VALLE", o "BOGOTÁ, D.C."
     vs "BOGOTA D.C."); se normalizan (mayúsculas, sin tildes, sin comas) y
     se resuelven 2 casos especiales con un diccionario de equivalencias.
     Guainía y Vaupés no tienen empresas en el dataset y quedan sin color
     (gris claro) en el mapa — no es un error, es la ausencia real de datos
     para esos 2 departamentos.

CONTENIDO DE ESTA PESTAÑA (en orden)
────────────────────────────────────────────────────────────────────────────
  1. Tabla "Tipo de datos originales"        → como llegan desde data_loader_V.
  2. Tabla "Nuevos tipos de datos asignados"  → tipo correcto por variable.
  3. Reestructuración de variables:
       · Texto (RAZÓN SOCIAL, SUPERVISOR, REGIÓN, DEPARTAMENTO DOMICILIO,
         CIUDAD DOMICILIO, CIIU, MACROSECTOR) → MAYÚSCULAS SIN TILDES.
       · Variables financieras (15 columnas, 2022-2023-2024) → se deja
         explícito que son cifras en MILES DE PESOS COLOMBIANOS (COP) y que
         el signo "-" al inicio de una cifra indica un valor NEGATIVO
         (pérdida, patrimonio negativo, etc.), no un separador visual.
  4. Tabla de calidad del dataset ya transformado: Nulos, Campos vacíos y
     Valores únicos por variable.
  5. Dos gráficas de barras (% de nulos y % de campos vacíos por variable).
  6. Vista con formato de moneda (solo para presentación) de las 15
     variables financieras: signo "$" al frente y separador de miles.

El DataFrame ya transformado (financieras en int64) se guarda en
`st.session_state["df_limpio"]` para que las siguientes pestañas del EDA
y las fases posteriores lo reutilicen sin repetir esta limpieza. La vista
con formato de moneda es SOLO PARA VISUALIZACIÓN — no reemplaza a
`df_limpio`, porque convertir las cifras a texto ("$1,234,567") rompería
cualquier cálculo aritmético posterior (estadística descriptiva, ratios,
modelado). Esa vista formateada queda disponible en
`st.session_state["df_formateado"]` por si otra pestaña quiere mostrarla.

ORIGEN DE LOS VALORES VACÍOS EN "GRUPO EN NIIF" Y "Punto de entrada"
────────────────────────────────────────────────────────────────────────────
Se verificó directamente contra los dos archivos Excel ORIGINALES (antes de
cualquier fusión o filtro de completitud):

  · "Punto de entrada" (columna exclusiva del archivo 2023-2024): de las
    9.000 filas originales, 7 ya llegaban en blanco. De esas, 6 sobreviven
    en el dataset final de 7.533 empresas (una de las 7 no cumplía el
    criterio de completitud financiera de 3 años y quedó excluida por esa
    razón, no por este campo).
  · "GRUPO EN NIIF" (columna exclusiva del archivo 2022-2023): de las 9.000
    filas originales, 152 ya llegaban en blanco. De esas, 113 sobreviven en
    el dataset final.

Es decir: NO son vacíos introducidos por la fusión de fuentes ni por el
filtro de completitud de los 3 años (ese filtro solo exige datos completos
en las 15 variables financieras). Son huecos de captura que YA VENÍAN en el
archivo original — la entidad que reportó el dato no diligenció esa
clasificación para esas empresas. Por tratarse de un metadato categórico no
reportado (no de un valor numérico ausente), se registran en este módulo
como CAMPOS VACÍOS, no como valores nulos.

Además, dentro de "Punto de entrada" se detectó un valor atípico: el
registro numérico "414" en lugar de una de las categorías esperadas
(GRUPO 1/2/3, "10 NIIF Plenas...", etc.) — probablemente una referencia
truncada a la Resolución 414 de 2014 mal transcrita en el archivo fuente.
No se corrige automáticamente (no hay forma de inferir con certeza la
categoría correcta); queda documentado aquí para que se tenga en cuenta si
esta columna se usa más adelante para segmentar o modelar.
"""

import json
import os
import unicodedata

import numpy as np
import pandas as pd
# import plotly.colors as pc
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st

import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─── Constantes de columnas ───────────────────────────────────────────────────

COLUMNAS_RANKING = ["RANKING 2022 2023", "RANKING 2023 2024"]

COLUMNAS_FINANCIERAS_2024 = [
    "INGRESOS OPERACIONALES 2024*", "GANANCIA (PERDIDA) 2024",
    "TOTAL ACTIVOS 2024", "TOTAL PASIVOS 2024", "TOTAL PATRIMONIO 2024",
]
COLUMNAS_FINANCIERAS_2023 = [
    "INGRESOS OPERACIONALES 2023*", "GANANCIA (PERDIDA) 2023",
    "TOTAL ACTIVOS 2023", "TOTAL PASIVOS 2023", "TOTAL PATRIMONIO 2023",
]
COLUMNAS_FINANCIERAS_2022 = [
    "INGRESOS OPERACIONALES 2022", "GANANCIA (PERDIDA) 2022",
    "TOTAL ACTIVOS 2022", "TOTAL PASIVOS 2022", "TOTAL PATRIMONIO 2022",
]
COLUMNAS_FINANCIERAS = COLUMNAS_FINANCIERAS_2024 + COLUMNAS_FINANCIERAS_2023 + COLUMNAS_FINANCIERAS_2022

# Texto libre: quedan como object, pero SIN normalizar mayúsculas/tildes
# (NIT es un identificador numérico convertido a texto; "Punto de entrada"
# y "GRUPO EN NIIF" son metadatos de la fuente, no descriptores geográficos
# ni sectoriales).
# Metadatos de fuente: object, sin normalizar mayúsculas/tildes. Sus NaN de
# origen (ver docstring del módulo) se registran como CAMPOS VACÍOS, no como
# nulos, porque representan una categoría no diligenciada en la fuente.
COLUMNAS_METADATO_FUENTE = ["Punto de entrada", "GRUPO EN NIIF"]

COLUMNAS_OBJECT_SIN_NORMALIZAR = ["NIT"] + COLUMNAS_METADATO_FUENTE

# Categóricas descriptivas: object + MAYÚSCULAS SIN TILDES.
COLUMNAS_MAYUSCULA_SIN_TILDE = [
    "RAZON SOCIAL", "SUPERVISOR", "REGION", "DEPARTAMENTO DOMICILIO",
    "CIUDAD DOMICILIO", "CIIU", "MACROSECTOR",
]

COLUMNAS_OBJECT_TOTAL = COLUMNAS_OBJECT_SIN_NORMALIZAR + COLUMNAS_MAYUSCULA_SIN_TILDE

# Tamaño de fuente mínimo garantizado en todas las gráficas (consistencia
# tipográfica con el resto del proyecto).
FONT_SIZE = 14

# Paleta sobria y de alto contraste para los 6 MACROSECTORES (basada en la
# paleta Okabe-Ito, diseñada para ser nítida y distinguible incluso para
# personas con daltonismo). El mismo color identifica siempre al mismo
# macrosector en todas las gráficas de esta pestaña.
COLORES_MACROSECTOR = {
    "COMERCIO":              "#0072B2",  # azul
    "SERVICIOS":             "#56B4E9",  # azul cielo
    "MANUFACTURA":           "#CC79A7",  # púrpura
    "CONSTRUCCION":          "#E69F00",  # naranja
    "AGROPECUARIO":          "#009E73",  # verde
    "MINERO-HIDROCARBUROS":  "#D55E00",  # rojo teja
}

# Colores fijos para distinguir Promedio vs. Mediana en las gráficas
# financieras (independientes del color de macrosector).
COLOR_PROMEDIO = "#1D3557"   # azul marino
COLOR_MEDIANA = "#9CA3AF"    # gris sobrio

# Los valores financieros del dataset están en MILES de pesos colombianos.
# Para expresarlos en MILES DE MILLONES de COP se divide entre 1.000.000.
DIVISOR_MILES_DE_MILLONES = 1_000_000

ANIOS_DISPONIBLES = [2022, 2023, 2024]

# Paleta sobria y distintiva para las 7 REGIONES (colores medios/oscuros
# para garantizar buen contraste con texto blanco encima).
COLORES_REGION = {
    "BOGOTA - CUNDINAMARCA": "#1D3557",  # azul marino
    "ANTIOQUIA":             "#457B9D",  # azul acero
    "COSTA PACIFICA":        "#2A9D8F",  # verde azulado
    "COSTA ATLANTICA":       "#E63946",  # rojo
    "CENTRO - ORIENTE":      "#6A4C93",  # púrpura
    "SUR Y ORIENTE":         "#BC6C25",  # ocre/marrón
    "EJE CAFETERO":          "#606C38",  # oliva oscuro
}

# ─── Constantes para el mapa de Colombia (pestaña Distribución Geográfica) ──

NOMBRE_GEOJSON_DEPARTAMENTOS = "colombia_departamentos.geojson"

# Equivalencias entre el nombre de departamento del GeoJSON (normalizado:
# mayúsculas, sin tildes, sin comas) y el nombre usado en el dataset. La
# gran mayoría de los 31 departamentos coincide directamente tras
# normalizar; solo estos 2 requieren un ajuste manual.
EQUIVALENCIAS_DEPARTAMENTO_GEOJSON = {
    "ARCHIPIELAGO DE SAN ANDRES PROVIDENCIA Y SANTA CATALINA": "SAN ANDRES Y PROVIDENCIA",
    "VALLE DEL CAUCA": "VALLE",
}


def _layout_base(**kwargs) -> dict:
    """Layout Plotly con tipografía unificada (mín. 14 pt)."""
    base = dict(
        font=dict(size=FONT_SIZE),
        title_font=dict(size=FONT_SIZE + 2),
        legend=dict(font=dict(size=FONT_SIZE)),
        xaxis=dict(title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE)),
        yaxis=dict(title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE)),
    )
    base.update(kwargs)
    return base


# ─── Utilidad de normalización de texto ───────────────────────────────────────

def _quitar_tildes(texto: str) -> str:
    """Elimina tildes/diacríticos conservando el resto del texto tal cual."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )


def _normalizar_mayuscula_sin_tilde(serie: pd.Series) -> pd.Series:
    """Convierte una serie de texto a MAYÚSCULAS y sin tildes, preservando NaN."""
    return serie.apply(lambda x: _quitar_tildes(str(x)).upper().strip() if pd.notna(x) else x)


# ─── 1. Tipos de datos originales ────────────────────────────────────────────

def _tabla_tipos_originales(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Variable": df.columns,
        "Tipo de dato original": df.dtypes.astype(str).values,
    }).set_index("Variable")


# ─── 2. Reasignación de tipos + 3. Reestructuración de variables ────────────

def _clasificar_tipo_nuevo(col: str) -> str:
    if col in COLUMNAS_RANKING:
        return "int64"
    if col in COLUMNAS_FINANCIERAS:
        return "int64"
    if col in COLUMNAS_OBJECT_TOTAL:
        return "object"
    return "object"  # cualquier columna no contemplada explícitamente


def _justificacion_tipo(col: str) -> str:
    if col in COLUMNAS_RANKING:
        return "Posición entera dentro del ranking (1..7.533)"
    if col == "NIT":
        return "Identificador de empresa: no es una cantidad, no se opera aritméticamente"
    if col in COLUMNAS_MAYUSCULA_SIN_TILDE:
        return "Variable categórica/descriptiva de texto (normalizada en mayúsculas sin tildes)"
    if col in ("Punto de entrada", "GRUPO EN NIIF"):
        return "Metadato categórico propio de la fuente de origen"
    if col in COLUMNAS_FINANCIERAS:
        return "Cifra financiera en miles de COP, sin decimales"
    return "—"


def _tabla_tipos_nuevos(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Variable": df.columns,
        "Tipo de dato asignado": [_clasificar_tipo_nuevo(c) for c in df.columns],
        "Justificación": [_justificacion_tipo(c) for c in df.columns],
    }).set_index("Variable")


def _reasignar_y_transformar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica sobre una copia del DataFrame original:
      · Ranking y variables financieras → int64.
      · NIT, "Punto de entrada", "GRUPO EN NIIF" → object (sin normalizar).
      · Variables categóricas descriptivas → object, MAYÚSCULAS SIN TILDES.
    """
    df2 = df.copy()

    # Ranking → int64
    for col in COLUMNAS_RANKING:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").round(0).astype("int64")

    # Variables financieras → int64 (miles de COP, sin decimales)
    for col in COLUMNAS_FINANCIERAS:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").round(0).astype("int64")

    # NIT → object (identificador, no cantidad)
    if "NIT" in df2.columns:
        df2["NIT"] = pd.to_numeric(df2["NIT"], errors="coerce").round(0).astype("Int64").astype(str)

    # Metadatos de fuente → object, sin normalizar mayúsculas/tildes.
    # Los NaN de origen se registran como CAMPO VACÍO ("") en vez de nulo:
    # ver "ORIGEN DE LOS VALORES VACÍOS" en el docstring del módulo.
    for col in COLUMNAS_METADATO_FUENTE:
        if col in df2.columns:
            df2[col] = df2[col].astype(object).where(df2[col].notna(), "")

    # Categóricas descriptivas → object, MAYÚSCULAS SIN TILDES
    for col in COLUMNAS_MAYUSCULA_SIN_TILDE:
        if col in df2.columns:
            df2[col] = _normalizar_mayuscula_sin_tilde(df2[col].astype(object))

    return df2


# ─── 4. Calidad del dataset transformado ─────────────────────────────────────

def _campos_vacios(serie: pd.Series) -> int:
    """
    Cuenta celdas de texto en blanco ('' tras strip) DISTINTAS de NaN.
    Para columnas numéricas (int64) no aplica el concepto → 0.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return 0
    return int(serie.dropna().astype(str).str.strip().eq("").sum())


def _tabla_calidad(df: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for col in df.columns:
        filas.append({
            "Variable": col,
            "Nulos": int(df[col].isna().sum()),
            "Campos vacíos": _campos_vacios(df[col]),
            "Valores únicos": int(df[col].nunique(dropna=True)),
        })
    tabla = pd.DataFrame(filas)
    tabla["% Nulos"] = (tabla["Nulos"] / len(df) * 100).round(2)
    tabla["% Campos vacíos"] = (tabla["Campos vacíos"] / len(df) * 100).round(2)
    return tabla.set_index("Variable")


# ─── 5. Gráficas de barras (% nulos, % campos vacíos) ────────────────────────

def _grafico_porcentaje(tabla_calidad: pd.DataFrame, columna_pct: str, titulo: str, color: str):
    datos = tabla_calidad[tabla_calidad[columna_pct] > 0][columna_pct].sort_values(ascending=False)
    if datos.empty:
        st.success(f"✅ No se detectaron {titulo.lower()} en ninguna variable.")
        return

    fig = px.bar(
        x=datos.index, y=datos.values,
        labels={"x": "Variable", "y": "% del total de registros"},
        title=titulo,
        text=datos.values.round(2),
        color_discrete_sequence=[color],
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(**_layout_base(
        height=420, showlegend=False,
        xaxis=dict(tickangle=-35, title_font=dict(size=FONT_SIZE), tickfont=dict(size=FONT_SIZE)),
    ))
    st.plotly_chart(fig, use_container_width=True)


# ─── 6. Vista con formato de moneda (solo presentación) ──────────────────────

def _formatear_moneda_cop(valor) -> str:
    """
    '$' al frente + separador de miles. El signo negativo se deja delante
    del '$' (p. ej. -$1,234,567) para que sea visualmente inconfundible,
    consistente con la convención ya explicada de que el "-" indica un
    valor negativo.
    """
    if pd.isna(valor):
        return ""
    signo = "-" if valor < 0 else ""
    return f"{signo}${abs(valor):,.0f}"


def _construir_vista_formateada(df_limpio: pd.DataFrame) -> pd.DataFrame:
    """
    Copia de `df_limpio` en la que las 15 variables financieras se muestran
    como texto con formato de moneda ($ + separador de miles). Es SOLO PARA
    VISUALIZACIÓN: al convertir la cifra a texto deja de ser apta para
    cálculos aritméticos, por eso no reemplaza a `df_limpio`.
    """
    df_fmt = df_limpio.copy()
    for col in COLUMNAS_FINANCIERAS:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(_formatear_moneda_cop)
    return df_fmt


# ─── Función principal de la pestaña ──────────────────────────────────────────

def mostrar_transformacion_limpieza(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renderiza el contenido completo de la pestaña "Transformación y Limpieza
    de Datos" y retorna el DataFrame ya transformado (para que otras
    pestañas/fases lo reutilicen).
    """
    st.subheader("🧹 Transformación y Limpieza de Datos")
    st.markdown(
        """
        Antes de explorar el dataset, se documentan los tipos de dato con los que llega
        el archivo consolidado, se reasigna el tipo correcto a cada variable y se
        normaliza el texto de las variables categóricas.
        """
    )

    # 1) Tipos originales
    st.markdown("#### 1️⃣ Tipo de datos originales")
    st.dataframe(_tabla_tipos_originales(df), use_container_width=True)

    # 2) Nuevos tipos asignados
    st.markdown("#### 2️⃣ Nuevos tipos de datos asignados")
    st.dataframe(_tabla_tipos_nuevos(df), use_container_width=True)

    # 3) Reestructuración de variables
    df_limpio = _reasignar_y_transformar(df)

    st.markdown("#### 3️⃣ Reestructuración de variables")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Variables categóricas → MAYÚSCULAS sin tildes**")
        ejemplo_cat = df_limpio[COLUMNAS_MAYUSCULA_SIN_TILDE].head(5)
        st.dataframe(ejemplo_cat, use_container_width=True)
    with col2:
        st.markdown("**Variables financieras (2022, 2023 y 2024) → `int64`**")
        st.info(
            "💰 Las 15 variables financieras (Ingresos Operacionales, Ganancia/Pérdida, "
            "Total Activos, Total Pasivos y Total Patrimonio, para cada uno de los 3 años) "
            "están expresadas en **miles de pesos colombianos (COP)**.\n\n"
            "El signo **\"-\"** al inicio de una cifra indica un valor **negativo** "
            "(p. ej. una pérdida en Ganancia/Pérdida, o un patrimonio negativo), "
            "no es un separador visual ni un dato faltante."
        )
        ejemplo_fin = df_limpio[["GANANCIA (PERDIDA) 2024", "TOTAL PATRIMONIO 2024"]].head(5)
        st.dataframe(ejemplo_fin, use_container_width=True)

    # 4) Tabla de calidad del dataset transformado
    st.markdown("---")
    st.markdown("#### 4️⃣ Calidad del dataset transformado: Nulos, Campos vacíos y Valores únicos")
    tabla_calidad = _tabla_calidad(df_limpio)
    st.dataframe(
        tabla_calidad[["Nulos", "% Nulos", "Campos vacíos", "% Campos vacíos", "Valores únicos"]],
        use_container_width=True,
    )

    total_nulos = int(tabla_calidad["Nulos"].sum())
    total_vacios = int(tabla_calidad["Campos vacíos"].sum())
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("🔴 Total valores nulos", f"{total_nulos:,}")
    col_b.metric("🟠 Total campos vacíos", f"{total_vacios:,}")
    col_c.metric("🔵 Registros del dataset", f"{len(df_limpio):,}")

    with st.expander("ℹ️ ¿De dónde salen los campos vacíos de 'Punto de entrada' y 'GRUPO EN NIIF'?"):
        st.markdown(
            """
            Se verificó directamente contra los dos archivos Excel **originales** (antes de
            fusionar fuentes o filtrar por completitud financiera):

            - **`Punto de entrada`** (solo existe en el archivo 2023-2024): de las 9.000 filas
              originales, **7** ya llegaban en blanco. De esas, **6** sobreviven en este dataset
              de 7.533 empresas.
            - **`GRUPO EN NIIF`** (solo existe en el archivo 2022-2023): de las 9.000 filas
              originales, **152** ya llegaban en blanco. De esas, **113** sobreviven aquí.

            Es decir: **no** son huecos que se hayan generado al fusionar las dos fuentes ni al
            aplicar el filtro de completitud de los 3 años (ese filtro solo exige datos completos
            en las 15 variables financieras). Ya venían vacíos en el archivo original — la entidad
            que reportó el dato no diligenció esa clasificación para esas empresas. Por tratarse de
            una categoría no reportada (no de una cifra ausente), aquí se registran como
            **campos vacíos**, no como valores nulos.

            **Dato atípico detectado:** dentro de `Punto de entrada` aparece el registro numérico
            **"414"** en lugar de una de las categorías esperadas (GRUPO 1/2/3, "10 NIIF Plenas...",
            etc.) — probablemente una referencia truncada a la *Resolución 414 de 2014*, mal
            transcrita en el archivo fuente. No se corrige automáticamente (no hay forma de inferir
            con certeza la categoría correcta); queda documentado para tenerlo en cuenta si esta
            columna se usa más adelante para segmentar o modelar.
            """
        )

    # 5) Gráficas de % nulos y % campos vacíos
    st.markdown("#### 5️⃣ Porcentaje de Nulos y de Campos Vacíos por Variable")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        _grafico_porcentaje(
            tabla_calidad, "% Nulos",
            "Porcentaje de Valores Nulos por Variable", "#c0392b",
        )
    with col_g2:
        _grafico_porcentaje(
            tabla_calidad, "% Campos vacíos",
            "Porcentaje de Campos Vacíos por Variable", "#e08e0b",
        )

    # 6) Vista con formato de moneda (solo presentación)
    st.markdown("---")
    st.markdown("#### 6️⃣ Vista con Formato de Moneda (Miles de COP)")
    st.caption(
        "Únicamente para presentación: los valores se muestran como texto "
        "(\"$1,234,567\") con el signo negativo delante del \"$\" cuando aplica. "
        "El dataset que usan las siguientes fases (`df_limpio`) conserva las "
        "15 variables en `int64` para poder seguir operando con ellas."
    )
    df_formateado = _construir_vista_formateada(df_limpio)
    columnas_vista = ["NIT", "RAZON SOCIAL"] + COLUMNAS_FINANCIERAS_2024
    st.dataframe(df_formateado[columnas_vista].head(10), use_container_width=True)
    with st.expander("Ver las 15 variables financieras con formato de moneda completas"):
        st.dataframe(
            df_formateado[["NIT", "RAZON SOCIAL"] + COLUMNAS_FINANCIERAS].head(20),
            use_container_width=True,
        )

    st.session_state["df_limpio"] = df_limpio
    st.session_state["df_formateado"] = df_formateado
    return df_limpio


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA: 🏭 Distribución Sectorial
# ═══════════════════════════════════════════════════════════════════════════

def _orden_macrosectores(df: pd.DataFrame) -> list[str]:
    """
    Orden fijo de los macrosectores (de mayor a menor número de empresas),
    reutilizado en TODAS las gráficas de esta pestaña para que la lectura
    sea consistente de una gráfica a otra.
    """
    return df["MACROSECTOR"].value_counts().index.tolist()


# ─── 1. Conteo de empresas por macrosector (barras horizontales) ────────────

def _grafico_conteo_macrosector(df: pd.DataFrame, orden: list[str]):
    conteo = df["MACROSECTOR"].value_counts().reindex(orden)

    fig = go.Figure(
        go.Bar(
            x=conteo.values,
            y=conteo.index,
            orientation="h",
            marker_color=[COLORES_MACROSECTOR.get(m, "#888888") for m in conteo.index],
            text=[f"{v:,}" for v in conteo.values],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(size=20, color="white", weight="bold"),
        )
    )
    fig.update_layout(**_layout_base(
        title="Número de Empresas por Macrosector",
        xaxis_title="Número de empresas",
        yaxis_title=None,
        height=430,
        margin=dict(l=10, r=60, t=60, b=40),
        yaxis=dict(autorange="reversed", tickfont=dict(size=FONT_SIZE + 1)),
        xaxis=dict(tickfont=dict(size=FONT_SIZE)),
    ))
    st.plotly_chart(fig, use_container_width=True)


# ─── 2. Porcentaje de empresas por macrosector (circular) ────────────────────

def _grafico_pie_macrosector(df: pd.DataFrame, orden: list[str]):
    conteo = df["MACROSECTOR"].value_counts().reindex(orden)

    fig = go.Figure(
        go.Pie(
            labels=conteo.index,
            values=conteo.values,
            hole=0.42,
            marker=dict(
                colors=[COLORES_MACROSECTOR.get(m, "#888888") for m in conteo.index],
                line=dict(color="white", width=2),
            ),
            textinfo="percent",
            textfont=dict(size=20, color="white", weight="bold"),
            insidetextorientation="horizontal",
            hovertemplate="%{label}<br>%{value:,} empresas (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(**_layout_base(
        title="Participación (%) de Empresas por Macrosector",
        height=430,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(font=dict(size=FONT_SIZE), orientation="v", yanchor="middle", y=0.5),
    ))
    st.plotly_chart(fig, use_container_width=True)


# ─── 3/4. Promedio y mediana por macrosector, por año (barras agrupadas) ────

# def _resumen_promedio_mediana(df: pd.DataFrame, columna: str, orden: list[str]) -> pd.DataFrame:
#     resumen = (
#         df.groupby("MACROSECTOR")[columna]
#         .agg(Promedio="mean", Mediana="median")
#         .reindex(orden)
#         / DIVISOR_MILES_DE_MILLONES
#     )
#     return resumen.reset_index()


# def _grafico_promedio_mediana(df: pd.DataFrame, columna: str, anio: int, variable_label: str, orden: list[str]):
#     resumen = _resumen_promedio_mediana(df, columna, orden)

def _resumen_promedio_mediana(df: pd.DataFrame, columna: str, orden: list[str], variable_label: str) -> pd.DataFrame:
    """
    Promedio y mediana por macrosector para una variable/año, EXCLUYENDO los
    outliers extremos fuera del rango fijo (`VARIABLE_BINS`) ya establecido
    para esa variable en el histograma y el box plot de "Variables
    Financieras" — mismo criterio, para que todas las gráficas de la
    pestaña EDA coincidan entre sí.
    """
    bordes = VARIABLE_BINS[variable_label]
    limite_inferior, limite_superior = bordes[0], bordes[-1]

    serie_mm = df[columna] / DIVISOR_MILES_DE_MILLONES
    mascara = (serie_mm >= limite_inferior) & (serie_mm <= limite_superior)
    df_filtrado = df.loc[mascara, ["MACROSECTOR", columna]]

    resumen = (
        df_filtrado.groupby("MACROSECTOR")[columna]
        .agg(Promedio="mean", Mediana="median")
        .reindex(orden)
        / DIVISOR_MILES_DE_MILLONES
    )
    return resumen.reset_index()


def _grafico_promedio_mediana(df: pd.DataFrame, columna: str, anio: int, variable_label: str, orden: list[str]):
    resumen = _resumen_promedio_mediana(df, columna, orden, variable_label)
    
    resumen_largo = resumen.melt(
        id_vars="MACROSECTOR", value_vars=["Promedio", "Mediana"],
        var_name="Métrica", value_name="Valor",
    )

    fig = px.bar(
        resumen_largo, x="MACROSECTOR", y="Valor", color="Métrica",
        barmode="group",
        category_orders={"MACROSECTOR": orden},
        color_discrete_map={"Promedio": COLOR_PROMEDIO, "Mediana": COLOR_MEDIANA},
        text=resumen_largo["Valor"].map(lambda v: f"{v:,.1f}"),
        title=f"{variable_label} por Macrosector — {anio}",
    )
    fig.update_traces(textposition="outside", textfont=dict(size=13))
    fig.update_layout(**_layout_base(
        height=440,
        xaxis_title="Macrosector",
        yaxis_title="Miles de Millones COP",
        xaxis=dict(tickangle=-15, tickfont=dict(size=FONT_SIZE)),
        legend=dict(title=None, font=dict(size=FONT_SIZE), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
    ))
    bordes = VARIABLE_BINS[variable_label]
    total = df[columna].notna().sum()
    incluidas = ((df[columna] / DIVISOR_MILES_DE_MILLONES >= bordes[0]) & (df[columna] / DIVISOR_MILES_DE_MILLONES <= bordes[-1])).sum()
    if incluidas < total:
        st.caption(
            f"⚠️ Se aplica el mismo rango fijo del histograma y el box plot ({bordes[0]:,} a "
            f"{bordes[-1]:,} Miles de Millones COP): incluye {incluidas:,} de {total:,} "
            f"empresas — las {total - incluidas:,} restantes son outliers extremos (del orden "
            f"de billones de pesos) y no se tienen en cuenta en el cálculo del promedio/mediana."
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Gráfica de evolución (comparación entre los 3 años) ────────────────────

# def _preparar_evolucion(df: pd.DataFrame, columnas_por_anio: dict[int, str], orden: list[str]) -> pd.DataFrame:
#     filas = []
#     for anio, columna in columnas_por_anio.items():
#         resumen = _resumen_promedio_mediana(df, columna, orden)

def _preparar_evolucion(df: pd.DataFrame, columnas_por_anio: dict[int, str], orden: list[str], variable_label: str) -> pd.DataFrame:
    filas = []
    for anio, columna in columnas_por_anio.items():
        resumen = _resumen_promedio_mediana(df, columna, orden, variable_label)
        
        resumen["Año"] = anio
        filas.append(resumen)
    return pd.concat(filas, ignore_index=True)


def _grafico_evolucion(df_evolucion: pd.DataFrame, metrica: str, variable_label: str, orden: list[str]):
    fig = go.Figure()
    for macrosector in orden:
        datos = df_evolucion[df_evolucion["MACROSECTOR"] == macrosector].sort_values("Año")
        fig.add_trace(go.Scatter(
            x=datos["Año"], y=datos[metrica],
            mode="lines+markers+text",
            name=macrosector,
            line=dict(color=COLORES_MACROSECTOR.get(macrosector, "#888888"), width=3),
            marker=dict(size=9),
            text=[f"{v:,.1f}" for v in datos[metrica]],
            textposition="top center",
            textfont=dict(size=11),
        ))

    fig.update_layout(**_layout_base(
        title=f"Evolución del {metrica} de {variable_label} por Macrosector (2022–2024)",
        xaxis_title="Año",
        yaxis_title="Miles de Millones COP",
        height=460,
        xaxis=dict(tickmode="array", tickvals=[2022, 2023, 2024], tickfont=dict(size=FONT_SIZE)),
        legend=dict(font=dict(size=FONT_SIZE), orientation="v", yanchor="middle", y=0.5, x=1.02),
        margin=dict(r=160),
    ))
    st.plotly_chart(fig, use_container_width=True)


# ─── Bloque reutilizable: 3 pestañas por año + gráfica de evolución ─────────

def _bloque_variable_financiera(
    df: pd.DataFrame,
    columnas_por_anio: dict[int, str],
    variable_label: str,
    orden: list[str],
    key_prefix: str,
):
    st.markdown(f"##### {variable_label} por Macrosector, año a año")
    tabs_anio = st.tabs([str(a) for a in ANIOS_DISPONIBLES])
    for tab, anio in zip(tabs_anio, ANIOS_DISPONIBLES):
        with tab:
            _grafico_promedio_mediana(df, columnas_por_anio[anio], anio, variable_label, orden)

    st.markdown(f"##### Evolución de {variable_label} — comparación 2022-2023-2024")
    metrica = st.radio(
        "Métrica a comparar en la evolución:",
        ["Promedio", "Mediana"],
        horizontal=True,
        key=f"{key_prefix}_metrica_evolucion",
    )
    # df_evolucion = _preparar_evolucion(df, columnas_por_anio, orden)
    df_evolucion = _preparar_evolucion(df, columnas_por_anio, orden, variable_label)
    _grafico_evolucion(df_evolucion, metrica, variable_label, orden)


# ─── Función principal de la pestaña ──────────────────────────────────────────

def mostrar_distribucion_sectorial(df: pd.DataFrame):
    """
    Renderiza la pestaña "🏭 Distribución Sectorial" del EDA: composición del
    dataset por macrosector (conteo y participación %), y el comportamiento
    de Ingresos Operacionales y Ganancia/Pérdida por macrosector a lo largo
    de los tres años (2022, 2023, 2024), con una gráfica de evolución que
    permite comparar los tres años entre sí.
    """
    st.subheader("🏭 Distribución Sectorial")
    st.markdown(
        "Composición del dataset de **7.533 empresas** por macrosector, y el comportamiento "
        "de sus principales variables financieras —Ingresos Operacionales y Ganancia/Pérdida— "
        "en miles de millones de COP, a lo largo de 2022, 2023 y 2024."
    )

    orden = _orden_macrosectores(df)

    st.markdown("#### 1️⃣ ¿Cuántas empresas hay en cada Macrosector?")
    col1, col2 = st.columns(2)
    with col1:
        _grafico_conteo_macrosector(df, orden)
    with col2:
        _grafico_pie_macrosector(df, orden)

    st.markdown("---")
    st.markdown("#### 2️⃣ Ingresos Operacionales por Macrosector")
    _bloque_variable_financiera(
        df,
        columnas_por_anio={
            2022: "INGRESOS OPERACIONALES 2022",
            2023: "INGRESOS OPERACIONALES 2023*",
            2024: "INGRESOS OPERACIONALES 2024*",
        },
        variable_label="Ingresos Operacionales",
        orden=orden,
        key_prefix="ingresos",
    )

    st.markdown("---")
    st.markdown("#### 3️⃣ Ganancia (Pérdida) por Macrosector")
    _bloque_variable_financiera(
        df,
        columnas_por_anio={
            2022: "GANANCIA (PERDIDA) 2022",
            2023: "GANANCIA (PERDIDA) 2023",
            2024: "GANANCIA (PERDIDA) 2024",
        },
        variable_label="Ganancia (Pérdida)",
        orden=orden,
        key_prefix="ganancia",
    )


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA: 🗺️ Distribución Geográfica
# ═══════════════════════════════════════════════════════════════════════════

def _conteo_por_region(df: pd.DataFrame) -> pd.Series:
    """Número de empresas por región, ordenado de mayor a menor."""
    return df["REGION"].value_counts()


# ─── 1. Barras verticales: empresas por región ──────────────────────────────

def _grafico_barras_region(conteo: pd.Series):
    fig = go.Figure(
        go.Bar(
            x=conteo.index,
            y=conteo.values,
            marker_color=[COLORES_REGION.get(r, "#888888") for r in conteo.index],
            text=[f"{v:,}" for v in conteo.values],
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(size=20, color="white", weight="bold"),
        )
    )
    fig.update_layout(**_layout_base(
        title="Número de Empresas por Región",
        xaxis_title="Región",
        yaxis_title="Número de empresas",
        height=520,
        xaxis=dict(tickangle=-20, tickfont=dict(size=FONT_SIZE + 2)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE + 1)),
        title_font=dict(size=FONT_SIZE + 6),
        margin=dict(t=70, b=90),
    ))
    st.plotly_chart(fig, use_container_width=True)


# ─── 2. Treemap: empresas por región ─────────────────────────────────────────

def _grafico_treemap_region(conteo: pd.Series):
    colores = [COLORES_REGION.get(r, "#888888") for r in conteo.index]
    fig = go.Figure(
        go.Treemap(
            labels=conteo.index,
            parents=[""] * len(conteo),
            values=conteo.values,
            marker=dict(
                colors=colores,
                line=dict(color="white", width=3),
                pad=dict(t=6, l=6, r=6, b=6),
            ),
            text=[f"{v:,} empresas" for v in conteo.values],
            texttemplate="<b>%{label}</b><br>%{text}",
            textfont=dict(size=22, color="white", weight="bold"),
            textposition="middle center",
            hovertemplate="%{label}<br>%{value:,} empresas<extra></extra>",
        )
    )
    fig.update_layout(**_layout_base(
        title="Participación de Empresas por Región (Treemap)",
        title_font=dict(size=FONT_SIZE + 6),
        height=520,
        margin=dict(t=70, l=10, r=10, b=10),
        uniformtext=dict(minsize=15, mode="show"),
    ))
    st.plotly_chart(fig, use_container_width=True)


# ─── 3. Heatmap: empresas por Región × Macrosector ──────────────────────────

def _grafico_heatmap_region_macrosector(df: pd.DataFrame, orden_region: list[str], orden_macrosector: list[str]):
    tabla = pd.crosstab(df["REGION"], df["MACROSECTOR"]).reindex(index=orden_region, columns=orden_macrosector).fillna(0).astype(int)

    fig = go.Figure(
        go.Heatmap(
            z=tabla.values,
            x=tabla.columns,
            y=tabla.index,
            colorscale="Blues",
            colorbar=dict(title="Empresas", tickfont=dict(size=FONT_SIZE)),
            xgap=3, ygap=3,
            hovertemplate="Región: %{y}<br>Macrosector: %{x}<br>Empresas: %{z:,}<extra></extra>",
        )
    )

    # Color del texto dinámico: blanco sobre celdas oscuras, oscuro sobre
    # celdas claras, para que el número SIEMPRE sea legible.
    umbral = tabla.values.max() * 0.5
    anotaciones = []
    for i, region in enumerate(tabla.index):
        for j, sector in enumerate(tabla.columns):
            valor = tabla.values[i, j]
            color_texto = "white" if valor > umbral else "#1a1a1a"
            anotaciones.append(dict(
                x=sector, y=region, text=f"{valor:,}",
                showarrow=False,
                font=dict(size=17, color=color_texto, weight="bold"),
            ))

    fig.update_layout(**_layout_base(
        title="Número de Empresas por Región y Macrosector",
        title_font=dict(size=FONT_SIZE + 6),
        height=520,
        xaxis=dict(title="Macrosector", tickangle=-15, tickfont=dict(size=FONT_SIZE + 1), side="bottom"),
        yaxis=dict(title="Región", tickfont=dict(size=FONT_SIZE + 1)),
        margin=dict(t=70, l=10, r=10, b=90),
        annotations=anotaciones,
    ))
    st.plotly_chart(fig, use_container_width=True)


# ─── 4. Mapas de Colombia: por Departamento y por Región ────────────────────

def _quitar_tildes_geo(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )


def _normalizar_nombre_departamento(nombre: str) -> str:
    limpio = _quitar_tildes_geo(nombre).upper().strip().replace(",", "")
    limpio = " ".join(limpio.split())
    return EQUIVALENCIAS_DEPARTAMENTO_GEOJSON.get(limpio, limpio)


def _resolver_ruta_geojson() -> str:
    """Busca el GeoJSON de departamentos en las mismas ubicaciones donde
    `data_loader_V.py` busca el archivo consolidado."""
    dir_modulo  = os.path.dirname(os.path.abspath(__file__))
    dir_padre   = os.path.dirname(dir_modulo)
    dir_trabajo = os.getcwd()

    candidatos = [
        os.path.join(dir_modulo,  NOMBRE_GEOJSON_DEPARTAMENTOS),
        os.path.join(dir_padre,   NOMBRE_GEOJSON_DEPARTAMENTOS),
        os.path.join(dir_trabajo, NOMBRE_GEOJSON_DEPARTAMENTOS),
    ]
    for ruta in candidatos:
        if os.path.isfile(ruta):
            return ruta

    rutas_str = "\n  • ".join(candidatos)
    raise FileNotFoundError(
        f"No se encontró '{NOMBRE_GEOJSON_DEPARTAMENTOS}' en ninguna de las rutas buscadas:\n  • {rutas_str}"
    )


@st.cache_data
def _cargar_geojson_departamentos() -> dict:
    with open(_resolver_ruta_geojson(), encoding="utf-8") as f:
        return json.load(f)


def _construir_tabla_departamentos(df: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    """
    Una fila por CADA departamento del GeoJSON (los 33, incluidos Guainía,
    Vaupés y el archipiélago de San Andrés), con su número de empresas
    individual y la región del dataset a la que pertenece. Si el
    departamento no tiene ninguna empresa registrada, no hay región que
    inferir y se marca explícitamente como sin registro.
    """
    conteo_depto = df["DEPARTAMENTO DOMICILIO"].value_counts().to_dict()
    mapa_depto_region = (
        df.groupby("DEPARTAMENTO DOMICILIO")["REGION"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    filas = []
    for feature in geojson["features"]:
        nombre_geojson = feature["properties"]["DPTO_CNMBR"]
        nombre_normalizado = _normalizar_nombre_departamento(nombre_geojson)
        empresas = conteo_depto.get(nombre_normalizado)
        region = mapa_depto_region.get(nombre_normalizado)

        if empresas is None:
            hover = f"<b>{nombre_geojson}</b><br>No hay registro de empresas para este departamento."
        else:
            hover = f"<b>{nombre_geojson}</b><br>Región: {region}<br>{empresas:,} empresas"

        filas.append({
            "Departamento_GeoJSON": nombre_geojson,
            "REGION": region,
            "Empresas": empresas if empresas is not None else 0,
            "Hover": hover,
        })
    return pd.DataFrame(filas)


def _construir_tabla_regiones(df: pd.DataFrame, geojson: dict) -> pd.DataFrame:
    """
    Una fila por CADA departamento del GeoJSON, pero coloreada según el
    total de empresas de SU REGIÓN (no del departamento individual), para
    visualizar la distribución por región sobre el mapa real de Colombia.
    """
    mapa_depto_region = (
        df.groupby("DEPARTAMENTO DOMICILIO")["REGION"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    conteo_region = _conteo_por_region(df).to_dict()

    filas = []
    for feature in geojson["features"]:
        nombre_geojson = feature["properties"]["DPTO_CNMBR"]
        nombre_normalizado = _normalizar_nombre_departamento(nombre_geojson)
        region = mapa_depto_region.get(nombre_normalizado)
        empresas = conteo_region.get(region) if region else None

        if empresas is None:
            hover = f"<b>{nombre_geojson}</b><br>No hay registro de empresas para este departamento."
        else:
            hover = f"<b>{nombre_geojson}</b><br>Región: {region}<br>{empresas:,} empresas en la región"

        filas.append({
            "Departamento_GeoJSON": nombre_geojson,
            "REGION": region,
            "Empresas": empresas,
            "Hover": hover,
        })
    return pd.DataFrame(filas)


# Color "divergente" (contrastante frente a la escala secuencial de datos)
# para señalar departamentos SIN empresas registradas.
COLOR_SIN_REGISTRO = "#4B5563"

# Coordenadas aproximadas del archipiélago de San Andrés y Providencia,
# usadas para superponer un círculo grande y legible (la isla es minúscula
# frente al resto del país y, a escala nacional, su polígono real es casi
# invisible sin este refuerzo visual).
COORD_SAN_ANDRES = (12.5847, -81.7006)


def _color_para_valor(colorscale, valor: float, zmin: float, zmax: float) -> str:
    """
    Devuelve el color exacto que le correspondería a `valor` dentro de la
    misma escala de colores (colorscale) y el mismo rango (zmin/zmax) que
    usa el mapa — para que el círculo de San Andrés se pinte con el color
    que de verdad representa su cifra, no un color arbitrario.
    """
    frac = 0.0 if zmax == zmin else (valor - zmin) / (zmax - zmin)
    frac = min(max(frac, 0.0), 1.0)
    return pc.sample_colorscale(colorscale, [frac])[0]


def _agregar_marcador_san_andres(fig: go.Figure, tabla: pd.DataFrame,
                                  colorscale, zmin: float, zmax: float, valor_color: float):
    """
    Superpone un círculo grande sobre el archipiélago de San Andrés y
    Providencia. El relleno se colorea según `valor_color` (el número de
    empresas de la Región Sur y Oriente, a la que pertenece San Andrés),
    evaluado en la MISMA escala de colores y rango que usa el mapa — así el
    círculo se ve en el mismo tono "bajo" en ambos mapas. El texto emergente
    (hover) sigue mostrando la cifra real y específica de ese mapa. No
    modifica el tamaño ni el encuadre del mapa principal.
    """
    fila = tabla[tabla["Departamento_GeoJSON"].str.contains("SAN ANDR", case=False, na=False)]
    if fila.empty:
        return
    hover = fila.iloc[0]["Hover"]
    color_relleno = _color_para_valor(colorscale, valor_color, zmin, zmax)
    lat, lon = COORD_SAN_ANDRES

    fig.add_trace(go.Scattergeo(
        lat=[lat], lon=[lon],
        mode="markers+text",
        marker=dict(size=34, color=color_relleno, line=dict(color="#1a1a1a", width=2.5)),
        text=["<b>San Andrés y<br>Providencia</b>"],
        textposition="top center",
        textfont=dict(size=14, color="#1a1a1a", weight="bold"),
        hovertext=[hover],
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ))


def _grafico_mapa_departamento(df: pd.DataFrame, geojson: dict):
    tabla = _construir_tabla_departamentos(df, geojson)
    colores_borde = [COLORES_REGION.get(r, "#BBBBBB") for r in tabla["REGION"]]

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=tabla["Departamento_GeoJSON"],
            z=tabla["Empresas"],
            featureidkey="properties.DPTO_CNMBR",
            colorscale="YlOrRd",
            marker=dict(line=dict(color=colores_borde, width=1.8)),
            colorbar_title="Empresas",
            hovertext=tabla["Hover"],
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
    conteo_sur_oriente = df["REGION"].value_counts().get("SUR Y ORIENTE", 0)
    _agregar_marcador_san_andres(
        fig, tabla, colorscale="YlOrRd",
        zmin=tabla["Empresas"].min(), zmax=tabla["Empresas"].max(),
        valor_color=conteo_sur_oriente,
    )
    fig.update_layout(**_layout_base(
        title="Empresas por Departamento",
        title_font=dict(size=FONT_SIZE + 4),
        height=600,
        margin=dict(t=60, l=10, r=10, b=10),
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "El color de relleno representa el número de empresas de cada departamento. "
        "El borde de color indica, de forma tenue, la REGIÓN a la que pertenece — al pasar "
        "el cursor (o tocar) un departamento se muestra su región y su cifra exacta; los "
        "departamentos sin empresas registradas lo indican explícitamente. San Andrés y "
        "Providencia se muestra con un círculo (su polígono real es casi invisible a esta "
        "escala), coloreado según su propia cifra de empresas."
    )


def _grafico_mapa_region(df: pd.DataFrame, geojson: dict):
    tabla = _construir_tabla_regiones(df, geojson)
    tiene_datos = tabla["Empresas"].notna()
    tabla_con_datos = tabla[tiene_datos]
    tabla_sin_datos = tabla[~tiene_datos]

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=tabla_con_datos["Departamento_GeoJSON"],
            z=tabla_con_datos["Empresas"],
            featureidkey="properties.DPTO_CNMBR",
            colorscale="Blues",
            marker_line_color="white",
            marker_line_width=1.2,
            colorbar_title="Empresas",
            hovertext=tabla_con_datos["Hover"],
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )

    # Guainía y Vaupés no pertenecen a ninguna región del dataset (no tienen
    # empresas registradas): se pintan en un color divergente y sólido para
    # que sigan siendo visibles en el mapa, distinguiéndolos claramente de
    # la escala de azules usada para las regiones con datos.
    if len(tabla_sin_datos) > 0:
        fig.add_trace(go.Choropleth(
            geojson=geojson,
            locations=tabla_sin_datos["Departamento_GeoJSON"],
            z=[1] * len(tabla_sin_datos),
            featureidkey="properties.DPTO_CNMBR",
            colorscale=[[0, COLOR_SIN_REGISTRO], [1, COLOR_SIN_REGISTRO]],
            showscale=False,
            marker_line_color="white",
            marker_line_width=1.2,
            hovertext=tabla_sin_datos["Hover"],
            hovertemplate="%{hovertext}<extra></extra>",
            name="Sin registro de empresas",
            showlegend=True,
        ))

    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
    conteo_sur_oriente = df["REGION"].value_counts().get("SUR Y ORIENTE", 0)
    _agregar_marcador_san_andres(
        fig, tabla, colorscale="Blues",
        zmin=tabla_con_datos["Empresas"].min(), zmax=tabla_con_datos["Empresas"].max(),
        valor_color=conteo_sur_oriente,
    )
    fig.update_layout(**_layout_base(
        title="Empresas por Región",
        title_font=dict(size=FONT_SIZE + 4),
        height=600,
        margin=dict(t=60, l=10, r=10, b=10),
        legend=dict(font=dict(size=FONT_SIZE, color="#1a1a1a"), yanchor="bottom", y=0.02, xanchor="left", x=0.02,
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#9CA3AF", borderwidth=1),
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Cada departamento se colorea según el total de empresas de la REGIÓN a la que "
        "pertenece (no de su departamento individual) — todos los departamentos de una "
        "misma región comparten color. Guainía y Vaupés se muestran en gris porque no "
        "tienen empresas registradas en el dataset (no pertenecen a ninguna región). San "
        "Andrés y Providencia se muestra con un círculo, coloreado según la cifra de su "
        "región (Sur y Oriente)."
    )


def _nota_departamentos_sur_oriente():
    """Nota grande y legible, requerida explícitamente por el usuario,
    debajo de los dos mapas de Colombia."""
    st.markdown(
        """
        <div style="background:#fff4e5; border-left:6px solid #BC6C25; border-radius:6px;
                    padding:18px 22px; margin-top:20px; font-size:1.08rem; line-height:1.65;
                    color:#3a2a12;">
        📌 <b>Nota metodológica:</b> en el dataset original, los departamentos de
        <b>Chocó</b>, <b>La Guajira</b>, <b>Cesar</b>, <b>Tolima</b> y <b>San Andrés y Providencia</b> aparecen clasificados
        dentro del listado de empresas de la región <b>"Sur y Oriente"</b>, además de los departamantos propios de esta región.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _grafico_mapas_colombia(df: pd.DataFrame):
    try:
        geojson = _cargar_geojson_departamentos()
    except FileNotFoundError as e:
        st.warning(
            f"⚠️ No se pudo cargar el mapa: {e}\n\n"
            f"Copia el archivo `{NOMBRE_GEOJSON_DEPARTAMENTOS}` a la misma carpeta del proyecto "
            f"para ver los mapas de Colombia."
        )
        return

    col_mapa1, col_mapa2 = st.columns(2)
    with col_mapa1:
        _grafico_mapa_departamento(df, geojson)
    with col_mapa2:
        _grafico_mapa_region(df, geojson)

    _nota_departamentos_sur_oriente()


# ─── Función principal de la pestaña ──────────────────────────────────────────

def mostrar_distribucion_geografica(df: pd.DataFrame):
    """
    Renderiza la pestaña "🗺️ Distribución Geográfica" del EDA: número de
    empresas por región mediante barras verticales, un Treemap, un heatmap
    cruzando Región × Macrosector, y dos mapas reales de Colombia (por
    departamento y por región).
    """
    st.subheader("🗺️ Distribución Geográfica")
    st.markdown(
        "Distribución de las **7.533 empresas** del dataset entre las **7 regiones** "
        "en las que se agrupan los 31 departamentos de domicilio."
    )

    conteo = _conteo_por_region(df)
    orden_region = conteo.index.tolist()
    orden_macrosector = _orden_macrosectores(df)

    st.markdown("#### 1️⃣ Empresas por Región")
    _grafico_barras_region(conteo)

    st.markdown("#### 2️⃣ Participación por Región (Treemap)")
    _grafico_treemap_region(conteo)

    st.markdown("#### 3️⃣ Empresas por Región y Macrosector")
    _grafico_heatmap_region_macrosector(df, orden_region, orden_macrosector)

    st.markdown("#### 4️⃣ Mapas de Colombia")
    _grafico_mapas_colombia(df)


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA: 💰 Variables Financieras
# ═══════════════════════════════════════════════════════════════════════════

# Variable financiera (etiqueta) → columna correspondiente en cada año.
VARIABLES_FINANCIERAS = {
    "Ingresos Operacionales": {
        2022: "INGRESOS OPERACIONALES 2022",
        2023: "INGRESOS OPERACIONALES 2023*",
        2024: "INGRESOS OPERACIONALES 2024*",
    },
    "Ganancia (Pérdida)": {
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

# Color sobrio y distinto por año, reutilizado en el histograma y en la
# gráfica de comparación entre años.
COLOR_ANIO = {2022: "#264653", 2023: "#2A9D8F", 2024: "#E76F51"}

# Rangos FIJOS de cada barra del histograma, en Miles de Millones COP,
# iguales para los 3 años (así los histogramas y la gráfica de comparación
# son directamente comparables entre 2022, 2023 y 2024). El primer borde de
# Ingresos Operacionales y de Total Activos coincide con el valor mínimo
# real de esa variable en el dataset; el resto de bordes son múltiplos
# exactos del ancho de barra indicado.
VARIABLE_BINS = {
    "Ingresos Operacionales": [0.004] + list(range(10, 510, 10)),
    # "Ganancia (Pérdida)": list(range(-20, 51, 5)),
    "Ganancia (Pérdida)": [-250] + list(range(-225, 400, 5)),
    # "Total Activos": [1] + list(range(10, 701, 10)),
    "Total Activos": [0.06] + list(range(5, 1000, 5)),
    "Total Pasivos": list(range(0, 1000, 5)),
    # "Total Patrimonio": list(range(-50, 1001, 50)),
    "Total Patrimonio": [-250] + list(range(-225, 400, 5))
}


def _serie_en_miles_de_millones(df: pd.DataFrame, columna: str) -> pd.Series:
    return (df[columna] / DIVISOR_MILES_DE_MILLONES).dropna()


def _contar_por_bins(serie: pd.Series, bordes: list[float]) -> pd.DataFrame:
    """
    Cuenta cuántas empresas caen en cada rango fijo (`bordes`, límites de
    barra en Miles de Millones COP). Retorna una fila por rango, con su
    etiqueta ("19 - 30"), su punto medio (para graficar) y el conteo.
    """
    conteos, _ = np.histogram(serie, bins=bordes)
    etiquetas = [f"{bordes[i]:,} - {bordes[i + 1]:,}" for i in range(len(bordes) - 1)]
    medios = [(bordes[i] + bordes[i + 1]) / 2 for i in range(len(bordes) - 1)]
    return pd.DataFrame({"Rango": etiquetas, "Medio": medios, "Empresas": conteos})


# ─── 1. Histograma: distribución de una variable en un año ──────────────────

def _grafico_histograma(df: pd.DataFrame, variable_label: str, anio: int):
    columna = VARIABLES_FINANCIERAS[variable_label][anio]
    serie = _serie_en_miles_de_millones(df, columna)
    bordes = VARIABLE_BINS[variable_label]
    tabla = _contar_por_bins(serie, bordes)

    fig = go.Figure(
        go.Bar(
            x=tabla["Medio"],
            y=tabla["Empresas"],
            width=[bordes[i + 1] - bordes[i] for i in range(len(bordes) - 1)],
            marker_color=COLOR_ANIO[anio],
            marker_line=dict(color="white", width=0.6),
            customdata=tabla["Rango"],
            hovertemplate=(
                f"<b>{variable_label} — {anio}</b><br>"
                "Rango: %{customdata} Miles de Millones COP<br>"
                "Empresas: %{y:,}<extra></extra>"
            ),
        )
    )
    fig.update_layout(**_layout_base(
        title=f"Distribución de {variable_label} — {anio}",
        title_font=dict(size=FONT_SIZE + 4),
        xaxis_title="Miles de Millones COP",
        yaxis_title="Número de empresas",
        height=460,
        xaxis=dict(tickfont=dict(size=FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE)),
        bargap=0.02,
        showlegend=False,
    ))
    st.plotly_chart(fig, use_container_width=True)

    incluidas = int(tabla["Empresas"].sum())
    total = len(serie)
    if incluidas < total:
        st.caption(
            f"⚠️ Rango fijo de {variable_label} entre {bordes[0]:,} y {bordes[-1]:,} Miles de "
            f"Millones COP: incluye {incluidas:,} de {total:,} empresas — las {total - incluidas:,} "
            f"restantes tienen valores fuera de este rango (por lo general, empresas atípicamente "
            f"grandes o pequeñas) y no aparecen en esta gráfica."
        )
    else:
        st.caption(
            f"Rango fijo de {variable_label} entre {bordes[0]:,} y {bordes[-1]:,} Miles de Millones "
            f"COP, igual en los tres años, para que las gráficas sean directamente comparables."
        )


# ─── 2. Comparación entre años (misma escala del histograma) ────────────────

def _grafico_comparacion_anios(df: pd.DataFrame, variable_label: str):
    bordes = VARIABLE_BINS[variable_label]

    fig = go.Figure()
    for anio in ANIOS_DISPONIBLES:
        serie = _serie_en_miles_de_millones(df, VARIABLES_FINANCIERAS[variable_label][anio])
        tabla = _contar_por_bins(serie, bordes)
        fig.add_trace(go.Scatter(
            x=tabla["Medio"], y=tabla["Empresas"],
            mode="lines+markers",
            name=str(anio),
            line=dict(color=COLOR_ANIO[anio], width=3.5, shape="spline"),
            marker=dict(size=6),
            customdata=tabla["Rango"],
            hovertemplate=(
                f"<b>{anio}</b><br>Rango: %{{customdata}} Miles de Millones COP<br>"
                "Empresas: %{y:,}<extra></extra>"
            ),
        ))

    fig.update_layout(**_layout_base(
        title=f"Comparación entre Años — {variable_label}",
        title_font=dict(size=FONT_SIZE + 4),
        xaxis_title="Miles de Millones COP",
        yaxis_title="Número de empresas",  # misma escala/unidad que el histograma
        height=460,
        legend=dict(title="Año", font=dict(size=FONT_SIZE)),
        xaxis=dict(tickfont=dict(size=FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE)),
    ))
    st.plotly_chart(fig, use_container_width=True)


# ─── 3/4. Box plots por Macrosector y por Región ─────────────────────────────

def _rango_y_ajustado_por_bigotes(df_plot: pd.DataFrame, columna_grupo: str, orden: list[str]) -> tuple[float, float] | None:
    """
    Calcula un rango de eje Y ajustado a los bigotes (fences) reales de cada
    grupo -en vez del mínimo/máximo global, que queda dominado por unos
    pocos outliers dispersos-, para que las cajas (Q1-Q3, mediana) se vean
    grandes y legibles. Los puntos que caigan fuera de este rango simplemente
    no se muestran, pero siguen formando parte del cálculo de la caja.
    """
    limites_inf, limites_sup = [], []
    for grupo in orden:
        datos = df_plot.loc[df_plot[columna_grupo] == grupo, "Valor"].dropna()
        if len(datos) < 4:
            continue
        q1, q3 = datos.quantile(0.25), datos.quantile(0.75)
        iqr = q3 - q1
        limites_inf.append(max(datos.min(), q1 - 1.5 * iqr))
        limites_sup.append(min(datos.max(), q3 + 1.5 * iqr))

    if not limites_inf:
        return None

    minimo, maximo = min(limites_inf), max(limites_sup)
    margen = (maximo - minimo) * 0.15
    return minimo - margen, maximo + margen


# def _grafico_boxplot(df: pd.DataFrame, variable_label: str, anio: int,
#                       columna_grupo: str, orden: list[str], paleta: dict, titulo_eje_x: str):
    
def _grafico_boxplot(df: pd.DataFrame, variable_label: str, anio: int,
                      columna_grupo: str, orden: list[str], paleta: dict, titulo_eje_x: str,
                      key_prefix: str = "boxplot"):
    columna_valor = VARIABLES_FINANCIERAS[variable_label][anio]
    bordes = VARIABLE_BINS[variable_label]
    limite_inferior, limite_superior = bordes[0], bordes[-1]

    df_plot = df[[columna_grupo, columna_valor]].copy()
    df_plot["Valor"] = df_plot[columna_valor] / DIVISOR_MILES_DE_MILLONES

    total = len(df_plot)
    # Mismo rango fijo que el histograma de esta variable: los valores fuera
    # de este rango (outliers extremos) se EXCLUYEN por completo del cálculo
    # de la caja — no solo de la vista — para que Q1, Q3, mediana y los
    # bigotes (fences) se dibujen a una escala legible en todos los casos.
    df_plot = df_plot[(df_plot["Valor"] >= limite_inferior) & (df_plot["Valor"] <= limite_superior)]
    incluidas = len(df_plot)

    fig = go.Figure()
    for grupo in orden:
        datos_grupo = df_plot.loc[df_plot[columna_grupo] == grupo, "Valor"]
        color = paleta.get(grupo, "#888888")
        fig.add_trace(go.Box(
            y=datos_grupo,
            name=grupo,
            boxpoints="outliers",          # solo los outliers DENTRO del rango fijo, no todos los puntos
            marker=dict(color=color, size=5, opacity=0.65),
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.55,
        ))

    # Para "Ganancia (Pérdida)" el mínimo/máximo global queda dominado por
    # unos pocos outliers dispersos (una empresa con pérdida o ganancia
    # atípicamente grande), lo que aplasta visualmente las cajas de todos
    # los demás grupos. Se acota el eje Y a los bigotes reales para que las
    # cajas se vean más amplias y legibles; los outliers que caigan fuera
    # de este rango ajustado simplemente no se dibujan (pero siguen
    # formando parte del cálculo de la caja).
    yaxis_config = dict(tickfont=dict(size=FONT_SIZE))
    if variable_label == "Ganancia (Pérdida)":
        rango_ajustado = _rango_y_ajustado_por_bigotes(df_plot, columna_grupo, orden)
        if rango_ajustado is not None:
            yaxis_config["range"] = list(rango_ajustado)

    fig.update_layout(**_layout_base(
        title=f"{variable_label} por {titulo_eje_x} — {anio}",
        title_font=dict(size=FONT_SIZE + 4),
        xaxis_title=titulo_eje_x,
        yaxis_title="Miles de Millones COP",
        height=1000,
        showlegend=False,
        xaxis=dict(tickangle=-15, tickfont=dict(size=FONT_SIZE + 1)),
        yaxis=yaxis_config,
    ))
    # st.plotly_chart(fig, use_container_width=True)
    
    st.plotly_chart(
        fig, use_container_width=True,
        key=f"{key_prefix}_{columna_grupo}_{variable_label}_{anio}",
    )

    if variable_label == "Ganancia (Pérdida)":
        st.caption(
            "🔎 El eje Y de esta gráfica se acota a los bigotes (fences) reales de cada caja "
            "para que se vean grandes y legibles — algunos outliers puntuales de ganancia o "
            "pérdida extrema quedan fuera de la vista, pero siguen incluidos en el cálculo de "
            "cada caja (Q1, Q3, mediana)."
        )

    if incluidas < total:
        st.caption(
            f"⚠️ Se aplica el mismo rango fijo del histograma ({limite_inferior:,} a "
            f"{limite_superior:,} Miles de Millones COP): incluye {incluidas:,} de {total:,} "
            f"empresas — las {total - incluidas:,} restantes son outliers extremos fuera de "
            f"ese rango y no se tienen en cuenta en esta gráfica."
        )


# ─── Función principal de la pestaña ──────────────────────────────────────────

def mostrar_variables_financieras(df: pd.DataFrame):
    """
    Renderiza la pestaña "💰 Variables Financieras" del EDA: histograma y
    comparación entre años para la variable seleccionada, y dos box plots
    (por Macrosector y por Región) para el año y la variable seleccionados.
    """
    st.subheader("💰 Variables Financieras")
    st.markdown(
        "Distribución de las 5 variables financieras clave —Ingresos Operacionales, "
        "Ganancia/Pérdida, Total Activos, Total Pasivos y Total Patrimonio— en miles de "
        "millones de COP, para 2022, 2023 y 2024."
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        variable_label = st.selectbox(
            "Variable financiera:", list(VARIABLES_FINANCIERAS.keys()), key="fin_variable",
        )
    with col_sel2:
        anio = st.radio("Año:", ANIOS_DISPONIBLES, horizontal=True, key="fin_anio")

    orden_macrosector = _orden_macrosectores(df)
    orden_region = _conteo_por_region(df).index.tolist()

    st.markdown("#### 1️⃣ Distribución del Número de Empresas")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        _grafico_histograma(df, variable_label, anio)
    with col_h2:
        _grafico_comparacion_anios(df, variable_label)

    st.markdown("#### 2️⃣ Dispersión por Macrosector")
    # _grafico_boxplot(df, variable_label, anio, "MACROSECTOR", orden_macrosector, COLORES_MACROSECTOR, "Macrosector")
    
    _grafico_boxplot(df, variable_label, anio, "MACROSECTOR", orden_macrosector, COLORES_MACROSECTOR, "Macrosector", key_prefix="finvars")
      
    st.markdown("#### 3️⃣ Dispersión por Región")
    # _grafico_boxplot(df, variable_label, anio, "REGION", orden_region, COLORES_REGION, "Región")
    
    _grafico_boxplot(df, variable_label, anio, "REGION", orden_region, COLORES_REGION, "Región", key_prefix="finvars")


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA: 📈 Comparativo 2022 vs 2023 vs 2024
# ═══════════════════════════════════════════════════════════════════════════

# Etiqueta de fila solicitada para esta tabla (distinta de la etiqueta usada
# en la pestaña "Variables Financieras": aquí "Ganancia/Pérdida" en vez de
# "Ganancia (Pérdida)", y "Total activos" en minúscula, tal como se pidió).
_ETIQUETA_FILA_COMPARATIVO = {
    "Ingresos Operacionales": "Ingresos Operacionales",
    "Ganancia (Pérdida)": "Ganancia/Pérdida",
    "Total Activos": "Total activos",
    "Total Pasivos": "Total Pasivos",
    "Total Patrimonio": "Total Patrimonio",
}


def _variacion_porcentual(valor_anterior: float, valor_actual: float) -> float:
    """(actual - anterior) / |anterior| * 100. Si el valor anterior es 0,
    no hay una variación porcentual bien definida → se retorna NaN."""
    if valor_anterior == 0 or pd.isna(valor_anterior) or pd.isna(valor_actual):
        return float("nan")
    return (valor_actual - valor_anterior) / abs(valor_anterior) * 100


def _promedio_sin_outliers(df: pd.DataFrame, variable_label: str, anio: int) -> float:
    """
    Promedio de una variable financiera para un año, EXCLUYENDO los outliers
    extremos fuera del rango fijo ya establecido para el histograma y el box
    plot de esa misma variable (`VARIABLE_BINS`) — mismo criterio, para que
    esta tabla sea consistente con el resto de la pestaña EDA. Así, por
    ejemplo, las 3 empresas con Ingresos Operacionales 2022 del orden de
    decenas de billones de COP quedan fuera del cálculo del promedio.
    """
    columna = VARIABLES_FINANCIERAS[variable_label][anio]
    bordes = VARIABLE_BINS[variable_label]
    limite_inferior, limite_superior = bordes[0], bordes[-1]

    serie = df[columna] / DIVISOR_MILES_DE_MILLONES
    serie_filtrada = serie[(serie >= limite_inferior) & (serie <= limite_superior)]
    return serie_filtrada.mean()


def _tabla_comparativo_anios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la tabla comparativa: promedio de cada variable financiera en
    cada uno de los 3 años (en Miles de Millones COP) y su variación
    porcentual año a año (2022→2023 y 2023→2024). Los outliers extremos
    (fuera del rango fijo de `VARIABLE_BINS` de cada variable) se excluyen
    del cálculo del promedio.
    """
    filas = []
    for variable_label in VARIABLES_FINANCIERAS:
        promedio_2022 = _promedio_sin_outliers(df, variable_label, 2022)
        promedio_2023 = _promedio_sin_outliers(df, variable_label, 2023)
        promedio_2024 = _promedio_sin_outliers(df, variable_label, 2024)

        filas.append({
            "Variable": _ETIQUETA_FILA_COMPARATIVO[variable_label],
            "Promedio 2022 (MM COP)": promedio_2022,
            "Promedio 2023 (MM COP)": promedio_2023,
            "Promedio 2024 (MM COP)": promedio_2024,
            "Variación (%) 2022-2023": _variacion_porcentual(promedio_2022, promedio_2023),
            "Variación (%) 2023-2024": _variacion_porcentual(promedio_2023, promedio_2024),
        })

    return pd.DataFrame(filas).set_index("Variable")


def _colorear_variacion(valor: float) -> str:
    """Verde para variación positiva, rojo para negativa, gris para 0/NaN
    — consistente con el color de fuente de las tablas, sin distraer con
    fondos llamativos."""
    if pd.isna(valor):
        return "color: #888888;"
    if valor > 0:
        return "color: #1B7A43; font-weight: 700;"
    if valor < 0:
        return "color: #C0392B; font-weight: 700;"
    return "color: #555555;"


def _tabla_comparativo_estilizada(tabla: pd.DataFrame):
    columnas_variacion = ["Variación (%) 2022-2023", "Variación (%) 2023-2024"]
    columnas_promedio = ["Promedio 2022 (MM COP)", "Promedio 2023 (MM COP)", "Promedio 2024 (MM COP)"]

    estilo = (
        tabla.style
        .format({c: "{:,.1f}" for c in columnas_promedio})
        .format({c: "{:+.2f}%" for c in columnas_variacion}, na_rep="N/D")
        .map(_colorear_variacion, subset=columnas_variacion)
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "14px"), ("text-align", "center")]},
            {"selector": "td", "props": [("font-size", "14px"), ("text-align", "center")]},
        ])
    )
    st.dataframe(estilo, use_container_width=True)


def _grafico_variacion(tabla: pd.DataFrame, columna: str, titulo: str):
    """
    Barras verticales con la variación porcentual de cada variable
    financiera para el periodo indicado. Color por signo (verde = creció,
    rojo = cayó) para que se lea de un vistazo; el nombre de cada variable
    queda justo debajo de su barra, en el eje X.
    """
    datos = tabla[columna]

    fig = go.Figure(
        go.Bar(
            x=datos.index,
            y=datos.values,
            marker_color=["#1B7A43" if v >= 0 else "#C0392B" for v in datos.values],
            text=[f"{v:+.2f}%" for v in datos.values],
            textposition="outside",
            textfont=dict(size=16, color="#1a1a1a", weight="bold"),
        )
    )
    fig.update_layout(**_layout_base(
        title=titulo,
        title_font=dict(size=FONT_SIZE + 4),
        xaxis_title="Variable",
        yaxis_title="Variación (%)",
        height=460,
        showlegend=False,
        xaxis=dict(tickangle=-15, tickfont=dict(size=FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE), zeroline=True, zerolinecolor="#888888", zerolinewidth=1.5),
        margin=dict(t=70, b=90),
    ))
    st.plotly_chart(fig, use_container_width=True)


def _resumen_grupo_variacion(df: pd.DataFrame, variable_label: str, columna_grupo: str, orden: list[str]) -> pd.DataFrame:
    """
    Para cada grupo (macrosector o región), calcula el promedio de la
    variable en cada uno de los 3 años (Miles de Millones COP) —excluyendo
    los mismos outliers extremos fuera del rango fijo (`VARIABLE_BINS`)
    usado en el resto de la pestaña— y su variación porcentual año a año.
    """
    bordes = VARIABLE_BINS[variable_label]
    limite_inferior, limite_superior = bordes[0], bordes[-1]

    filas = []
    for grupo in orden:
        fila = {columna_grupo: grupo}
        for anio in ANIOS_DISPONIBLES:
            columna = VARIABLES_FINANCIERAS[variable_label][anio]
            serie = df.loc[df[columna_grupo] == grupo, columna] / DIVISOR_MILES_DE_MILLONES
            serie_filtrada = serie[(serie >= limite_inferior) & (serie <= limite_superior)]
            fila[anio] = serie_filtrada.mean()
        fila["Variación 2022-2023"] = _variacion_porcentual(fila[2022], fila[2023])
        fila["Variación 2023-2024"] = _variacion_porcentual(fila[2023], fila[2024])
        filas.append(fila)

    return pd.DataFrame(filas)


def _mezclar_color(color_hex: str, color_objetivo_hex: str, fraccion: float) -> str:
    """Mezcla `color_hex` hacia `color_objetivo_hex` en la proporción `fraccion`
    (0 = color original, 1 = color objetivo). Usado para generar tonos claro/
    oscuro de un mismo color base, uno por año."""
    r1, g1, b1 = pc.hex_to_rgb(color_hex)
    r2, g2, b2 = pc.hex_to_rgb(color_objetivo_hex)
    r = round(r1 + (r2 - r1) * fraccion)
    g = round(g1 + (g2 - g1) * fraccion)
    b = round(b1 + (b2 - b1) * fraccion)
    return f"rgb({r},{g},{b})"


def _tonos_por_anio(color_base: str) -> dict[int, str]:
    """3 tonos de un mismo color base -uno por año-: 2022 más claro (mezclado
    con blanco), 2023 el color base tal cual, 2024 más oscuro (mezclado con
    negro). Así cada región conserva su color identificador de 'Distribución
    Geográfica', diferenciando el año por la intensidad del tono."""
    return {
        2022: _mezclar_color(color_base, "#FFFFFF", 0.45),
        2023: color_base,
        2024: _mezclar_color(color_base, "#000000", 0.30),
    }


def _formato_max4_decimales(valor: float) -> str:
    """Formatea un número con HASTA 4 decimales, recortando ceros sobrantes
    (p. ej. 64.1000 → '64.1', 64.0 → '64')."""
    if pd.isna(valor):
        return "N/D"
    texto = f"{valor:.4f}".rstrip("0").rstrip(".")
    return texto if texto not in ("", "-") else "0"


def _color_variacion_texto(valor: float) -> str:
    """Verde si la variación es positiva, rojo si es negativa (mismo criterio
    de color que el resto de la pestaña 'Comparativo')."""
    if pd.isna(valor):
        return "#888888"
    return "#1B7A43" if valor >= 0 else "#C0392B"


def _grafico_variacion_por_grupo(
    df: pd.DataFrame, variable_label: str, columna_grupo: str, orden: list[str],
    obtener_color_anio, mostrar_anio_en_barra: bool = False, mostrar_leyenda: bool = True,
):
    """
    Para una variable financiera: 3 barras (2022, 2023, 2024) por grupo
    (macrosector o región) en el eje Y izquierdo (Miles de Millones COP, de
    10 en 10). La variación porcentual entre años ya NO se dibuja con
    líneas: aparece como una etiqueta de texto (verde si creció, rojo si
    cayó) DENTRO de las barras, justo en el límite entre cada par de barras
    contiguas. Al pasar el cursor sobre cualquier barra, la cifra se
    muestra con un máximo de 4 decimales.

    `obtener_color_anio(grupo, anio)` retorna el color de la barra de ese
    grupo y ese año — así el mismo esqueleto de gráfica sirve tanto para
    macrosector (color fijo por año, con leyenda) como para región (tono
    del color de la región, distinto por año; en ese caso no hay un color
    único por año que una leyenda pueda representar, así que en su lugar
    se escribe el año directamente sobre cada barra).
    """
    resumen = _resumen_grupo_variacion(df, variable_label, columna_grupo, orden)

    # Posiciones numéricas explícitas (en vez de categóricas) para poder
    # calcular con precisión el punto medio ("límite") entre cada par de
    # barras contiguas del mismo grupo.
    centros = list(range(len(orden)))
    ancho_barra = 0.26
    desplazamiento = 0.27
    x_por_anio = {
        2022: [c - desplazamiento for c in centros],
        2023: [float(c) for c in centros],
        2024: [c + desplazamiento for c in centros],
    }

    fig = go.Figure()

    for anio in ANIOS_DISPONIBLES:
        valores = resumen[anio].tolist()
        colores = [obtener_color_anio(grupo, anio) for grupo in orden]
        if mostrar_anio_en_barra:
            texto_barra = [f"<b>{anio}</b><br>{v:,.1f}" for v in valores]
        else:
            texto_barra = [f"{v:,.1f}" for v in valores]
        fig.add_trace(go.Bar(
            name=str(anio),
            x=x_por_anio[anio], y=valores,
            width=ancho_barra,
            marker_color=colores,
            text=texto_barra,
            textposition="outside",
            textfont=dict(size=15, color="#FFFFFF"),
            hovertext=[
                f"<b>{grupo}</b><br>{anio}: {_formato_max4_decimales(v)} Miles de Millones COP"
                for grupo, v in zip(orden, valores)
            ],
            hovertemplate="%{hovertext}<extra></extra>",
        ))

    # Etiquetas de variación DENTRO de las barras, en el límite entre cada
    # par contiguo (2022↔2023 y 2023↔2024).
    anotaciones = []
    for i, grupo in enumerate(orden):
        v2022, v2023, v2024 = resumen.loc[i, 2022], resumen.loc[i, 2023], resumen.loc[i, 2024]
        var_22_23 = resumen.loc[i, "Variación 2022-2023"]
        var_23_24 = resumen.loc[i, "Variación 2023-2024"]

        x_limite_1 = (x_por_anio[2022][i] + x_por_anio[2023][i]) / 2
        y_limite_1 = min(v2022, v2023) * 0.5 if pd.notna(v2022) and pd.notna(v2023) else 0
        anotaciones.append(dict(
            x=x_limite_1, y=y_limite_1, text=f"{var_22_23:+.2f}%", showarrow=False,
            font=dict(size=15, color=_color_variacion_texto(var_22_23), weight="bold"),
            bgcolor="rgba(255,255,255,0.55)",
        ))

        x_limite_2 = (x_por_anio[2023][i] + x_por_anio[2024][i]) / 2
        y_limite_2 = min(v2023, v2024) * 0.5 if pd.notna(v2023) and pd.notna(v2024) else 0
        anotaciones.append(dict(
            x=x_limite_2, y=y_limite_2, text=f"{var_23_24:+.2f}%", showarrow=False,
            font=dict(size=15, color=_color_variacion_texto(var_23_24), weight="bold"),
            bgcolor="rgba(255,255,255,0.55)",
        ))

    fig.update_layout(**_layout_base(
        title=f"{variable_label} por {'Macrosector' if columna_grupo == 'MACROSECTOR' else 'Región'} — 2022, 2023 y 2024",
        title_font=dict(size=FONT_SIZE + 4),
        barmode="group",
        height=500,
        showlegend=mostrar_leyenda,
        legend=dict(font=dict(size=FONT_SIZE), orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        annotations=anotaciones,
        xaxis=dict(
            title="Macrosector" if columna_grupo == "MACROSECTOR" else "Región",
            tickmode="array", tickvals=centros, ticktext=orden,
            tickangle=-15, tickfont=dict(size=FONT_SIZE),
        ),
        yaxis=dict(title="Miles de Millones COP", dtick=10, tickfont=dict(size=FONT_SIZE)),
        margin=dict(t=70 if mostrar_leyenda else 60),
    ))
    st.plotly_chart(fig, use_container_width=True)


def mostrar_comparativo(df: pd.DataFrame):
    """
    Renderiza la pestaña "📈 Comparativo 2022 vs 2023 vs 2024" del EDA.
    Primer bloque: tabla de promedios por variable financiera y su
    variación porcentual año a año.
    """
    st.subheader("📈 Comparativo 2022 vs 2023 vs 2024")
    st.markdown(
        "Promedio de cada variable financiera clave —en Miles de Millones COP— para las "
        "7.533 empresas del dataset, en cada uno de los tres años, junto con su variación "
        "porcentual año a año."
    )

    st.markdown("#### 1️⃣ Promedios y Variación Porcentual por Variable")
    tabla = _tabla_comparativo_anios(df)
    _tabla_comparativo_estilizada(tabla)
    st.caption(
        "Variación (%) = (valor del año más reciente − valor del año anterior) / "
        "|valor del año anterior| × 100. En verde: crecimiento; en rojo: caída.\n\n"
        "Los promedios excluyen los outliers extremos fuera del rango fijo ya establecido "
        "para cada variable en el histograma y el box plot de la pestaña anterior (p. ej., "
        "las 3 empresas con Ingresos Operacionales 2022 del orden de decenas de billones de "
        "COP no se tienen en cuenta en este cálculo)."
    )

    st.markdown("#### 2️⃣ Variación Porcentual por Variable, Año a Año")
    col_izq, col_der = st.columns(2)
    with col_izq:
        _grafico_variacion(tabla, "Variación (%) 2022-2023", "Variación 2022 → 2023")
    with col_der:
        _grafico_variacion(tabla, "Variación (%) 2023-2024", "Variación 2023 → 2024")

    st.markdown("#### 3️⃣ Variación por Macrosector")
    st.caption(
        "Las cifras corresponden al promedio de cada variable por año para cada Macrosector. "
        "Cada gráfica excluye los mismos outliers extremos que el histograma y el box plot "
        "de 'Variables Financieras'."
    )
    orden_macrosector = _orden_macrosectores(df)
    tabs_macrosector = st.tabs(list(VARIABLES_FINANCIERAS.keys()))
    for tab, variable_label in zip(tabs_macrosector, VARIABLES_FINANCIERAS.keys()):
        with tab:
            _grafico_variacion_por_grupo(
                df, variable_label, "MACROSECTOR", orden_macrosector,
                obtener_color_anio=lambda grupo, anio: COLOR_ANIO[anio],
            )

    st.markdown("#### 4️⃣ Variación por Región")
    st.caption(
        "Las cifras corresponden al promedio de cada variable por año para cada Región. "
        "Cada barra conserva el color identificador de esa región (el mismo de la pestaña "
        "'Distribución Geográfica'), en un tono más claro para 2022 y más oscuro para 2024. "
        "Cada gráfica excluye los mismos outliers extremos que el histograma y el box plot "
        "de 'Variables Financieras'."
    )
    orden_region = _conteo_por_region(df).index.tolist()
    tabs_region = st.tabs(list(VARIABLES_FINANCIERAS.keys()))
    for tab, variable_label in zip(tabs_region, VARIABLES_FINANCIERAS.keys()):
        with tab:
            _grafico_variacion_por_grupo(
                df, variable_label, "REGION", orden_region,
                obtener_color_anio=lambda grupo, anio: _tonos_por_anio(
                    COLORES_REGION.get(grupo, "#888888")
                )[anio],
                mostrar_anio_en_barra=True,
                mostrar_leyenda=False,
            )
            
# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA: 🔗 Correlaciones
# ═══════════════════════════════════════════════════════════════════════════

def _etiquetas_columnas_corr() -> dict[str, str]:
    """Nombre corto ('Ingresos 22') para cada una de las 15 columnas
    financieras (5 variables × 3 años), usado como etiqueta de fila/columna
    en la matriz de correlación."""
    abreviaturas = {
        "Ingresos Operacionales": "Ingresos",
        "Ganancia (Pérdida)": "Ganancia",
        "Total Activos": "Activos",
        "Total Pasivos": "Pasivos",
        "Total Patrimonio": "Patrimonio",
    }
    etiquetas = {}
    for variable_label, columnas_por_anio in VARIABLES_FINANCIERAS.items():
        for anio, columna in columnas_por_anio.items():
            etiquetas[columna] = f"{abreviaturas[variable_label]} {str(anio)[-2:]}"
    return etiquetas


def _grafico_matriz_correlacion(df: pd.DataFrame):
    etiquetas = _etiquetas_columnas_corr()
    columnas = list(etiquetas.keys())
    matriz = df[columnas].corr()
    matriz.index = matriz.columns = [etiquetas[c] for c in columnas]

    fig = go.Figure(
        go.Heatmap(
            z=matriz.values, x=matriz.columns, y=matriz.index,
            zmin=-1, zmax=1, colorscale="RdBu",
            colorbar=dict(title="r", tickfont=dict(size=FONT_SIZE)),
            xgap=2, ygap=2,
            hovertemplate="%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>",
        )
    )
    anotaciones = []
    for i, fila in enumerate(matriz.index):
        for j, col in enumerate(matriz.columns):
            valor = matriz.values[i, j]
            color_texto = "white" if abs(valor) > 0.55 else "#1a1a1a"
            anotaciones.append(dict(
                x=col, y=fila, text=f"{valor:.2f}", showarrow=False,
                font=dict(size=10, color=color_texto),
            ))

    fig.update_layout(**_layout_base(
        title="Matriz de Correlación — 15 Variables Financieras (2022-2023-2024)",
        title_font=dict(size=FONT_SIZE + 4),
        height=750,
        xaxis=dict(tickangle=-45, tickfont=dict(size=FONT_SIZE - 2)),
        yaxis=dict(tickfont=dict(size=FONT_SIZE - 2), autorange="reversed"),
        margin=dict(t=70, l=10, r=10, b=90),
        annotations=anotaciones,
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    
def _mapa_columna_variable() -> dict[str, str]:
    """Columna financiera → nombre de la variable base (para buscar su
    rango de exclusión de outliers en VARIABLE_BINS)."""
    mapa = {}
    for variable_label, columnas_por_anio in VARIABLES_FINANCIERAS.items():
        for _anio, columna in columnas_por_anio.items():
            mapa[columna] = variable_label
    return mapa


def _grafico_dispersion(df: pd.DataFrame, columna_x: str, columna_y: str,
                         etiqueta_x: str, etiqueta_y: str, orden_macrosector: list[str]):
    """
    Una sub-gráfica de dispersión POR CADA MACROSECTOR (en vez de una sola
    gráfica con todos los sectores superpuestos), cada una con su propio
    color (misma paleta que 'Distribución Sectorial'), su propia línea de
    tendencia (regresión lineal simple, sin dependencias externas) y su
    propio coeficiente de correlación de Pearson. Excluye los mismos
    outliers extremos (VARIABLE_BINS) que el resto de la pestaña EDA.
    """
    mapa_var = _mapa_columna_variable()
    bordes_x = VARIABLE_BINS[mapa_var[columna_x]]
    bordes_y = VARIABLE_BINS[mapa_var[columna_y]]

    df_plot = df[["MACROSECTOR", columna_x, columna_y]].copy()
    df_plot["X"] = df_plot[columna_x] / DIVISOR_MILES_DE_MILLONES
    df_plot["Y"] = df_plot[columna_y] / DIVISOR_MILES_DE_MILLONES
    total = len(df_plot)
    df_plot = df_plot[
        (df_plot["X"] >= bordes_x[0]) & (df_plot["X"] <= bordes_x[-1]) &
        (df_plot["Y"] >= bordes_y[0]) & (df_plot["Y"] <= bordes_y[-1])
    ]
    incluidas = len(df_plot)

    columnas_grid = 3
    filas_grid = -(-len(orden_macrosector) // columnas_grid)  # redondeo hacia arriba

    fig = make_subplots(
        rows=filas_grid, cols=columnas_grid,
        subplot_titles=orden_macrosector,
        horizontal_spacing=0.08, vertical_spacing=0.18,
    )

    for i, macrosector in enumerate(orden_macrosector):
        fila, col = i // columnas_grid + 1, i % columnas_grid + 1
        datos = df_plot[df_plot["MACROSECTOR"] == macrosector]
        color = COLORES_MACROSECTOR.get(macrosector, "#888888")

        fig.add_trace(
            go.Scatter(
                x=datos["X"], y=datos["Y"], mode="markers", showlegend=False,
                marker=dict(size=6, color=color, opacity=0.65),
                hovertemplate=(
                    f"<b>{macrosector}</b><br>{etiqueta_x}: %{{x:,.1f}}<br>"
                    f"{etiqueta_y}: %{{y:,.1f}}<extra></extra>"
                ),
            ),
            row=fila, col=col,
        )

        if len(datos) > 2:
            pendiente, intercepto = np.polyfit(datos["X"], datos["Y"], 1)
            x_linea = np.linspace(datos["X"].min(), datos["X"].max(), 50)
            y_linea = pendiente * x_linea + intercepto
            r = datos["X"].corr(datos["Y"])
            fig.add_trace(
                go.Scatter(
                    x=x_linea, y=y_linea, mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color="#FFFFFF", width=2, dash="dash"),
                ),
                row=fila, col=col,
            )
            fig.layout.annotations[i].text = f"{macrosector}  (r = {r:.2f})"

    for ann in fig.layout.annotations:
        ann.font.size = FONT_SIZE - 1
    fig.update_xaxes(title_text=etiqueta_x, tickfont=dict(size=FONT_SIZE - 3), title_font=dict(size=FONT_SIZE - 3))
    fig.update_yaxes(title_text=etiqueta_y, tickfont=dict(size=FONT_SIZE - 3), title_font=dict(size=FONT_SIZE - 3))

    fig.update_layout(**_layout_base(
        title=f"{etiqueta_x} vs. {etiqueta_y}, por Macrosector",
        title_font=dict(size=FONT_SIZE + 4),
        height=290 * filas_grid,
        showlegend=False,
        margin=dict(t=90),
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Cada panel muestra únicamente las empresas de ese macrosector, con su propia línea "
        f"de tendencia y su propio coeficiente r — así se evita superponer varios sectores en "
        f"una sola gráfica. Incluye {incluidas:,} de {total:,} empresas en total — "
        f"{total - incluidas:,} quedaron fuera por estar más allá del rango fijo establecido "
        f"para alguna de las dos variables."
    )    


# def mostrar_correlaciones(df: pd.DataFrame):
#     """Renderiza la pestaña "🔗 Correlaciones" del EDA."""
#     st.subheader("🔗 Correlaciones")
#     st.markdown(
#         "Correlación de Pearson entre las 15 variables financieras (5 variables × 3 años: "
#         "2022, 2023 y 2024). La escala de color junto a la gráfica indica el sentido y la "
#         "fuerza de cada relación (de -1 a +1)."
#     )
#     st.markdown("#### 1️⃣ Matriz de Correlación")
#     _grafico_matriz_correlacion(df)
            

def mostrar_correlaciones(df: pd.DataFrame):
    """Renderiza la pestaña "🔗 Correlaciones" del EDA."""
    st.subheader("🔗 Correlaciones")
    st.markdown(
        "Correlación de Pearson entre las 15 variables financieras (5 variables × 3 años: "
        "2022, 2023 y 2024). La escala de color junto a la gráfica indica el sentido y la "
        "fuerza de cada relación (de -1 a +1)."
    )
    st.markdown("#### 1️⃣ Matriz de Correlación")
    _grafico_matriz_correlacion(df)

    st.markdown("#### 2️⃣ Relación entre Dos Variables")
    st.caption(
        "Cada punto es una empresa. La línea punteada es la tendencia lineal (regresión "
        "simple); r es el coeficiente de correlación de Pearson entre las dos variables "
        "elegidas."
    )
    etiquetas = _etiquetas_columnas_corr()
    columnas = list(etiquetas.keys())
    opciones = [etiquetas[c] for c in columnas]
    etiqueta_a_columna = {etiquetas[c]: c for c in columnas}

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        etiqueta_x = st.selectbox("Variable eje X:", opciones, index=0, key="corr_x")
    with col_sel2:
        etiqueta_y = st.selectbox("Variable eje Y:", opciones, index=1, key="corr_y")

    orden_macrosector = _orden_macrosectores(df)
    _grafico_dispersion(
        df, etiqueta_a_columna[etiqueta_x], etiqueta_a_columna[etiqueta_y],
        etiqueta_x, etiqueta_y, orden_macrosector,
    )
    
def mostrar_valores_atipicos(df: pd.DataFrame):
    """
    Renderiza la pestaña "⚠️ Valores Atípicos" del EDA (referencia:
    `_seccion_outliers` de eda_IV.py). Reutiliza `_grafico_boxplot` de
    "Variables Financieras" para el box plot, así las cajas se ven con el
    mismo tamaño legible (outliers extremos excluidos del cálculo, eje Y
    ajustado a los bigotes reales para "Ganancia (Pérdida)").
    """
    st.subheader("⚠️ Detección de Valores Atípicos (Outliers)")
    st.markdown(
        "Identificación de empresas con valores atípicos (outliers) en cada variable "
        "financiera, según el criterio estadístico IQR (rango intercuartílico), para "
        "cada uno de los tres años."
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        variable_label = st.selectbox(
            "Variable financiera:", list(VARIABLES_FINANCIERAS.keys()), key="out_variable",
        )
    with col_sel2:
        anio = st.radio("Año:", ANIOS_DISPONIBLES, horizontal=True, key="out_anio")

    columna = VARIABLES_FINANCIERAS[variable_label][anio]
    datos = (df[columna] / DIVISOR_MILES_DE_MILLONES).dropna()
    q1, q3 = datos.quantile(0.25), datos.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = datos[(datos < lim_inf) | (datos > lim_sup)]

    st.markdown("#### 1️⃣ Resumen de Outliers (criterio IQR)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Outliers (IQR)", f"{len(outliers):,}")
    col2.metric("% del Total", f"{len(outliers) / len(datos) * 100:.2f}%")
    col3.metric("Rango IQR", f"{iqr:,.1f} Miles de Millones COP")

    st.markdown("#### 2️⃣ Box Plot por Macrosector")
    st.caption(
        "Mismo criterio que la pestaña 'Variables Financieras': los outliers extremos "
        "fuera del rango fijo de esta variable no entran en el cálculo de la caja, para "
        "que Q1, Q3, mediana y bigotes se vean a una escala legible."
    )
    orden_macrosector = _orden_macrosectores(df)
    # _grafico_boxplot(df, variable_label, anio, "MACROSECTOR", orden_macrosector, COLORES_MACROSECTOR, "Macrosector")
    
    _grafico_boxplot(df, variable_label, anio, "MACROSECTOR", orden_macrosector, COLORES_MACROSECTOR, "Macrosector", key_prefix="outliers")

    st.markdown("#### 3️⃣ Top 10 Empresas con Mayores Valores")
    top10 = df.nlargest(10, columna)[["RAZON SOCIAL", "MACROSECTOR", "REGION", columna]].copy()
    top10[columna] = (top10[columna] / DIVISOR_MILES_DE_MILLONES).round(2)
    top10.columns = ["Empresa", "Macrosector", "Región", f"{variable_label} (Miles de Millones COP)"]
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)
 
# ─── Entrada pública de la fase EDA (envuelve la pestaña en st.tabs) ─────────

def mostrar_eda(df: pd.DataFrame):
    """
    Punto de entrada de la fase EDA para el registro `_SECCIONES` de
    `new_main_streamlit.py`. Contiene las pestañas construidas hasta el
    momento; las siguientes se irán agregando a este mismo `st.tabs(...)`.
    """
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")

    # tabs = st.tabs([
    #     "🧹 Transformación y Limpieza de Datos",
    #     "🏭 Distribución Sectorial",
    #     "🗺️ Distribución Geográfica",
    #     "💰 Variables Financieras",
    #     "📈 Comparativo 2022 vs 2023 vs 2024",
    # ])
    # with tabs[0]:
    #     df_limpio = mostrar_transformacion_limpieza(df)
    # with tabs[1]:
    #     mostrar_distribucion_sectorial(df_limpio)
    # with tabs[2]:
    #     mostrar_distribucion_geografica(df_limpio)
    # with tabs[3]:
    #     mostrar_variables_financieras(df_limpio)
    # with tabs[4]:
    #     mostrar_comparativo(df_limpio)
    
    tabs = st.tabs([
        "🧹 Transformación y Limpieza de Datos",
        "🏭 Distribución Sectorial",
        "🗺️ Distribución Geográfica",
        "💰 Variables Financieras",
        "📈 Comparativo 2022 vs 2023 vs 2024",
        "🔗 Correlaciones",
        "⚠️ Valores Atípicos",
    ])
    with tabs[0]:
        df_limpio = mostrar_transformacion_limpieza(df)
    with tabs[1]:
        mostrar_distribucion_sectorial(df_limpio)
    with tabs[2]:
        mostrar_distribucion_geografica(df_limpio)
    with tabs[3]:
        mostrar_variables_financieras(df_limpio)
    with tabs[4]:
        mostrar_comparativo(df_limpio)
    with tabs[5]:
        mostrar_correlaciones(df_limpio)
    with tabs[6]:
        mostrar_valores_atipicos(df_limpio)