"""
SCRIPT: generar_dataset_consolidado.py
ETAPA CRISP-DM: Preparación de los Datos (Data Preparation)
PROPÓSITO: Ejecutar UNA SOLA VEZ la fusión de las dos fuentes Excel
           (2023-2024 y 2022-2023) y aplicar el criterio de completitud
           financiera de 3 años, generando el archivo consolidado que
           la aplicación Streamlit carga en producción.

CRITERIO DE COMPLETITUD
────────────────────────────────────────────────────────────────────────────
Una empresa solo permanece en el dataset final si tiene valor NO NULO en
las 5 variables financieras (Ingresos Operacionales, Ganancia/Pérdida,
Total Activos, Total Pasivos, Total Patrimonio) para CADA UNO de los tres
años evaluados (2022, 2023 y 2024). No se admiten espacios vacíos en estas
15 columnas: es un AND estricto, no un OR.

Esto equivale a un INNER JOIN real por NIT entre ambas fuentes (una empresa
que no aparece en una de las dos fuentes automáticamente pierde el año que
esa fuente reportaba, y por tanto queda excluida) seguido de un
`dropna(subset=<15 columnas financieras>)`.

RESULTADO (verificado sobre los archivos actuales del proyecto)
────────────────────────────────────────────────────────────────────────────
    Unión total (outer join, con huecos)                  : 10.448 empresas
    Con datos completos en 2024 (5 variables)              :  8.994 empresas
    Con datos completos en 2023 (5 variables)              : 10.402 empresas
    Con datos completos en 2022 (5 variables)              :  8.968 empresas
    Con los 3 años completos simultáneamente (dataset final):  7.533 empresas

COLUMNAS DESCRIPTIVAS EXCLUSIVAS DE UNA FUENTE
────────────────────────────────────────────────────────────────────────────
"Punto de entrada" (solo fuente 2023-2024) y "GRUPO EN NIIF" (solo fuente
2022-2023) se CONSERVAN en el dataset final tal cual, aunque no todas las
7.533 empresas tengan dato en ellas (son metadatos informativos, no
variables financieras sujetas al criterio de completitud de 3 años).

CRITERIO DE RANKING (verificado empíricamente sobre las fuentes originales)
────────────────────────────────────────────────────────────────────────────
Se comprobó, ordenando cada archivo fuente por sus Ingresos Operacionales y
comparando contra su columna "RANKING"/"Ranking" original, que el criterio
es el mismo en ambos archivos:

    Posición = orden DESCENDENTE de Ingresos Operacionales del año más
    reciente del periodo cubierto por ese archivo (mayor ingreso = mejor
    posición), arrancando en 1.001 y terminando en 10.000 en las fuentes
    originales (posiciones 1-1.000 pertenecen a un listado de "Top 1000"
    publicado aparte, que no forma parte de estos archivos).

      · Fuente A (2023-2024): "RANKING" = orden desc. por
        INGRESOS OPERACIONALES 2024*.  Coincidencia verificada: 100%.
      · Fuente B (2022-2023): "Ranking" = orden desc. por
        INGRESOS OPERACIONALES 2023 (columna propia de esa fuente).
        Coincidencia verificada: 100%.

Esas columnas originales (offset 1001-10000, calculadas sobre el universo
de ~9.000 empresas de cada fuente) YA NO representan una posición válida
sobre el dataset final (7.533 empresas con los 3 años completos). Por eso
se RECALCULAN dos columnas nuevas, con el mismo criterio metodológico
(orden descendente por ingresos del año más reciente del periodo), pero
aplicado al universo final consolidado, y se insertan antes de "NIT":

    · "RANKING 2022 2023" = posición por INGRESOS OPERACIONALES 2023*
      descendente (año más reciente del periodo 2022-2023), 1..7.533.
    · "RANKING 2023 2024" = posición por INGRESOS OPERACIONALES 2024*
      descendente (año más reciente del periodo 2023-2024), 1..7.533.

Las columnas crudas de origen ("RANKING_2023_2024" / "RANKING_2022_2023",
heredadas literalmente de cada archivo fuente) se retiran del archivo final
para no dejar dos rankings con semántica distinta y nombres casi idénticos.

USO
────────────────────────────────────────────────────────────────────────────
    python generar_dataset_consolidado.py

Genera: df_empresas_2022_2023_2024_consolidado.xlsx en el directorio actual.
Este archivo es el que consume `data_loader_V.py` como fuente por defecto
de la aplicación Streamlit.
"""

import os
import sys

import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader_IV as dl  # reutiliza la lógica de fusión ya validada

NOMBRE_SALIDA = "df_empresas_2022_2023_2024_consolidado.xlsx"

# ─── Formato replicado de los archivos Excel originales ──────────────────────
# Verificado con openpyxl sobre df_9000_empresas_2023_2024.xlsx / _2022_2023.xlsx:
#   · Encabezado de campo: negrita, texto blanco, relleno azul FF003A8A.
#   · Cifras financieras: formato contable con separador de miles y "-" para cero
#     ('_-* #,##0_-;-* #,##0_-;_-* "-"_-;_-@_-').
#   · NIT / Ranking: numérico plano (General), con ancho de columna suficiente
#     para no mostrarse truncado ni en notación científica.
_AZUL_CAMPO = "FF003A8A"
_FORMATO_CONTABLE = '_-* #,##0_-;\\-* #,##0_-;_-* "-"_-;_-@_-'
_BORDE_FINO = Side(style="thin", color="FFBFBFBF")
_BORDE_CELDA = Border(left=_BORDE_FINO, right=_BORDE_FINO, top=_BORDE_FINO, bottom=_BORDE_FINO)

# Colores de la fila superior de agrupación por año (para distinguir de un
# vistazo a qué año corresponde cada bloque de 5 variables financieras).
_COLOR_GRUPO = {
    "RANKING": "FFBF8F00",
    "INFORMACIÓN GENERAL": "FFD9D9D9",
    "2024": "FF003A8A",
    "2023": "FF2E75B6",
    "2022": "FF5B9BD5",
    "METADATOS DE FUENTE": "FFA6A6A6",
}


def _grupo_de_columna(col: str) -> str:
    if col in ("RANKING 2022 2023", "RANKING 2023 2024"):
        return "RANKING"
    if col in dl.COLUMNAS_FINANCIERAS_2024:
        return "2024"
    if col in dl.COLUMNAS_FINANCIERAS_2023:
        return "2023"
    if col in dl.COLUMNAS_FINANCIERAS_2022:
        return "2022"
    if col in ("Punto de entrada", "GRUPO EN NIIF"):
        return "METADATOS DE FUENTE"
    return "INFORMACIÓN GENERAL"


def _aplicar_formato_excel(ruta: str, df: pd.DataFrame) -> None:
    """
    Reabre el archivo recién escrito por pandas y le aplica el estilo visual
    de los archivos fuente originales:
      · Fila 1: agrupación merged por año (nueva, para ubicar rápido cada bloque).
      · Fila 2: encabezado de campo (azul, negrita, blanco) — igual al original.
      · Columnas financieras: formato contable con separador de miles.
      · Anchos de columna ajustados al contenido (evita NIT en notación científica).
      · Paneles inmovilizados para no perder de vista NIT / Razón Social al desplazar.
    """
    wb = openpyxl.load_workbook(ruta)
    ws = wb.active

    columnas = list(df.columns)
    n_cols = len(columnas)

    # Insertar una fila arriba del encabezado actual para la agrupación por año
    ws.insert_rows(1)

    font_header = Font(name="Aptos", size=10, bold=True, color="FFFFFFFF")
    fill_header = PatternFill(fill_type="solid", fgColor=_AZUL_CAMPO)
    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # ── Fila 2: encabezados de campo (estilo idéntico al original) ───────────
    for idx, col in enumerate(columnas, start=1):
        cell = ws.cell(row=2, column=idx, value=col)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = align_center
        cell.border = _BORDE_CELDA
    ws.row_dimensions[2].height = 38.25

    # ── Fila 1: agrupación por año (celdas combinadas) ────────────────────────
    grupos_col = [_grupo_de_columna(c) for c in columnas]
    inicio = 0
    while inicio < n_cols:
        grupo = grupos_col[inicio]
        fin = inicio
        while fin + 1 < n_cols and grupos_col[fin + 1] == grupo:
            fin += 1
        col_ini_letra = get_column_letter(inicio + 1)
        col_fin_letra = get_column_letter(fin + 1)
        if col_ini_letra != col_fin_letra:
            ws.merge_cells(f"{col_ini_letra}1:{col_fin_letra}1")
        cell = ws.cell(row=1, column=inicio + 1, value=grupo)
        cell.font = Font(name="Aptos", size=11, bold=True, color="FFFFFFFF")
        cell.fill = PatternFill(fill_type="solid", fgColor=_COLOR_GRUPO[grupo])
        cell.alignment = align_center
        for c in range(inicio + 1, fin + 2):
            ws.cell(row=1, column=c).border = _BORDE_CELDA
        inicio = fin + 1
    ws.row_dimensions[1].height = 22

    # ── Formato de celdas de datos + anchos de columna ────────────────────────
    columnas_financieras = set(
        dl.COLUMNAS_FINANCIERAS_2024 + dl.COLUMNAS_FINANCIERAS_2023 + dl.COLUMNAS_FINANCIERAS_2022
    )
    fila_datos_inicio = 3  # datos empiezan después de las 2 filas de encabezado

    for idx, col in enumerate(columnas, start=1):
        letra = get_column_letter(idx)

        if col in columnas_financieras:
            ancho = max(len(col) + 2, 16)
            for row in range(fila_datos_inicio, ws.max_row + 1):
                celda = ws.cell(row=row, column=idx)
                celda.number_format = _FORMATO_CONTABLE
                celda.border = _BORDE_CELDA
        elif col == "NIT":
            ancho = 14  # suficiente para 9-10 dígitos sin notación científica
            for row in range(fila_datos_inicio, ws.max_row + 1):
                ws.cell(row=row, column=idx).border = _BORDE_CELDA
        elif col in ("RANKING 2022 2023", "RANKING 2023 2024"):
            ancho = max(len(col) + 2, 13)
            for row in range(fila_datos_inicio, ws.max_row + 1):
                ws.cell(row=row, column=idx).border = _BORDE_CELDA
        else:
            # columnas de texto: ancho según el contenido más largo (muestreado)
            muestra = df[col].dropna().astype(str)
            max_dato = muestra.str.len().max() if len(muestra) else 0
            ancho = min(max(len(col), int(max_dato) if pd.notna(max_dato) else 0) + 2, 60)
            for row in range(fila_datos_inicio, ws.max_row + 1):
                ws.cell(row=row, column=idx).border = _BORDE_CELDA

        ws.column_dimensions[letra].width = ancho

    # ── Paneles inmovilizados: encabezados + columnas NIT/Razón Social ───────
    ws.freeze_panes = "C3"

    wb.save(ruta)


def generar(
    ruta_2023_2024: str = None,
    ruta_2022_2023: str = None,
    ruta_salida: str = NOMBRE_SALIDA,
) -> pd.DataFrame:
    """
    Ejecuta la fusión completa (data_loader_IV) y aplica el filtro de
    completitud financiera de 3 años. Guarda el resultado en Excel y lo
    retorna como DataFrame.
    """
    if ruta_2023_2024 is None:
        ruta_2023_2024 = dl._resolver_ruta_default(dl.NOMBRE_ARCHIVO_2023_2024)
    if ruta_2022_2023 is None:
        ruta_2022_2023 = dl._resolver_ruta_default(dl.NOMBRE_ARCHIVO_2022_2023)

    df_a = dl._cargar_fuente_A(ruta_2023_2024)
    df_b = dl._cargar_fuente_B(ruta_2022_2023)
    df = dl._fusionar_fuentes(df_a, df_b)
    df = dl._limpiar_categoricas(df)

    n_union = len(df)

    columnas_financieras_3_años = (
        dl.COLUMNAS_FINANCIERAS_2024 + dl.COLUMNAS_FINANCIERAS_2023 + dl.COLUMNAS_FINANCIERAS_2022
    )
    df_completo = df.dropna(subset=columnas_financieras_3_años).reset_index(drop=True)

    n_final = len(df_completo)
    n_excluidas = n_union - n_final

    print(f"Empresas en la unión (outer join)       : {n_union:,}")
    print(f"Empresas excluidas (sin 3 años completos): {n_excluidas:,}")
    print(f"Empresas en el dataset final             : {n_final:,}")

    # ── Ranking recalculado sobre el universo final (mismo criterio de las
    # fuentes originales: orden descendente por Ingresos Operacionales del
    # año más reciente de cada periodo), insertado antes de NIT ─────────────
    df_completo["RANKING 2022 2023"] = (
        df_completo["INGRESOS OPERACIONALES 2023*"].rank(ascending=False, method="min").astype(int)
    )
    df_completo["RANKING 2023 2024"] = (
        df_completo["INGRESOS OPERACIONALES 2024*"].rank(ascending=False, method="min").astype(int)
    )
    # Retirar las columnas de ranking crudas heredadas de cada fuente (ya no
    # representan una posición válida sobre este universo de 7.533 empresas).
    df_completo.drop(columns=["RANKING_2023_2024", "RANKING_2022_2023"], inplace=True)

    columnas_orden = (
        ["RANKING 2022 2023", "RANKING 2023 2024"]
        + [c for c in df_completo.columns if c not in ("RANKING 2022 2023", "RANKING 2023 2024")]
    )
    df_completo = df_completo[columnas_orden]

    df_completo.to_excel(ruta_salida, index=False, engine="openpyxl")
    _aplicar_formato_excel(ruta_salida, df_completo)
    print(f"\n✅ Archivo generado con formato: {ruta_salida}")

    return df_completo


if __name__ == "__main__":
    generar()