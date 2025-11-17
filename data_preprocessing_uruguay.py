import pandas as pd
import glob
import os
import gc
import numpy as np

folder = r"C:\Users\JP\OneDrive - Auburn University\Research - port shipping cost\dataset\uruguay"
all_files = glob.glob(os.path.join(folder, "*.csv"))

df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

df.columns = ["dia", "mes", "anio", "num_aceptacion", "tipo_operacion", "aduana", "importador", "rut",
            "dv_rut", "via_transporte", "compania_transporte", "incoterms", "num_item", "partida_arancelaria",
            "desc_arancelaria", "mercancia", "pais_origen", "pais_adquisicion", "pais_procedencia",
            "cantidad_comercial", "unidad_comercial", "cantidad_estadistica", "unidad_estadistica",
            "fob_item_usd", "flete_item_usd", "seguro_item_usd", "cif_item_usd", "fob_unit_usd",
            "cantidad_unitaria", "peso_neto_item", "nombre_despachante", "cod_pago_1", "pct_pago_1",
            "valor_pagado_1", "cod_pago_2", "pct_pago_2", "valor_pagado_2", "cod_pago_3", "pct_pago_3",
            "valor_pagado_3", "cod_pago_4", "pct_pago_4", "valor_pagado_4", "cod_pago_5", "pct_pago_5",
            "valor_pagado_5", "peso_bruto_total"]

for col in df.select_dtypes(include='float').columns:
    df[col] = pd.to_numeric(df[col], downcast='float')
for col in df.select_dtypes(include='int').columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')


df = df.applymap(lambda x: x.replace('Ñ', 'N') if isinstance(x, str) else x)
df = df.applymap(lambda x: x.replace('Ã‘', 'N') if isinstance(x, str) else x)


df.rename(columns={'dia': 'day', 'mes': 'month', 'anio': 'year'}, inplace=True)
df['FECHA'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')




df = df.groupby("num_aceptacion", as_index=False).agg({
    "day": "first",
    "month": "first",
    "year": "first",
    "FECHA": "first",
    "tipo_operacion": "first",
    "aduana": "first",
    "importador": "first",
    "rut": "first",
    'dv_rut': 'first',
    'via_transporte': 'first',
    'compania_transporte': 'first',
    'incoterms': 'first',
    'num_item': 'nunique',
    'partida_arancelaria': 'nunique',
    'desc_arancelaria': 'first',
    'mercancia': 'first',
    'pais_origen': 'first',
    'pais_adquisicion': 'first',
    'pais_procedencia': 'first',
    'cantidad_comercial': 'sum',
    'unidad_comercial': 'first',
    'cantidad_estadistica': 'sum',
    'unidad_estadistica': 'first',
    'fob_item_usd': 'sum',
    'flete_item_usd': 'sum',
    'seguro_item_usd': 'sum',
    'cif_item_usd': 'sum',
    'fob_unit_usd': 'mean',#delete
    'cantidad_unitaria': 'sum',#delete
    'peso_neto_item': 'sum',
    'nombre_despachante': 'first',#delete
    'cod_pago_1': 'first',
    'pct_pago_1': 'first',
    'valor_pagado_1': 'sum',
    'cod_pago_2': 'first',
    'pct_pago_2': 'first',
    'valor_pagado_2': 'sum',
    'cod_pago_3': 'first',
    'pct_pago_3': 'first',
    'valor_pagado_3': 'sum',
    'cod_pago_4': 'first',
    'pct_pago_4': 'first',
    'valor_pagado_4': 'sum',
    'cod_pago_5': 'first',
    'pct_pago_5': 'first',
    'valor_pagado_5': 'sum',
    'peso_bruto_total': 'first'
})

df = df.drop(columns=["rut", "dv_rut", "via_transporte","compania_transporte","desc_arancelaria","mercancia","pais_adquisicion",
                      "fob_unit_usd","cantidad_unitaria","nombre_despachante","cod_pago_1","pct_pago_1","cod_pago_2","pct_pago_2","cod_pago_3","pct_pago_3",
                      "cod_pago_4","pct_pago_4","cod_pago_5","pct_pago_5","valor_pagado_1","valor_pagado_2","valor_pagado_3","valor_pagado_4","valor_pagado_5"])


df = df.drop(columns=["pais_procedencia","unidad_comercial","unidad_estadistica","cantidad_estadistica","cantidad_comercial"])





df["TEU"] = np.ceil(df["peso_bruto_total"] / 14000)
df["flete_per_TEU"] = df["flete_item_usd"] / df["TEU"]

df = df[df["flete_per_TEU"] > 50]

df = df[df["flete_per_TEU"] < 500000]

# import matplotlib.pyplot as plt
#
# plt.plot(df["flete_per_TEU"])
# plt.xlabel("Index")
# plt.ylabel("Flete (USD)")
# plt.title("Flete por ítem")
# plt.show()
#
# df["flete_item_usd"].hist(bins=50)
# plt.show()

# Calcular frecuencias y porcentajes
frecuencias = df['pais_origen'].value_counts(normalize=True)
menos_frecuentes = frecuencias[frecuencias < 0.01].index
df['pais_origen'] = df['pais_origen'].replace(menos_frecuentes, 'other_countries')

frecuencias = df['tipo_operacion'].value_counts(normalize=True)
menos_frecuentes = frecuencias[frecuencias < 0.05].index
df['tipo_operacion'] = df['tipo_operacion'].replace(menos_frecuentes, 'other_types_of_operations')

frecuencias = df['incoterms'].value_counts(normalize=True)
menos_frecuentes = frecuencias[frecuencias < 0.05].index
df['incoterms'] = df['incoterms'].replace(menos_frecuentes, 'other_incoterms')

region_map = {
    # ---------- FAR EAST (FE) ----------
    "República Popular de China": "FE",
    "Republica Popular de China": "FE",
    "China": "FE",
    "Taiwán": "FE",
    "Taiwan": "FE",
    "Japon": "FE",
    "Japan": "FE",
    "Hong Kong": "FE",
    "República de Corea del Sur": "FE",
    "Republica de Corea del Sur": "FE",
    "Republica de Corea": "FE",
    "Corea del Sur": "FE",
    "República Democrática Popular de Corea Norte": "FE",
    "Republica Democratica Popular de Corea Norte": "FE",
    "Vietnam": "FE",
    "Malasia": "FE",
    "Tailandia": "FE",
    "Indonesia": "FE",
    "Singapur": "FE",
    "Filipinas": "FE",
    "Camboya": "FE",
    "Sri Lanka": "FE",
    "Macao": "FE",
    "Mongolia": "FE",
    "Brunei Darussalam": "FE",
    "Islas Maldivas": "FE",

    # ---------- NORTH AMERICA (NA) ----------
    "Estados Unidos": "NA",
    "USA": "NA",
    "Canada": "NA",
    "Canadá": "NA",
    "Mexico": "NA",
    "México": "NA",
    "Puerto Rico": "NA",
    "Jamaica": "NA",
    "Islas Vírgenes Británicas": "NA",
    "Islas Bahamas": "NA",
    "Cuba": "NA",
    "Belice": "NA",
    "Honduras": "NA",
    "Guatemala": "NA",
    "El Salvador": "NA",
    "Nicaragua": "NA",
    "Costa Rica": "NA",
    "Panama": "NA",
    "Zona del Canal de Panama": "NA",
    "Republica Dominica": "NA",
    "Republica Dominicana": "NA",
    "Barbados": "NA",
    "Antigua y Barbuda": "NA",
    "Granada": "NA",
    "San Cristobal y Nevis": "NA",
    "San Vicente y Las Granadinas": "NA",
    "Trinidad y Tobago": "NA",

    # ---------- SOUTH AMERICA (SA) ----------
    "Brasil": "SA",
    "Argentina": "SA",
    "Uruguay": "SA",
    "Chile": "SA",
    "Colombia": "SA",
    "Peru": "SA",
    "Perú": "SA",
    "Ecuador": "SA",
    "Bolivia": "SA",
    "Paraguay": "SA",
    "Venezuela": "SA",
    "Guyana": "SA",
    "Suriname": "SA",

    # ---------- NORTHERN EUROPE (NE) ----------
    "Alemania": "NE",
    "Suecia": "NE",
    "Finlandia": "NE",
    "Dinamarca": "NE",
    "Noruega": "NE",
    "Paises Bajos": "NE",
    "Países Bajos": "NE",
    "Holanda": "NE",
    "Bélgica": "NE",
    "Belgica": "NE",
    "Estonia": "NE",
    "Letonia": "NE",
    "Lituania": "NE",
    "Reino Unido": "NE",
    "Irlanda": "NE",
    "Islandia": "NE",
    "Polonia": "NE",
    "Republica Eslovaquia": "NE",
    "Republica Checa": "NE",
    "República Checa": "NE",
    "Luxemburgo": "NE",
    "Austria": "NE",
    "Federacion Rusia": "NE",
    "Bielorrusia": "NE",
    "Ucrania": "NE",

    # ---------- SOUTHERN EUROPE (SE) ----------
    "Espana": "SE",
    "España": "SE",
    "Italia": "SE",
    "Portugal": "SE",
    "Grecia": "SE",
    "Turquia": "SE",
    "Turquía": "SE",
    "Malta": "SE",
    "Chipre": "SE",
    "Andorra": "SE",
    "San Marino": "SE",
    "Montenegro": "SE",
    "Albania": "SE",
    "Macedonia": "SE",
    "Croacia": "SE",
    "Eslovenia": "SE",
    "Serbia": "SE",
    "Bosnia y Herzegovina": "SE",
    "Hungria": "SE",
    "Hungría": "SE",
    "Rumania": "SE",
}

df["region"] = df["pais_origen"].map(region_map).fillna("Other")


# df.to_csv("processed_df_uruguay.csv", index=False)

# CONSTRUCT NEW DF WITH DAILY AGREGGATED DATA
df["WEEK"] = df["FECHA"].dt.to_period("W").dt.start_time

weekly_region_mean = (
    df.groupby(["WEEK", "region"])["flete_per_TEU"]
    .mean()
    .reset_index()
    .rename(columns={"flete_per_TEU": "region_mean_flete_per_TEU"})
)

weekly_region_pivot = weekly_region_mean.pivot(
    index="WEEK", columns="region", values="region_mean_flete_per_TEU"
)

# FOR DAILY USED FECHA< FOR WEEKLY USE WEEK

daily_df = df.groupby("WEEK", as_index=False).agg({
    "FECHA": "first",
    "num_aceptacion": "nunique",
    "tipo_operacion": "nunique",
    "aduana": "first",
    "importador": "nunique",
    'incoterms': 'first',
    'num_item':'sum',
    'partida_arancelaria': 'nunique',
    'pais_origen': 'first',
    'fob_item_usd': ['mean', 'min', 'max'],
    'flete_item_usd': ['mean', 'min', 'max'],
    'seguro_item_usd': ['mean', 'min', 'max'],
    'cif_item_usd': ['mean', 'min', 'max'],
    'fob_item_usd': ['mean', 'min', 'max'],
    'TEU': 'sum',
    'flete_per_TEU': ['mean', 'min', 'max']

})



rename_dict = {

    "WEEK/": "WEEK",
    "FECHA_first": "FECHA",

    "num_aceptacion_nunique": "N_DOCS",
    "tipo_operacion_first": "TIPO_OPERACION",
    "aduana_first": "ADUANA",
    "importador_nunique": "N_IMPORTADORES",
    "incoterms_first": "INCOTERMS",
    "num_item_sum": "TOTAL_ITEMS",
    "partida_arancelaria_nunique": "N_PARTIDAS",
    "pais_origen_first": "PAIS_ORIGEN",

    # FOB
    "fob_item_usd_mean": "MEAN_FOB_USD",
    "fob_item_usd_min": "MIN_FOB_USD",
    "fob_item_usd_max": "MAX_FOB_USD",

    # FLETE
    "flete_item_usd_mean": "MEAN_FLETE_USD",
    "flete_item_usd_min": "MIN_FLETE_USD",
    "flete_item_usd_max": "MAX_FLETE_USD",

    # SEGURO
    "seguro_item_usd_mean": "MEAN_SEGURO_USD",
    "seguro_item_usd_min": "MIN_SEGURO_USD",
    "seguro_item_usd_max": "MAX_SEGURO_USD",

    # CIF
    "cif_item_usd_mean": "MEAN_CIF_USD",
    "cif_item_usd_min": "MIN_CIF_USD",
    "cif_item_usd_max": "MAX_CIF_USD",

    # TEU
    "teu_sum": "TOTAL_TEU",

    # FLETES NORMALIZADOS
    "flete_per_teu_mean": "MEAN_FLETE_POR_TEU",
    "flete_per_teu_min": "MIN_FLETE_POR_TEU",
    "flete_per_teu_max": "MAX_FLETE_POR_TEU",
}

daily_df.columns = [
    f"{col[0]}_{col[1]}" if col[1] != "" else col[0]
    for col in daily_df.columns
]




daily_df = daily_df.rename(columns=rename_dict)

daily_df = daily_df.merge(weekly_region_pivot, on="WEEK", how="left")



def pivot_and_merge(main_df, series_df, group_col, value_col, agg_func='sum'):
    """Helper to perform pivot-like aggregation and merge."""
    pivot_df = series_df.groupby(['FECHA', group_col])[value_col].agg(agg_func).unstack(fill_value=0)
    pivot_df = pivot_df.add_prefix(f'COUNT_{group_col}_')  # Add prefix for clarity
    return main_df.merge(pivot_df, on='FECHA', how='left')


daily_df = pivot_and_merge(daily_df, df, 'pais_origen', 'TEU')
daily_df = pivot_and_merge(daily_df, df, 'tipo_operacion', 'TEU')
daily_df = pivot_and_merge(daily_df, df, 'incoterms', 'TEU')
daily_df = pivot_and_merge(daily_df, df, 'aduana', 'TEU')




# ADD MACRO DATA


original_columns = set(daily_df.columns)

folder = r"C:\Users\JP\OneDrive - Auburn University\Research - port shipping cost\dataset\macrodata"

for filename in os.listdir(folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder, filename)
        df_macro = pd.read_csv(file_path)

        if 'Date' in df_macro.columns:
            df_macro['FECHA'] = pd.to_datetime(df_macro['Date'], errors='coerce')
            df_macro = df_macro.sort_values('FECHA').drop(columns=['Date'])
            df_macro[['Close', 'Volume']] = df_macro[['Close', 'Volume']].fillna(method='ffill').fillna(method='bfill')
            df_macro['Close_pct_change'] = df_macro['Close'].pct_change().fillna(0)

            # df_macro.to_csv("test_macro.csv", index=False)

            if 'Close' in df_macro.columns and 'Volume' in df_macro.columns:
                df_macro[['Close', 'Volume']] = df_macro[['Close', 'Volume']].fillna(method='ffill')

                name = filename.replace('.csv', '').lower()  # e.g. 'data_gold'
                macro_filtered = df_macro[['FECHA', 'Close', 'Volume', 'Close_pct_change']].rename(
                    columns={
                        'Close': f'{name}_price',
                        'Volume': f'{name}_volume',
                        'Close_pct_change': f'{name}_pct_change'
                    })

                daily_df = pd.merge(daily_df, macro_filtered, how='left', on='FECHA')

new_columns = [col for col in daily_df.columns if col not in original_columns]
daily_df[new_columns] = daily_df[new_columns].fillna(method='ffill').fillna(method='bfill')




daily_df.to_csv("weekly_uruguay_data.csv", index=False)

