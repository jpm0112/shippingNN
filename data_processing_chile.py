import pandas as pd
import glob
import os
import gc

folder = r"C:\Users\JP\OneDrive - Auburn University\Research - port shipping cost\dataset\container_data"
all_files = glob.glob(os.path.join(folder, "*.csv"))

df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)


df.columns = [
    'DIA', 'MES', 'ANO', 'ADUANA', 'NUMERO DE ACEPTACION',
    'RUT PROBABLE IMPORTADOR', 'DIGITO VERIFICADOR RUT',
    'PROBABLE IMPORTADOR', 'PARTIDA ARANCELARIA', 'PRODUCTO', 'MARCA',
    'VARIEDAD', 'DESCRIPCION', 'PAIS DE ORIGEN', 'PAIS DE ADQUISICION',
    'VIA DE TRANSPORTE', 'FORMA PAGO', 'PUERTO DE EMBARQUE',
    'PUERTO DE DESEMBARQUE', 'COMPANIA DE TRANSPORTE', 'TIPO DE CARGA',
    'TIPO DE BULTO', 'PESO BRUTO TOTAL', 'CLAUSULA', 'IMPUESTO', 'CANTIDAD',
    'UNIDAD', 'US$ FOB', 'US$ FLETE', 'US$ SEGURO', 'US$ CIF',
    'US$ CIF UNIT', 'TIPO DE OPERACION', 'DESCRIPCION ARANCELARIA',
    'NUM DE ITEM', 'PAIS COMPANIA DE TRANSPORTE', 'IMPUESTO US$',
    'CANTIDAD DE BULTO', 'ZONA ECONOMICA', 'CLAVE ECONOMICA IMPORTADOR',
    'ALMACEN', 'FECHA DE ALMACEN', 'NRO DE MANIFIESTO',
    'FECHA DE MANIFIESTO', 'NRO DOC. TRANSPORTE', 'FECHA DOC. TRANSPORTE',
    'ITEMS TOTALES', 'FOB TOTAL', 'FLETE TOTAL', 'SEGURO TOTAL',
    'CIF TOTAL', 'TOTAL IVA', 'US$ FOB UNIT', 'ACUERDO COMERCIAL',
    'CANTIDAD UNIDADES FISICAS', 'UNIDAD DE MEDIDA FISICA',
    'ESTADO DE MERCANCIA', 'EMISOR'
]

for col in df.select_dtypes(include='float').columns:
    df[col] = pd.to_numeric(df[col], downcast='float')
for col in df.select_dtypes(include='int').columns:
    df[col] = pd.to_numeric(df[col], downcast='integer')

df['TIPO DE BULTO'] = df['TIPO DE BULTO'].astype('category')

# df = df.applymap(lambda x: x.replace('Ñ', 'N') if isinstance(x, str) else x)
# df = df.applymap(lambda x: x.replace('Ã‘', 'N') if isinstance(x, str) else x)




# inlude only container rows
contenedores = [
    'CONTENEDOR 20',
    'CONTENEDOR 40',
    'CONTENEDOR REFRIGERADO 40',
    'CONTENEDOR NO REFRIGERADO',
    'CONTENEDOR REFRIGERADO 20'
]
# filters the whole df to include container data only
df = df[df['TIPO DE BULTO'].isin(contenedores)].copy()

df.rename(columns={'DIA': 'day', 'MES': 'month', 'ANO': 'year'}, inplace=True)
df['FECHA'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')

df['FECHA_DOC_TRANSPORTE'] = pd.to_datetime(df['FECHA DOC. TRANSPORTE'], format='%d%m%Y', errors='coerce')

df['DIFF FECHA DIN Y DOC TRANSPORTE'] = (df['FECHA'] - df['FECHA_DOC_TRANSPORTE']).dt.days

df['FOB_POR_BULTO'] = df['FOB TOTAL'] / df['CANTIDAD DE BULTO']
df['FLETE_POR_BULTO'] = df['FLETE TOTAL'] / df['CANTIDAD DE BULTO']
df['SEGURO_POR_BULTO'] = df['SEGURO TOTAL'] / df['CANTIDAD DE BULTO']

df['PESO BRUTO POR CONTENEDOR'] = df['PESO BRUTO TOTAL'] / df['CANTIDAD DE BULTO']
df['ITEMS POR CONTENEDOR'] = df['ITEMS TOTALES'] / df['CANTIDAD DE BULTO']

df['TOTAL_TEU_REFRIGERADOS'] = (
    df['CONTENEDOR REFRIGERADO 20'] + 2 *df['CONTENEDOR REFRIGERADO 40']
)




# Calcular frecuencias y porcentajes
frecuencias = df['PUERTO DE EMBARQUE'].value_counts(normalize=True) * 100
menos_frecuentes = frecuencias[frecuencias < 10].index
df['PUERTO DE EMBARQUE'] = df['PUERTO DE EMBARQUE'].replace(menos_frecuentes, 'other_ports')

# Calcular frecuencias y porcentajes
frecuencias = df['PUERTO DE DESEMBARQUE'].value_counts(normalize=True) * 100
# Identificar los que representan menos del 0.5%
menos_frecuentes = frecuencias[frecuencias < 10].index
# Reemplazar en el DataFrame
df['PUERTO DE DESEMBARQUE'] = df['PUERTO DE DESEMBARQUE'].replace(menos_frecuentes, 'other_ports')

# Calcular frecuencias y porcentajes
frecuencias = df['PAIS DE ORIGEN'].value_counts(normalize=True) * 100
menos_frecuentes = frecuencias[frecuencias < 0.5].index
df['PAIS DE ORIGEN'] = df['PAIS DE ORIGEN'].replace(menos_frecuentes, 'other_countries')



# COMPANY PRE-PROCESSING:

# GROUP ALL THE OBSERVATIONS WITH THE SAME COMPANY (but with different names)
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].fillna('other_countries')
maersk_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'MAERSK' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(maersk_variants, 'MAERSK')
# Identify and replace ZIM-related variants
zim_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'ZIM' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(zim_variants, 'ZIM')
# Identify and replace MSC-related variants
msc_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'MSC' in name or 'MEDITERRANEAN' in name or 'MEDIT.' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(msc_variants, 'MSC')
pil_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'PIL' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(pil_variants, 'PIL')
hyundai_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'HYUNDAI' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(hyundai_variants, 'HYUNDAI')
cosco_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'COSCO' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(cosco_variants, 'COSCO')
oocl_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'OOCL' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(oocl_variants, 'OOCL')
hapag_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'HAPAG' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(hapag_variants, 'HAPAG-LLOYD')
one_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'ONE' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(one_variants, 'ONE')
hamburg_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'HAMBURG' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(hamburg_variants, 'HAMBURG')
agunsa_variants = [name for name in df['COMPANIA DE TRANSPORTE'].unique() if 'AGUNSA' in name]
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(agunsa_variants, 'AGUNSA')

replacements = {
    'WAN HAI': ['WAN HAI. NAVEPAC','WAN HAI LINES', 'WAN HAI LINE', 'WAN HAI LINES LTD', 'WAN HAI LINES LTDA', 'WHL WAN HAI LINE', 'WAN HAI LINE LINES L'],
    'COSCO': ['COSCO SHIPPING', 'COSCO SHIPPING LINE', 'COSCO SHIPPING CO', 'COSCO SHIPPING LINES', 'COSCO SHIPPIN LINES', 'COSCO SHIPPING BULK', 'COSCO CHILE', 'COSCO CONTAINER', 'COSCO LINE', 'COSCO LINE.'],
    'CMA CGM': ['CMA CGM', 'CMA CGM CHILE', 'CMA-CGM', 'CMACGM', 'CMA CGM S.A.', 'CMA CGM CHILR LTDA', 'CMA  CGM', 'CMA CMG'],
    'EVERGREEN': ['EVERGREEN  INTL.','EVERGREEN MARINE', 'EVERGREEN LINE', 'EVERGREEN CHILE', 'EVERGREEN INTL.', 'EVER GREEN', 'EVERGREEN SHIPPING A', 'EVERGEEN SHIPPING AG','EVERGREEN  LINE'],
    'ONE': ['O.N.E.','ONE OCEAN','OCEAN NETWORK', 'OCEAN NEXTWORK EXPRE', 'OCEAN NETWORK EXP.', 'ONE OCEAN NETWORK EX', 'ONE NETWORK EXPRESS', 'ONE - OCEAN NETWORK', 'ONE-OCEAN NETWORK EX', 'OCEAN  NETWORK EXP', 'OCEAN  NETWORK EXPRE', 'OCEAN NERTWORK EXPRE', 'ONE LINE', 'ONE LINES', 'OCEAN EXPRESS', 'OCEAN NET.EXPRESS'],
    'YANG MING': ['YANG MING', 'YAN MING', 'YANGMING', 'YANG MING MARINE', 'YANG MING LINE', 'YANG  MING'],
    'HAPAG-LLOYD': ['HAPAG LLOYD', 'HAPAG LLOYD CHILE', 'HAPAG LLOYD SPA', 'HAPAG LLOYD HAMBURGO', 'HAPAG LLOYD HAMBURG', 'HAPAG-LLOYD','HAPAG LLO'],
    'PACIFIC INT. LINE': ['PACIFIC INT LINE', 'PACIFIC INT.LINE', 'PACIFIC INTER. LINE', 'PACIFIC INTL LINE', 'PACIFIC INTER LINE',
                          'PACIFIC INT. LINES(P)', 'PACIFIC INTL.LINES(P', 'PACIFIC INT. LINES (', 'PACIFIC INT. LINES L',
                          'PACIFIC INT,LINE', 'PACIFIC INTERNAT.LIN', 'PACIFIC INTERN.LINE', 'PACIFIC INTER.LINES',
                          'PACIFIC INTERNATIONA'],
    'HYUNDAI': ['HYUNDAI MERCHANT', 'HYUNDAI MERCHANT MAR', 'HMM HYUNDAI MERCHANT', 'HYNDAI MERCHANT MARI', 'HMM', 'HYUNDAY'],
    'MAERSK': ['MAERKS CHILE S.A.', 'MAERKS LINE S.A'],
    'MSC': ['MEDITERR.SHIPPING CO','M.S.C.', 'MEDITERRAN SHIPPING', 'MEDITERRANEA SHIPPIN','MED.SHIPP.','MEDITER.SH.CO.S.A.','MED.SHIPPING COMPANY','MARITIMA DEL MEDITER','MEDITERRANEO CARGO S'],
    'PIL': ['PIL-PACIFIC INTERNAT', 'PIL PACIFIC INTERNAT', 'PIL (PACIFIC INTERNA', 'P.I.L.', 'P.I.L','PACIFIC INT. LINE','PACIFIC INT. LINES(P','PACIFIC INT. LINE','PACIFIC INT. LINE','PACIFIC INT. LINE'],
    'OOCL': ['O.O.C.L.','OOCL ORIENT OVERSEAS', 'OOCL  ORIENT OVERSEA', 'ORIENT OVERSEAS CONT', 'ORIENTAL OVERSEAS CO','OOCL SHIPPING','OOCL CONT LINE','OOCL LOGISTICS LIMIT'],
    'ULTRAMAR': ['ULTRAMAR AG.M.LTDA.','ULTRAMAR AGENCIA MAR', 'ULTRAMAR AGENCIAS MA'],
    'SPARX': ['SPARX LOGISTIC CHILE', 'SPARX LOGISTICS CHIL'],
    'SAVINO DEL BENE': ['SAVINO DEL BENE', 'SAVINO DEL BENE CHIL','SAVINO DEL BENE'],
    'AGUNSA': ['AGENCIA UNIVERSALES', 'AGENCIAS UNIVERSALES', 'AGENCIAS UNIVER', 'AGUNSA UNIVERSALES','AGUNSA LOGISTICS']
}
def reemplazar_nombre_compania(nombre):
    for estandar, variantes in replacements.items():
        if any(var in nombre for var in variantes):
            return estandar
    return nombre
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].astype(str).apply(reemplazar_nombre_compania)
# Calcular frecuencias y porcentajes
frecuencias = df['COMPANIA DE TRANSPORTE'].value_counts(normalize=True) * 100
# Identificar los que representan menos del 0.5%
menos_frecuentes = frecuencias[frecuencias < 0.5].index
# create the other_countries category
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(menos_frecuentes, 'other_countries')
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace('NO EXISTE', 'other_countries')







# drop some columns
df = df.drop(columns=['ADUANA','DIGITO VERIFICADOR RUT','PROBABLE IMPORTADOR','PRODUCTO', 'MARCA', 'VARIEDAD',
                      'DESCRIPCION','VIA DE TRANSPORTE','DESCRIPCION ARANCELARIA','NUM DE ITEM',
                      'FORMA PAGO','TIPO DE CARGA','TIPO DE OPERACION','PAIS COMPANIA DE TRANSPORTE','ALMACEN',
                      'ZONA ECONOMICA','CLAVE ECONOMICA IMPORTADOR','ACUERDO COMERCIAL','EMISOR','ESTADO DE MERCANCIA',
                      'NRO DOC. TRANSPORTE','FECHA DE MANIFIESTO','FECHA DE ALMACEN','FECHA DOC. TRANSPORTE',
                      'NRO DE MANIFIESTO','IMPUESTO US$','IMPUESTO US$','IMPUESTO','TOTAL IVA'])


df = df.drop(columns=['CIF TOTAL','US$ CIF','US$ CIF UNIT','US$ FOB UNIT','CANTIDAD','US$ FOB','US$ FLETE','US$ SEGURO',
                      'FECHA_DOC_TRANSPORTE','PAIS DE ADQUISICION','FOB TOTAL','FLETE TOTAL','SEGURO TOTAL',
                      'day','month','year','PESO BRUTO TOTAL','UNIDAD','ITEMS TOTALES'])

df.to_csv("processed_df.csv", index=False)



# CONSTRUCT NEW DF WITH DAILY AGREGGATED DATA

# create a df for the aggregated data by day
daily_df = pd.DataFrame({'FECHA': df['FECHA'].dropna().sort_values().unique()})
din_diario = df.groupby('FECHA')['NUMERO DE ACEPTACION'].nunique().reset_index()
din_diario.rename(columns={'NUMERO DE ACEPTACION': 'DIN_UNICOS_DIARIOS'}, inplace=True)
# Unir al DataFrame diario
daily_df = pd.merge(daily_df, din_diario, on='FECHA', how='left')

# UNIQUE COUNT OF IMPORTER COMPANY AND PARTIDA ARACELARIA
rut_diario = df.groupby('FECHA')['RUT PROBABLE IMPORTADOR'].nunique().reset_index()
rut_diario.rename(columns={'RUT PROBABLE IMPORTADOR': 'RUTS_UNICOS_DIARIOS'}, inplace=True)

partida_diario = df.groupby('FECHA')['PARTIDA ARANCELARIA'].nunique().reset_index()
partida_diario.rename(columns={'PARTIDA ARANCELARIA': 'PARTIDAS_UNICAS_DIARIAS'}, inplace=True)

daily_df = daily_df.merge(rut_diario, on='FECHA', how='left')
daily_df = daily_df.merge(partida_diario, on='FECHA', how='left')





# TOTAL CONTAINER COUNT
total_contenedores_diario = df.groupby('FECHA')['CANTIDAD DE BULTO'].sum().reset_index()
total_contenedores_diario.rename(columns={'CANTIDAD DE BULTO': 'TOTAL_TEUS'}, inplace=True)
daily_df = pd.merge(daily_df, total_contenedores_diario, on='FECHA', how='left')

# total container refrigerados
total_contenedores_diario_ref = df.groupby('FECHA')['TOTAL_TEU_REFRIGERADOS'].sum().reset_index()
total_contenedores_diario_ref.rename(columns={'TOTAL_TEU_REFRIGERADOS': 'TOTAL_TEUS'}, inplace=True)
daily_df = pd.merge(daily_df, total_contenedores_diario_ref, on='FECHA', how='left')




cols = ['DIFF FECHA DIN Y DOC TRANSPORTE','FOB_POR_BULTO', 'SEGURO_POR_BULTO', 'FLETE_POR_BULTO',
        'PESO BRUTO POR CONTENEDOR', 'ITEMS POR CONTENEDOR']
agg_funcs = ['mean', 'min', 'max']

for col in cols:
    agg_df = df.groupby('FECHA')[col].agg(agg_funcs).reset_index()
    agg_df.columns = ['FECHA'] + [f'{func.upper()}_{col}' for func in agg_funcs]
    daily_df = pd.merge(daily_df, agg_df, how='left', on='FECHA')



# CLAUSULA count
#todo I can group some categories in others
clausula_container_count = df.groupby(['FECHA', 'CLAUSULA'])['CANTIDAD DE BULTO'].sum().reset_index()
pivot_clausula_count = clausula_container_count.pivot(index='FECHA', columns='CLAUSULA', values='CANTIDAD DE BULTO').fillna(0).reset_index()
daily_df = pd.merge(daily_df, pivot_clausula_count, how='left', on='FECHA')




# containers per company count
company_container_count = df.groupby(['FECHA', 'COMPANIA DE TRANSPORTE'])['CANTIDAD DE BULTO'].sum().reset_index()
pivot_company_count = company_container_count.pivot(index='FECHA', columns='COMPANIA DE TRANSPORTE', values='CANTIDAD DE BULTO').fillna(0).reset_index()
daily_df = pd.merge(daily_df, pivot_company_count, how='left', on='FECHA')

# containers per container type count
container_type_container_count = df.groupby(['FECHA', 'TIPO DE BULTO'])['CANTIDAD DE BULTO'].sum().reset_index()
pivot_container_type_count = container_type_container_count.pivot(index='FECHA', columns='TIPO DE BULTO',
                                                    values='CANTIDAD DE BULTO').fillna(0).reset_index()
daily_df = pd.merge(daily_df, pivot_container_type_count, how='left', on='FECHA')


# container per origin country
container_country_count = df.groupby(['FECHA', 'PAIS DE ORIGEN'])['CANTIDAD DE BULTO'].sum().reset_index()
pivot_country_count = container_country_count.pivot(index='FECHA', columns='PAIS DE ORIGEN',
                                                                  values='CANTIDAD DE BULTO').fillna(0).reset_index()
daily_df = pd.merge(daily_df, pivot_country_count, how='left', on='FECHA')


del df
gc.collect()

# Agrupar por FECHA, PUERTO DE EMBARQUE y DESEMBARQUE, sumando CANTIDAD DE BULTO
# containers_by_route = df.groupby(['FECHA', 'PUERTO DE EMBARQUE', 'PUERTO DE DESEMBARQUE'])['CANTIDAD DE BULTO'].sum().reset_index()
# containers_by_route['RUTA'] = containers_by_route['PUERTO DE EMBARQUE'].str.lower().str.strip() + " - " + containers_by_route['PUERTO DE DESEMBARQUE'].str.lower().str.strip()
# pivot_routes = containers_by_route.pivot_table(index='FECHA', columns='RUTA', values='CANTIDAD DE BULTO', fill_value=0).reset_index()
# daily_df = pd.merge(daily_df, pivot_routes, on='FECHA', how='left')




#ADD MACRO DATA


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






daily_df.to_csv("test_daily.csv", index=False)





# numeric_df = daily_df.select_dtypes(include='number')
#
# # Compute correlation matrix
# correlation_matrix = numeric_df.corr()
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Set up the plot
# plt.figure(figsize=(14, 12))
# sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", square=True)
# plt.title("Correlation Matrix Heatmap", fontsize=16)
# plt.tight_layout()
# plt.show()
#
# (
# # check grouped mean values
# df.groupby('TIPO DE BULTO')[['PESO BRUTO TOTAL','FLETE TOTAL']].mean()