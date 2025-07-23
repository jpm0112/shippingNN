import pandas as pd
import numpy as np
import glob
import os
import gc

# --- 1. Configuration & Constants ---

# Define file paths
DATA_FOLDER = r"C:\Users\JP\OneDrive - Auburn University\Research - port shipping cost\dataset\container_data"
MACRO_FOLDER = r"C:\Users\JP\OneDrive - Auburn University\Research - port shipping cost\dataset\macrodata"
OUTPUT_PROCESSED_DF = "processed_df.csv"
OUTPUT_DAILY_DF = "daily_df.csv"

# Define column names and types for faster loading and consistency
# EFFICIENCY: Specifying dtypes on read avoids costly type inference and reduces memory.
DTYPES = {
    'DIA': 'int8', 'MES': 'int8', 'ANO': 'int16',
    'CANTIDAD DE BULTO': 'float32', 'PESO BRUTO TOTAL': 'float32',
    'FOB TOTAL': 'float32', 'FLETE TOTAL': 'float32', 'SEGURO TOTAL': 'float32',
    'ITEMS TOTALES': 'float32'
}
NEW_COLUMNS = [
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

# Consolidated mapping for transport companies
# EFFICIENCY: Consolidating all replacements into one dict allows for a single, fast operation.
COMPANY_REPLACEMENTS = {
    'MAERSK': ['MAERSK', 'MAERKS'],
    'ZIM': ['ZIM'],
    'MSC': ['MSC', 'MEDITERRANEAN', 'MEDITERR.'],
    'PIL': ['PIL', 'PACIFIC INT'],
    'HYUNDAI': ['HYUNDAI', 'HMM'],
    'COSCO': ['COSCO'],
    'OOCL': ['OOCL', 'ORIENT OVERSEAS'],
    'HAPAG-LLOYD': ['HAPAG'],
    'ONE': ['ONE', 'OCEAN NETWORK'],
    'HAMBURG': ['HAMBURG'],
    'AGUNSA': ['AGUNSA', 'AGENCIA UNIVERSALES'],
    'WAN HAI': ['WAN HAI'],
    'CMA CGM': ['CMA CGM', 'CMA-CGM', 'CMACGM'],
    'EVERGREEN': ['EVERGREEN', 'EVER GREEN'],
    'YANG MING': ['YANG MING', 'YAN MING', 'YANGMING'],
    'ULTRAMAR': ['ULTRAMAR'],
    'SPARX': ['SPARX'],
    'SAVINO DEL BENE': ['SAVINO DEL BENE']
}


# --- 2. Helper Functions ---

def group_infrequent(series: pd.Series, threshold: float = 0.05) -> pd.Series:
    """Groups infrequent categories into 'OTHERS'."""
    counts = series.value_counts(normalize=True)
    infrequent = counts[counts < threshold].index
    return series.replace(infrequent, 'OTHERS')


def standardize_company_names(series: pd.Series, replacements: dict) -> pd.Series:
    """Standardizes company names using a dictionary of keywords."""
    # EFFICIENCY: np.select is a vectorized conditional operation, much faster than .apply()
    s = series.str.upper().fillna('OTHERS')
    conditions = [s.str.contains(key) for key in replacements.keys()]
    choices = list(replacements.values())
    return pd.Series(np.select(conditions, choices, default=s), index=series.index)


# --- 3. Data Loading and Initial Processing ---

print("Loading and concatenating files...")
all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
# EFFICIENCY: Using a generator expression can be slightly more memory-friendly
df_list = (pd.read_csv(f, dtype=DTYPES, low_memory=False) for f in all_files)
df = pd.concat(df_list, ignore_index=True)
df.columns = NEW_COLUMNS

print("Filtering for container data...")
contenedores = [
    'CONTENEDOR 20', 'CONTENEDOR 40', 'CONTENEDOR REFRIGERADO 40',
    'CONTENEDOR NO REFRIGERADO', 'CONTENEDOR REFRIGERADO 20'
]
df['TIPO DE BULTO'] = df['TIPO DE BULTO'].astype('category')
df = df[df['TIPO DE BULTO'].isin(contenedores)].copy()

print("Cleaning data and creating features...")
# Date handling
df['FECHA'] = pd.to_datetime(df[['ANO', 'MES', 'DIA']].rename(columns={'ANO': 'year', 'MES': 'month', 'DIA': 'day'}),
                             errors='coerce')
df['FECHA_DOC_TRANSPORTE'] = pd.to_datetime(df['FECHA DOC. TRANSPORTE'], format='%d%m%Y', errors='coerce')
df['DIFF FECHA DIN Y DOC TRANSPORTE'] = (df['FECHA'] - df['FECHA_DOC_TRANSPORTE']).dt.days

# New features per container
for col in ['FOB TOTAL', 'FLETE TOTAL', 'SEGURO TOTAL', 'PESO BRUTO TOTAL', 'ITEMS TOTALES']:
    new_col_name = col.replace(' TOTAL', '').replace(' ', '_') + '_POR_BULTO'
    # Use np.divide for safe division (handles division by zero)
    df[new_col_name] = np.divide(df[col], df['CANTIDAD DE BULTO'])

# Clean categorical columns
df['PUERTO DE EMBARQUE'] = group_infrequent(df['PUERTO DE EMBARQUE'], threshold=0.05)
df['PUERTO DE DESEMBARQUE'] = group_infrequent(df['PUERTO DE DESEMBARQUE'], threshold=0.05)
df['PAIS DE ORIGEN'] = group_infrequent(df['PAIS DE ORIGEN'], threshold=0.05)

# Standardize company names
# EFFICIENCY: This single call replaces all the previous loops and the slow .apply() function.
company_keyword_map = {keyword: std_name for std_name, keywords in COMPANY_REPLACEMENTS.items() for keyword in keywords}
df['COMPANIA DE TRANSPORTE'] = standardize_company_names(df['COMPANIA DE TRANSPORTE'], company_keyword_map)
df['COMPANIA DE TRANSPORTE'] = group_infrequent(df['COMPANIA DE TRANSPORTE'], threshold=0.005)
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace('NO EXISTE', 'OTHERS').astype('category')

print("Dropping unnecessary columns...")
cols_to_drop = [
    'ADUANA', 'DIGITO VERIFICADOR RUT', 'PROBABLE IMPORTADOR', 'PRODUCTO', 'MARCA', 'VARIEDAD',
    'DESCRIPCION', 'VIA DE TRANSPORTE', 'DESCRIPCION ARANCELARIA', 'NUM DE ITEM',
    'FORMA PAGO', 'TIPO DE CARGA', 'TIPO DE OPERACION', 'PAIS COMPANIA DE TRANSPORTE', 'ALMACEN',
    'ZONA ECONOMICA', 'CLAVE ECONOMICA IMPORTADOR', 'ACUERDO COMERCIAL', 'EMISOR', 'ESTADO DE MERCANCIA',
    'NRO DOC. TRANSPORTE', 'FECHA DE MANIFIESTO', 'FECHA DE ALMACEN', 'FECHA DOC. TRANSPORTE',
    'NRO DE MANIFIESTO', 'IMPUESTO US$', 'IMPUESTO', 'TOTAL IVA', 'CIF TOTAL', 'US$ CIF',
    'US$ CIF UNIT', 'US$ FOB UNIT', 'CANTIDAD', 'US$ FOB', 'US$ FLETE', 'US$ SEGURO',
    'FECHA_DOC_TRANSPORTE', 'PAIS DE ADQUISICION', 'FOB TOTAL', 'FLETE TOTAL', 'SEGURO TOTAL',
    'DIA', 'MES', 'ANO', 'PESO BRUTO TOTAL', 'UNIDAD', 'ITEMS TOTALES'
]
df.drop(columns=cols_to_drop, inplace=True)
df.to_csv(OUTPUT_PROCESSED_DF, index=False)

# --- 4. Daily Aggregation ---
print("Aggregating data by day...")

# EFFICIENCY: A single, comprehensive groupby.agg is much faster than multiple group-by and merge operations.
agg_dict = {
    'NUMERO DE ACEPTACION': pd.Series.nunique,
    'RUT PROBABLE IMPORTADOR': pd.Series.nunique,
    'PARTIDA ARANCELARIA': pd.Series.nunique,
    'DIFF FECHA DIN Y DOC TRANSPORTE': ['mean', 'min', 'max'],
    'FOB_POR_BULTO': ['mean', 'min', 'max'],
    'FLETE_POR_BULTO': ['mean', 'min', 'max'],
    'SEGURO_POR_BULTO': ['mean', 'min', 'max'],
    'PESO_BRUTO_POR_BULTO': ['mean', 'min', 'max'],
    'ITEMS_POR_BULTO': ['mean', 'min', 'max'],
}

daily_df = df.groupby('FECHA').agg(agg_dict).reset_index()

# Flatten the multi-level column index
daily_df.columns = ['_'.join(col).strip('_').upper() for col in daily_df.columns.values]
daily_df.rename(columns={'FECHA': 'FECHA'}, inplace=True)  # Fix column name after join


# EFFICIENCY: Use unstack() for pivoting, which is often faster than pivot_table.
def pivot_and_merge(main_df, series_df, group_col, value_col, agg_func='sum'):
    """Helper to perform pivot-like aggregation and merge."""
    pivot_df = series_df.groupby(['FECHA', group_col])[value_col].agg(agg_func).unstack(fill_value=0)
    pivot_df = pivot_df.add_prefix(f'COUNT_{group_col}_')  # Add prefix for clarity
    return main_df.merge(pivot_df, on='FECHA', how='left')


daily_df = pivot_and_merge(daily_df, df, 'CLAUSULA', 'CANTIDAD DE BULTO')
daily_df = pivot_and_merge(daily_df, df, 'COMPANIA DE TRANSPORTE', 'CANTIDAD DE BULTO')
daily_df = pivot_and_merge(daily_df, df, 'TIPO DE BULTO', 'CANTIDAD DE BULTO')
daily_df = pivot_and_merge(daily_df, df, 'PAIS DE ORIGEN', 'CANTIDAD DE BULTO')

del df
gc.collect()

# --- 5. Merge Macroeconomic Data ---
print("Merging macroeconomic data...")
original_columns = set(daily_df.columns)
for filename in os.listdir(MACRO_FOLDER):
    if filename.endswith(".csv"):
        file_path = os.path.join(MACRO_FOLDER, filename)
        df_macro = pd.read_csv(file_path, parse_dates=['Date'])
        df_macro.rename(columns={'Date': 'FECHA'}, inplace=True)
        df_macro = df_macro.sort_values('FECHA')

        df_macro[['Close', 'Volume']] = df_macro[['Close', 'Volume']].fillna(method='ffill').fillna(method='bfill')
        df_macro['Close_pct_change'] = df_macro['Close'].pct_change().fillna(0)

        name = filename.replace('.csv', '').lower()
        macro_filtered = df_macro[['FECHA', 'Close', 'Volume', 'Close_pct_change']].rename(
            columns={'Close': f'{name}_price', 'Volume': f'{name}_volume', 'Close_pct_change': f'{name}_pct_change'}
        )
        daily_df = pd.merge(daily_df, macro_filtered, how='left', on='FECHA')

# Forward-fill and back-fill any gaps created by merging time-series data
new_columns = [col for col in daily_df.columns if col not in original_columns]
daily_df[new_columns] = daily_df[new_columns].fillna(method='ffill').fillna(method='bfill')

# --- 6. Save Final Output ---
print(f"Saving final daily aggregated data to {OUTPUT_DAILY_DF}...")
daily_df.to_csv(OUTPUT_DAILY_DF, index=False)
print("Done.")