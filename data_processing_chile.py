import pandas as pd
import glob
import os
import gc

folder = r"C:\Users\JP\OneDrive - Auburn University\Research - port shipping cost\dataset\container_data"
all_files = glob.glob(os.path.join(folder, "*.csv"))

df_list = [pd.read_csv(f) for f in all_files[0:3]]
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

df = df.applymap(lambda x: x.replace('Ñ', 'N') if isinstance(x, str) else x)
df = df.applymap(lambda x: x.replace('Ã‘', 'N') if isinstance(x, str) else x)




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

df = df.drop(columns=["DIGITO VERIFICADOR RUT", 'PRODUCTO', 'MARCA', 'VARIEDAD', 'DESCRIPCION',
                      'VIA DE TRANSPORTE','FORMA PAGO','TIPO DE CARGA',"ZONA ECONOMICA",
                      'CLAVE ECONOMICA IMPORTADOR','ALMACEN','FECHA DE ALMACEN','ACUERDO COMERCIAL',
                      'ESTADO DE MERCANCIA',"EMISOR",'PROBABLE IMPORTADOR','PAIS DE ADQUISICION',
                      'TIPO DE OPERACION','DESCRIPCION ARANCELARIA'])

df = df.groupby("NUMERO DE ACEPTACION", as_index=False).agg({
    'FECHA': 'first',
    'day': 'first',
    'month': 'first',
    'year': 'first',
    'ADUANA': 'first',
    'RUT PROBABLE IMPORTADOR': 'first',
    'PARTIDA ARANCELARIA': 'nunique',
    'PAIS DE ORIGEN': 'first',
    'PUERTO DE EMBARQUE': 'first',
    'PUERTO DE DESEMBARQUE': 'first',
    'COMPANIA DE TRANSPORTE': 'first',
    'TIPO DE BULTO': 'first',
    'PESO BRUTO TOTAL': 'first',
    'CLAUSULA': 'first',
    'IMPUESTO': 'first',
    'CANTIDAD': 'sum',
    'UNIDAD': 'first',
    'US$ FOB': 'sum',
    'US$ FLETE': 'sum',
    'US$ SEGURO': 'sum',
    'US$ CIF': 'sum',
    'US$ CIF UNIT': 'first',
    'NUM DE ITEM': 'max',
    'PAIS COMPANIA DE TRANSPORTE': 'first',
    'IMPUESTO US$': 'sum',
    'CANTIDAD DE BULTO': 'first',
    'NRO DE MANIFIESTO': 'first',
    'NRO DOC. TRANSPORTE': 'first',
    'FECHA DOC. TRANSPORTE': 'first',
    'ITEMS TOTALES': 'first',
    'FOB TOTAL': 'first',
    'FLETE TOTAL': 'first',
    'SEGURO TOTAL': 'first',
    'CIF TOTAL': 'first',
    'TOTAL IVA': 'first',
    'US$ FOB UNIT': 'first',
    'CANTIDAD UNIDADES FISICAS': 'first',
    'UNIDAD DE MEDIDA FISICA': 'first',
    'FECHA': 'first'

})



df['FECHA_DOC_TRANSPORTE'] = pd.to_datetime(df['FECHA DOC. TRANSPORTE'], format='%d%m%Y', errors='coerce')

df['DIFF FECHA DIN Y DOC TRANSPORTE'] = (df['FECHA'] - df['FECHA_DOC_TRANSPORTE']).dt.days

map_teu = {
    "CONTENEDOR 20": 1,
    "CONTENEDOR 40": 2,
    "CONTENEDOR REFRIGERADO 40": 2,
    "CONTENEDOR REFRIGERADO 20": 1
}
df["CANTIDAD DE BULTO"] = pd.to_numeric(df["CANTIDAD DE BULTO"], errors="raise")
df["TEU_factor"] = df["TIPO DE BULTO"].map(map_teu).astype("float64")
df["TEU"] = df["CANTIDAD DE BULTO"].astype("float64") * df["TEU_factor"]

df.drop(columns=['US$ FOB', 'US$ FLETE', 'US$ SEGURO', 'US$ CIF', 'NUM DE ITEM', 'CANTIDAD', 'UNIDAD','TEU_factor'], inplace=True)



df['FOB_per_TEU'] = df['FOB TOTAL'] / df['TEU']
df['FLETE_per_TEU'] = df['FLETE TOTAL'] / df['TEU']
df['SEGURO_per_TEU'] = df['SEGURO TOTAL'] / df['TEU']

df['PESO BRUTO PER TEU'] = df['PESO BRUTO TOTAL'] / df['TEU']
df['ITEMS PER TEU'] = df['ITEMS TOTALES'] / df['TEU']

grouped_coasts = {
    'GALVESTON': 'NAE',
    'OTROS PTOS VENEZUELA': 'SAE',
    'PUERTO ANGAMOS': 'SAW',
    'SAN  ANTONIO': 'SAW',
    'OTROS PTOS.INGLATERR': 'NE',
    'OTROS PTOS.AMERICA': 'NAE',
    'LISBOA': 'SE',
    'ZEEBRUGGE': 'NE',
    'NAPOLES': 'SE',
    'DAIREN': 'FE',
    'SALVADOR': 'SAE',
    'CHARLESTON': 'NAE',
    'KEELUNG': 'FE',
    'OTROS PTOS. CHILENOS': 'SAW',
    'OTROS PTOS.DE ITALIA': 'SE',
    'NEW HAVEN': 'NAE',
    'OTROS PTOS. COLOMBIA': 'SAW',
    'HOUSTON': 'NAE',
    'OTROS PTOS. DE PERU': 'SAW',
    'SALERNO': 'SE',
    'OTROS PTOS.EUROPA': 'NE',
    'CUTTER COVE': 'NAW',
    'PALENA CARRENLEUFU': 'SAW',
    'SEATTLE': 'NAW',
    'LOS ANGELES': 'NAW',
    'RIJEKA': 'SE',
    'CUXHAVEN': 'NE',
    'OTROS PTOS.HOLANDA': 'NE',
    'AMSTERDAM': 'NE',
    'TORONTO': 'NAE',
    'PORT ARTHUR': 'NAE',
    'OTROS PTOS.TAIWAN': 'FE',
    'LIECHTENSTEIN': 'NE',
    'OTROS PTOS.BRASIL': 'SAE',
    'MENDOZA': 'SAE',
    'OTROS PTOS. ECUADOR': 'SAW',
    'BALTIMORE': 'NAE',
    'CORONEL': 'SAW',
    'AEROPUERTO COM. A. MERINO B.': 'SAW',
    'OTROS PTOS PORTUGAL': 'SE',
    'OTROS PTOS.PORTUGAL': 'SE',
    'PORTLAND': 'NAW',
    'KOTKA': 'NE',
    'OULO': 'NE',
    'OSLO': 'NE',
    'OTROS PTO.NORUEGA': 'NE',
    'HELSINKI': 'NE',
    'SANTOS': 'SAE',
    'BILBAO': 'SE',
    'GOTEMBURGO': 'NE',
    'FRANKFURT': 'NE',
    'OLDENBURG': 'NE',
    'LE HAVRE': 'NE',
    'YOKOHAMA': 'FE',
    'CRISTOBAL': 'NAE',
    'COSTA DEL PACIFICO': 'NAW',
    'GNL MEJILLONES': 'SAW',
    'CURACAO': 'NAE',
    'OTROS PTOS. CHILENOS': 'SAW',
    'SAN FRANCISCO': 'NAW',
    'ROSTOCK': 'NE',
    'BARRANQUILLA': 'NAE',
    'CIUDAD DEL CABO': 'AF',
    'HELSIMBORG': 'NE',
    'AUGUSTA': 'SE',
    'OTROS PTOS. PANAMA': 'NAE',
    'OTROS PTOS.ALEMANIA': 'NE',
    'OTROS PTOS BULGARIA': 'SE',
    'OTROS PTOS.SUECIA': 'NE',
    'PARANAGUA': 'SAE',
    'OSAKA': 'FE',
    'OTROS PTO.SUD AFRIC': 'AF',
    'VANCOUVER': 'NAW',
    'SALINAS': 'SAW',
    'COLUMBRES': 'SE',
    'NEW YORK': 'NAE',
    'PUERTOS DEL GOLFO': 'NAE',
    'COSTA DE ATLANTICO': 'NAE',
    'TAMPA': 'NAE',
    'QUEBEC': 'NAE',
    'OTROS PUERTOS CANADA': 'NAE',
    'AMBERES': 'NE',
    'LIVERPOOL': 'NE',
    'SAVONA': 'SE',
    'OTROS PTOS.ESPANA': 'SE',
    'LIORNA,LIVORNO': 'SE',
    'OTROS PTO.OCEANIA': 'OC',
    'NORFOLK': 'NAE',
    'VALENCIA': 'SE',
    'COLON': 'NAE',
    'OTROS PTOS.FRANCIA': 'SE',
    'OTROS PTOS.ARGENTINA': 'SAE',
    'VERACRUZ': 'NAE',
    'BREMEN': 'NE',
    'OTROS PTOS PORTUGAL': 'SE',
    'OTROS PTOS.ESPANA': 'SE',
    'AALBORG': 'NE',
    'OTROS PTOS.JAPONESES': 'FE',
    'BUENAVENTURA': 'SAW',
    'OTROS PTO.BANGLADESH': 'ME',
    'OTROS PTOS. ECUADOR ': 'SAW',
    'OTROS PTOS.TAIWAN ': 'FE',
    'COATZACOALES': 'NAE',
    'PANAMA': 'NAW',
    'OTROS PTOS.DE COREA': 'FE',
    'OTROS PTOS.ARGENTINA ': 'SAE',
    'IQUIQUE': 'SAW',
    'RIO DE JANEIRO': 'SAE',
    'RIO GRANDE DEL SUR': 'SAE',
    'BUENOS AIRES': 'SAE',
    'OTROS PTOS.URUGUAY': 'SAE',
    'MONTEVIDEO': 'SAE',
    'BAHIA BLANCA': 'SAE',
    'SAO PAULO': 'SAE',
    'ARICA': 'SAW',
    'MEJILLONES': 'SAW',
    'PATILLOS': 'SAW',
    'CALDERA': 'SAW',
    'SAN VICENTE': 'SAW',
    'LIRQUEN': 'SAW',
    'TALCAHUANO': 'SAW',
    'VALPARAISO': 'SAW',
    'SAN  ANTONIO': 'SAW',
    'GUAYACAN': 'SAW',
    'OTROS PTOS. DE PERU': 'SAW',
    'CALLAO': 'SAW',
    'ILO': 'SAW',
    'GUAYAQUIL': 'SAW',
    'OTROS PTOS. ECUADOR': 'SAW',
    'OTROS PTOS.BRASIL': 'SAE',
    'SALVADOR': 'SAE',
    'SANTOS': 'SAE',
    'PARANAGUA': 'SAE',
    'RIO GRANDE DEL SUR': 'SAE',
    'AMSTERDAM': 'NE',
    'ROTTERDAM': 'NE',
    'HAMBURGO': 'NE',
    'LONDRES': 'NE',
    'LE HAVRE ': 'NE',
    'LA PALLICE': 'NE',
    'ZEEBRUGGE ': 'NE',
    'BREMEN ': 'NE',
    'CUXHAVEN ': 'NE',
    'ROSTOCK ': 'NE',
    'FRANKFURT ': 'NE',
    'NUREMBERG': 'NE',
    'OLDENBURG ': 'NE',
    'OTROS PTOS.ALEMANIA ': 'NE',
    'OTROS PTOS.HOLANDA ': 'NE',
    'OTROS PTO.BELGICA': 'NE',
    'OTROS PTOS.INGLATERR ': 'NE',
    'COPENHAGEN': 'NE',
    'AARHUS': 'NE',
    'AALBORG ': 'NE',
    'GOTEMBURGO ': 'NE',
    'HELSIMBORG ': 'NE',
    'OSLO ': 'NE',
    'STAVANGER': 'NE',
    'HELSINKI ': 'NE',
    'KOTKA ': 'NE',
    'OULO ': 'NE',
    'OTROS PTO.NORUEGA ': 'NE',
    'OTROS  PTO.DINAMARCA': 'NE',
    'OTROS PTOS.SUECIA ': 'NE',
    'OTROS PTO.FINLANDIA': 'NE',
    'GENOVA': 'SE',
    'VALENCIA ': 'SE',
    'BARCELONA': 'SE',
    'BILBAO ': 'SE',
    'ALGECIRAS': 'SE',
    'CADIZ': 'SE',
    'SEVILLA': 'SE',
    'LISBOA ': 'SE',
    'SETUBAL': 'SE',
    'MARSELLA': 'SE',
    'LIORNA,LIVORNO ': 'SE',
    'SAVONA ': 'SE',
    'AUGUSTA ': 'SE',
    'NAPOLES ': 'SE',
    'SALERNO ': 'SE',
    'CONSTANZA': 'SE',
    'VARNA': 'SE',
    'RIJEKA ': 'SE',
    'OTROS PTOS.PORTUGAL ': 'SE',
    'OTROS PTOS.ESPANA ': 'SE',
    'OTROS PTOS.DE ITALIA ': 'SE',
    'OTROS PTOS.FRANCIA ': 'SE',
    'OTROS PTOS BULGARIA ': 'SE',
    'OTROS PTO.DE RUMANIA': 'SE',
    'HONG KONG': 'FE',
    'SHANGAI': 'FE',
    'OTROS PTOS.DE CHINA': 'FE',
    'DAIREN ': 'FE',
    'KAOHSIUNG': 'FE',
    'KEELUNG ': 'FE',
    'OTROS PTOS.TAIWAN  ': 'FE',
    'BUSAN CY (PUSAN)': 'FE',
    'OTROS PTOS.DE COREA ': 'FE',
    'YOKOHAMA ': 'FE',
    'KOBE': 'FE',
    'OSAKA ': 'FE',
    'NAGOYA': 'FE',
    'SHIMIZUI': 'FE',
    'FUKUYAMA': 'FE',
    'MOJI': 'FE',
    'OTROS PTOS.JAPONESES ': 'FE',
    'MANILA': 'FE',
    'OTROS PTOS.FILIPINAS': 'FE',
    'OTROS PTO.SINGAPURE': 'FE',
    'OTROS PTO.ASIATICOS': 'FE',
    'OTROS PTO.IRAN NO ES': 'ME',
    'OTROS PTO.INDIA NO E': 'FE',
    'CALCUTA': 'FE',
    'OTROS PTO.BANGLADESH ': 'FE',
    'DURBAM': 'AF',
    'CIUDAD DEL CABO ': 'AF',
    'OTROS PTO.DE AFRICA': 'AF',
    'OTROS PTO.SUD AFRIC ': 'AF',
    'SIDNEY': 'OC',
    'ADELAIDA': 'OC',
    'PREMANTLE': 'OC',
    'OTROS PTO.AUSTRALIA': 'OC',
    'OTROS PTO.OCEANIA ': 'OC',
    'LONG BEACH': 'NAW',
    'OAKLAND': 'NAW',
    'SEATTLE ': 'NAW',
    'SAN DIEGO': 'NAW',
    'SAN FRANCISCO ': 'NAW',
    'PORTLAND ': 'NAW',
    'VANCOUVER ': 'NAW',
    'BALBOA': 'NAW',
    'MANZANILLO': 'NAW',
    'MAZATLAN': 'NAW',
    'GUAYMAS': 'NAW',
    'COSTA DEL PACIFICO ': 'NAW',
    'OTROS PUERTOS MEXICO': 'NAW',
    'NEW YORK ': 'NAE',
    'MIAMI': 'NAE',
    'PHILADELPHIA': 'NAE',
    'EVERGLADES': 'NAE',
    'CHARLESTON ': 'NAE',
    'SAVANAH': 'NAE',
    'HOUSTON ': 'NAE',
    'NORFOLK ': 'NAE',
    'BALTIMORE ': 'NAE',
    'SAINT JOHN': 'NAE',
    'MOBILE': 'NAE',
    'NEW ORLEANS': 'NAE',
    'JACKSONVILLE': 'NAE',
    'TAMPA ': 'NAE',
    'PORT ARTHUR ': 'NAE',
    'GALVESTON ': 'NAE',
    'BOSTON': 'NAE',
    'HALIFAX': 'NAE',
    'MONTREAL': 'NAE',
    'QUEBEC ': 'NAE',
    'TORONTO ': 'NAE',
    'PITTSBURGH': 'NAE',
    'MILWAUKEE': 'NAE',
    'NEW HAVEN ': 'NAE',
    'COLON ': 'NAE',
    'CRISTOBAL ': 'NAE',
    'CURACAO ': 'NAE',
    'PUERTOS DEL GOLFO ': 'NAE',
    'GOLFO DE MEXICO': 'NAE',
    'COATZACOALES ': 'NAE',
    'VERACRUZ ': 'NAE',
    'TAMPICO ': 'NAE',
    'OTROS PUERTOS CANADA ': 'NAE',
    'OTROS PUERTOS EE.UU.': 'NAE',
    'EEUU': 'NAE',
    'BARRANQUILLA ': 'NAE',
    'LA GUAYRA': 'NAE',
    'OTROS ANT.HOLANDESA': 'NAE',
    'PANAMA ': 'NAW',
    'COSTA DE ATLANTICO ': 'NAE',
    'OTROS PTOS. COLOMBIA ': 'SAW',
    'AEROPUERTO COM. A. MERINO B. ': 'SAW',
    'CORONEL ': 'SAW',
    'COLUMBRES ': 'SE',
    '-': 'NAE',
    'LIECHTENSTEIN ': 'NE',
    'CUTTER COVE ': 'NAW',
    'TERR. ANTARTICO CHILE': 'SAW',
    'OTROS PTO.DE RUMANIA ': 'SE',
    'OTROS PTO.FINLANDIA ': 'NE',
    'OTROS  PTO.DINAMARCA ': 'NE',
    'OTROS PTOS BULGARIA  ': 'SE',
    'OTROS PTOS.DE CHINA ': 'FE',
    'HONG KONG ': 'FE',
    'JURELES': 'NAW',
    'TAMPICO': 'NAE',
    'ANTOFAGASTA': 'SAW',
    'URUGUAY': 'SAE',
    'ZONA FRANCA PUNTA ARENAS': 'SAW',
    'SAN PEDRO DE ATACAMA': 'SAW',
    'ZONA FRANCA IQUIQUE': 'SAW',
    'OLLAGUE': 'SAW',
    'POSEIDON': 'SAW',
    'PASO GUANACO SONSO': 'SAW',
    'SOCOMPA': 'SAW',
    'CABO NEGRO': 'SAW',
}
df['coast'] = df['PUERTO DE EMBARQUE'].map(grouped_coasts)

# Calcular frecuencias y porcentajes
frecuencias = df['PUERTO DE EMBARQUE'].value_counts(normalize=True)
menos_frecuentes = frecuencias[frecuencias < 0.050].index
df['PUERTO DE EMBARQUE'] = df['PUERTO DE EMBARQUE'].replace(menos_frecuentes, 'other_ports')

# Calcular frecuencias y porcentajes
frecuencias = df['PUERTO DE DESEMBARQUE'].value_counts(normalize=True)
# Identificar los que representan menos del 0.5%
menos_frecuentes = frecuencias[frecuencias < 0.010].index
# Reemplazar en el DataFrame
df['PUERTO DE DESEMBARQUE'] = df['PUERTO DE DESEMBARQUE'].replace(menos_frecuentes, 'other_ports')

# Calcular frecuencias y porcentajes

df = df[df["PAIS DE ORIGEN"] != "ORIGEN O DESTINO NO PRECISADO"]
frecuencias = df['PAIS DE ORIGEN'].value_counts(normalize=True)
menos_frecuentes = frecuencias[frecuencias < 0.005].index
df['PAIS DE ORIGEN'] = df['PAIS DE ORIGEN'].replace(menos_frecuentes, 'other_countries')

frecuencias = df['CLAUSULA'].value_counts(normalize=True)
menos_frecuentes = frecuencias[frecuencias < 0.01].index
df['CLAUSULA'] = df['CLAUSULA'].replace(menos_frecuentes, 'OTRO')

frecuencias = df['ADUANA'].value_counts(normalize=True)
menos_frecuentes = frecuencias[frecuencias < 0.1].index
df['ADUANA'] = df['ADUANA'].replace(menos_frecuentes, 'OTRA')



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
frecuencias = df['COMPANIA DE TRANSPORTE'].value_counts(normalize=True)
# Identificar los que representan menos del 0.5%
menos_frecuentes = frecuencias[frecuencias < 0.005].index
# create the other_countries category
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace(menos_frecuentes, 'other_companies')
df['COMPANIA DE TRANSPORTE'] = df['COMPANIA DE TRANSPORTE'].replace('NO EXISTE', 'other_companies')



df = df.drop(columns=['NRO DE MANIFIESTO','CANTIDAD UNIDADES FISICAS','UNIDAD DE MEDIDA FISICA'])



df = df[df["FLETE_per_TEU"] > 100]
df = df[df["FLETE_per_TEU"] < 500000]

df["FECHA"] = pd.to_datetime(df["FECHA DOC. TRANSPORTE"], format="%d%m%Y", errors="coerce")

df["WEEK"] = df["FECHA"].dt.to_period("W").dt.start_time

weekly_df = df.groupby("WEEK", as_index=False).agg({
    'NUMERO DE ACEPTACION': 'nunique',
    'FECHA': 'first',
    'RUT PROBABLE IMPORTADOR': 'nunique',
    'PARTIDA ARANCELARIA': 'nunique',
    'PESO BRUTO TOTAL': 'mean',
    'US$ CIF UNIT': 'mean',
    'IMPUESTO US$': 'mean',
    'CANTIDAD DE BULTO': 'mean',
    'NRO DOC. TRANSPORTE': 'nunique',
    'ITEMS TOTALES': 'mean',
    'FOB TOTAL': 'mean',
    'FLETE TOTAL': 'mean',
    'SEGURO TOTAL': 'mean',
    'CIF TOTAL': 'mean',
    'TOTAL IVA': 'mean',
    'US$ FOB UNIT': 'mean',
    'DIFF FECHA DIN Y DOC TRANSPORTE': 'mean',
    'TEU': 'mean',
    'TEU':'min',
    'TEU':'max',
    'FOB_per_TEU': 'mean',
    'FLETE_per_TEU': 'mean',
    'SEGURO_per_TEU': 'mean',
    'PESO BRUTO PER TEU': 'mean',
    'ITEMS PER TEU': 'mean',
    })


def pivot_and_merge(main_df, series_df, group_col, value_col, agg_func='sum'):
    """Helper to perform pivot-like aggregation and merge."""
    pivot_df = series_df.groupby(['WEEK', group_col])[value_col].agg(agg_func).unstack(fill_value=0)

    func_name = agg_func if isinstance(agg_func, str) else agg_func.__name__
    prefix = f"{func_name.upper()}_{value_col}_"
    pivot_df = pivot_df.add_prefix(prefix)  # Add prefix for clarity
    return main_df.merge(pivot_df, on='WEEK', how='left')


weekly_df = pivot_and_merge(weekly_df, df, 'ADUANA', 'TEU')
weekly_df = pivot_and_merge(weekly_df, df, 'PAIS DE ORIGEN', 'TEU')
weekly_df = pivot_and_merge(weekly_df, df, 'PAIS DE ORIGEN', 'FLETE_per_TEU', agg_func='mean')
weekly_df = pivot_and_merge(weekly_df, df, 'PUERTO DE EMBARQUE', 'TEU')
weekly_df = pivot_and_merge(weekly_df, df, 'PUERTO DE DESEMBARQUE', 'TEU')
weekly_df = pivot_and_merge(weekly_df, df, 'COMPANIA DE TRANSPORTE', 'TEU')
weekly_df = pivot_and_merge(weekly_df, df, 'COMPANIA DE TRANSPORTE', 'FLETE_per_TEU', agg_func='mean')
weekly_df = pivot_and_merge(weekly_df, df, 'CLAUSULA', 'TEU')
weekly_df = pivot_and_merge(weekly_df, df, 'TIPO DE BULTO', 'TEU')
weekly_df = pivot_and_merge(weekly_df, df, 'coast', 'TEU')
weekly_df = pivot_and_merge(weekly_df, df, 'coast', 'FLETE_per_TEU', agg_func='mean')


weekly_df['FECHA']


#ADD MACRO DATA


original_columns = set(weekly_df.columns)

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


            if 'Close' in df_macro.columns and 'Volume' in df_macro.columns:
                df_macro[['Close', 'Volume']] = df_macro[['Close', 'Volume']].fillna(method='ffill')

                name = filename.replace('.csv', '').lower()  # e.g. 'data_gold'
                macro_filtered = df_macro[['FECHA', 'Close', 'Volume', 'Close_pct_change']].rename(
                    columns={
                        'Close': f'{name}_price',
                        'Volume': f'{name}_volume',
                        'Close_pct_change': f'{name}_pct_change'
                    })
                macro_filtered["WEEK"] = macro_filtered["FECHA"].dt.to_period("W").dt.start_time
                weekly_macro_df = macro_filtered.groupby("WEEK", as_index=False).agg({
                    'FECHA': 'first',
                    f'{name}_price': 'first',
                    f'{name}_volume': 'sum',
                })

                col = f"{name}_weekly_pct_change"
                weekly_macro_df[col] = weekly_macro_df[f"{name}_price"].pct_change()
                weekly_macro_df = weekly_macro_df.drop(columns=['FECHA'])


                weekly_df = pd.merge(weekly_df, weekly_macro_df, how='left', on='WEEK')


new_columns = [col for col in weekly_df.columns if col not in original_columns]
weekly_df[new_columns] = weekly_df[new_columns].fillna(method='ffill').fillna(method='bfill')






weekly_df.to_csv("chile_data.csv", index=False)



