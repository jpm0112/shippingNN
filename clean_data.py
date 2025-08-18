import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("test_daily.csv")

df_with_target = df.copy()

# df = df.drop(columns=['TOTAL_TEUS'])



corr = df.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', annot=False, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()




df = df.drop(columns=['CONTENEDOR REFRIGERADO 40', 'CONTENEDOR REFRIGERADO 20'])

df = df.drop(columns=['wti_oil_price','wti_oil_pct_change'])

df = df.drop(columns=['CONTENEDOR 40'])

df = df.drop(columns=['CONTENEDOR 20'])

df = df.drop(columns=['DIN_UNICOS_DIARIOS'])

df = df.drop(columns=['RUTS_UNICOS_DIARIOS'])

df = df.drop(columns=['PARTIDAS_UNICAS_DIARIAS'])

df = df.drop(columns=['yang_ming_price'])

df = df.drop(columns=['FOB'])

df = df.drop(columns=['data_gold_price'])

df = df.drop(columns=['wan_hai_price'])

df = df.drop(columns=['zim_price'])

df = df.drop(columns=['cny_price'])





corr = df.corr(numeric_only=True)
corr_pairs = corr.abs().unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1]  # remove self-correlation
top_corr = corr_pairs.drop_duplicates().head(10)
print(top_corr)
lower_corr = corr_pairs.drop_duplicates().tail(10)
print(lower_corr)

df.to_csv("test_daily.csv", index=False)
