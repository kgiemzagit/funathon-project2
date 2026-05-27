# %%
# import mlflow
import polars as pol
from dotenv import load_dotenv

load_dotenv(override=True)
url = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
df = pol.read_parquet(url)

# print((df.head(10)))
# with pol.Config(tbl_rows=100):
#     print(df)

n_classes = len(df.unique("code"))
# wyswietla tabelke
print(df.unique("code"))
print(n_classes)
# print(df.count())
# print(len(df))

# %%
# print(2-2)
