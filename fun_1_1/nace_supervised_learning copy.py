# %% 3.1
'''
# import mlflow
import polars as pol
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv(override=True)
url = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
df = pol.read_parquet(url)

# print((df.head(10)))
# with pol.Config(tbl_rows=100):
#     print(df)

n_classes = len(df.unique("code"))
# wyswietla tabelke
# print(df.unique("code"))
print(n_classes)
# print(df.count())
# print(len(df))

# 3.1 podzial
y = df["label"].to_numpy()
x = df.drop("label").to_numpy()

x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.7, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem,  test_size=0.5, random_state=42)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
print(x_test.shape, y_test.shape)'''
# %% 3.2
# import mlflow
import polars as pol
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

load_dotenv(override=True)
url = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
df = pol.read_parquet(url)

# print((df.head(10)))
# with pol.Config(tbl_rows=100):
#     print(df)

n_classes = len(df.unique("code"))
# wyswietla tabelke
# print(df.unique("code"))
# print(df.count())
# print(len(df))

# 3.1 podzial
y = df["label"].to_numpy()
x = df.drop("label").to_numpy()

x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.7, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem,  test_size=0.5, random_state=42)

x_train_codes = x_train[:, 0]
print(type(x_train))
# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# print(x_test.shape, y_test.shape)

le = LabelEncoder()

le.fit(x_train_codes)
print(le.inverse_transform([218]))
# print(x_train_encoded)
# sprawdzenie

# print(type(df['code']))
# print(type(x_train_codes))

all_codes_set = set(df['code'])
x_train_codes_set = set(x_train_codes)
diff = all_codes_set - x_train_codes_set

print(len(diff))

# %% 3.3
import polars as pol
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder

load_dotenv(override=True)
url = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
df = pol.read_parquet(url)

# print((df.head(10)))
# with pol.Config(tbl_rows=100):
#     print(df)

n_classes = len(df.unique("code"))
# wyswietla tabelke
# print(df.unique("code"))
# print(df.count())
# print(len(df))

# 3.1 podzial
y = df["label"].to_numpy()
x = df.drop("label").to_numpy()

x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.7, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem,  test_size=0.5, random_state=42)

x_train_codes = x_train[:, 0]
print(type(x_train))
# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# print(x_test.shape, y_test.shape)

le = LabelEncoder()

le.fit(x_train_codes)
print(le.inverse_transform([218]))
print(len(le.classes_))
# print(x_train_encoded)
# sprawdzenie

# print(type(df['code']))
# print(type(x_train_codes))

all_codes_set = set(df['code'])
x_train_codes_set = set(x_train_codes)
diff = all_codes_set - x_train_codes_set

if diff:
    print("Uwaga:", len(diff))
else:
    print("Jest ok.")

value_encoder = ValueEncoder(label_encoder=le)

# print(value_encode)

# %% 4.3
import polars as pol
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder
from torchTextClassifiers.tokenizers import WordPieceTokenizer

load_dotenv(override=True)
url = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
df = pol.read_parquet(url)

# print((df.head(10)))
# with pol.Config(tbl_rows=100):
#     print(df)

n_classes = len(df.unique("code"))
# wyswietla tabelke
# print(df.unique("code"))
# print(df.count())
# print(len(df))

# 3.1 podzial
y = df["label"].to_numpy()
x = df.drop("label").to_numpy()

x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.7, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem,  test_size=0.5, random_state=42)

x_train_codes = x_train[:, 0]
print(type(x_train))
# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# print(x_test.shape, y_test.shape)

le = LabelEncoder()

le.fit(x_train_codes)
print(le.inverse_transform([218]))
print(len(le.classes_))
# print(x_train_encoded)
# sprawdzenie

# print(type(df['code']))
# print(type(x_train_codes))

all_codes_set = set(df['code'])
x_train_codes_set = set(x_train_codes)
diff = all_codes_set - x_train_codes_set

if diff:
    print("Uwaga:", len(diff))
else:
    print("Jest ok.")

value_encoder = ValueEncoder(label_encoder=le)

x_train_name = [str(s) for s in x_train[:, 1].tolist()]

wordPieceTokenizer = WordPieceTokenizer(vocab_size=5000)
wordPieceTokenizer.train(x_train_name)

# Look at an example of tokenization
print("Raw text", x_train_name[10])
print(
    "Tokens id:",
    wordPieceTokenizer.tokenize(x_train_name[10]).input_ids.squeeze(0)
)
print(
    "Tokens:",
    wordPieceTokenizer.tokenizer.convert_ids_to_tokens(
        wordPieceTokenizer.tokenize(x_train_name[10]).input_ids.squeeze(0)
    )
)
# %% 6.1
import polars as pol
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers import ModelConfig, torchTextClassifiers

load_dotenv(override=True)
url = "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
df = pol.read_parquet(url)

# print((df.head(10)))
# with pol.Config(tbl_rows=100):
#     print(df)

n_classes = len(df.unique("code"))
# wyswietla tabelke
# print(df.unique("code"))
# print(df.count())
# print(len(df))

# 3.1 podzial
y = df["label"].to_numpy()
x = df.drop("label").to_numpy()

x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.7, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem,  test_size=0.5, random_state=42)

x_train_codes = x_train[:, 0]
print(type(x_train))
# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# print(x_test.shape, y_test.shape)

le = LabelEncoder()

le.fit(x_train_codes)
print(le.inverse_transform([218]))
print(len(le.classes_))
# print(x_train_encoded)
# sprawdzenie

# print(type(df['code']))
# print(type(x_train_codes))

all_codes_set = set(df['code'])
x_train_codes_set = set(x_train_codes)
diff = all_codes_set - x_train_codes_set

if diff:
    print("Uwaga:", len(diff))
else:
    print("Jest ok.")

value_encoder = ValueEncoder(label_encoder=le)

x_train_name = [str(s) for s in x_train[:, 1].tolist()]

wordPieceTokenizer = WordPieceTokenizer(vocab_size=5000)
wordPieceTokenizer.train(x_train_name)

# Look at an example of tokenization
print("Raw text", x_train_name[10])
print(
    "Tokens id:",
    wordPieceTokenizer.tokenize(x_train_name[10]).input_ids.squeeze(0)
)
print(
    "Tokens:",
    wordPieceTokenizer.tokenizer.convert_ids_to_tokens(
        wordPieceTokenizer.tokenize(x_train_name[10]).input_ids.squeeze(0)
    )
)

modelConfig = ModelConfig(embedding_dim=96, num_classes=n_classes)
torchTextClassifiers = torchTextClassifiers(
    tokenizer=wordPieceTokenizer,
    model_config=modelConfig,
    value_encoder=value_encoder)

# wordPieceTokenizer.tokenizer.convert_ids_to_tokens(wordPieceTokenizer.tokenize(x_train_name))

# print(value_encode)