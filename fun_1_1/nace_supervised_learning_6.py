# %%
import polars as pol
import mlflow
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchTextClassifiers.value_encoder import ValueEncoder
from torchTextClassifiers.tokenizers import WordPieceTokenizer
from torchTextClassifiers import ModelConfig, torchTextClassifiers, TrainingConfig

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
y = df["code"].to_numpy()     # ← kody SIC (to przewidujemy)
x = df["label"].to_numpy()

x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=0.7, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_rem, y_rem, test_size=0.5, random_state=42)

x_train_codes = y_train
print(type(y_train))
# print(x_train.shape, y_train.shape)
# print(x_val.shape, y_val.shape)
# print(x_test.shape, y_test.shape)

le = LabelEncoder()

le.fit(y_train)
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

x_train_texts = [str(s) for s in x_train]

wordPieceTokenizer = WordPieceTokenizer(vocab_size=5000)
wordPieceTokenizer.train(x_train_texts)

# Look at an example of tokenization
print("Raw text", x_train_texts[10])
print(
    "Tokens id:",
    wordPieceTokenizer.tokenize(x_train_texts[10]).input_ids.squeeze(0)
)
print(
    "Tokens:",
    wordPieceTokenizer.tokenizer.convert_ids_to_tokens(
        wordPieceTokenizer.tokenize(x_train_texts[10]).input_ids.squeeze(0)
    )
)

modelConfig = ModelConfig(embedding_dim=96, num_classes=n_classes)
torchTextClassifiers = torchTextClassifiers(
    tokenizer=wordPieceTokenizer,
    model_config=modelConfig,
    value_encoder=value_encoder)

training_config = TrainingConfig(lr=0.0005, batch_size=128, num_epochs=1)

mlflow.set_experiment("funathon-2026-project2")
mlflow.pytorch.autolog()

with mlflow.start_run() as run:
    # This should take approximately 1-2mn
    torchTextClassifiers.train(
        X_train=x_train,
        y_train=y_train,
        training_config=training_config,
        X_val=x_val,
        y_val=y_val,
        verbose=True,
    )

    mlflow.log_artifacts(
        training_config.save_path,   # local folder produced by ttc.train()
        artifact_path="model_artifacts",
    )

# wordPieceTokenizer.tokenizer.convert_ids_to_tokens(wordPieceTokenizer.tokenize(x_train_name))

# print(value_encode)
