from openai import OpenAI
import pandas as pd
from set_key import set_api_key_from_file

def get_embedding(text, model="text-embedding-ada-002"):
    client = OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

set_api_key_from_file()
df = pd.read_csv('SS_StructuredQA.csv')

embeddings = df["Question"].apply(get_embedding).tolist()

embedding_df = pd.DataFrame(embeddings)

embedding_df.to_csv('Embeddings.csv', index=False)
