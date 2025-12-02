import pandas as pd
import numpy as np
from openai import OpenAI
import gradio as gr
from set_key import set_api_key_from_file
set_api_key_from_file()
client = OpenAI()

# Load Q&A and Embeddings

qa_df = pd.read_csv("SS_StructuredQA.csv")
questions = qa_df["Question"].tolist()
answers = qa_df["Answer"].tolist()

embeddings = pd.read_csv("Embeddings.csv").values.astype("float32")

# Cosine similarity function

def cosine_similarity(query_vec, matrix):
    q = query_vec / np.linalg.norm(query_vec)
    m = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.dot(m, q)

# Chatbot function

def query_bot(message, history):
    # Embed user query
    q_emb = client.embeddings.create(
        input=[message],
        model="text-embedding-ada-002"
    ).data[0].embedding

    q_emb = np.array(q_emb).astype("float32")

    # Compute similarity
    sims = cosine_similarity(q_emb, embeddings)

    # Top 3 most similar Q&A entries
    top_idxs = sims.argsort()[-3:][::-1]

    retrieved = [
        f"Q: {questions[i]}\nA: {answers[i]}"
        for i in top_idxs
    ]

    context = "\n\n".join(retrieved)

    prompt = f"""
    User asked: {message}

    Here is the relevant info from our database:
    {context}

    Answer using ONLY this info.
    """

    # Call GPT to generate final answer
    response = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )

    return response.output_text

# Launch Chatbot UI

gr.ChatInterface(query_bot).launch(share=True, server_port=7860)


