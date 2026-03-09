import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load vector database
with open("sql_assistant_vector_db.pkl", "rb") as f:
    vector_db = pickle.load(f)

def cosine_similarity(query_vector, stored_vectors):

    dot_product = np.dot(stored_vectors, query_vector)
    query_norm = np.linalg.norm(query_vector)
    stored_norms = np.linalg.norm(stored_vectors, axis=1)

    similarity = dot_product / (stored_norms * query_norm)

    return similarity

def retrieve_chunks(query, top_k=3):

    query_embedding = embedding_model.encode(query)
    query_embedding = np.array(query_embedding).astype("float32")

    scores = cosine_similarity(query_embedding, vector_db["embeddings"])

    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved_chunks = [vector_db["texts"][i] for i in top_indices]

    return retrieved_chunks

def build_prompt(query, retrieved_chunks):

    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are a SQL interview assistant.

Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer clearly in a short professional interview style.
"""

    return prompt

def generate_answer(prompt):

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = llm.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.3
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

def ask_sql_question(question):

    chunks = retrieve_chunks(question)

    prompt = build_prompt(question, chunks)

    answer = generate_answer(prompt)

    return answer