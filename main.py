from fastapi import FastAPI
from pydantic import BaseModel

from rag_engine import ask_sql_question

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask_sql")
def ask_sql(data: QuestionRequest):

    answer = ask_sql_question(data.question)

    return {"answer": answer}