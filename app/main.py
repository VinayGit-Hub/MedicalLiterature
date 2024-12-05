from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import pipeline
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db_path = "embeddings/"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory=db_path, embedding_function=embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

qa_pipeline = pipeline("text-generation", model="gpt2", device=-1)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    try:
        logger.info("Processing question: %s", query.question)

        # Retrieve relevant chunks
        relevant_chunks = retriever.invoke(query.question)
        if not relevant_chunks:
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        context = "\n".join([chunk.page_content for chunk in relevant_chunks])
        logger.info(f"Retrieved context: {context}")

        # Generate answer
        prompt = f"Question: {query.question}\nAnswer:"    #Context:\n{context}\n\n
        response = qa_pipeline(
            prompt, 
            max_new_tokens=150,  # Adjust max tokens for generation
            truncation=True,     # Ensure truncation for long inputs
            pad_token_id=50256   # Set padding token for gpt2
        )
        answer = response[0]["generated_text"]
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")