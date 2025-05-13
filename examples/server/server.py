#!/usr/bin/env python3
import duckdb
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI()

con = duckdb.connect(config = {"allow_unsigned_extensions": "true"})

# Load the Quackformers extension
# To test locally
# con.execute("LOAD '../../build/release/quackformers.duckdb_extension';")
con.execute("INSTALL quackformers;")
con.execute("LOAD quackformers;")



class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedded_text: list

@app.post("/embed_jina", response_model=EmbedResponse)
def embed_text_jina(request: EmbedRequest):
    # Use the Quackformers embed_jina function
    result = con.execute(f"SELECT embed_jina('{request.text}')").fetchone()
    return {"embedded_text": result[0]}

@app.post("/embed", response_model=EmbedResponse)
def embed_text(request: EmbedRequest):
    # Use the Quackformers embed function
    result = con.execute(f"SELECT embed('{request.text}')").fetchone()
    return {"embedded_text": result[0]}

@app.get("/")
def root():
    return {"message": "Welcome to the DuckDB Quackformers API!"}

if __name__ == "__main__":
    print("Starting FastAPI server on http://0.0.0.0:8080/")
    uvicorn.run(app, host="0.0.0.0", port=8080)
