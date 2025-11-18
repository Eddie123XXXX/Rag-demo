# RAG Demo Script

Python experiment that demonstrates a miniature Retrieval-Augmented Generation (RAG) flow end to end:

- extracts text from a PDF (`fitz` / PyMuPDF),
- chunks it into overlapping windows,
- generates OpenAI embeddings for every chunk,
- runs semantic search over the chunk embeddings for a user query stored in `data/val.json`,
- builds a context prompt with the top‑k chunks,
- calls an OpenAI chat model to answer strictly from that context.

Everything lives in `rag-demo.py`.

## Requirements

- Python 3.8+ (script tested with 3.8.2)
- OpenAI API key (`OPENAI_API_KEY`)
- Packages: `PyMuPDF`, `numpy`, `openai`, `python-dotenv`

Install with:

```bash
python3 -m pip install PyMuPDF numpy openai python-dotenv
```

## Project Layout

```
rag-demo/
├── rag-demo.py
└── data/
    ├── AI_Information.pdf   # source PDF used in the demo
    └── val.json             # contains sample questions
```

> Adjust the PDF path in `rag-demo.py` if your file lives somewhere else.

## Environment Setup

1. Create a virtual environment (optional but recommended).
2. Install the dependencies listed above.
3. Create a `.env` file in the project root with:

   ```bash
   OPENAI_API_KEY=sk-your-key-here
   ```

4. Place your source PDF at `data/AI_Information.pdf`.
5. Ensure `data/val.json` contains an object array with a `question` field, e.g.

   ```json
   [
     { "question": "What is Retrieval-Augmented Generation?" }
   ]
   ```

## How It Works

1. **PDF ingestion** – `extract_text_from_pdf` reads the entire PDF and concatenates the text.
2. **Chunking** – `chunk_text` slices the text into 1000‑character windows with 200 characters overlap for better context continuity.
3. **Embedding creation** – `create_embeddings` calls `client.embeddings.create(...)` on every chunk (default model `text-embedding-ada-002`).
4. **Semantic search** – `semantic_search` computes cosine similarity between a query embedding and the chunk embeddings returned in `response.data`, picking the top `k` chunks.
5. **Answer generation** – `generate_response` sends a system guardrail plus the concatenated top chunks to `client.chat.completions.create(...)` (default model `gpt-4o`) and prints the model’s answer.

Key sections are marked in `rag-demo.py`:

```1:70:rag-demo.py
# PDF extraction & chunking utilities ...
```

```79:176:rag-demo.py
# Embedding generation, semantic search, and loading query data ...
```

```184:215:rag-demo.py
# Prompt construction and chat completion call ...
```

## Running the Script

```bash
python3 rag-demo.py
```

Console output:

1. Prints your OpenAI key (if loaded correctly; remove `print(api_key)` in production).
2. Displays the model’s final answer derived from the top retrieved chunks.

## Customization Tips

- **Chunk parameters** – change `chunk_text(..., n=1000, overlap=200)` to suit document length.
- **OpenAI models** – swap `text-embedding-ada-002` or `gpt-4o` for models available in your account.
- **Top‑k retrieval** – adjust `semantic_search(..., k=2)` to provide more or fewer context chunks.
- **Guardrails** – edit `system_prompt` to fit your compliance rules; currently it enforces “answer only from provided context”.

## Troubleshooting

- **ImportError: fitz** – install PyMuPDF (`pip install PyMuPDF`); the import name is `fitz`.
- **File not found** – verify the absolute path set for `pdf_path` and that `data/val.json` exists.
- **API errors** – ensure `OPENAI_API_KEY` has access to the chosen models and that your billing/quota is active.

Enjoy iterating on this lightweight RAG proof of concept!

