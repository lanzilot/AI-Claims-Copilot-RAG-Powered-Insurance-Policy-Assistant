import re
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
from pypdf import PdfReader

DB_DIR = "vector_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("hmo_policies")


def extract_article_title(text):
    match = re.search(r"(ARTICLE\s+[IVXLC]+\.\s+[A-Z0-9\s,\-\/&]+)", text)
    if match:
        return match.group(1).strip()
    return "Unknown Article"


def ingest_pdf(file_path):
    file_path = Path(file_path)
    reader = PdfReader(str(file_path))

    existing_count = collection.count()
    chunk_count = 0

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text or not text.strip():
            continue

        article_title = extract_article_title(text)

        embedding = model.encode(text).tolist()

        collection.add(
            ids=[f"{file_path.stem}_page_{page_num}_{existing_count}_{chunk_count}"],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "source": file_path.name,
                "page": page_num,
                "article": article_title,
                "file_type": "pdf"
            }]
        )

        chunk_count += 1

    return chunk_count