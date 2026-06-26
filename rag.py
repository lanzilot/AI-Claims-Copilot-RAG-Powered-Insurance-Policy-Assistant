import chromadb
import requests
from sentence_transformers import SentenceTransformer
from datetime import datetime
import csv
import os

DB_DIR = "vector_db"
AUDIT_FILE = "audit_logs/claims_audit.csv"

PROMPT_VERSION = "claims_prompt_v3"
MODEL_VERSION = "phi3"
KNOWLEDGE_BASE_VERSION = "kb_v1"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection("hmo_policies")

def retrieve_context(question, top_k=2):
    query_embedding = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "", []

    question_lower = question.lower()

    keyword_boosts = [
        "article iv",
        "membership fees",
        "membership fees and charges",
        "fees and charges",
        "sec. 4.1",
        "membership fee",

        "article ix",
        "schedule of benefits",
        "sec. 9.1",
        "sec. 9.2",
        "out-patient consultations",
        "dental services",

        "article vii",
        "membership eligibility",
        "sec. 7.1",
        "sec. 7.2",

        "article xi",
        "exclusions",
        "limitations",
        "sec. 11.1",
        "sec. 11.2"
    ]

    scored_chunks = []

    for doc, meta in zip(documents, metadatas):
        doc_lower = doc.lower()
        article_lower = meta.get("article", "").lower()

        score = 0

        for keyword in keyword_boosts:
            if keyword in question_lower and keyword in doc_lower:
                score += 20

            if keyword in question_lower and keyword in article_lower:
                score += 30

        if "membership fees" in question_lower and "article iv" in doc_lower:
            score += 50

        if "schedule of benefits" in question_lower and "article ix" in doc_lower:
            score += 50

        if "exclusion" in question_lower and "article xi" in doc_lower:
            score += 50

        page = meta.get("page", "?")
        scored_chunks.append((score, page, doc, meta))

    scored_chunks = sorted(scored_chunks, key=lambda x: x[0], reverse=True)

    seen_pages = set()
    selected_chunks = []

    for score, page, doc, meta in scored_chunks:
        if page not in seen_pages:
            seen_pages.add(page)
            selected_chunks.append((score, doc, meta))

        if len(selected_chunks) >= top_k:
            break

    context_parts = []
    sources = []

    for score, doc, meta in selected_chunks:
        source = meta.get("source", "Unknown source")
        page = meta.get("page", "?")
        article = meta.get("article", "Unknown Article")

        sources.append(f"{source}, Page {page}, {article}")

        context_parts.append(
            f"Source: {source}, Page {page}\n"
            f"Article: {article}\n"
            f"Relevance Score: {score}\n"
            f"Policy Text:\n{doc}"
        )

    return "\n\n---\n\n".join(context_parts), sources

def ask_ollama(question, context, sources):
    if not context.strip():
        return "I cannot find sufficient information in the policy documents."

    prompt = f"""
You are an HMO Claims Copilot.

Use only the policy context.
Give specific bullet points.
Include article/section numbers if available.
Do not invent information.
If not found, say: I cannot find sufficient information in the policy documents.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL_VERSION,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 300,
                    "temperature": 0.1
                }
            },
            timeout=300
        )

        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.ReadTimeout:
        return "The AI model took too long to respond. Try asking a shorter or more specific question."

    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Please make sure Ollama is running."

    except Exception as e:
        return f"Unexpected AI error: {str(e)}"


def save_audit(question, answer, sources):
    os.makedirs("audit_logs", exist_ok=True)

    file_exists = os.path.exists(AUDIT_FILE)

    with open(AUDIT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "question",
                "answer",
                "sources",
                "prompt_version",
                "model_version",
                "knowledge_base_version"
            ])

        writer.writerow([
            datetime.now().isoformat(),
            question,
            answer,
            ", ".join(sources),
            PROMPT_VERSION,
            MODEL_VERSION,
            KNOWLEDGE_BASE_VERSION
        ])


def answer_claim_question(question):
    context, sources = retrieve_context(question)
    answer = ask_ollama(question, context, sources)
    save_audit(question, answer, sources)

    return answer, sources, context