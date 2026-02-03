# -----------------------------
# Silence HF / asyncio noise
# -----------------------------
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stderr = open("/dev/null", "w")

import os
os.environ["GRADIO_SSR_MODE"] = "false"

import torch
import gradio as gr
import faiss
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

TECH_PATH = "./tech_lora"
RESEARCH_PATH = "./research_lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

router = SentenceTransformer("all-MiniLM-L6-v2")

tech_seed = ["debug python","write code","fix bug","algorithm"]
research_seed = ["summarize paper","research","experiment"]

tech_emb = router.encode(tech_seed, normalize_embeddings=True).mean(axis=0)
research_emb = router.encode(research_seed, normalize_embeddings=True).mean(axis=0)

docs = [
    "LoRA adapts low rank matrices instead of full model weights.",
    "Transformers use self attention.",
    "FAISS enables vector similarity search."
]

doc_emb = router.encode(docs, normalize_embeddings=True)
index = faiss.IndexFlatIP(doc_emb.shape[1])
index.add(doc_emb)

history = []

def route(prompt):
    e = router.encode([prompt], normalize_embeddings=True)[0]
    return "tech" if np.dot(e, tech_emb) > np.dot(e, research_emb) else "research"

def retrieve(prompt):
    e = router.encode([prompt], normalize_embeddings=True)
    _, ids = index.search(e, 2)
    return "\n".join([docs[i] for i in ids[0]])

def load_lora(path):
    return PeftModel.from_pretrained(base_model, path)

def chat(prompt):
    start = time.time()

    domain = route(prompt)

    history.append(prompt)
    context = "\n".join(history[-3:])

    rag = retrieve(prompt)

    full = f"""
Context:
{rag}

Conversation:
{context}

User:
{prompt}
"""

    model = load_lora(TECH_PATH if domain=="tech" else RESEARCH_PATH)

    inputs = tokenizer(full, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)

    txt = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return f"[{domain.upper()} | {time.time()-start:.2f}s]\n\n{txt}"

demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=4),
    outputs="text",
    title="Multi-LoRA RAG Assistant (Qwen CPU)",
    description="Embedding routing + RAG + dual LoRA adapters"
)

demo.queue()
demo.launch()
