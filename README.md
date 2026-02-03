
---

## 🔧 How It Works

### 1. Semantic Routing

Each incoming prompt is embedded using MiniLM and compared against domain seed vectors:

- Technical queries → TECH LoRA  
- Academic / research queries → RESEARCH LoRA  

This enables automatic domain selection without user input.

---

### 2. Retrieval-Augmented Generation

A lightweight FAISS index injects relevant background context (LoRA, transformers, RAG concepts) before generation.

This improves grounding while keeping inference efficient.

---

### 3. Parameter-Efficient Fine-Tuning

Rather than retraining the full model:

- LoRA adapters are trained offline in Colab  
- Only adapter weights are loaded at inference time  
- The base model remains frozen  

This dramatically reduces compute and storage requirements.

---

## 🧱 Design Decisions

### Single Frozen Base Model

Instead of training or deploying multiple full models, this project uses a single frozen Qwen base model with lightweight LoRA adapters. This significantly reduces memory usage and simplifies deployment while still enabling domain specialization.

---

### Dual LoRA Adapters

Two separate LoRA adapters are trained for different domains:

- TECH LoRA for programming and technical reasoning  
- RESEARCH LoRA for academic writing and summarization  

This approach enables parameter-efficient multi-domain behavior without duplicating the base model.

---

### Embedding-Based Routing

User prompts are embedded using MiniLM and compared against domain seed vectors. This allows the system to automatically select the appropriate LoRA adapter per request, removing the need for manual mode selection and improving user experience.

---

### Lightweight RAG with FAISS

A small FAISS vector index injects relevant background context before generation. This improves factual grounding while keeping the system simple and CPU-friendly.

Rather than building a full document pipeline, this project demonstrates how minimal RAG can still add value in production-style systems.

---

### CPU-First Deployment

The entire system is designed to run on CPU using Hugging Face Spaces. This constraint influenced model choice, adapter size, and retrieval design, emphasizing practical deployment over raw model scale.

The goal is to demonstrate real-world feasibility, not just experimentation on GPUs.

---

## 🚀 Future Improvements

- Add a learned router model instead of seed-vector similarity for more accurate domain classification  
- Support additional LoRA adapters (e.g., data science, product writing, tutoring)  
- Replace static RAG documents with user-uploaded knowledge bases  
- Add streaming token generation for improved UI responsiveness  
- Persist conversation history using a vector memory store  
- Introduce evaluation metrics for routing accuracy and response quality  
- Add authentication and rate limiting for production deployment  
- Containerize the application with Docker for cloud portability  

These extensions would move the system closer to a full production LLM platform.

---

## 🏃 Running Locally

```bash
pip install -r requirements.txt
python app.py
```

🧪 Example Prompts

Technical:

Write a Python function to detect cycles in a linked list.


Research:

Compare transformer architectures with recurrent neural networks.


Mixed:

Explain LoRA, then summarize your explanation like a research abstract.

📚 Training

LoRA adapters were trained offline using:

Code instruction datasets (technical domain)

Summarization / research-style datasets (academic domain)

Training notebooks are intentionally kept separate.
This repository focuses on inference, routing, and deployment.

💼 Resume Summary

Built a modular LLM system using Qwen with dual LoRA adapters, embedding-based routing, FAISS-powered RAG, and Gradio UI. Deployed on Hugging Face Spaces with CPU inference, demonstrating parameter-efficient fine-tuning, semantic specialization, and production-style ML system design.

🎯 What This Project Demonstrates

Practical PEFT (LoRA) implementation

Multi-adapter routing

Retrieval-Augmented Generation

End-to-end deployment

ML systems engineering beyond basic prompting

📎 Links

Live Demo:
https://huggingface.co/spaces/SaiTejaSrivilli/multi-lora-rag-assistant

GitHub Repository:
(this repo)
