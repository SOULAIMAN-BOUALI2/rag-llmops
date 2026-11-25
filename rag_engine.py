import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import time
import yaml
import fitz
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from pydantic import BaseModel, Field
import ollama

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from llmops_layer import GuardrailsEngine, SemanticCache

# 1. Configuration OpenTelemetry -> Arize Phoenix
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces"))
)
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("production-rag-llmops")

# 2. Schéma de sortie strict (Output Guardrail)
class RAGStructuredOutput(BaseModel):
    answer: str = Field(description="Réponse précise basée uniquement sur le contexte.")
    cited_pages: list[int] = Field(description="Numéros de pages sources utilisés.")
    confidence: float = Field(description="Score de confiance entre 0.0 et 1.0.")
    has_sufficient_context: bool = Field(description="True si le contexte répond à la question.")

class AdvancedLocalRAG:
    def __init__(self, pdf_path: str, llm_model: str = "qwen2.5-coder:7b", prompt_file: str = "prompts.yaml"):
        self.pdf_path = pdf_path
        self.llm_model = llm_model
        self.prompt_config = self._load_prompts(prompt_file)
        
        # Modèles d'embedding multilingue + Cross-Encoder Reranker
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="pdf_docs_v3")
        self.cache = SemanticCache(threshold=0.92)
        
        self.all_chunks = []
        self.all_metas = []
        self.bm25 = None
        self._ingest_pdf()

    def _load_prompts(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        ver = data["active_version"]
        cfg = data["versions"][ver]
        cfg["version"] = ver
        return cfg

    def _ingest_pdf(self, chunk_size: int = 750, overlap: int = 150):
        doc = fitz.open(self.pdf_path)
        ids = []
        chunk_id = 0

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i : i + chunk_size].strip()
                if len(chunk) > 50:
                    self.all_chunks.append(chunk)
                    self.all_metas.append({"page": page_num, "source": self.pdf_path})
                    ids.append(f"chunk_{chunk_id}")
                    chunk_id += 1

        if self.all_chunks:
            # Indexation Vectorielle (Dense)
            embeddings = self.embedder.encode(self.all_chunks).tolist()
            self.collection.add(
                documents=self.all_chunks, embeddings=embeddings, metadatas=self.all_metas, ids=ids
            )
            # Indexation Lexicale (BM25 Sparse)
            tokenized_corpus = [doc.lower().split() for doc in self.all_chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"[INFO] {len(self.all_chunks)} chunks indexés (Hybrid Dense + BM25)")

    def _hybrid_retrieve_and_rerank(self, query: str, q_emb: list[float], top_k: int = 3):
        # Étape A : Récupération hybride (Dense + BM25)
        with tracer.start_as_current_span("hybrid_retrieval") as ret_span:
            ret_span.set_attribute("openinference.span.kind", "RETRIEVER")
            ret_span.set_attribute("input.value", query)
            
            # Top 6 Dense
            dense_res = self.collection.query(query_embeddings=[q_emb], n_results=6)
            dense_docs = dense_res["documents"][0]
            dense_metas = dense_res["metadatas"][0]
            
            # Top 6 BM25
            bm25_scores = self.bm25.get_scores(query.lower().split())
            top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:6]
            
            # Fusion & Déduplication
            candidates = {}
            for doc, meta in zip(dense_docs, dense_metas):
                candidates[doc] = meta
            for idx in top_bm25_idx:
                candidates[self.all_chunks[idx]] = self.all_metas[idx]
                
            candidate_list = list(candidates.items())
            ret_span.set_attribute("retrieval.candidates_count", len(candidate_list))

        # Étape B : Reranking par Cross-Encoder
        with tracer.start_as_current_span("cross_encoder_rerank") as rerank_span:
            rerank_span.set_attribute("openinference.span.kind", "RERANKER")
            rerank_span.set_attribute("reranker.model_name", "ms-marco-MiniLM-L-6-v2")
            
            pairs = [[query, doc] for doc, _ in candidate_list]
            scores = self.reranker.predict(pairs)
            
            scored_candidates = sorted(
                zip(candidate_list, scores), key=lambda x: x[1], reverse=True
            )[:top_k]
            
            final_docs = [item[0][0] for item in scored_candidates]
            final_metas = [item[0][1] for item in scored_candidates]
            final_scores = [round(float(item[1]), 4) for item in scored_candidates]
            
            # Attributs OpenInference pour afficher les chunks dans Arize Phoenix
            for idx, (doc, meta, score) in enumerate(zip(final_docs, final_metas, final_scores)):
                rerank_span.set_attribute(f"retrieval.documents.{idx}.document.content", doc)
                rerank_span.set_attribute(f"retrieval.documents.{idx}.document.score", score)
                rerank_span.set_attribute(f"retrieval.documents.{idx}.document.metadata", str(meta))

        return final_docs, final_metas, final_scores

    def query(self, question: str, top_k: int = 3) -> dict:
        with tracer.start_as_current_span("rag_pipeline_request") as root_span:
            root_span.set_attribute("openinference.span.kind", "CHAIN")
            root_span.set_attribute("input.value", question)
            root_span.set_attribute("prompt.version", self.prompt_config["version"])
            t_start = time.perf_counter()

            # 1. Input Guardrail
            with tracer.start_as_current_span("input_guardrail") as guard_span:
                guard_span.set_attribute("openinference.span.kind", "GUARDRAIL")
                guard_res = GuardrailsEngine.check_input(question)
                guard_span.set_attribute("guardrail.allowed", guard_res["allowed"])
                if not guard_res["allowed"]:
                    root_span.set_attribute("status", "BLOCKED")
                    return {
                        "status": "BLOCKED",
                        "reason": guard_res["reason"],
                        "answer": "Requête bloquée par la politique de sécurité.",
                        "metrics": {"total_latency_ms": round((time.perf_counter() - t_start) * 1000, 2)}
                    }
                clean_question = guard_res["sanitized_text"]

            # 2. Semantic Cache
            q_emb = self.embedder.encode([clean_question]).tolist()[0]
            with tracer.start_as_current_span("semantic_cache") as cache_span:
                cached = self.cache.get(q_emb)
                cache_span.set_attribute("cache.hit", cached is not None)
                if cached:
                    cached["metrics"]["total_latency_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
                    root_span.set_attribute("output.value", cached["answer"])
                    return cached

            # 3. Hybrid Search + Reranking
            t0 = time.perf_counter()
            docs, metas, rerank_scores = self._hybrid_retrieve_and_rerank(clean_question, q_emb, top_k=top_k)
            t_retrieval = time.perf_counter() - t0

            # 4. Génération LLM avec Structured Output (Output Guardrail)
            context_str = "\n\n".join([f"[Page {m['page']}] {d}" for d, m in zip(docs, metas)])
            user_prompt = self.prompt_config["template"].format(context=context_str, question=clean_question)

            with tracer.start_as_current_span("ollama_llm_generation") as llm_span:
                llm_span.set_attribute("openinference.span.kind", "LLM")
                llm_span.set_attribute("llm.model_name", self.llm_model)
                llm_span.set_attribute("input.value", user_prompt)
                
                t1 = time.perf_counter()
                response = ollama.chat(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": self.prompt_config["system"]},
                        {"role": "user", "content": user_prompt}
                    ],
                    format=RAGStructuredOutput.model_json_schema(),
                    options={"temperature": 0.0}
                )
                t_generation = time.perf_counter() - t1
                
                structured = RAGStructuredOutput.model_validate_json(response["message"]["content"])
                
                llm_span.set_attribute("llm.token_count.prompt", response.get("prompt_eval_count", 0))
                llm_span.set_attribute("llm.token_count.completion", response.get("eval_count", 0))
                llm_span.set_attribute("output.value", structured.answer)

            result = {
                "status": "SUCCESS",
                "cache_hit": False,
                "prompt_version": self.prompt_config["version"],
                "question": clean_question,
                "answer": structured.answer,
                "confidence": structured.confidence,
                "cited_pages": structured.cited_pages,
                "contexts": docs,
                "rerank_scores": rerank_scores,
                "metrics": {
                    "retrieval_latency_ms": round(t_retrieval * 1000, 2),
                    "generation_latency_ms": round(t_generation * 1000, 2),
                    "total_latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
                    "tokens_out": response.get("eval_count", 0)
                }
            }

            self.cache.set(q_emb, result)
            root_span.set_attribute("output.value", structured.answer)
            return result