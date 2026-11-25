import re
import numpy as np


class GuardrailsEngine:
    INJECTION_PATTERNS = [
        r"ignore (all |previous |the )?instructions",
        r"oublie (toutes )?les instructions",
        r"system prompt",
        r"you are now",
        r"bypass",
    ]

    @classmethod
    def check_input(cls, text: str) -> dict:
        lower_text = text.lower()
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, lower_text):
                return {
                    "allowed": False,
                    "reason": f"Prompt injection détectée (pattern: {pattern})",
                    "sanitized_text": text,
                }

        # Masquage PII (E-mails & Téléphones marocains)
        sanitized = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL_MASQUÉ]", text)
        sanitized = re.sub(r"(\+212|0)[5-7]\d{8}", "[TEL_MASQUÉ]", sanitized)

        return {"allowed": True, "reason": "OK", "sanitized_text": sanitized}


class SemanticCache:
    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self.entries = []

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def get(self, query_embedding: list[float]):
        q_vec = np.array(query_embedding)
        for cached_vec, cached_resp in self.entries:
            sim = self._cosine_sim(q_vec, cached_vec)
            if sim >= self.threshold:
                resp_copy = cached_resp.copy()
                resp_copy["cache_hit"] = True
                resp_copy["cache_similarity"] = round(sim, 4)
                return resp_copy
        return None

    def set(self, query_embedding: list[float], response: dict):
        self.entries.append((np.array(query_embedding), response))