import os, json, random
from typing import List, Dict, Any
import requests
from loguru import logger

class LLMProposer:
    """
    OpenAI-only. If OPENAI_API_KEY missing, returns [].
    JSON-hardened: expects a JSON list of formulas where each formula is either
      [cond, 'AND'|'OR', cond] or [feature, '>'|'<' , threshold]
    """
    def __init__(self, state):
        self.state = state
        self.provider = (os.environ.get("LLM_PROVIDER","openai") or "openai").lower()

    def _local_suggestions(self, base_features: List[str], k: int = 0) -> List[List[Any]]:
        # disabled by policy; return []
        return []

    def _remote_suggestions(self, base_features: List[str], k: int = 3) -> List[List[Any]]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key or self.provider != "openai" or k<=0:
            return []
        sys_prompt = "Generate boolean trading rules using only the given feature names. Return STRICT JSON."
        user = {
            "features": base_features,
            "k": k,
            "format": "Return a JSON list of formulas where each formula is either [cond, 'AND'|'OR', cond] or [feature, '>'|'<' , threshold]."
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model":"gpt-4o-mini","messages":[{"role":"system","content":sys_prompt},{"role":"user","content":json.dumps(user)}],
                   "temperature": 0.7}
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
            r.raise_for_status()
            j = r.json()
            txt = (j.get("choices") or [{}])[0].get("message", {}).get("content", "[]")
            start = txt.find("["); end = txt.rfind("]")
            if start == -1 or end == -1:
                return []
            suggestions = json.loads(txt[start:end+1])
            cleaned = []
            for f in suggestions:
                if isinstance(f, list) and len(f)>=3:
                    if isinstance(f[0], list) or (isinstance(f[0], str) and f[1] in (">","<")):
                        cleaned.append(f)
            self.state.r.incr("llm:proposer:success") if getattr(self.state,'r',None) else None
            return cleaned[:k]
        except requests.HTTPError as he:
            code = he.response.status_code if he.response else 'NA'
            logger.warning(f"LLM proposer HTTP {code}")
            if getattr(self.state,'r',None):
                self.state.r.incr("llm:proposer:http_errors")
            return []
        except Exception as e:
            logger.warning(f"LLM proposer error: {type(e).__name__}")
            if getattr(self.state,'r',None):
                self.state.r.incr("llm:proposer:errors")
            return []

    def propose(self, base_features: List[str], k: int = 3) -> List[List[Any]]:
        return self._remote_suggestions(base_features, k=k)
