import os, json, random, time, hashlib
from typing import List, Dict, Any
import requests
from loguru import logger

class LLMProposer:
    """OpenAI-only proposer with hardened JSON pipeline.

    PIT: Deterministic post-processing; no future leakage.
    Guardrails: feature whitelist, threshold clamps, dedupe, metrics.
    """
    def __init__(self, state):
        self.state = state
        self.provider = (os.environ.get("LLM_PROVIDER","openai") or "openai").lower()

    def _local_suggestions(self, base_features: List[str], k: int = 0) -> List[List[Any]]:
        # disabled by policy; return []
        return []

    @staticmethod
    def sanitize_formulas(base_features: List[str], suggestions: List[List[Any]],
                          suppressed: List[str] | None = None,
                          threshold_min: float = -5.0, threshold_max: float = 5.0,
                          max_nodes: int = 31) -> List[List[Any]]:
        """Validate & clamp formulas.

        PIT: Pure transformation of already-produced suggestions.
        No side effects; deterministic given inputs.
        """
        whitelist = set(base_features)
        suppressed_set = set(suppressed or [])
        cleaned: List[List[Any]] = []
        seen_hashes = set()
        import json as _json, hashlib as _hashlib

        def _valid(node) -> bool:
            if not isinstance(node, list):
                return False
            # condition
            if len(node) >=3 and isinstance(node[0], str) and node[1] in ('>','<'):
                if node[0] not in whitelist or node[0] in suppressed_set:
                    return False
                try:
                    float(node[2])
                except Exception:
                    return False
                return True
            if len(node) >=3 and isinstance(node[0], list) and isinstance(node[2], list) and node[1] in ('AND','OR'):
                return _valid(node[0]) and _valid(node[2])
            return False

        def _clamp(node):
            if isinstance(node, list) and len(node)>=3 and isinstance(node[0], str) and node[1] in ('>','<'):
                try:
                    val = float(node[2])
                    node[2] = max(threshold_min, min(threshold_max, val))
                except Exception:
                    node[2] = 0.0
            elif isinstance(node, list) and len(node)>=3:
                if isinstance(node[0], list): _clamp(node[0])
                if isinstance(node[2], list): _clamp(node[2])

        def _count_nodes(node) -> int:
            if not isinstance(node, list): return 0
            if len(node) >=3 and isinstance(node[0], str) and node[1] in ('>','<'):
                return 1
            if len(node) >=3:
                return 1 + _count_nodes(node[0]) + _count_nodes(node[2])
            return 0

        for f in suggestions:
            if not _valid(f):
                continue
            if _count_nodes(f) > max_nodes:
                continue
            _clamp(f)
            canon = _json.dumps(f, separators=(',',':'))
            h = _hashlib.sha256(canon.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            cleaned.append(f)
        return cleaned

    def _remote_suggestions(self, base_features: List[str], k: int = 3, threshold_min: float = -5.0, threshold_max: float = 5.0) -> List[List[Any]]:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key or self.provider != "openai" or k<=0:
            return []
        whitelist = sorted(set(base_features))
        sys_prompt = (
            "You generate ONLY raw JSON (no prose). Return a JSON list of boolean formulas. "
            "Grammar: A condition is [feature, '>'|'<' , number]. A formula is either a condition or [formula,'AND'|'OR',formula]. "
            "Constraints: use ONLY these features: " + ",".join(whitelist) + ". "
            "Numbers should be within [-5,5]. Do not explain."
        )
        user = {"k": k, "features": whitelist}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model":"gpt-4o-mini","messages":[{"role":"system","content":sys_prompt},{"role":"user","content":json.dumps(user)}],
                   "temperature": 0.4, "max_tokens": 400}
        t0 = time.monotonic()
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
            latency_ms = int((time.monotonic()-t0)*1000)
            if getattr(self.state,'r',None):
                self.state.r.hincrby("llm:proposer:latency_ms_buckets", str(min( (latency_ms//100)*100, 2000)), 1)
            r.raise_for_status()
            j = r.json()
            txt = (j.get("choices") or [{}])[0].get("message", {}).get("content", "[]")
            # Strip fences / whitespace
            txt = txt.strip().strip('`')
            # Locate outermost JSON list
            start = txt.find('['); end = txt.rfind(']')
            if start == -1 or end == -1:
                if getattr(self.state,'r',None): self.state.r.incr("llm:proposer:rejected")
                return []
            raw_slice = txt[start:end+1]
            try:
                suggestions = json.loads(raw_slice)
            except Exception:
                if getattr(self.state,'r',None): self.state.r.incr("llm:proposer:rejected")
                return []
            cleaned: List[List[Any]] = []
            seen_hashes = set()
            pipe = getattr(self.state,'r',None)
            existing = set(pipe.smembers('llm:proposer:seen')) if pipe else set()

            for f in suggestions:
                # use sanitizer (suppressed awareness via Redis keys if present)
                suppressed = []
                if getattr(self.state, 'get', None):
                    try:
                        suppressed = self.state.get('eil:continuity_suppress') or []
                        # include currently gated off features
                        fg = self.state.get('eil:feature_gate_diag') or {}
                        for feat, meta in fg.items():
                            if not meta.get('keep', False):
                                suppressed.append(feat)
                    except Exception:
                        suppressed = []
                sanitized = self.sanitize_formulas(base_features, [f], suppressed, threshold_min, threshold_max)
                if not sanitized:
                    continue
                f = sanitized[0]
                canon = json.dumps(f, separators=(',',':'))
                h = hashlib.sha256(canon.encode()).hexdigest()
                if h in existing or h in seen_hashes:
                    continue
                seen_hashes.add(h)
                cleaned.append(f)
            if pipe:
                if cleaned:
                    pipe.incr("llm:proposer:success")
                    for f in cleaned:
                        pipe.rpush("llm:proposals", json.dumps(f, separators=(',',':')))
                    # add hashes to SET (trim if >5k)
                    if seen_hashes:
                        pipe.sadd('llm:proposer:seen', *list(seen_hashes))
                rejected = len(suggestions) - len(cleaned)
                if rejected>0: pipe.incrby("llm:proposer:rejected", rejected)
                pipe.hincrby("llm:proposer:char_usage", "total_chars", len(txt))
                pipe.execute()
            return cleaned[:k]
        except requests.HTTPError as he:
            if getattr(self.state,'r',None):
                self.state.r.incr("llm:proposer:http_errors")
            logger.warning(f"LLM proposer HTTP {getattr(he.response,'status_code', 'NA')}")
            return []
        except Exception as e:
            if getattr(self.state,'r',None):
                self.state.r.incr("llm:proposer:errors")
            logger.warning(f"LLM proposer error: {type(e).__name__}")
            return []

    def propose(self, base_features: List[str], k: int = 3, threshold_min: float=-5.0, threshold_max: float=5.0) -> List[List[Any]]:
        return self._remote_suggestions(base_features, k=k, threshold_min=threshold_min, threshold_max=threshold_max)
