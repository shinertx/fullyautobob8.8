import os, json, random, time, hashlib
from typing import List, Dict, Any
import requests
from loguru import logger

class LLMProposer:
    """OpenAI-only proposer with hardened JSON pipeline.

    PIT: Deterministic post-processing; no future leakage.
    Guardrails: feature whitelist, threshold clamps, dedupe, metrics.
    """
    def __init__(self, state, cfg: Dict[str, Any] | None = None):
        self.state = state
        self.cfg = cfg or {}
        self.llm_cfg = self.cfg.get('llm', {})
        self.provider = self.llm_cfg.get("provider", "openai").lower()

    def _local_suggestions(self, base_features: List[str], k: int = 0) -> List[List[Any]]:
        # disabled by policy; return []
        return []

    @staticmethod
    def sanitize_formulas(base_features: List[str], suggestions: List[List[Any]],
                          suppressed: List[str] | None = None,
                          threshold_min: float = -5.0, threshold_max: float = 5.0,
                          max_nodes: int = 31,
                          feature_stats: Dict[str, Dict[str, float]] | None = None) -> List[List[Any]]:
        """Validate & clamp formulas.

        PIT: Pure transformation of already-produced suggestions.
        No side effects; deterministic given inputs.
        """
        whitelist = set(base_features)
        suppressed_set = set(suppressed or [])
        cleaned: List[List[Any]] = []
        seen_hashes = set()
        import json as _json, hashlib as _hashlib

        def _normalize_op(op: str) -> str:
            if not isinstance(op, str):
                return ""
            opu = op.strip().upper()
            if opu in ("AND", "OR"):
                return opu
            if op.strip() in ('>', '<', '>=', '<=', '==', '!='):
                return op.strip()
            return ""

        def _coerce_number(val: Any) -> float | None:
            try:
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    return float(val.strip())
            except Exception:
                return None
            return None

        def _valid(node) -> bool:
            if not isinstance(node, list):
                return False
            # condition
            if len(node) >=3 and isinstance(node[0], str) and _normalize_op(node[1]) in ('>','<','>=','<=','==','!='):
                if node[0] not in whitelist or node[0] in suppressed_set:
                    return False
                if _coerce_number(node[2]) is None:
                    return False
                return True
            if len(node) >=3 and isinstance(node[0], list) and isinstance(node[2], list) and _normalize_op(node[1]) in ('AND','OR'):
                return _valid(node[0]) and _valid(node[2])
            return False

        def _clamp(node):
            if isinstance(node, list) and len(node)>=3 and isinstance(node[0], str) and _normalize_op(node[1]) in ('>','<','>=','<=','==','!='):
                feat = node[0]
                num = _coerce_number(node[2])
                if num is None:
                    node[2] = 0.0
                    return
                # Prefer empirical quantiles per feature if available
                if feature_stats and isinstance(feature_stats.get(feat), dict):
                    st = feature_stats.get(feat, {})
                    lo = st.get('q10', st.get('min', threshold_min))
                    hi = st.get('q90', st.get('max', threshold_max))
                    try:
                        lo = float(lo); hi = float(hi)
                        if lo > hi:
                            lo, hi = hi, lo
                    except Exception:
                        lo, hi = threshold_min, threshold_max
                    node[1] = _normalize_op(node[1]) or '>'
                    node[2] = min(max(num, lo), hi)
                else:
                    node[1] = _normalize_op(node[1]) or '>'
                    node[2] = max(threshold_min, min(threshold_max, num))
            elif isinstance(node, list) and len(node)>=3:
                if isinstance(node[1], str) and _normalize_op(node[1]) in ('AND','OR'):
                    node[1] = _normalize_op(node[1])
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

        model = self.llm_cfg.get("model", "gpt-4o-mini")
        temperature = self.llm_cfg.get("temperature", 0.4)
        max_tokens = self.llm_cfg.get("max_tokens", 400)

        run_details = {
            "timestamp": time.time(),
            "inputs": {"base_features": base_features, "k": k, "threshold_min": threshold_min, "threshold_max": threshold_max},
            "prompt": {"system": sys_prompt, "user": json.dumps(user)},
            "api_call": {"model": model, "temperature": temperature, "max_tokens": max_tokens},
            "api_response": {},
            "processing_steps": [],
            "final_output": [],
            "metrics": {},
            "errors": [],
        }

        t0 = time.monotonic()
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": model, "messages":[{"role":"system","content":sys_prompt},{"role":"user","content":json.dumps(user)}],
                       "temperature": temperature, "max_tokens": max_tokens}

            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
            latency_ms = int((time.monotonic()-t0)*1000)
            run_details["api_response"]["latency_ms"] = latency_ms
            run_details["api_response"]["status_code"] = r.status_code

            if getattr(self.state,'r',None):
                self.state.r.hincrby("llm:proposer:latency_ms_buckets", str(min( (latency_ms//100)*100, 2000)), 1)
            r.raise_for_status()
            j = r.json()
            txt = (j.get("choices") or [{}])[0].get("message", {}).get("content", "[]")
            run_details["api_response"]["raw_content"] = txt

            # Strip fences / whitespace
            txt = txt.strip().strip('`')
            # Locate outermost JSON list
            start = txt.find('['); end = txt.rfind(']')
            if start == -1 or end == -1:
                if getattr(self.state,'r',None): self.state.r.incr("llm:proposer:rejected")
                run_details["errors"].append("Could not find JSON list in LLM response.")
                return []
            raw_slice = txt[start:end+1]
            try:
                suggestions = json.loads(raw_slice)
                run_details["metrics"]["initial_suggestion_count"] = len(suggestions)
            except Exception as e:
                if getattr(self.state,'r',None): self.state.r.incr("llm:proposer:rejected")
                run_details["errors"].append(f"JSON parsing failed: {e}")
                run_details["processing_steps"].append({"original_slice": raw_slice, "outcome": "REJECTED", "reason": "Invalid JSON"})
                return []

            cleaned: List[List[Any]] = []
            seen_hashes = set()
            r = getattr(self.state,'r',None)
            try:
                existing = set(r.smembers('llm:proposer:seen')) if r else set()
            except Exception:
                existing = set()

            for f in suggestions:
                step_detail = {"original_suggestion": f, "outcome": "REJECTED"}
                suppressed = []
                if getattr(self.state, 'get', None):
                    try:
                        suppressed = self.state.get('eil:continuity_suppress') or []
                        fg = self.state.get('eil:feature_gate_diag') or {}
                        for feat, meta in fg.items():
                            if not meta.get('keep', False):
                                suppressed.append(feat)
                    except Exception:
                        suppressed = []

                # Pull empirical feature stats when available to clamp thresholds to realistic ranges
                feat_stats = {}
                try:
                    if getattr(self.state, 'get', None):
                        raw_stats = self.state.get('eil:feature_stats') or {}
                        if isinstance(raw_stats, dict):
                            # raw_stats may include 'keep' flags per feature; pass-through is fine
                            feat_stats = raw_stats
                except Exception:
                    feat_stats = {}
                sanitized = self.sanitize_formulas(base_features, [f], suppressed, threshold_min, threshold_max, feature_stats=feat_stats)
                if not sanitized:
                    step_detail["reason"] = "Sanitization failed"
                    step_detail["details"] = {"suppressed_features": suppressed}
                    run_details["processing_steps"].append(step_detail)
                    continue

                f_sanitized = sanitized[0]
                canon = json.dumps(f_sanitized, separators=(',',':'))
                h = hashlib.sha256(canon.encode()).hexdigest()
                if h in existing or h in seen_hashes:
                    step_detail["reason"] = "Duplicate"
                    step_detail["details"] = {"hash": h, "is_new_duplicate": h in seen_hashes, "is_existing_duplicate": h in existing}
                    run_details["processing_steps"].append(step_detail)
                    continue

                step_detail["outcome"] = "ACCEPTED"
                step_detail["sanitized_suggestion"] = f_sanitized
                step_detail["details"] = {"hash": h}
                run_details["processing_steps"].append(step_detail)
                seen_hashes.add(h)
                cleaned.append(f_sanitized)

            run_details["final_output"] = cleaned
            run_details["metrics"]["accepted_count"] = len(cleaned)
            run_details["metrics"]["rejected_count"] = run_details["metrics"].get("initial_suggestion_count", 0) - len(cleaned)

            if r:
                if cleaned:
                    try:
                        r.incr("llm:proposer:success")
                    except Exception:
                        pass
                    for f in cleaned:
                        try:
                            r.rpush("llm:proposals", json.dumps(f, separators=(',',':')))
                        except Exception:
                            pass
                    if seen_hashes:
                        try:
                            r.sadd('llm:proposer:seen', *list(seen_hashes))
                        except Exception:
                            pass
                rejected = len(suggestions) - len(cleaned)
                if rejected>0:
                    try:
                        r.incrby("llm:proposer:rejected", rejected)
                    except Exception:
                        pass
                try:
                    r.hincrby("llm:proposer:char_usage", "total_chars", len(txt))
                except Exception:
                    pass
            return cleaned[:k]
        except requests.HTTPError as he:
            if getattr(self.state,'r',None):
                self.state.r.incr("llm:proposer:http_errors")
            error_msg = f"LLM proposer HTTP {getattr(he.response,'status_code', 'NA')}"
            logger.warning(error_msg)
            run_details["errors"].append(error_msg)
            return []
        except Exception as e:
            if getattr(self.state,'r',None):
                self.state.r.incr("llm:proposer:errors")
            error_msg = f"LLM proposer error: {type(e).__name__}: {e}"
            logger.warning(error_msg)
            run_details["errors"].append(error_msg)
            return []
        finally:
            # This is the key: log the detailed trace of the entire run.
            logger.bind(run_details=run_details).info("LLMProposer run complete.")

    def propose(self, base_features: List[str], k: int = 3, threshold_min: float=-5.0, threshold_max: float=5.0) -> List[List[Any]]:
        return self._remote_suggestions(base_features, k=k, threshold_min=threshold_min, threshold_max=threshold_max)
