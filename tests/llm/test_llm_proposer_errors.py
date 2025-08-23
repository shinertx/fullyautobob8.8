import types
from v26meme.llm.proposer import LLMProposer

class FakeState:
    def __init__(self): self.store={}
    def get(self,k): return self.store.get(k)
    def set(self,k,v): self.store[k]=v

class DummyResp:
    def __init__(self, text: str, status: int=200):
        self.text = text; self.status_code=status
    def json(self):
        raise ValueError("invalid json")

def test_llm_proposer_handles_missing_key(monkeypatch):
    st = FakeState()
    p = LLMProposer(st)
    # Ensure no key in env
    import os; monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    out = list(p.propose(['f1','f2'], k=2))
    assert out == []

def test_llm_proposer_handles_bad_json(monkeypatch):
    st = FakeState(); p = LLMProposer(st)
    import os; monkeypatch.setenv('OPENAI_API_KEY','sk-test')
    def fake_post(*a, **k):
        return DummyResp('{"not":"list"}', 200)
    import requests
    monkeypatch.setattr(requests, 'post', fake_post)
    out = list(p.propose(['fa'], k=1))
    assert out == []

