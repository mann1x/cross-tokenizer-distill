"""Unit tests for ctd.teachers — no live HTTP needed.

Mocks requests.post to validate:
- OllamaTeacher: N HTTP calls per N-token text, correct prompt prefixes,
  correct top-K extraction.
- OpenAICompletionsTeacher: 1 call per text, correct echo+logprobs parse.
- _str_to_id round-trip via vocab dict.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from ctd.teachers.ollama import OllamaTeacher
from ctd.teachers.openai_compat import OpenAICompletionsTeacher


class FakeTokenizer:
    """Minimal HF-tokenizer-like stub.

    Vocab: maps tokens to small integer IDs so str_to_id round-trips.
    encode/decode use whitespace splitting.
    """

    def __init__(self):
        self._vocab = {
            "<bos>": 0, "def": 1, " hello": 2, "(": 3, ")": 4,
            ":": 5, "\n": 6, "    ": 7, "return": 8, " 1": 9,
            "Hello": 10, "world": 11, "!": 12,
        }
        self._inv = {v: k for k, v in self._vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        out = []
        i = 0
        while i < len(text):
            for tok, tid in self._vocab.items():
                if text[i:].startswith(tok):
                    out.append(tid)
                    i += len(tok)
                    break
            else:
                i += 1
        return out

    def decode(self, ids):
        return "".join(self._inv.get(i, "?") for i in ids)


def _ollama_response(top_token, alts):
    return {
        "logprobs": [
            {
                "token": top_token,
                "logprob": 0.0,
                "top_logprobs": [
                    {"token": t, "logprob": lp, "bytes": list(t.encode())}
                    for t, lp in alts
                ],
            }
        ]
    }


def test_ollama_one_call_per_token():
    tok = FakeTokenizer()
    text = "def hello"
    teacher_ids = tok.encode(text)
    assert teacher_ids == [1, 2]

    backend = OllamaTeacher(
        base_url="http://x", model_id="m", tokenizer=tok, top_k=3
    )
    expected_prompts = ["", "def"]
    posted_prompts = []

    def fake_post(url, json, timeout):
        posted_prompts.append(json["prompt"])
        m = MagicMock()
        m.raise_for_status = lambda: None
        # Different alts per position for assertion clarity.
        if json["prompt"] == "":
            m.json = lambda: _ollama_response("def", [("def", -0.1), (" hello", -2.0)])
        else:
            m.json = lambda: _ollama_response(" hello", [(" hello", -0.05), ("(", -3.0)])
        return m

    with patch("ctd.teachers.ollama.requests.post", side_effect=fake_post):
        ids, per_pos = backend.get_token_topk(text)

    assert ids == teacher_ids
    assert posted_prompts == expected_prompts
    assert len(per_pos) == 2
    assert per_pos[0].topk_ids == [1, 2]  # def, " hello"
    assert per_pos[0].topk_logprobs == [-0.1, -2.0]
    assert per_pos[1].topk_ids == [2, 3]  # " hello", "("


def test_ollama_unknown_token_skipped():
    tok = FakeTokenizer()
    backend = OllamaTeacher(base_url="http://x", model_id="m", tokenizer=tok, top_k=5)

    def fake_post(url, json, timeout):
        m = MagicMock()
        m.raise_for_status = lambda: None
        m.json = lambda: _ollama_response(
            "def", [("def", -0.1), ("UNKNOWN_TOKEN_NOT_IN_VOCAB", -2.0), (" hello", -3.0)]
        )
        return m

    with patch("ctd.teachers.ollama.requests.post", side_effect=fake_post):
        _, per_pos = backend.get_token_topk("def")

    # UNKNOWN should be skipped; we keep def + " hello".
    assert per_pos[0].topk_ids == [1, 2]
    assert per_pos[0].topk_logprobs == [-0.1, -3.0]


def test_openai_compat_one_call_per_text():
    tok = FakeTokenizer()
    text = "def hello"
    teacher_ids = tok.encode(text)
    assert teacher_ids == [1, 2]

    posted = []

    def fake_post(url, headers, json, timeout):
        posted.append((url, json))
        m = MagicMock()
        m.raise_for_status = lambda: None
        m.json = lambda: {
            "choices": [{
                "logprobs": {
                    "tokens": ["def", " hello"],
                    "token_logprobs": [None, -0.05],
                    "top_logprobs": [
                        None,  # echo: first position has no preceding context
                        {" hello": -0.05, "(": -3.0, ":": -4.5},
                    ],
                    "text_offset": [0, 3],
                }
            }]
        }
        return m

    backend = OpenAICompletionsTeacher(
        base_url="https://api.fake.com/v1",
        model_id="fake-model",
        tokenizer=tok,
        api_key="sk-test",
        top_k=5,
    )

    with patch("ctd.teachers.openai_compat.requests.post", side_effect=fake_post):
        ids, per_pos = backend.get_token_topk(text)

    # Exactly one HTTP call (vs N for Ollama).
    assert len(posted) == 1
    url, body = posted[0]
    assert url.endswith("/completions")
    assert body["echo"] is True
    assert body["prompt"] == text
    assert body["logprobs"] == 5

    assert ids == teacher_ids
    assert len(per_pos) == 2
    # First position is the placeholder (echo's leading None).
    assert per_pos[0].topk_ids == []
    # Second position has 3 alternatives, all in vocab.
    assert sorted(per_pos[1].topk_ids) == sorted([2, 3, 5])
    assert sorted(per_pos[1].topk_logprobs) == sorted([-0.05, -3.0, -4.5])


def test_openai_compat_missing_api_key_raises():
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="api_key"):
        OpenAICompletionsTeacher(
            base_url="https://x", model_id="m", tokenizer=tok, api_key=None
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
