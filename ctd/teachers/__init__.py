"""Remote teacher backends for CTD precompute.

Supports running precompute against teachers we don't host locally —
hosted APIs (Together / Fireworks / DeepInfra) or local Ollama.
The cache shape produced is identical to ctd.precompute output, so
trainers don't need any changes.
"""

from ctd.teachers.base import PositionLogprobs, TeacherBackend
from ctd.teachers.ollama import OllamaTeacher
from ctd.teachers.openai_compat import OpenAICompletionsTeacher

__all__ = [
    "PositionLogprobs",
    "TeacherBackend",
    "OllamaTeacher",
    "OpenAICompletionsTeacher",
]
