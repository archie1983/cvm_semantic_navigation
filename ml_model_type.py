from enum import Enum

class MLModelType(Enum):
    LLM = 1 # LLM, e.g. Mistral, Gemma, LLama3
    CVM = 2 # CVM, e.g. Meta Chameleon, Ms Florence, Moondream
