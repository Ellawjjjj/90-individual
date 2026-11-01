"""Visual grounding (referring expression comprehension) evaluation and fine-tuning
toolkit for Qwen-family vision-language models.

Modules that do not need torch (safe to import in tests / lightweight tools):
    parse, metrics, prompts, data
Modules that need torch / transformers:
    models, train_utils
"""

__version__ = "0.1.0"
