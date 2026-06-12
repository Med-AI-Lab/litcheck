import os
import json
import time
import requests
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from metapub import PubMedFetcher
from .llm import llm
from .search import Article


# ─────────────────────────────────────────────
# PICO-декомпозиция
# ─────────────────────────────────────────────

_PICO_SYSTEM = """You are a biomedical research expert specialised in evidence-based medicine.
Given either a medical hypothesis OR a combination of an article title and abstract,
extract its PICO components.

Return ONLY a JSON object with exactly these keys:
  "population":    who are the subjects (patients, organisms, samples, etc.),
  "intervention":  what is being tested or applied (drug, procedure, exposure, etc.),
  "comparison":    what it is compared to (placebo, standard care, another intervention);
                   use null if no comparison is stated,
  "outcome":       what is measured or expected to change,
  "notes":         any important context that does not fit P/I/C/O (optional, may be null).

All values must be concise strings (1-2 sentences max) or null. English only.
No explanation outside the JSON."""


@dataclass
class PicoResult:
    population:   str
    intervention: str
    comparison:   Optional[str]
    outcome:      str
    notes:        Optional[str] = None

    def __str__(self) -> str:
        lines = [
            f"P (Population)   : {self.population}",
            f"I (Intervention) : {self.intervention}",
            f"C (Comparison)   : {self.comparison or '—'}",
            f"O (Outcome)      : {self.outcome}",
        ]
        if self.notes:
            lines.append(f"Notes            : {self.notes}")
        return "\n".join(lines)


def extract_pico(
    hypothesis: Optional[str] = None,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
) -> PicoResult:
    """
    Извлекает PICO-компоненты из гипотезы или из названия + абстракта статьи.

    Передайте ЛИБО `hypothesis`, ЛИБО `title` (и опционально `abstract`).

    Args:
        hypothesis: Медицинская гипотеза (любой язык).
        title:      Название статьи.
        abstract:   Абстракт статьи (опционально, но улучшает качество).

    Returns:
        PicoResult с полями population, intervention, comparison, outcome, notes.

    Raises:
        ValueError: если не передан ни один из входных параметров.
    """
    if hypothesis:
        user_msg = f"Medical hypothesis:\n{hypothesis}"
    elif title:
        user_msg = f"Article title:\n{title}"
        if abstract:
            user_msg += f"\n\nAbstract:\n{abstract}"
    else:
        raise ValueError("Передайте hypothesis или title (с опциональным abstract).")

    raw = llm(_PICO_SYSTEM, user_msg, max_tokens=512)
    data = json.loads(raw)

    return PicoResult(
        population   = data.get("population", ""),
        intervention = data.get("intervention", ""),
        comparison   = data.get("comparison"),
        outcome      = data.get("outcome", ""),
        notes        = data.get("notes"),
    )