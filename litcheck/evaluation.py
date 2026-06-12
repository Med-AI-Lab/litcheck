


import json
from dataclasses import dataclass
from .llm import llm
from .search import Article
from .pico import PicoResult, extract_pico


# ─────────────────────────────────────────────
# Оценка гипотезы: типы данных
# ─────────────────────────────────────────────

# Допустимые значения для уровней перекрытия и оценок
OverlapLevel = str   # "full" | "partial" | "none"
ScoreLevel   = str   # "low" | "medium" | "high"


@dataclass
class PaperComparison:
    """Результат сравнения PICO гипотезы с PICO одной статьи."""
    pmid:                 str
    title:                str
    paper_pico:           PicoResult
    population_overlap:   OverlapLevel   # full / partial / none
    intervention_overlap: OverlapLevel
    comparison_overlap:   OverlapLevel
    outcome_overlap:      OverlapLevel
    population_note:      str            # одно предложение объяснения
    intervention_note:    str
    comparison_note:      str
    outcome_note:         str

    def __str__(self) -> str:
        return (
            f"PMID {self.pmid}: {self.title[:80]}\n"
            f"  P overlap: {self.population_overlap:7s} — {self.population_note}\n"
            f"  I overlap: {self.intervention_overlap:7s} — {self.intervention_note}\n"
            f"  C overlap: {self.comparison_overlap:7s} — {self.comparison_note}\n"
            f"  O overlap: {self.outcome_overlap:7s} — {self.outcome_note}"
        )


@dataclass
class EvaluationDimension:
    score:     ScoreLevel   # low / medium / high
    reasoning: str          # 2-4 предложения

    def __str__(self) -> str:
        return f"[{self.score.upper()}] {self.reasoning}"


@dataclass
class HypothesisEvaluation:
    """Итоговая оценка гипотезы по трём измерениям."""
    hypothesis_pico: PicoResult
    comparisons:     list[PaperComparison]   # промежуточные результаты
    novelty:         EvaluationDimension
    feasibility:     EvaluationDimension
    impact:          EvaluationDimension

    def __str__(self) -> str:
        lines = [
            "─" * 65,
            "HYPOTHESIS PICO",
            "─" * 65,
            str(self.hypothesis_pico),
            "",
            "─" * 65,
            f"PER-PAPER COMPARISONS  ({len(self.comparisons)} papers)",
            "─" * 65,
        ]
        for i, c in enumerate(self.comparisons, 1):
            lines.append(f"\n[{i}] {c}")
        lines += [
            "",
            "─" * 65,
            "EVALUATION",
            "─" * 65,
            f"Novelty     : {self.novelty}",
            f"Feasibility : {self.feasibility}",
            f"Impact      : {self.impact}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Оценка гипотезы: LLM-промпты и функции
# ─────────────────────────────────────────────

_COMPARE_SYSTEM = """You are a biomedical research expert specialised in evidence-based medicine.
You will be given the PICO decomposition of a research hypothesis and the PICO decomposition
of a published article. Compare them component by component.

Return ONLY a JSON object with exactly these keys:
  "population_overlap":   "full" | "partial" | "none",
  "population_note":      one sentence explaining the overlap,
  "intervention_overlap": "full" | "partial" | "none",
  "intervention_note":    one sentence explaining the overlap,
  "comparison_overlap":   "full" | "partial" | "none",
  "comparison_note":      one sentence explaining the overlap,
  "outcome_overlap":      "full" | "partial" | "none",
  "outcome_note":         one sentence explaining the overlap.

Definitions:
  full    — the paper covers essentially the same concept as the hypothesis component.
  partial — there is meaningful overlap but also clear differences.
  none    — no meaningful overlap.

No explanation outside the JSON."""

_SYNTHESIZE_SYSTEM = """You are a biomedical research expert specialised in evidence-based medicine.
You will be given the PICO decomposition of a research hypothesis and a list of
per-paper PICO comparison summaries derived from a literature search.

Evaluate the hypothesis on three dimensions and return ONLY a JSON object with:
  "novelty": {
      "score":     "low" | "medium" | "high",
      "reasoning": "2-4 sentences"
  },
  "feasibility": {
      "score":     "low" | "medium" | "high",
      "reasoning": "2-4 sentences"
  },
  "impact": {
      "score":     "low" | "medium" | "high",
      "reasoning": "2-4 sentences"
  }

Scoring guidance:
  novelty     — HIGH means the hypothesis addresses a gap not covered by existing literature;
                LOW means it largely replicates known work.
  feasibility — HIGH means the methods/interventions are well established in the literature;
                LOW means they are speculative or untested.
  impact      — HIGH means the outcome is clinically or scientifically significant;
                LOW means the outcome has limited broader relevance.

No explanation outside the JSON."""


MAX_EVAL_PAPERS = 20   # максимальное число статей для оценки


def _compare_pico(
    hypothesis_pico: PicoResult,
    paper: Article,
    paper_pico: PicoResult,
) -> PaperComparison:
    """Сравнивает PICO гипотезы с PICO одной статьи."""
    def _fmt(p: PicoResult) -> str:
        return (
            f"P: {p.population}\n"
            f"I: {p.intervention}\n"
            f"C: {p.comparison or 'not stated'}\n"
            f"O: {p.outcome}"
        )

    user_msg = (
        f"HYPOTHESIS PICO:\n{_fmt(hypothesis_pico)}"
        f"\n\nPAPER PICO (PMID {paper.pmid}):\n{_fmt(paper_pico)}"
    )
    raw  = llm(_COMPARE_SYSTEM, user_msg, max_tokens=512)
    data = json.loads(raw)

    return PaperComparison(
        pmid                 = paper.pmid,
        title                = paper.title,
        paper_pico           = paper_pico,
        population_overlap   = data.get("population_overlap",   "none"),
        intervention_overlap = data.get("intervention_overlap", "none"),
        comparison_overlap   = data.get("comparison_overlap",   "none"),
        outcome_overlap      = data.get("outcome_overlap",      "none"),
        population_note      = data.get("population_note",      ""),
        intervention_note    = data.get("intervention_note",    ""),
        comparison_note      = data.get("comparison_note",      ""),
        outcome_note         = data.get("outcome_note",         ""),
    )


def _synthesize_evaluation(
    hypothesis_pico: PicoResult,
    comparisons: list[PaperComparison],
) -> tuple[EvaluationDimension, EvaluationDimension, EvaluationDimension]:
    """Синтезирует финальную оценку по всем сравнениям."""
    def _fmt_comparison(c: PaperComparison) -> str:
        return (
            f"PMID {c.pmid} | "
            f"P:{c.population_overlap} I:{c.intervention_overlap} "
            f"C:{c.comparison_overlap} O:{c.outcome_overlap} | "
            f"{c.population_note} {c.intervention_note} "
            f"{c.comparison_note} {c.outcome_note}"
        )

    comparisons_text = "\n".join(_fmt_comparison(c) for c in comparisons)
    user_msg = (
        f"HYPOTHESIS PICO:\n"
        f"P: {hypothesis_pico.population}\n"
        f"I: {hypothesis_pico.intervention}\n"
        f"C: {hypothesis_pico.comparison or 'not stated'}\n"
        f"O: {hypothesis_pico.outcome}\n\n"
        f"LITERATURE COMPARISONS ({len(comparisons)} papers):\n"
        f"{comparisons_text}"
    )

    raw  = llm(_SYNTHESIZE_SYSTEM, user_msg, max_tokens=768)
    data = json.loads(raw)

    def _dim(key: str) -> EvaluationDimension:
        d = data.get(key, {})
        return EvaluationDimension(
            score     = d.get("score",     "medium"),
            reasoning = d.get("reasoning", ""),
        )

    return _dim("novelty"), _dim("feasibility"), _dim("impact")


def evaluate_hypothesis(
    hypothesis: str,
    articles: list[Article],
) -> HypothesisEvaluation:
    """
    Оценивает гипотезу по критериям новизны, выполнимости и значимости
    на основе найденной литературы.

    Пайплайн:
      1. PICO гипотезы.
      2. PICO каждой из top-N статей (по rank_score, затем по порядку).
      3. Попарное сравнение PICO гипотезы с каждой статьёй.
      4. Финальный синтез → novelty / feasibility / impact.

    Args:
        hypothesis: Медицинская гипотеза (любой язык).
        articles:   Список статей, отсортированных по релевантности
                    (как возвращает semantic_pubmed_search).

    Returns:
        HypothesisEvaluation с промежуточными сравнениями и итоговыми оценками.
    """
    ranked = sorted(
        articles,
        key=lambda a: a.rank_score if a.rank_score is not None else -1,
        reverse=True,
    )
    top_papers = ranked[:MAX_EVAL_PAPERS]
    total      = len(top_papers)

    print("=" * 65)
    print("ОЦЕНКА ГИПОТЕЗЫ")
    print("=" * 65)

    # ── Шаг 1. PICO гипотезы ─────────────────────────────────────────────
    print("⚙️  [1] PICO гипотезы...")
    hypothesis_pico = extract_pico(hypothesis=hypothesis)

    # ── Шаг 2 + 3. PICO и сравнение для каждой статьи ────────────────────
    comparisons: list[PaperComparison] = []
    for i, paper in enumerate(top_papers, 1):
        print(f"📄 [{i}/{total}] PMID {paper.pmid}: извлечение PICO и сравнение...")
        try:
            paper_pico = extract_pico(title=paper.title, abstract=paper.abstract)
            comparison = _compare_pico(hypothesis_pico, paper, paper_pico)
            comparisons.append(comparison)
        except Exception as exc:
            print(f"   ⚠️  Пропущено: {exc}")

    # ── Шаг 4. Синтез ────────────────────────────────────────────────────
    print("🤖 Финальный синтез оценки...")
    novelty, feasibility, impact = _synthesize_evaluation(hypothesis_pico, comparisons)

    return HypothesisEvaluation(
        hypothesis_pico = hypothesis_pico,
        comparisons     = comparisons,
        novelty         = novelty,
        feasibility     = feasibility,
        impact          = impact,
    )