"""
Семантический поиск медицинских статей в PubMed с расслаблением запроса.

Пайплайн:
1. LLM извлекает ключевые слова → строим первичный PubMed-запрос.
2. Запускаем поиск. Если результаты найдены — готово.
3. Если пусто → удаляем дженерик-слова (эвристика) → повторяем один раз.
4. Если всё ещё пусто → LLM генерирует 3 уровня запроса (specific / moderate / broad)
   → пробуем каждый по очереди, останавливаемся на первом непустом.
5. Если все попытки провалились → возвращаем пустой список + сообщение.

Результат содержит поле `search_path`, которое сообщает, каким путём были найдены статьи.

Переменные окружения:
    OPENROUTER_API_KEY  — ключ OpenRouter (обязательно)
    NCBI_API_KEY        — ключ NCBI (опционально, увеличивает лимиты)
"""


import os
import json
import time
import requests
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from metapub import PubMedFetcher
from .llm import llm


# ─────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────

NCBI_API_KEY       = os.getenv("NCBI_API_KEY", "")

MAX_RESULTS   = 10
REQUEST_DELAY = 0.5   # секунд между запросами NCBI

# Дженерик-слова, которые не несут специфической медицинской нагрузки
GENERIC_TERMS: set[str] = {
    "patients", "patient", "study", "studies", "effect", "effects",
    "role", "mechanism", "mechanisms", "association", "associations",
    "analysis", "disease", "condition", "conditions", "factor", "factors",
    "response", "responses", "level", "levels", "increase", "decrease",
    "change", "changes", "treatment", "treatments", "outcome", "outcomes",
    "risk", "risks", "human", "humans", "clinical", "related", "based",
    "induced", "mediated", "involved", "using", "potential", "novel",
    "recent", "current", "review", "meta-analysis", "systematic",
}

# ─────────────────────────────────────────────
# Типы данных
# ─────────────────────────────────────────────

class SearchPath(str, Enum):
    ORIGINAL  = "original"           # первый запрос сразу дал результат
    RELAXED   = "relaxed_generic"    # после удаления дженерик-слов
    ZOOM_SPECIFIC = "zoom_specific"  # уровень 1 от LLM
    ZOOM_MODERATE = "zoom_moderate"  # уровень 2 от LLM
    ZOOM_BROAD    = "zoom_broad"     # уровень 3 от LLM
    NOT_FOUND = "not_found"          # все попытки провалились


@dataclass
class Article:
    pmid:    str
    title:   str
    authors: list[str]     = field(default_factory=list)
    journal: str           = ""
    year:    Optional[str] = None
    abstract: str          = ""
    doi:     Optional[str] = None
    url:     str           = ""
    # заполняется после ранжирования
    rank_score:  Optional[float] = field(default=None, repr=False)
    rank_reason: str             = field(default="",   repr=False)

    def __str__(self) -> str:
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        score_str = (
            f"\n  Релевантность: {self.rank_score:.2f} — {self.rank_reason}"
            if self.rank_score is not None else ""
        )
        return "\n".join([
            f"PMID     : {self.pmid}",
            f"Заголовок: {self.title}",
            f"Авторы   : {authors_str or '—'}",
            f"Журнал   : {self.journal or '—'} ({self.year or '?'})",
            f"DOI      : {self.doi or '—'}",
            f"URL      : {self.url}",
            f"Аннотация: {self.abstract[:300]}{'…' if len(self.abstract) > 300 else ''}",
        ]) + score_str


@dataclass
class SearchResult:
    articles:    list[Article]
    search_path: SearchPath
    query_used:  str   # итоговый запрос, который дал результат

    def __str__(self) -> str:
        return (
            f"Путь поиска : {self.search_path.value}\n"
            f"Запрос      : {self.query_used}\n"
            f"Найдено     : {len(self.articles)} статей"
        )

# ─────────────────────────────────────────────
# Шаг 1. Извлечение ключевых слов
# ─────────────────────────────────────────────

_KW_SYSTEM = """You are a biomedical research expert.
Given a medical hypothesis, extract search keywords for PubMed.

Return ONLY a JSON object with:
  "keywords": list of 5-10 MeSH-style terms/phrases (English),
  "pubmed_query": a PubMed boolean query using AND/OR and optional
                  [MeSH Terms] or [Title/Abstract] field tags.

No explanation outside the JSON.

Example:
{
  "keywords": ["type 2 diabetes", "gut microbiome", "insulin resistance"],
  "pubmed_query": "(type 2 diabetes[MeSH Terms]) AND (gut microbiome[Title/Abstract] OR intestinal microbiota[MeSH Terms]) AND insulin resistance[MeSH Terms]"
}"""


def _extract_keywords(hypothesis: str) -> tuple[list[str], str]:
    """Возвращает (keywords, pubmed_query)."""
    raw = llm(_KW_SYSTEM, f"Medical hypothesis:\n{hypothesis}", max_tokens=512)
    data = json.loads(raw)
    keywords: list[str] = data.get("keywords", [])
    query: str = data.get("pubmed_query", " AND ".join(keywords))
    return keywords, query

# ─────────────────────────────────────────────
# Шаг 3 (опция 6). Удаление дженерик-слов
# ─────────────────────────────────────────────

def _is_generic(term: str) -> bool:
    """Возвращает True, если все значимые слова термина — дженерики."""
    words = {w.lower().strip("[]()") for w in term.split()}
    # термин считается дженерик-словом, если ВСЕ его слова есть в GENERIC_TERMS
    return words.issubset(GENERIC_TERMS)


def _drop_generic_keywords(keywords: list[str]) -> list[str]:
    """Фильтрует дженерик-термины из списка ключевых слов."""
    filtered = [kw for kw in keywords if not _is_generic(kw)]
    # защита: оставляем хотя бы одно слово
    return filtered if filtered else keywords


def _rebuild_query(keywords: list[str]) -> str:
    """
    Строит простой AND-запрос из оставшихся ключевых слов.
    Каждое слово ищется по Title/Abstract для максимального покрытия.
    """
    parts = [f'("{kw}"[Title/Abstract])' for kw in keywords]
    return " AND ".join(parts)


# ─────────────────────────────────────────────
# Шаг 4 (опция 3). Генерация zoom-уровней
# ─────────────────────────────────────────────

_ZOOM_SYSTEM = """You are a biomedical librarian expert in PubMed search strategy.
Given a medical hypothesis and its extracted keywords, generate 3 PubMed queries
at different specificity levels to progressively relax the search.

Return ONLY a JSON object with exactly these keys:
  "specific": PubMed query — keeps the 2-3 most important core concepts,
              still uses MeSH or Title/Abstract field tags.
  "moderate": PubMed query — uses broader MeSH parent terms, fewer constraints.
  "broad":    PubMed query — 1-2 top-level concepts only, no field restrictions.

No explanation outside the JSON."""


def _generate_zoom_queries(hypothesis: str, keywords: list[str]) -> dict[str, str]:
    """Генерирует 3 уровня запроса: specific / moderate / broad."""
    user_msg = (
        f"Hypothesis:\n{hypothesis}\n\n"
        f"Extracted keywords: {', '.join(keywords)}"
    )
    raw = llm(_ZOOM_SYSTEM, user_msg, max_tokens=512)
    data = json.loads(raw)
    return {
        "specific": data.get("specific", ""),
        "moderate": data.get("moderate", ""),
        "broad":    data.get("broad", ""),
    }


# ─────────────────────────────────────────────
# PubMed поиск
# ─────────────────────────────────────────────

def _get_fetcher() -> PubMedFetcher:
    if NCBI_API_KEY:
        os.environ["NCBI_API_KEY"] = NCBI_API_KEY
    return PubMedFetcher()


def _query_pubmed(query: str, max_results: int = MAX_RESULTS) -> list[Article]:
    """Выполняет запрос к PubMed, возвращает список Article."""
    fetcher = _get_fetcher()
    print(f"  🔍 Запрос: {query}")

    pmids = fetcher.pmids_for_query(query, retmax=max_results)
    if not pmids:
        print("     → результатов нет")
        return []

    print(f"     → найдено PMID: {len(pmids)}")
    articles: list[Article] = []

    for pmid in pmids:
        try:
            art = fetcher.article_by_pmid(pmid)
            time.sleep(REQUEST_DELAY)
            authors = [str(a) for a in art.authors] if art.authors else []
            articles.append(Article(
                pmid     = str(pmid),
                title    = art.title or "",
                authors  = authors,
                journal  = art.journal or "",
                year     = str(art.year) if art.year else None,
                abstract = art.abstract or "",
                doi      = art.doi,
                url      = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            ))
        except Exception as exc:
            print(f"     ⚠️  PMID {pmid}: {exc}")

    return articles


# ─────────────────────────────────────────────
# Ранжирование (опционально)
# ─────────────────────────────────────────────

_RANK_SYSTEM = """You are a biomedical research assistant.
Given a medical hypothesis and a list of articles (PMID, title, abstract),
rank them by relevance to the hypothesis.

Return ONLY a JSON array, each element:
  {"pmid": "...", "score": <float 0-1>, "reason": "<one sentence>"}

Sorted descending by score. No explanation outside the JSON."""


def _rank_articles(hypothesis: str, articles: list[Article]) -> list[Article]:
    """Ранжирует статьи по релевантности гипотезе, возвращает отсортированный список."""
    if not articles:
        return articles

    articles_text = "\n\n".join(
        f"PMID {a.pmid}:\nTitle: {a.title}\nAbstract: {a.abstract[:400]}"
        for a in articles
    )
    user_msg = f"Hypothesis:\n{hypothesis}\n\nArticles:\n{articles_text}"

    try:
        raw = llm(_RANK_SYSTEM, user_msg, max_tokens=1024)
        ranked = json.loads(raw)
        rank_map = {r["pmid"]: r for r in ranked}
        for art in articles:
            info = rank_map.get(art.pmid, {})
            art.rank_score  = info.get("score")
            art.rank_reason = info.get("reason", "")
        articles.sort(key=lambda a: a.rank_score or 0, reverse=True)
    except Exception as exc:
        print(f"  ⚠️  Ранжирование не удалось: {exc}")

    return articles

# ─────────────────────────────────────────────
# Основной пайплайн
# ─────────────────────────────────────────────

def search_pubmed(
    hypothesis: str,
    max_results: int = MAX_RESULTS,
    rank: bool = True,
) -> SearchResult:
    """
    Полный пайплайн семантического поиска с расслаблением запроса.

    Args:
        hypothesis:  Медицинская гипотеза (любой язык).
        max_results: Максимальное число статей из PubMed.
        rank:        Ранжировать результаты через LLM.

    Returns:
        SearchResult с полями articles, search_path, query_used.
    """
    print("=" * 65)
    print("СЕМАНТИЧЕСКИЙ ПОИСК В PUBMED")
    print("=" * 65)
    print(f"Гипотеза: {hypothesis}\n")

    # ── Шаг 1. Извлечение ключевых слов ─────────────────────────────────
    print("⚙️  [1/4] Извлечение ключевых слов...")
    keywords, original_query = _extract_keywords(hypothesis)
    print(f"  Ключевые слова : {', '.join(keywords)}")
    print(f"  Исходный запрос: {original_query}")

    # ── Шаг 2. Первичный поиск ───────────────────────────────────────────
    print("\n🔎 [2/4] Первичный поиск...")
    articles = _query_pubmed(original_query, max_results)
    if articles:
        if rank:
            print("\n🤖 Ранжирование...")
            articles = _rank_articles(hypothesis, articles)
        return SearchResult(articles, SearchPath.ORIGINAL, original_query)

    # ── Шаг 3. Расслабление: удаление дженерик-слов ──────────────────────
    print("\n🔧 [3/4] Расслабление запроса: удаление дженерик-слов...")
    clean_keywords = _drop_generic_keywords(keywords)
    removed = set(keywords) - set(clean_keywords)
    if removed:
        print(f"  Удалены дженерики: {', '.join(removed)}")
    else:
        print("  Дженерик-слов не найдено, запрос не изменился.")

    relaxed_query = _rebuild_query(clean_keywords)
    print(f"  Расслабленный запрос: {relaxed_query}")

    articles = _query_pubmed(relaxed_query, max_results)
    if articles:
        if rank:
            print("\n🤖 Ранжирование...")
            articles = _rank_articles(hypothesis, articles)
        return SearchResult(articles, SearchPath.RELAXED, relaxed_query)

    # ── Шаг 4. Zoom-уровни от LLM ────────────────────────────────────────
    print("\n🌐 [4/4] Генерация zoom-запросов через LLM...")
    zoom = _generate_zoom_queries(hypothesis, keywords)

    zoom_levels = [
        (SearchPath.ZOOM_SPECIFIC, zoom["specific"]),
        (SearchPath.ZOOM_MODERATE, zoom["moderate"]),
        (SearchPath.ZOOM_BROAD,    zoom["broad"]),
    ]

    for path, query in zoom_levels:
        if not query:
            continue
        print(f"\n  Уровень [{path.value}]:")
        articles = _query_pubmed(query, max_results)
        if articles:
            if rank:
                print("\n🤖 Ранжирование...")
                articles = _rank_articles(hypothesis, articles)
            return SearchResult(articles, path, query)

    # ── Все попытки исчерпаны ─────────────────────────────────────────────
    print("\n❌ Статьи не найдены ни на одном уровне расслабления.")
    return SearchResult([], SearchPath.NOT_FOUND, "")