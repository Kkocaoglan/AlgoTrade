"""
sentiment.py -- BERTurk + keyword-fallback sentiment scoring for BIST stocks
-----------------------------------------------------------------------------
Primary  : savasy/bert-base-turkish-sentiment-cased  (~88% accuracy)
Fallback : keyword-based scoring (~62% accuracy)

Public API (unchanged — backward compatible with loop_trader.py):
  score_news(title, symbol) -> (sentiment: str, score: int)
    sentiment : "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    score     : int in roughly [-3, +3]
                  >= +1 → POSITIVE, <= -1 → NEGATIVE, 0 → NEUTRAL
                  Forced-exit threshold: score <= -2

New API:
  SentimentAnalyzer         — class with .score(), .batch_score(), .classify()
  BERT_AVAILABLE            — module-level flag
  read_recent_news(minutes) — unchanged

Test: py -3.12 sentiment.py
"""

import sys, json
from datetime import datetime, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from zoneinfo import ZoneInfo
    TZ_IST = ZoneInfo("Europe/Istanbul")
except Exception:
    TZ_IST = None

BASE_DIR = Path(__file__).parent
NEWS_LOG = BASE_DIR / "results" / "news_log.jsonl"

# =============================================================================
# SECTOR MAP  (matches loop_trader.py SECTOR_MAP exactly)
# =============================================================================

SECTOR_MAP = {
    "YKBNK": "banka",     "AKBNK": "banka",
    "ISCTR": "banka",     "GARAN": "banka",
    "TUPRS": "enerji",    "PETKM": "enerji",
    "TAVHL": "havacilik", "FROTO": "otomotiv",
    "TCELL": "telekom",   "ASELS": "savunma",
    "BIMAS": "perakende", "MGROS": "perakende",
    "ENKAI": "insaat",    "EKGYO": "insaat",
}

# =============================================================================
# KEYWORD TABLES  (fallback; used when BERT unavailable)
# =============================================================================

POSITIVE_KEYWORDS: dict[str, list[str]] = {
    "banka":     ["kar artisi", "temettu", "temett\u00fc", "buyume", "b\u00fcy\u00fcme",
                  "kredi artisi", "faiz indirimi"],
    "enerji":    ["petrol yukseldi", "ham petrol", "uretim artisi"],
    "savunma":   ["ihracat anlasmasi", "ihracat anla\u015fmas\u0131",
                  "nato", "savunma butcesi", "savunma b\u00fct\u00e7esi", "siparis"],
    "perakende": ["ciro artisi", "enflasyon dustu", "tuketici guveni",
                  "t\u00fcketici g\u00fcveni"],
    "insaat":    ["konut satisi artti", "faiz indirimi", "toki", "tok\u0130"],
    "havacilik": ["yolcu artisi", "turizm rekoru", "kapasite artisi"],
    "otomotiv":  ["ihracat artisi", "uretim artisi", "siparis"],
    "telekom":   ["abone artisi", "gelir artisi", "buyume", "b\u00fcy\u00fcme"],
    "genel":     ["guclu buyume", "g\u00fc\u00e7l\u00fc b\u00fcy\u00fcme",
                  "ihracat rekoru", "cari fazla"],
}

NEGATIVE_KEYWORDS: dict[str, list[str]] = {
    "banka":     ["zarar", "takipteki kredi", "bddk cezasi", "faiz artisi"],
    "enerji":    ["petrol dustu", "vergi artisi", "ithalat kisitlamasi"],
    "savunma":   ["savunma kesintisi", "ihracat yasagi", "ateskes"],
    "perakende": ["enflasyon yukseldi", "tuketim dustu", "t\u00fcketim d\u00fc\u015ft\u00fc",
                  "ciro dustu", "ciro d\u00fc\u015ft\u00fc"],
    "insaat":    ["deprem", "faiz artisi", "konut satisi dustu"],
    "havacilik": ["grev", "yakit zammi", "yak\u0131t zamm\u0131",
                  "yolcu dustu", "yolcu d\u00fc\u015ft\u00fc"],
    "otomotiv":  ["grev", "ihracat dustu", "uretim dustu"],
    "telekom":   ["para cezasi", "lisans iptal", "abone kaybi"],
    "genel":     ["resesyon", "kriz", "doviz krizi", "d\u00f6viz krizi",
                  "devaluasyon", "deval\u00fcasyon"],
}

# =============================================================================
# HELPERS
# =============================================================================

def _normalize(s: str) -> str:
    """Lowercase + Turkish->ASCII substitution for robust keyword matching."""
    return (
        s.lower()
         .replace("\u00fc", "u").replace("\u00f6", "o")
         .replace("\u015f", "s").replace("\u00e7", "c")
         .replace("\u011f", "g").replace("\u0131", "i")
         .replace("\u0130", "i")
         .replace("\u015e", "s").replace("\u00c7", "c")
         .replace("\u00dc", "u").replace("\u00d6", "o")
         .replace("\u011e", "g")
    )


def _now_naive() -> datetime:
    if TZ_IST:
        return datetime.now(TZ_IST).replace(tzinfo=None)
    return datetime.now()


# =============================================================================
# SENTIMENT ANALYZER CLASS
# =============================================================================

class SentimentAnalyzer:
    """
    Two-tier sentiment scorer:
      Tier 1: savasy/bert-base-turkish-sentiment-cased  (BERT, ~88% accuracy)
      Tier 2: keyword-based fallback (~62% accuracy)

    .score(text)           -> float  [-1.0 very negative, +1.0 very positive]
    .batch_score(texts)    -> list[float]
    .classify(score)       -> "POSITIVE" | "NEUTRAL" | "NEGATIVE"
    """

    # Classification thresholds for float score
    POS_THRESHOLD = 0.30
    NEG_THRESHOLD = -0.30

    def __init__(self):
        self.bert_available = False
        self.pipe = None
        self._bert_initialized = False   # lazy: load on first score() call, not at import

    def _load_bert(self):
        import platform, os
        if platform.system() == "Darwin":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            from transformers import pipeline as hf_pipeline
            self.pipe = hf_pipeline(
                "text-classification",
                model="savasy/bert-base-turkish-sentiment-cased",
                device=-1,          # CPU — stable on machines with small VRAM
            )
            self.bert_available = True
            print("[BERT] savasy/bert-base-turkish-sentiment-cased loaded")
        except Exception as e:
            self.bert_available = False
            print("[BERT] Disabled — using keyword fallback")

    def _ensure_bert(self):
        """Load BERT on first call; no-op after that."""
        if not self._bert_initialized:
            self._load_bert()
            self._bert_initialized = True

    # ------------------------------------------------------------------
    def score(self, text: str) -> float:
        """
        Score a single headline.
        Returns float in [-1.0, +1.0].
        """
        self._ensure_bert()
        if self.bert_available:
            return self._bert_score(text)
        return self._keyword_score(text)

    def _bert_score(self, text: str) -> float:
        """Run BERT pipeline; truncate at 512 chars before tokenization."""
        try:
            result = self.pipe(text[:512])[0]
            label  = result["label"].lower()   # "positive" or "negative"
            conf   = float(result["score"])    # 0.0 → 1.0
            return conf if label == "positive" else -conf
        except Exception:
            # Silently fall back to keyword on inference error
            return self._keyword_score(text)

    def _keyword_score(self, text: str, symbol: str | None = None) -> float:
        """
        Keyword-based scoring, returns float in [-1.0, +1.0].
        Internally computes the same int count as the legacy score_news(),
        then normalises to float (clamp ±3 → ±1.0).
        """
        sector   = SECTOR_MAP.get(symbol or "", "genel") if symbol else "genel"
        norm     = _normalize(text)
        pos_kws  = POSITIVE_KEYWORDS.get(sector, []) + POSITIVE_KEYWORDS.get("genel", [])
        neg_kws  = NEGATIVE_KEYWORDS.get(sector, []) + NEGATIVE_KEYWORDS.get("genel", [])
        pos_hits = sum(1 for kw in pos_kws if _normalize(kw) in norm)
        neg_hits = sum(1 for kw in neg_kws if _normalize(kw) in norm)
        raw      = pos_hits - neg_hits
        # Normalise: clamp to [-3, 3] then scale to [-1, 1]
        return max(-1.0, min(1.0, raw / 3.0))

    # ------------------------------------------------------------------
    def batch_score(self, texts: list[str]) -> list[float]:
        """Score multiple headlines at once (batched BERT is faster than one-by-one)."""
        if not texts:
            return []
        self._ensure_bert()
        if self.bert_available:
            try:
                results = self.pipe([t[:512] for t in texts], batch_size=8)
                scores  = []
                for r in results:
                    label = r["label"].lower()
                    conf  = float(r["score"])
                    scores.append(conf if label == "positive" else -conf)
                return scores
            except Exception:
                pass   # fall through to keyword
        return [self._keyword_score(t) for t in texts]

    # ------------------------------------------------------------------
    def classify(self, score: float) -> str:
        """Map a float score to a sentiment label."""
        if score > self.POS_THRESHOLD:
            return "POSITIVE"
        if score < self.NEG_THRESHOLD:
            return "NEGATIVE"
        return "NEUTRAL"


# =============================================================================
# MODULE-LEVEL INSTANCE + FLAG
# =============================================================================

analyzer       = SentimentAnalyzer()   # module-level singleton (reuse; don't re-instantiate)
_analyzer      = analyzer              # internal alias kept for backward compat
BERT_AVAILABLE = analyzer.bert_available   # importable flag for news_filter.py


# =============================================================================
# PUBLIC API  (backward-compatible — loop_trader.py uses these)
# =============================================================================

def score_news(title: str, symbol: str | None) -> tuple[str, int]:
    """
    Score a news headline.  Hybrid: keyword-primary, BERT-extended.

    Strategy:
      1. Always compute keyword score (sector-aware, domain-specific).
      2. When keyword score != 0 → use it directly (reliable for known signals).
      3. When keyword score == 0 AND BERT available → use BERT to cover headlines
         the sparse keyword list misses.  BERT score is clamped to [-2, +2] to
         prevent false forced-exits caused by the general-Turkish model's known
         domain mismatch on financial text.

    This design means:
      - Domain vocabulary (profits, dividends, BDDK fines, …) triggers keywords.
      - Macro / monetary policy text (no keyword match) is covered by BERT.
      - BERT never overrides a keyword signal, preventing false position exits
        from the model's high false-negative rate on financial headlines.

    Returns
    -------
    (sentiment, score)
      sentiment : "POSITIVE" | "NEGATIVE" | "NEUTRAL"
      score     : int in roughly [-3, +3]
                    >= +1 → POSITIVE, <= -1 → NEGATIVE, 0 → NEUTRAL
                    Forced-exit threshold used by loop_trader: score <= -2
    """
    # -- Step 1: keyword score (always computed, sector-aware) ---------------
    sector   = SECTOR_MAP.get(symbol or "", "genel") if symbol else "genel"
    norm     = _normalize(title)
    pos_kws  = POSITIVE_KEYWORDS.get(sector, []) + POSITIVE_KEYWORDS.get("genel", [])
    neg_kws  = NEGATIVE_KEYWORDS.get(sector, []) + NEGATIVE_KEYWORDS.get("genel", [])
    pos_hits = [kw for kw in pos_kws if _normalize(kw) in norm]
    neg_hits = [kw for kw in neg_kws if _normalize(kw) in norm]
    kw_int   = len(pos_hits) - len(neg_hits)

    # -- Step 2: decide which score to use ------------------------------------
    analyzer._ensure_bert()   # lazy-load BERT if not yet done
    if kw_int != 0 or not analyzer.bert_available:
        # Keywords have a signal → trust them (domain-specific)
        int_score = kw_int
    else:
        # No keyword match → let BERT extend coverage
        bert_float = analyzer._bert_score(title)
        # Clamp to [-2, +2]: score -3 triggers forced position exit,
        # which must not fire from a general-purpose model's domain mismatch
        int_score = max(-2, min(2, int(round(bert_float * 2))))

    if int_score >= 1:
        sentiment = "POSITIVE"
    elif int_score <= -1:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"

    return sentiment, int_score


def read_recent_news(minutes: int = 30) -> list[dict]:
    """
    Read all news items from results/news_log.jsonl newer than `minutes` minutes.
    Returns list sorted oldest-first.  Returns [] if file missing or empty.
    """
    if not NEWS_LOG.exists():
        return []

    cutoff = _now_naive() - timedelta(minutes=minutes)
    items: list[dict] = []

    try:
        with open(NEWS_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    ts   = datetime.fromisoformat(item.get("timestamp", ""))
                    if ts >= cutoff:
                        items.append(item)
                except (json.JSONDecodeError, ValueError):
                    continue
    except Exception:
        return []

    return items


# =============================================================================
# TEST  (py -3.12 sentiment.py)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("sentiment.py -- test modu")
    mode = "BERT (savasy/bert-base-turkish-sentiment-cased)" if BERT_AVAILABLE else "KEYWORD FALLBACK"
    print(f"Mod: {mode}")
    print("=" * 60)

    test_cases = [
        ("AKBNK", "Akbank rekor kar acikladi, temettu artisi bekleniyor"),
        ("AKBNK", "BDDK cezasi sonrasi Akbank hisseleri geriledi"),
        ("AKBNK", "Akbank genel kurulda temett\u00fc onayi verdi"),
        ("TUPRS", "Brent petrol yukseldi, ham petrol uretim artisi"),
        ("TUPRS", "Petrol dustu, rafineriler zarar aciklamayi bekliyor"),
        ("ASELS", "NATO savunma butcesi artisi siparis rekoru"),
        ("ASELS", "Savunma kesintisi ihracat yasagina yol acabilir"),
        ("TAVHL", "Yolcu artisi turizm rekoru ile havacilik canlanadi"),
        ("TAVHL", "Grev tehdidi kapilari kapatiyor, yolcu dustu"),
        ("EKGYO", "Faiz indirimi konut satisi artti, TOKI destegi"),
        ("EKGYO", "Deprem sonrasi faiz artisi konut satisi dustu"),
        (None,    "Doviz krizi resesyon enflasyon kriz haberleri"),
        (None,    "Cari fazla ihracat rekoru guclu buyume"),
        ("GARAN", "Garanti neutralne bir aciklama yapti"),
    ]

    print(f"\n{'Sembol':<8} {'Skor':>5}  {'Duygu':<10}  Baslik")
    print("-" * 60)
    for sym, title in test_cases:
        sent, score = score_news(title, sym)
        sym_str   = sym if sym else "MAKRO"
        sign      = "+" if score > 0 else ""
        title_out = title[:45].encode("ascii", "replace").decode("ascii")
        print(f"{sym_str:<8} {sign}{score:>4}   {sent:<10}  {title_out}")

    # ---- SentimentAnalyzer direct tests ----
    print()
    print("--- SentimentAnalyzer.score() (float) ---")
    sa = _analyzer
    float_tests = [
        "AKBNK guclu kar artisi beklentileri asti",
        "TCMB faiz karari piyasalarda sert satis yaratti",
        "Merkez Bankasi faizi sabit tuttu",
    ]
    for t in float_tests:
        fs   = sa.score(t)
        cls  = sa.classify(fs)
        tshort = t[:50].encode("ascii", "replace").decode("ascii")
        print(f"  {fs:+.3f}  {cls:<10}  {tshort}")

    # ---- Recent news file ----
    print()
    print("--- Son 24 saatteki haberler (news_log.jsonl) ---")
    recent = read_recent_news(minutes=60 * 24)
    if not recent:
        print("  (haber yok veya dosya bos)")
    else:
        for item in recent[:8]:
            sym_str   = (item.get("symbol") or "MAKRO")[:6]
            title_out = (item.get("title", ""))[:50].encode("ascii", "replace").decode("ascii")
            ts        = item.get("timestamp", "")[:16]
            sent, sc  = score_news(item.get("title", ""), item.get("symbol"))
            sign      = "+" if sc > 0 else ""
            print(f"  [{ts}] [{sym_str:<6}] {sign}{sc:>2}  {sent:<8}  {title_out}")

    print()
    print("Test tamamlandi.")
