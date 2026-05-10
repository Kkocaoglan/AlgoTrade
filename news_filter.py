"""
news_filter.py -- BIST Haber & KAP Bildiri Filtresi v2
-------------------------------------------------------
Sources (priority order):
  1. KAP JSON API  https://www.kap.org.tr/tr/api/disclosures   (every 60s)
     fallback: HTML scrape https://www.kap.org.tr/tr/bildirim-sorgu
  2. Investing.com TR RSS  (every 120s)
  3. TCMB          https://www.tcmb.gov.tr/...                 (every 300s)

Output : results/news_log.jsonl  (one JSON per line)
Usage  : py -3.12 news_filter.py
         py -3.12 news_filter.py --once   (single pass, first 5 per source)
"""

import sys, time, json, gzip as _gzip
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from sentiment import analyzer as _sa, BERT_AVAILABLE

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from zoneinfo import ZoneInfo
    TZ_IST = ZoneInfo("Europe/Istanbul")
except Exception:
    TZ_IST = None

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR       = Path(__file__).parent
RESULTS        = BASE_DIR / "results"
RESULTS.mkdir(exist_ok=True)
NEWS_LOG_JSONL = RESULTS / "news_log.jsonl"

ONCE_MODE = "--once" in sys.argv

SYMBOLS = {
    "YKBNK", "AKBNK", "ISCTR", "GARAN",
    "TUPRS", "PETKM",
    "TAVHL", "FROTO",
    "TCELL", "ASELS",
    "BIMAS", "MGROS",
    "ENKAI", "EKGYO",
}

INTERVAL_KAP       = 60
INTERVAL_INVESTING = 120
INTERVAL_TCMB      = 300

KAP_API_URL    = "https://www.kap.org.tr/tr/api/disclosures"
KAP_SCRAPE_URL = "https://www.kap.org.tr/tr/bildirim-sorgu"

INVESTING_URLS = [
    "https://tr.investing.com/rss/news.rss",
    "https://tr.investing.com/rss/market_overview_Bonds.rss",
]

TCMB_URL = "https://www.tcmb.gov.tr/wps/wcm/connect/tr/tcmb+tr/"

# Investing.com: only log items containing at least one of these keywords
INVESTING_FILTER_KW = [
    "tcmb", "faiz", "merkez bankasi", "bddk", "enflasyon",
    "petrol", "dolar", "nato", "savunma", "turizm",
]

# Keyword -> category mapping (checked in order)
MACRO_KW_CATS = [
    (["tcmb", "merkez bankasi", "merkez bank", "bddk", "faiz", "enflasyon"], "faiz"),
    (["dolar", "euro", "kur ", "doviz", "doviz"],                             "kur"),
    (["petrol", "brent", "dogalgaz", "opec"],                                 "petrol"),
    (["nato", "savunma", "askeri", "silah", "harp"],                          "savunma"),
    (["turizm", "otel", "tatil", "sezon"],                                    "turizm"),
]

# KAP company code / title keyword -> BIST ticker
KAP_COMPANY_MAP = {
    "YKBNK": "YKBNK",  "AKBNK": "AKBNK",  "ISCTR": "ISCTR",  "GARAN": "GARAN",
    "TUPRS": "TUPRS",  "PETKM": "PETKM",  "TAVHL": "TAVHL",  "FROTO": "FROTO",
    "TCELL": "TCELL",  "ASELS": "ASELS",  "BIMAS": "BIMAS",  "MGROS": "MGROS",
    "ENKAI": "ENKAI",  "EKGYO": "EKGYO",
    # Common name variants seen in KAP titles
    "YAPI KREDI":   "YKBNK",
    "YAPIKREDI":    "YKBNK",
    "IS BANKASI":   "ISCTR",
    "ISBANK":       "ISCTR",
    "GARANTI":      "GARAN",
    "GARANTI BBVA": "GARAN",
    "TUPRAS":       "TUPRS",
    "PETKIM":       "PETKM",
    "TAV HAVALIMANLARI": "TAVHL",
    "FORD OTOSAN":  "FROTO",
    "TURKCELL":     "TCELL",
    "ASELSAN":      "ASELS",
    "BIM MAGAZALAR": "BIMAS",
    "MIGROS":       "MGROS",
    "ENKA INSAAT":  "ENKAI",
    "ENKA":         "ENKAI",
    "EMLAK KONUT":  "EKGYO",
}

# =============================================================================
# HELPERS
# =============================================================================

_seen: set = set()
_last_kap:       float = 0.0
_last_investing: float = 0.0
_last_tcmb:      float = 0.0


def _now():
    return datetime.now(TZ_IST) if TZ_IST else datetime.now()

def _ts() -> str:
    return _now().strftime("%Y-%m-%dT%H:%M:%S")

def _hms() -> str:
    return _now().strftime("%H:%M:%S")

def _req(url: str, accept: str = "application/json, */*") -> urllib.request.Request:
    return urllib.request.Request(url, headers={
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36",
        "Accept":          accept,
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Connection":      "keep-alive",
    })

def _categorize(text: str) -> str:
    tl = text.lower()
    for kws, cat in MACRO_KW_CATS:
        if any(kw in tl for kw in kws):
            return cat
    return "genel"

def _sym_from_code(code: str):
    if not code:
        return None
    c = code.upper().strip()
    return KAP_COMPANY_MAP.get(c)

def _sym_from_title(title: str):
    tu = title.upper()
    # Direct ticker match first
    for sym in SYMBOLS:
        # word-boundary-ish: check it's not buried in a longer word
        idx = tu.find(sym)
        if idx >= 0:
            before = tu[idx - 1] if idx > 0 else " "
            after  = tu[idx + len(sym)] if idx + len(sym) < len(tu) else " "
            if not before.isalpha() and not after.isalpha():
                return sym
    # Name-based match
    for alias, sym in KAP_COMPANY_MAP.items():
        if len(alias) > 4 and alias in tu:
            return sym
    return None

def _decompress(raw: bytes) -> bytes:
    """Decompress gzip bytes if needed."""
    if raw[:2] == b'\x1f\x8b':
        try:
            return _gzip.decompress(raw)
        except Exception:
            pass
    return raw

def _dedup_key(item: dict) -> str:
    url   = item.get("url", "")
    title = item.get("title", "")[:100]
    return url if url else title

def _is_new(item: dict) -> bool:
    key = _dedup_key(item)
    if not key or key in _seen:
        return False
    _seen.add(key)
    return True

def _log_item(item: dict):
    with open(NEWS_LOG_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def _print_item(item: dict):
    sym_str = item.get("symbol") or "MAKRO"
    title   = (item.get("title") or "")[:80]
    cat     = item.get("category", "?")
    src     = item.get("source", "?")
    # ASCII-safe: replace non-ASCII
    title_ascii = title.encode("ascii", "replace").decode("ascii")
    print(f"    [{src:<14}] [{sym_str:<6}] [{cat:<15}] {title_ascii}")


# =============================================================================
# SOURCE 1 — KAP JSON API
# =============================================================================

def _parse_kap_record(rec: dict):
    """
    Extract fields from a KAP API record dict.
    Returns a log item dict or None if not in our universe.
    """
    # member_company_code is the primary identifier per spec
    code = (
        rec.get("member_company_code") or
        rec.get("memberCompanyCode")   or
        rec.get("companyCode")         or
        rec.get("stockCode")           or
        ""
    )
    sym = _sym_from_code(str(code))

    title = str(
        rec.get("title")            or
        rec.get("disclosureTitle")  or
        rec.get("baslik")           or
        ""
    ).strip()

    # Try title-based symbol detection if code lookup failed
    if not sym and title:
        sym = _sym_from_title(title)

    if not sym:
        return None  # not our universe

    dtype = str(
        rec.get("disclosure_type") or
        rec.get("disclosureType")  or
        rec.get("type")            or
        ""
    ).strip()

    pub = str(
        rec.get("published_at")  or
        rec.get("publishDate")   or
        rec.get("publishedAt")   or
        rec.get("tarih")         or
        ""
    ).strip()

    disc_id = rec.get("id") or rec.get("disclosureId") or ""
    url = str(rec.get("url") or rec.get("link") or "")
    if not url and disc_id:
        url = f"https://www.kap.org.tr/tr/bildirim/{disc_id}"

    return {
        "timestamp":       _ts(),
        "source":          "KAP",
        "symbol":          sym,
        "title":           title,
        "category":        "KAP_bildirimi",
        "url":             url,
        "disclosure_type": dtype,
        "published_at":    pub,
    }


def _fetch_kap_api() -> list[dict]:
    try:
        with urllib.request.urlopen(_req(KAP_API_URL), timeout=8) as resp:
            raw = _decompress(resp.read()).decode("utf-8", errors="replace")
        data = json.loads(raw)

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            for key in ("data", "disclosures", "basic", "items", "results"):
                if key in data and isinstance(data[key], list):
                    records = data[key]
                    break
            else:
                records = list(data.values())[0] if data else []
                if not isinstance(records, list):
                    records = []
        else:
            records = []

        items = []
        for rec in records:
            if isinstance(rec, dict):
                parsed = _parse_kap_record(rec)
                if parsed:
                    items.append(parsed)
        return items

    except urllib.error.HTTPError as e:
        if e.code == 666:
            print(f"  [KAP API] Bot-engeli (HTTP 666) -- Google News fallback")
        else:
            print(f"  [KAP API] HTTP {e.code}: {e.reason}")
        return []
    except urllib.error.URLError as e:
        print(f"  [KAP API] Baglanti hatasi: {e.reason}")
        return []
    except (json.JSONDecodeError, TimeoutError, OSError):
        print(f"  [KAP API] Zaman asimi / parse hatasi")
        return []
    except Exception as e:
        print(f"  [KAP API] Hata: {e}")
        return []


def _fetch_kap_scrape() -> list[dict]:
    if not HAS_BS4:
        print("  [KAP Scrape] beautifulsoup4 yok -- pip install beautifulsoup4")
        return []
    try:
        with urllib.request.urlopen(
            _req(KAP_SCRAPE_URL, accept="text/html, */*"), timeout=20
        ) as resp:
            raw = _decompress(resp.read()).decode("utf-8", errors="replace")

        soup  = BeautifulSoup(raw, "html.parser")
        items = []

        # Try successive selectors until one yields rows
        rows = (
            soup.select("div.w-clearfix.w-inline-block.comp-row") or
            soup.select("div.comp-row")                           or
            soup.select("div[class*='disclosure']")               or
            soup.select("tr")[1:]  # table fallback, skip header
        )

        for row in rows[:40]:
            text = row.get_text(separator=" ", strip=True)
            if len(text) < 8:
                continue
            sym = _sym_from_title(text)
            if not sym:
                continue
            # Try to find a link
            a_tag = row.find("a", href=True)
            url   = ""
            if a_tag:
                href = a_tag["href"]
                url  = href if href.startswith("http") else f"https://www.kap.org.tr{href}"
            items.append({
                "timestamp": _ts(),
                "source":    "KAP",
                "symbol":    sym,
                "title":     text[:200],
                "category":  "KAP_bildirimi",
                "url":       url,
            })

        return items

    except urllib.error.HTTPError as e:
        print(f"  [KAP Scrape] HTTP {e.code}: {e.reason}")
        return []
    except Exception as e:
        print(f"  [KAP Scrape] Hata: {e}")
        return []


def _fetch_kap_gnews() -> list[dict]:
    """
    Google News RSS fallback — queries 'SYMBOL KAP' for each of our 14 stocks.
    Returns recent KAP-related news items for stocks in our universe.
    """
    all_items = []
    sym_list  = sorted(SYMBOLS)
    batch_size = 4
    for i in range(0, len(sym_list), batch_size):
        batch = sym_list[i:i + batch_size]
        query = "+".join(batch) + "+KAP"
        url   = (f"https://news.google.com/rss/search"
                 f"?q={query}&hl=tr&gl=TR&ceid=TR:tr")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                raw = _decompress(resp.read())
            root = ET.fromstring(raw)
            for item in root.iter("item"):
                title = (item.findtext("title") or "").strip()
                link  = (item.findtext("link")  or "").strip()
                if not title:
                    continue
                sym = _sym_from_title(title)
                if not sym:
                    continue
                all_items.append({
                    "timestamp": _ts(),
                    "source":    "KAP-GNews",
                    "symbol":    sym,
                    "title":     title,
                    "category":  "KAP_bildirimi",
                    "url":       link,
                })
        except Exception as e:
            print(f"  [KAP-GNews] {e}")
        time.sleep(0.3)

    if all_items:
        print(f"  [KAP-GNews] {len(all_items)} ilgili haber")
    return all_items


def fetch_kap() -> list[dict]:
    """Try KAP API → HTML scrape → Google News (in order)."""
    items = _fetch_kap_api()
    if items:
        print(f"  [KAP API] {len(items)} ilgili bildiri")
        return items

    print(f"  [KAP API] Sonuc yok -- HTML scrape deneniyor...")
    items = _fetch_kap_scrape()
    if items:
        print(f"  [KAP Scrape] {len(items)} ilgili bildiri")
        return items

    print(f"  [KAP Scrape] Sonuc yok -- Google News fallback...")
    items = _fetch_kap_gnews()
    if not items:
        print(f"  [KAP] Tum kaynaklar basarisiz")
    return items


# =============================================================================
# SOURCE 2 — Investing.com TR RSS
# =============================================================================

def _parse_rss(url: str, source_name: str) -> list[dict]:
    try:
        with urllib.request.urlopen(
            _req(url, accept="application/rss+xml, application/xml, text/xml, */*"),
            timeout=15,
        ) as resp:
            raw = _decompress(resp.read())

        root  = ET.fromstring(raw)
        items = []
        for item in root.iter("item"):
            title = (item.findtext("title")       or "").strip()
            link  = (item.findtext("link")        or "").strip()
            desc  = (item.findtext("description") or "").strip()
            combined = (title + " " + desc).lower()
            if not any(kw in combined for kw in INVESTING_FILTER_KW):
                continue
            items.append({
                "timestamp": _ts(),
                "source":    source_name,
                "symbol":    None,
                "title":     title,
                "category":  _categorize(combined),
                "url":       link,
            })
        return items

    except urllib.error.HTTPError as e:
        print(f"  [{source_name}] HTTP {e.code}: {e.reason}")
        return []
    except ET.ParseError as e:
        print(f"  [{source_name}] XML parse hatasi: {e}")
        return []
    except Exception as e:
        print(f"  [{source_name}] Hata: {e}")
        return []


def fetch_investing() -> list[dict]:
    all_items = []
    for url in INVESTING_URLS:
        items = _parse_rss(url, "Investing-TR")
        all_items.extend(items)
    if all_items:
        print(f"  [Investing TR] {len(all_items)} ilgili haber")
    else:
        print(f"  [Investing TR] Ilgili haber bulunamadi (filtre veya erisim)")
    return all_items


# =============================================================================
# SOURCE 3 — TCMB
# =============================================================================

def fetch_tcmb() -> list[dict]:
    # TCMB RSS/HTML — try RSS parse first, then BS4 scrape
    items = []
    try:
        with urllib.request.urlopen(
            _req(TCMB_URL, accept="application/rss+xml, text/html, */*"),
            timeout=20,
        ) as resp:
            raw = _decompress(resp.read())

        # Attempt XML/RSS parse
        try:
            root = ET.fromstring(raw)
            for el in root.iter("item"):
                title = (el.findtext("title") or "").strip()
                link  = (el.findtext("link")  or TCMB_URL).strip()
                if title:
                    items.append({
                        "timestamp": _ts(),
                        "source":    "TCMB",
                        "symbol":    None,
                        "title":     title,
                        "category":  "faiz",
                        "url":       link,
                    })
            if items:
                print(f"  [TCMB] {len(items)} haber (RSS)")
                return items
        except ET.ParseError:
            pass

        # Fall back to HTML scrape
        if HAS_BS4:
            import re as _re
            date_pat = _re.compile(r"\d{2}\.\d{2}\.\d{4}|\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}")
            soup     = BeautifulSoup(raw, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                text = a_tag.get_text(strip=True)
                href = a_tag["href"]
                # Skip short nav items and fragment-only links
                if len(text) < 30 or href == "#" or href.startswith("#"):
                    continue
                # Prefer items whose surrounding context contains a date
                parent_text = ""
                p = a_tag.find_parent()
                if p:
                    parent_text = p.get_text(separator=" ", strip=True)
                url = href if href.startswith("http") else f"https://www.tcmb.gov.tr{href}"
                items.append({
                    "timestamp": _ts(),
                    "source":    "TCMB",
                    "symbol":    None,
                    "title":     text[:200],
                    "category":  "faiz",
                    "url":       url,
                    "_has_date": bool(date_pat.search(parent_text)),
                })
            # Prefer dated items, take up to 10
            items.sort(key=lambda x: -x.pop("_has_date", 0))
            items = items[:10]
            if items:
                print(f"  [TCMB] {len(items)} haber (HTML scrape)")
                return items

        print(f"  [TCMB] Icerik alinamadi (HTML + RSS basarisiz)")
        return []

    except urllib.error.HTTPError as e:
        print(f"  [TCMB] HTTP {e.code}: {e.reason}")
        return []
    except urllib.error.URLError as e:
        print(f"  [TCMB] Baglanti hatasi: {e.reason}")
        return []
    except Exception as e:
        print(f"  [TCMB] Hata: {e}")
        return []


# =============================================================================
# PROCESS + LOG
# =============================================================================

_BLOCK_SCORE = -0.35   # float score below this triggers entry-block flag in log


def _score_and_enrich(item: dict) -> dict:
    """
    Run sentiment scoring on item['title'], attach results to item dict.
    Uses BERTurk when available, keyword fallback otherwise.
    Mutates item in-place; returns it.
    """
    title  = item.get("title", "")
    symbol = item.get("symbol")

    float_score = _sa.score(title)
    label       = _sa.classify(float_score)
    item["sentiment_score"] = round(float_score, 4)
    item["sentiment_label"] = label
    return item


def _print_item_enriched(item: dict):
    """Print a news item with its sentiment score (replaces plain _print_item)."""
    sym_str = item.get("symbol") or "MAKRO"
    title   = (item.get("title") or "")[:70]
    cat     = item.get("category", "?")
    src     = item.get("source", "?")
    score   = item.get("sentiment_score")
    label   = item.get("sentiment_label", "?")

    title_ascii = title.encode("ascii", "replace").decode("ascii")

    if score is not None:
        flag = ""
        if score <= _BLOCK_SCORE:
            # Determine what action this would trigger
            if score <= -0.67:   # maps to int -2 → position exit threshold
                flag = " -> position exit triggered"
            else:
                flag = " -> entry block"
        score_str = f"score={score:+.2f} | {label}{flag}"
        print(f"    [{src:<14}] [{sym_str:<6}] {score_str:<40}  {title_ascii}")
    else:
        print(f"    [{src:<14}] [{sym_str:<6}] [{cat:<15}] {title_ascii}")


def process_items(items: list[dict], limit: int = 0) -> int:
    """
    Deduplicate, score sentiment, write new items to JSONL, print summary.
    limit>0: only print first `limit` items (--once mode).
    Returns count of new items written.
    """
    new_items = [it for it in items if _is_new(it)]
    for it in new_items:
        _score_and_enrich(it)
        _log_item(it)
    shown = new_items[:limit] if limit > 0 else new_items
    for it in shown:
        _print_item_enriched(it)
    if limit > 0 and len(new_items) > limit:
        print(f"    ... ve {len(new_items) - limit} daha")
    return len(new_items)


# =============================================================================
# SINGLE PASS (--once)
# =============================================================================

def run_once():
    global _last_kap, _last_investing, _last_tcmb
    t = time.time()
    print(f"\n[{_hms()}] --- KAP (JSON API + scrape fallback) ---")
    kap_items = fetch_kap()
    n_kap     = process_items(kap_items, limit=5)
    print(f"  => {n_kap} yeni KAP bildirisi kaydedildi")
    _last_kap = t

    print(f"\n[{_hms()}] --- Investing.com TR RSS ---")
    inv_items = fetch_investing()
    n_inv     = process_items(inv_items, limit=5)
    print(f"  => {n_inv} yeni Investing haberi kaydedildi")
    _last_investing = t

    print(f"\n[{_hms()}] --- TCMB ---")
    tcmb_items = fetch_tcmb()
    n_tcmb     = process_items(tcmb_items, limit=5)
    print(f"  => {n_tcmb} yeni TCMB haberi kaydedildi")
    _last_tcmb = t

    total = n_kap + n_inv + n_tcmb
    print(f"\n[{_hms()}] Toplam {total} yeni item -> {NEWS_LOG_JSONL}")


# =============================================================================
# CONTINUOUS LOOP
# =============================================================================

def run_loop():
    global _last_kap, _last_investing, _last_tcmb
    tick = 0
    print(f"[{_hms()}] Loop basliyor (Ctrl+C ile dur)...")

    while True:
        now  = time.time()
        tick += 1
        fired = False

        if now - _last_kap >= INTERVAL_KAP:
            print(f"\n[{_hms()}] Tick #{tick} -- KAP ({INTERVAL_KAP}s)")
            items = fetch_kap()
            n = process_items(items)
            if n:
                print(f"  => {n} yeni KAP bildirisi")
            _last_kap = now
            fired = True

        if now - _last_investing >= INTERVAL_INVESTING:
            print(f"\n[{_hms()}] Tick #{tick} -- Investing ({INTERVAL_INVESTING}s)")
            items = fetch_investing()
            n = process_items(items)
            if n:
                print(f"  => {n} yeni Investing haberi")
            _last_investing = now
            fired = True

        if now - _last_tcmb >= INTERVAL_TCMB:
            print(f"\n[{_hms()}] Tick #{tick} -- TCMB ({INTERVAL_TCMB}s)")
            items = fetch_tcmb()
            n = process_items(items)
            if n:
                print(f"  => {n} yeni TCMB haberi")
            _last_tcmb = now
            fired = True

        if not fired:
            next_kap  = max(0, INTERVAL_KAP       - (now - _last_kap))
            next_inv  = max(0, INTERVAL_INVESTING  - (now - _last_investing))
            next_tcmb = max(0, INTERVAL_TCMB       - (now - _last_tcmb))
            next_wake = min(next_kap, next_inv, next_tcmb)
            print(f"[{_hms()}] Bekleniyor... (sonraki: {next_wake:.0f}s)")

        time.sleep(60)


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BIST Haber Filtresi v2")
    print(f"KAP      : {KAP_API_URL}")
    print(f"           fallback: {KAP_SCRAPE_URL}")
    print(f"Investing: {INVESTING_URLS[0]}")
    print(f"TCMB     : {TCMB_URL}")
    print(f"Cikti    : {NEWS_LOG_JSONL}")
    print(f"BS4      : {'var' if HAS_BS4 else 'yok -- pip install beautifulsoup4'}")
    _sent_mode = "BERT (savasy/bert-base-turkish-sentiment-cased)" if BERT_AVAILABLE else "KEYWORD FALLBACK"
    print(f"Sentiment: {_sent_mode}")
    if ONCE_MODE:
        print("Mod      : --once (tek seferlik, max 5 per source)")
    else:
        print(f"Aralik   : KAP {INTERVAL_KAP}s / Investing {INTERVAL_INVESTING}s / TCMB {INTERVAL_TCMB}s")
    print("=" * 60)

    if ONCE_MODE:
        run_once()
        sys.exit(0)

    try:
        run_loop()
    except KeyboardInterrupt:
        print("\nHaber filtresi durduruldu (Ctrl+C)")
