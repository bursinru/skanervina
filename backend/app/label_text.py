"""Catalog-grounded scoring of PP-OCR label text.

SigLIP finds the right producer and series; the grape, cuvée and sweetness
printed on the label decide between bottles of one series. Every OCR word is
weighted by detector confidence, glyph size and distance from the frame centre
(neighbouring bottles sit at the edges), then matched against catalog fields
with IDF weights. The result is sparse per-slug evidence for the fusion step.
"""
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Letter and digit runs split apart: OCR glues «КРАСНОЕ2022» together.
TOKEN_RE = re.compile(r"[a-zа-яё]+|[0-9]+", re.IGNORECASE)
CAMEL = re.compile(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])")
YEAR_RE = re.compile(r"^(?:19[5-9]\d|20[0-4]\d)$")
CYRILLIC_RE = re.compile(r"[а-яё]")

TO_LATIN = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh",
    "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu",
    "я": "ya",
})
# Latin and Greek glyphs the recogniser emits for identical Cyrillic capitals.
TO_CYRILLIC = str.maketrans({
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м", "o": "о",
    "p": "р", "t": "т", "x": "х", "y": "у", "3": "з", "6": "б",
})
LOOKALIKE_ONLY = set("abcehkmoptxy")
GREEK = str.maketrans({
    "Α": "А", "Β": "В", "Γ": "Г", "Δ": "Д", "Ε": "Е", "Ζ": "З", "Η": "Н", "Κ": "К", "Λ": "Л",
    "Μ": "М", "Ο": "О", "Π": "П", "Ρ": "Р", "Τ": "Т", "Υ": "У", "Φ": "Ф", "Χ": "Х",
    "α": "а", "γ": "г", "δ": "д", "ε": "е", "κ": "к", "λ": "л", "μ": "м", "ο": "о",
    "π": "п", "ρ": "р", "τ": "т", "υ": "у", "φ": "ф", "χ": "х",
})

# Labels print grapes in Latin, the catalog in Russian (and the reverse).
TERMS = {
    "каберне": ("cabernet",), "совиньон": ("sauvignon",), "фран": ("franc",),
    "мерло": ("merlot",), "пино": ("pinot",), "нуар": ("noir",), "блан": ("blanc", "blancs"),
    "гри": ("gris",), "гриджио": ("grigio",), "шардоне": ("chardonnay",),
    "рислинг": ("riesling",), "сира": ("syrah", "shiraz", "шираз"), "мускат": ("muscat", "moscato"),
    "алиготе": ("aligote",), "ркацители": ("rkatsiteli", "rkaciteli"), "вионье": ("viognier",),
    "шенен": ("chenin",), "траминер": ("traminer", "gewurztraminer"), "мальбек": ("malbec",),
    "санджовезе": ("sangiovese",), "темпранильо": ("tempranillo",), "розе": ("rose", "rosé"),
    "руж": ("rouge",), "брют": ("brut",), "резерв": ("reserve", "reserva", "riserva"),
    "саперави": ("saperavi",), "кокур": ("kokur",), "красностоп": ("krasnostop",),
    "цимлянский": ("tsimlyansky",), "менье": ("meunier",), "мускатель": ("muscatel",),
    "гевюрцтраминер": ("gewurztraminer",), "гренаш": ("grenache",), "марселан": ("marselan",),
    "одесский": ("odessky",), "бастардо": ("bastardo",), "херес": ("sherry", "jerez"),
    "пти": ("petit",), "вердо": ("verdot",), "мурведр": ("mourvedre",), "кюве": ("cuvee",),
    "оранж": ("orange",), "игристое": ("sparkling", "spumante"),
}
CANONICAL = {variant: canonical for canonical, variants in TERMS.items() for variant in variants}
STOP = {
    "вино", "вина", "wine", "vino", "винодельня", "winery", "россия", "россии", "russia",
    "год", "урожая", "сорт", "сорта", "винограда", "the", "and", "of", "de", "из", "для",
    "выдержка", "vin", "вин", "защищенного", "географического", "указания", "наименования",
    "места", "происхождения", "произведено", "объем", "крепость", "продукт",
}

# Transliterated forms cover catalog slugs («...-polusuhoe-krasnoe-13»).
STYLE_GROUPS = {
    "sweetness": (
        ("extra_brut", ("экстра брют", "extra brut", "ekstra bryut")),
        ("brut", ("брют", "brut", "bryut")),
        ("semi_dry", ("полусухое", "polusuhoe", "semi dry", "demi sec")),
        ("semi_sweet", ("полусладкое", "polusladkoe", "semi sweet")),
        ("dry", ("сухое", "suhoe", "dry", "secco", "trocken")),
        ("sweet", ("сладкое", "sladkoe", "sweet", "десертное", "ликерное", "dolce")),
    ),
    "colour": (
        ("rose", ("розовое", "rozovoe", "rose", "rosé", "розе", "rosato", "rosado")),
        ("orange", ("оранжевое", "oranzhevoe", "orange")),
        ("red", ("красное", "krasnoe", "red", "rosso", "rouge", "tinto")),
        ("white", ("белое", "beloe", "white", "bianco", "blanc", "blanco")),
    ),
}


def _fold_word(raw: str) -> str:
    """A Cyrillic word typed with Latin look-alikes («CYXOE») becomes Cyrillic."""
    if CYRILLIC_RE.search(raw) or (len(raw) >= 3 and set(raw) <= LOOKALIKE_ONLY):
        return raw.translate(TO_CYRILLIC)
    return raw


def _roman(token: str) -> Optional[int]:
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
    if len(token) < 2 or any(ch not in values for ch in token):
        return None
    total = 0
    for current, following in zip(token, token[1:] + " "):
        value = values[current]
        total += -value if values.get(following, 0) > value else value
    return total if 1 < total < 200 else None


# pp_ocr folds «2024» with a stray letter into «2о24»; put the digits back.
DIGITISH_RE = re.compile(r"\b(?=[0-9оoзб]*[0-9])[0-9оoзб]{3,}\b")
DIGITS = str.maketrans({"о": "0", "o": "0", "з": "3", "б": "6"})


def raw_words(text: str) -> List[str]:
    text = CAMEL.sub(" ", unicodedata.normalize("NFKC", text).translate(GREEK))
    text = DIGITISH_RE.sub(lambda match: match.group(0).translate(DIGITS), text.casefold().replace("ё", "е"))
    return TOKEN_RE.findall(text)


# Sweetness and colour words are style evidence, not identity: they are on
# every label of a series, and the catalog category disagrees with the label
# often enough that they must not add to the name score.
STYLE_WORDS = {
    "сухое", "полусухое", "полусладкое", "сладкое", "брют", "экстра", "красное", "белое",
    "розовое", "оранжевое", "suhoe", "polusuhoe", "polusladkoe", "sladkoe", "bryut", "ekstra",
    "krasnoe", "beloe", "rozovoe", "oranzhevoe", "dry", "brut", "extra", "red", "white",
}


def variants(raw: str) -> Set[str]:
    """Comparable forms of one word: as read, script-folded, canonical grape, translit."""
    if len(raw) < 3 and not raw.isdigit():
        return set()
    forms = {raw}
    folded = _fold_word(raw)
    forms.add(folded)
    for form in tuple(forms):
        if form in CANONICAL:
            forms.add(CANONICAL[form])
    roman = _roman(raw)
    if roman is not None:
        forms.add(str(roman))
    for form in tuple(forms):
        if CYRILLIC_RE.search(form):
            forms.add(form.translate(TO_LATIN))
    if forms & STYLE_WORDS:
        return set()
    return {form for form in forms if form not in STOP and len(form) >= (2 if form.isdigit() else 3)}


def normalize_tokens(text: str) -> Set[str]:
    found: Set[str] = set()
    for raw in raw_words(text):
        found |= variants(raw)
    return found


def style_of(text: str) -> Dict[str, str]:
    folded = " " + " ".join(_fold_word(word) for word in raw_words(text)) + " "
    found = {}
    for group, styles in STYLE_GROUPS.items():
        for style, phrases in styles:
            if any(f" {phrase} " in folded for phrase in phrases):
                found[group] = style
                break
    return found


def _levenshtein(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, 1):
        current = [i]
        best = i
        for j, rch in enumerate(right, 1):
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (lch != rch))
            current.append(value)
            best = min(best, value)
        if best > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _deletes(word: str, depth: int) -> Set[str]:
    found = {word}
    frontier = {word}
    for _ in range(depth):
        frontier = {item[:i] + item[i + 1:] for item in frontier for i in range(len(item))}
        found |= frontier
    return found


@dataclass(frozen=True)
class Word:
    text: str
    weight: float


@dataclass(frozen=True)
class Evidence:
    score: float
    coverage: float
    matched: Tuple[str, ...]
    contradiction: float  # label weight a same-winery rival explains and this wine does not
    sweetness: int  # +1 label and catalog agree, -1 disagree, 0 unknown
    colour: int
    year: int


def ocr_words(lines: Sequence[dict], size: Tuple[int, int]) -> List[Word]:
    """PP-OCR lines -> weighted words. Edge text keeps a fifth of its weight."""

    width, height = max(1, size[0]), max(1, size[1])
    words = []
    for line in lines:
        box = line.get("box") or []
        if not box:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        cx = sum(xs) / len(xs) / width
        glyph = (max(ys) - min(ys)) / height
        centrality = math.exp(-((cx - 0.5) / 0.25) ** 2)
        size_prior = min(1.0, 0.4 + glyph * 12)
        weight = float(line.get("score") or 0.0) * (0.2 + 0.8 * centrality) * size_prior
        words.append(Word(str(line.get("text") or ""), weight))
    return words


def catalog_documents(wines: Iterable) -> Dict[str, List[Tuple[str, float]]]:
    """Catalog card fields with their identity weight."""

    documents = {}
    for wine in wines:
        documents[wine.slug] = [
            (wine.name, 2.0),
            (wine.winery or "", 1.5),
            (" ".join(wine.grapes or []), 0.8),
            (wine.slug.replace("-", " ").replace("_", " "), 0.8),
            (f"{wine.category or ''} {getattr(wine, 'color', '') or ''}", 0.5),
        ]
    return documents


def reference_documents(entries: Iterable[dict]) -> Dict[str, List[Tuple[str, float]]]:
    """Words read from each catalog photo, weighted like query words."""

    documents = {}
    for entry in entries:
        if entry.get("size") and entry.get("lines"):
            documents[entry["slug"]] = [(word.text, word.weight) for word in ocr_words(entry["lines"], entry["size"])]
    return documents


class TextIndex:
    """IDF-weighted inverted index over per-wine documents of (text, weight)."""

    def __init__(self, documents: Dict[str, List[Tuple[str, float]]], wines: Iterable = ()) -> None:
        self.fields: Dict[str, Dict[str, float]] = {}
        self.styles: Dict[str, Dict[str, str]] = {}
        self.years: Dict[str, Set[str]] = {}
        self.wineries: Dict[str, str] = {}
        frequency: Counter = Counter()
        for slug, parts in documents.items():
            weights: Dict[str, float] = {}
            for text, weight in parts:
                for token in normalize_tokens(text):
                    weights[token] = max(weights.get(token, 0.0), weight)
            self.fields[slug] = weights
            frequency.update(weights)
        for wine in wines:
            slug_text = wine.slug.replace("-", " ").replace("_", " ")
            # The catalog category is authoritative; name and slug fill the gaps.
            style = style_of(wine.category or "")
            for group, value in style_of(f"{wine.name} {slug_text}").items():
                style.setdefault(group, value)
            self.styles[wine.slug] = style
            self.wineries[wine.slug] = " ".join(sorted(normalize_tokens(wine.winery or "")))
            self.years[wine.slug] = {t for t in normalize_tokens(f"{wine.name} {slug_text}") if YEAR_RE.match(t)}
        total = len(self.fields)
        self.idf = {token: math.log((total + 1) / (count + 1)) + 1.0 for token, count in frequency.items()}
        self.postings: Dict[str, List[str]] = {}
        for slug, weights in self.fields.items():
            for token in weights:
                self.postings.setdefault(token, []).append(slug)
        self.norms = {
            slug: sum(weight * self.idf[token] for token, weight in weights.items() if not token.isdigit())
            for slug, weights in self.fields.items()
        }
        self._near: Dict[str, List[str]] = {}
        for token in self.idf:
            if len(token) >= 4 and not token.isdigit():
                for key in _deletes(token, 1 if len(token) <= 6 else 2):
                    self._near.setdefault(key, []).append(token)
        self._resolved: Dict[str, List[Tuple[str, float]]] = {}

    def _resolve(self, token: str) -> List[Tuple[str, float]]:
        """Exact token, nearest catalog token within 1-2 edits, or a glued pair."""
        if token in self.idf:
            return [(token, 1.0)]
        if token in self._resolved:
            return self._resolved[token]
        found: List[Tuple[str, float]] = []
        if len(token) >= 4 and not token.isdigit():
            limit = 1 if len(token) <= 6 else 2
            best, best_distance = None, limit + 1
            for key in _deletes(token, limit):
                for target in self._near.get(key, ()):
                    distance = _levenshtein(token, target, limit)
                    if distance < best_distance or (distance == best_distance and best and self.idf[target] < self.idf[best]):
                        best, best_distance = target, distance
            if best is not None:
                found = [(best, 0.6)]
            else:
                # «ПИНОНУАР»: the space between two words was lost.
                for cut in range(3, len(token) - 2):
                    left, right = token[:cut], token[cut:]
                    if left in self.idf and right in self.idf:
                        found = [(left, 0.8), (right, 0.8)]
                        break
        self._resolved[token] = found
        return found

    def score(self, words: Sequence[Word]) -> "LabelText":
        groups: Dict[str, Dict[str, float]] = {}
        votes: Dict[str, Dict[str, float]] = {}
        for word in words:
            for group, style in style_of(word.text).items():
                votes.setdefault(group, {}).setdefault(style, 0.0)
                votes[group][style] += word.weight
            for raw in raw_words(word.text):
                bucket = groups.setdefault(raw, {})
                for token in variants(raw):
                    parts = self._resolve(token)
                    for index, (key, factor) in enumerate(parts):
                        target = bucket if len(parts) == 1 else groups.setdefault(f"{raw}#{index}", {})
                        target[key] = max(target.get(key, 0.0), word.weight * factor)
        label_style = {
            group: max(styles, key=styles.get)
            for group, styles in votes.items()
            if max(styles.values()) >= 0.2
        }
        query_years = {token for bucket in groups.values() for token in bucket if YEAR_RE.match(token)}
        contributions: Dict[str, Dict[str, float]] = {}
        for raw, bucket in groups.items():
            best: Dict[str, float] = {}
            for token, weight in bucket.items():
                for slug in self.postings.get(token, ()):
                    value = self.fields[slug][token] * self.idf[token] * weight
                    if value > best.get(slug, 0.0):
                        best[slug] = value
            for slug, value in best.items():
                contributions.setdefault(slug, {})[raw] = value
        return LabelText(self, contributions, label_style, query_years)


class LabelText:
    """One label read against the catalog."""

    def __init__(self, index: "TextIndex", contributions, label_style, years) -> None:
        self.index = index
        self.contributions: Dict[str, Dict[str, float]] = contributions
        self.totals = {slug: sum(values.values()) for slug, values in contributions.items()}
        self.label_style: Dict[str, str] = label_style
        self.years: Set[str] = years

    def top(self, limit: int = 10) -> List[str]:
        return [slug for slug, _ in sorted(self.totals.items(), key=lambda item: item[1], reverse=True)[:limit]]

    def contradiction(self, slug: str, rivals: Iterable[str]) -> float:
        """Weight of label words that a same-winery rival explains and slug does not."""

        winery = self.index.wineries.get(slug)
        own = self.contributions.get(slug, {})
        missing: Dict[str, float] = {}
        for rival in rivals:
            if rival == slug or not winery or self.index.wineries.get(rival) != winery:
                continue
            for raw, value in self.contributions.get(rival, {}).items():
                if raw not in own and value > missing.get(raw, 0.0):
                    missing[raw] = value
        return sum(missing.values())

    def evidence(self, slug: str, rivals: Iterable[str] = ()) -> Evidence:
        total = self.totals.get(slug, 0.0)
        style = self.index.styles.get(slug, {})

        def agreement(group):
            if group not in self.label_style or group not in style:
                return 0
            return 1 if style[group] == self.label_style[group] else -1

        years = self.index.years.get(slug, set())
        year = 0
        if self.years and years:
            year = -1 if self.years.isdisjoint(years) else 1
        return Evidence(
            score=total,
            coverage=min(1.0, total / max(self.index.norms.get(slug, 0.0), 1e-6)),
            matched=tuple(sorted(self.contributions.get(slug, {}))),
            contradiction=self.contradiction(slug, rivals),
            sweetness=agreement("sweetness"),
            colour=agreement("colour"),
            year=year,
        )
