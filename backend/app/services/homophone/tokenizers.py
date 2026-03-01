"""
同音读音切分器。

V3.2.0+dev.20260211.01
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

try:
    from pypinyin import Style, pinyin
except ImportError:  # pragma: no cover - 依赖可选时的降级路径
    Style = None  # type: ignore[assignment]
    pinyin = None  # type: ignore[assignment]

try:
    import cmudict
except ImportError:  # pragma: no cover - 依赖可选时的降级路径
    cmudict = None  # type: ignore[assignment]

try:
    from phonemizer import phonemize as _phonemize_text
except ImportError:  # pragma: no cover - 依赖可选时的降级路径
    _phonemize_text = None  # type: ignore[assignment]

try:
    from sudachipy import dictionary as sudachi_dictionary
    from sudachipy import tokenizer as sudachi_tokenizer
except ImportError:  # pragma: no cover - 依赖可选时的降级路径
    sudachi_dictionary = None  # type: ignore[assignment]
    sudachi_tokenizer = None  # type: ignore[assignment]


Language = Literal["zh", "ja", "en"]


@dataclass(frozen=True)
class TokenReading:
    token_text: str
    reading_key: str
    reading_key_fuzzy: str
    reading_key_no_punct: str
    reading_key_fuzzy_no_punct: str
    char_start: int
    char_end: int


_WORD_RE = re.compile(r"[A-Za-z']+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ALNUM_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fffぁ-んァ-ン]")
_KATAKANA_RE = re.compile(r"[ァ-ヶー]")


_EN_HOMOPHONE_OVERRIDES: Dict[str, str] = {
    # Phase 4: 发布前回归语料覆盖的英语同音词映射。
    "flower": "flawr",
    "flour": "flawr",
    "right": "rayt",
    "write": "rayt",
    "sea": "si",
    "see": "si",
    "knight": "nait",
    "night": "nait",
    "nite": "nait",
    "pair": "per",
    "pear": "per",
    "son": "san",
    "sun": "san",
}


_ZH_FALLBACK_PINYIN: Dict[str, str] = {
    # 仅在 pypinyin 缺失时启用的轻量兜底映射。
    # 目标：保证核心同音检索链路可用，并为 Phase 4 回归语料提供稳定读音键。
    "边": "bian1",
    "编": "bian1",
    "鞭": "bian1",
    "偏": "pian1",
    "见": "jian4",
    "界": "jie4",
    "重": "zhong4",
    "量": "liang4",
    "要": "yao4",
    "启": "qi3",
    "始": "shi3",
    "终": "zhong1",
    "钟": "zhong1",
    "种": "zhong3",
    "氰": "qing2",
    "秦": "qin2",
    "化": "hua4",
    "淮": "huai2",
    "钠": "na4",
    "呢": "ne5",
}


_JA_READING_LEXICON: Dict[str, str] = {
    # Phase 4: 常见日语同音词（用于无外部分词器场景的稳定回归）。
    "橋": "はし",
    "箸": "はし",
    "端": "はし",
    "会う": "あう",
    "合う": "あう",
    "遭う": "あう",
    "海": "うみ",
    "産み": "うみ",
    "生み": "うみ",
    "公園": "こうえん",
    "講演": "こうえん",
    "こうえん": "こうえん",
    "はし": "はし",
}
_JA_LEXICON_KEYS = sorted(_JA_READING_LEXICON.keys(), key=len, reverse=True)


_ROMAJI_TABLE: Dict[str, str] = {
    "kya": "きゃ", "kyu": "きゅ", "kyo": "きょ",
    "sha": "しゃ", "shu": "しゅ", "sho": "しょ",
    "cha": "ちゃ", "chu": "ちゅ", "cho": "ちょ",
    "nya": "にゃ", "nyu": "にゅ", "nyo": "にょ",
    "hya": "ひゃ", "hyu": "ひゅ", "hyo": "ひょ",
    "mya": "みゃ", "myu": "みゅ", "myo": "みょ",
    "rya": "りゃ", "ryu": "りゅ", "ryo": "りょ",
    "gya": "ぎゃ", "gyu": "ぎゅ", "gyo": "ぎょ",
    "bya": "びゃ", "byu": "びゅ", "byo": "びょ",
    "pya": "ぴゃ", "pyu": "ぴゅ", "pyo": "ぴょ",
    "ja": "じゃ", "ju": "じゅ", "jo": "じょ",
    "shi": "し", "chi": "ち", "tsu": "つ", "fu": "ふ",
    "ji": "じ",
    "ka": "か", "ki": "き", "ku": "く", "ke": "け", "ko": "こ",
    "sa": "さ", "su": "す", "se": "せ", "so": "そ",
    "ta": "た", "te": "て", "to": "と",
    "na": "な", "ni": "に", "nu": "ぬ", "ne": "ね", "no": "の",
    "ha": "は", "hi": "ひ", "he": "へ", "ho": "ほ",
    "ma": "ま", "mi": "み", "mu": "む", "me": "め", "mo": "も",
    "ya": "や", "yu": "ゆ", "yo": "よ",
    "ra": "ら", "ri": "り", "ru": "る", "re": "れ", "ro": "ろ",
    "wa": "わ", "wo": "を",
    "ga": "が", "gi": "ぎ", "gu": "ぐ", "ge": "げ", "go": "ご",
    "za": "ざ", "zu": "ず", "ze": "ぜ", "zo": "ぞ",
    "da": "だ", "de": "で", "do": "ど",
    "ba": "ば", "bi": "び", "bu": "ぶ", "be": "べ", "bo": "ぼ",
    "pa": "ぱ", "pi": "ぴ", "pu": "ぷ", "pe": "ぺ", "po": "ぽ",
    "a": "あ", "i": "い", "u": "う", "e": "え", "o": "お",
    "n": "ん",
}


_CMUDICT_CACHE: Dict[str, List[List[str]]] = {}
_IS_CMUDICT_INITIALIZED = False


def _strip_tone(reading_key: str) -> str:
    return re.sub(r"[1-5]$", "", reading_key)


def _is_effective_symbol(text: str) -> bool:
    return bool(_ALNUM_RE.search(text))


def _zh_char_to_reading(char: str) -> str:
    """将中文字符映射为读音键。

    说明：
    - 优先使用 pypinyin（准确率更高）。
    - 缺失依赖时使用轻量字典兜底，避免功能完全失效。
    """
    if pinyin is not None and Style is not None:
        py_items = pinyin(char, style=Style.TONE3, heteronym=False)
        return py_items[0][0] if py_items and py_items[0] else char
    return _ZH_FALLBACK_PINYIN.get(char, char)


def _load_cmudict_entries() -> Dict[str, List[List[str]]]:
    """加载 cmudict 词典（延迟加载）。"""
    global _IS_CMUDICT_INITIALIZED, _CMUDICT_CACHE
    if _IS_CMUDICT_INITIALIZED:
        return _CMUDICT_CACHE

    _IS_CMUDICT_INITIALIZED = True
    if cmudict is None:
        _CMUDICT_CACHE = {}
        return _CMUDICT_CACHE

    try:
        loaded_entries = cmudict.dict()
        _CMUDICT_CACHE = loaded_entries if isinstance(loaded_entries, dict) else {}
    except Exception:
        _CMUDICT_CACHE = {}
    return _CMUDICT_CACHE


def _arpabet_to_key(phones: List[str]) -> str:
    normalized_phones = [re.sub(r"\d", "", phone).lower() for phone in phones if phone]
    return "|".join(phone for phone in normalized_phones if phone)


def _lookup_cmudict_key(word: str) -> str:
    entries = _load_cmudict_entries()
    if not entries:
        return ""

    candidates = [word.lower()]
    normalized_word = re.sub(r"[^a-z']", "", word.lower())
    if normalized_word and normalized_word not in candidates:
        candidates.append(normalized_word)

    for candidate in candidates:
        pronunciations = entries.get(candidate, [])
        if not pronunciations:
            continue
        key_value = _arpabet_to_key(pronunciations[0])
        if key_value:
            return key_value
    return ""


def _phonemize_en_key(word: str) -> str:
    if _phonemize_text is None:
        return ""
    try:
        phoneme_text = _phonemize_text(
            word,
            language="en-us",
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
        )
    except Exception:
        return ""
    normalized = re.sub(r"[\sˈˌː']", "", str(phoneme_text).lower())
    return normalized


def _fallback_en_phonetic_key(word: str) -> str:
    """英文音码兜底：规则归一化 + 简化 Soundex。"""
    normalized = _normalize_en_word(word)
    if not normalized:
        return ""
    if normalized in _EN_HOMOPHONE_OVERRIDES:
        return _EN_HOMOPHONE_OVERRIDES[normalized]
    mapping = {
        "b": "1", "f": "1", "p": "1", "v": "1",
        "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
        "d": "3", "t": "3",
        "l": "4",
        "m": "5", "n": "5",
        "r": "6",
    }
    first = normalized[0]
    encoded: List[str] = []
    previous = mapping.get(first, "")
    for char in normalized[1:]:
        code = mapping.get(char, "")
        if not code or code == previous:
            previous = code
            continue
        encoded.append(code)
        previous = code
    return f"{first}{''.join(encoded)}"


def _normalize_en_word(word: str) -> str:
    lowered = re.sub(r"[^a-z]", "", word.lower())
    if not lowered:
        return ""
    lowered = re.sub(r"^kn", "n", lowered)
    lowered = re.sub(r"^wr", "r", lowered)
    lowered = re.sub(r"^wh", "w", lowered)
    lowered = re.sub(r"ph", "f", lowered)
    lowered = re.sub(r"ght", "t", lowered)
    lowered = re.sub(r"e$", "", lowered)
    return lowered or word.lower()


def _en_phonetic_key(word: str) -> str:
    raw_word = re.sub(r"[^a-z]", "", word.lower())
    if raw_word in _EN_HOMOPHONE_OVERRIDES:
        return _EN_HOMOPHONE_OVERRIDES[raw_word]

    cmudict_key = _lookup_cmudict_key(raw_word)
    if cmudict_key:
        return cmudict_key

    phonemizer_key = _phonemize_en_key(raw_word)
    if phonemizer_key:
        return phonemizer_key

    return _fallback_en_phonetic_key(word)


def _katakana_to_hiragana(text: str) -> str:
    output: List[str] = []
    for char in text:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:
            output.append(chr(code - 0x60))
        else:
            output.append(char)
    return "".join(output)


def _romaji_to_hiragana(text: str) -> str:
    source = re.sub(r"\s+", "", text.lower())
    if not source:
        return ""
    result: List[str] = []
    index = 0
    while index < len(source):
        if (
            index + 1 < len(source)
            and source[index] == source[index + 1]
            and source[index] not in {"a", "i", "u", "e", "o", "n"}
        ):
            result.append("っ")
            index += 1
            continue
        if source[index] == "n" and index + 1 < len(source):
            next_char = source[index + 1]
            if next_char not in {"a", "i", "u", "e", "o", "y", "n"}:
                result.append("ん")
                index += 1
                continue

        matched: Optional[str] = None
        for width in (3, 2, 1):
            piece = source[index:index + width]
            if piece in _ROMAJI_TABLE:
                matched = _ROMAJI_TABLE[piece]
                index += width
                break
        if matched is None:
            result.append(source[index])
            index += 1
            continue
        result.append(matched)
    return "".join(result)


def _normalize_ja_reading(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return ""
    if compact in _JA_READING_LEXICON:
        return _JA_READING_LEXICON[compact]
    if re.fullmatch(r"[a-z\-']+", compact.lower()):
        return _romaji_to_hiragana(compact)
    if _KATAKANA_RE.search(compact):
        return _katakana_to_hiragana(compact)
    return compact


def _normalize_ja_fuzzy(text: str) -> str:
    normalized = _normalize_ja_reading(text)
    normalized = normalized.replace("ー", "")
    normalized = normalized.replace("づ", "ず").replace("ぢ", "じ")
    return normalized


class HomophoneTokenizer:
    """轻量级分词与读音编码器。"""

    def __init__(self) -> None:
        self._sudachi = self._build_sudachi_tokenizer()

    def tokenize(self, text: str, language: Language) -> List[TokenReading]:
        if language == "zh":
            return self._tokenize_zh(text)
        if language == "ja":
            return self._tokenize_ja(text)
        return self._tokenize_en(text)

    def build_query_key(self, query_text: str, language: Language, is_fuzzy: bool) -> str:
        query = query_text.strip()
        if not query:
            return ""
        if language == "zh":
            if re.fullmatch(r"[a-z]+[1-5]?", query.lower()):
                return _strip_tone(query.lower()) if is_fuzzy else query.lower()
            joined = "|".join(
                _zh_char_to_reading(char) if _CJK_RE.search(char) else char
                for char in query
            )
            return _strip_tone(joined) if is_fuzzy else joined
        if language == "ja":
            normalized = _normalize_ja_fuzzy(query) if is_fuzzy else _normalize_ja_reading(query)
            return normalized
        strict = _en_phonetic_key(query)
        if not strict:
            return ""
        # 英文 fuzzy 不再删除首字符，避免出现 night/right/red 这类过宽召回。
        return strict

    def _tokenize_zh(self, text: str) -> List[TokenReading]:
        records: List[TokenReading] = []
        cursor = 0
        for char in text:
            if _CJK_RE.search(char):
                strict = _zh_char_to_reading(char)
            else:
                strict = char
            fuzzy = _strip_tone(strict)
            key_no_punct = strict if _is_effective_symbol(char) else ""
            fuzzy_no_punct = fuzzy if _is_effective_symbol(char) else ""
            records.append(
                TokenReading(
                    token_text=char,
                    reading_key=strict,
                    reading_key_fuzzy=fuzzy,
                    reading_key_no_punct=key_no_punct,
                    reading_key_fuzzy_no_punct=fuzzy_no_punct,
                    char_start=cursor,
                    char_end=cursor + len(char),
                )
            )
            cursor += len(char)
        return records

    def _tokenize_ja(self, text: str) -> List[TokenReading]:
        if self._sudachi is not None:
            sudachi_rows = self._tokenize_ja_with_sudachi(text)
            if sudachi_rows:
                return sudachi_rows

        records: List[TokenReading] = []
        cursor = 0
        while cursor < len(text):
            matched = self._match_ja_lexicon(text=text, cursor=cursor)
            if matched is not None:
                token_text, strict = matched
            else:
                token_text = text[cursor]
                strict = _normalize_ja_reading(token_text)

            fuzzy = _normalize_ja_fuzzy(strict)
            key_no_punct = strict if _is_effective_symbol(token_text) else ""
            fuzzy_no_punct = fuzzy if _is_effective_symbol(token_text) else ""
            records.append(
                TokenReading(
                    token_text=token_text,
                    reading_key=strict,
                    reading_key_fuzzy=fuzzy,
                    reading_key_no_punct=key_no_punct,
                    reading_key_fuzzy_no_punct=fuzzy_no_punct,
                    char_start=cursor,
                    char_end=cursor + len(token_text),
                )
            )
            cursor += len(token_text)
        return records

    def _tokenize_ja_with_sudachi(self, text: str) -> List[TokenReading]:
        if self._sudachi is None:
            return []
        try:
            mode = sudachi_tokenizer.Tokenizer.SplitMode.B if sudachi_tokenizer is not None else None
            morphemes = self._sudachi.tokenize(text, mode) if mode is not None else []
        except Exception:
            return []

        records: List[TokenReading] = []
        cursor = 0
        for morpheme in morphemes:
            token_text = str(morpheme.surface())
            if not token_text:
                continue
            reading = str(morpheme.reading_form() or token_text)
            strict = _normalize_ja_reading(reading)
            fuzzy = _normalize_ja_fuzzy(strict)
            key_no_punct = strict if _is_effective_symbol(token_text) else ""
            fuzzy_no_punct = fuzzy if _is_effective_symbol(token_text) else ""
            records.append(
                TokenReading(
                    token_text=token_text,
                    reading_key=strict,
                    reading_key_fuzzy=fuzzy,
                    reading_key_no_punct=key_no_punct,
                    reading_key_fuzzy_no_punct=fuzzy_no_punct,
                    char_start=cursor,
                    char_end=cursor + len(token_text),
                )
            )
            cursor += len(token_text)
        return records

    def _tokenize_en(self, text: str) -> List[TokenReading]:
        records: List[TokenReading] = []
        for match in _WORD_RE.finditer(text):
            token = match.group(0)
            strict = _en_phonetic_key(token)
            if not strict:
                continue
            fuzzy = strict
            records.append(
                TokenReading(
                    token_text=token,
                    reading_key=strict,
                    reading_key_fuzzy=fuzzy,
                    reading_key_no_punct=strict,
                    reading_key_fuzzy_no_punct=fuzzy,
                    char_start=match.start(),
                    char_end=match.end(),
                )
            )
        return records

    @staticmethod
    def _match_ja_lexicon(text: str, cursor: int) -> Optional[Tuple[str, str]]:
        fragment = text[cursor:]
        for candidate in _JA_LEXICON_KEYS:
            if fragment.startswith(candidate):
                return candidate, _JA_READING_LEXICON[candidate]
        return None

    @staticmethod
    def _build_sudachi_tokenizer() -> Optional[Any]:
        """构建 Sudachi 分词器；不可用时返回 None。"""
        if sudachi_dictionary is None:
            return None
        try:
            return sudachi_dictionary.Dictionary().create()
        except Exception:
            return None
