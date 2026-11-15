import json
import os
import re
import sys
import time

import pandas as pd
import torch
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(HERE, "youtube_comments.csv")
OUTPUT_FILE = os.path.join(HERE, "youtube_comments_en.csv")
CACHE_FILE = os.path.join(HERE, "offline_translation_cache.json")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
BATCH_SIZE = 16
MAX_PIECE_CHARS = 200      # longest piece sent to the model at once

# Common Filipino words. A comment containing any of these is treated as Tagalog/Taglish.
# (Words that are also normal English words, like "at", "may", "man", are left out.)
TAGALOG_WORDS = set("""
ang ng mga sa na ko mo ka ako siya hindi di yung lang naman po talaga kasi dito natin tayo
kayo sila nila wala walang ba pa rin nga si ni ito yan ano bakit paano sana kaya pero dahil
kung para pag kami namin nyo niyo ninyo akin atin lahat ganito ganyan mahal presyo gasolina
pasada tigil sobra grabe sobrang bayad pera gobyerno nasa ay ung
naku kawawa tapos dapat pwede puwede gusto alam tama mali sige oo hindi nalang nlng lng nmn
""".split())
# Cebuano-only words -> translated as Cebuano instead of Tagalog.
CEBUANO_WORDS = set("dili naa kaayo unsa ngano gyud jud bitaw akong imong atong nimo nako diay usab karon".split())

LANGUAGES = {
    Language.ENGLISH: ("en", "eng_Latn"),
    Language.TAGALOG: ("tl", "tgl_Latn"),
    Language.SPANISH: ("es", "spa_Latn"),
    Language.INDONESIAN: ("id", "ind_Latn"),
    Language.MALAY: ("ms", "zsm_Latn"),
    Language.ITALIAN: ("it", "ita_Latn"),
    Language.FRENCH: ("fr", "fra_Latn"),
    Language.GERMAN: ("de", "deu_Latn"),
    Language.PORTUGUESE: ("pt", "por_Latn"),
    Language.VIETNAMESE: ("vi", "vie_Latn"),
    Language.RUSSIAN: ("ru", "rus_Cyrl"),
    Language.THAI: ("th", "tha_Thai"),
    Language.JAPANESE: ("ja", "jpn_Jpan"),
    Language.KOREAN: ("ko", "kor_Hang"),
    Language.CHINESE: ("zh", "zho_Hans"),
    Language.ARABIC: ("ar", "arb_Arab"),
    Language.HINDI: ("hi", "hin_Deva"),
}
DEFAULT = ("tl", "tgl_Latn")  # Philippine comments that aren't clearly something else


def detect(detector, text):
    """Return (short_code, nllb_code) or None if the comment has no words (emoji only, etc.)."""
    if not re.search(r"\w", text) or not re.search(r"[^\d\W]", text):
        return None
    top = detector.compute_language_confidence_values(text)[0]
    if not re.search(r"[A-Za-z]", text):
        return LANGUAGES[top.language]  # non-Latin scripts (Chinese, Japanese, ...) are detected reliably

    words = set(re.findall(r"[^\W\d_]+", text.lower()))
    if words & CEBUANO_WORDS:
        return ("ceb", "ceb_Latn")
    if words & TAGALOG_WORDS:
        return LANGUAGES[Language.TAGALOG]
    # No Filipino words: only trust the detector on longer comments.
    # Short ones ("Insane", "F USA!") are kept as English.
    if top.language != Language.ENGLISH and top.value >= 0.9 and len(words) >= 4:
        return LANGUAGES[top.language]
    return LANGUAGES[Language.ENGLISH]


def split_pieces(text):
    """Split a comment into single sentences. The model translates short pieces
    much more faithfully (long inputs make it skip parts or repeat itself)."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.7:
        text = text.lower()  # ALL-CAPS text confuses the model

    # Split at sentence endings and line breaks.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+|\n+", text)]
    # Drop pieces with no letters (emoji, "...") - the model would invent text for them.
    sentences = [" ".join(s.split()) for s in sentences if re.search(r"[^\W\d_]", s)]

    pieces = []
    for s in sentences:
        while len(s) > MAX_PIECE_CHARS:  # a very long run-on sentence: cut at a comma/space
            cut = max(s.rfind(",", 0, MAX_PIECE_CHARS), s.rfind(" ", 0, MAX_PIECE_CHARS))
            cut = cut if cut > 50 else MAX_PIECE_CHARS
            pieces.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if pieces and len(pieces[-1]) < 20:  # glue tiny bits like "Ayan na!" to the next sentence
            pieces[-1] = f"{pieces[-1]} {s}"
        elif s:
            pieces.append(s)
    return pieces or [" ".join(text.split())]


MAX_PIECES_PER_RUN = 4  # small groups keep memory use low on long comments
MAX_TRANSLATE_CHARS = 3000  # longer comments are kept untranslated (marked "too_long")


def translate_pieces(model, tokenizer, pieces, src_code):
    tokenizer.src_lang = src_code
    results = []
    for i in range(0, len(pieces), MAX_PIECES_PER_RUN):
        group = pieces[i:i + MAX_PIECES_PER_RUN]
        inputs = tokenizer(group, return_tensors="pt", padding=True, truncation=True, max_length=200)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=200,
                num_beams=2,
            )
        results.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return results


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def main():
    if not os.path.exists(INPUT_FILE):
        sys.exit("youtube_comments.csv not found. Run scrape.py first.")

    df = pd.read_csv(INPUT_FILE)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)
    print(f"{len(df)} comments loaded")

    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)

    # 1. Detect languages
    print("Detecting languages...")
    detector = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()
    to_translate = {}  # nllb_code -> list of texts
    for text in df["text"].unique():
        if text in cache:
            continue
        if len(text) > MAX_TRANSLATE_CHARS:
            # Very long comments (copy-pasted essays/spam) are kept as they are.
            cache[text] = {"en": text, "lang": "too_long"}
            continue
        lang = detect(detector, text)
        if lang is None or lang[0] == "en":
            cache[text] = {"en": text, "lang": lang[0] if lang else "none"}
        else:
            to_translate.setdefault(lang[1], []).append(text)
    save_cache(cache)

    total = sum(len(v) for v in to_translate.values())
    print(f"{total} comments need translating ({len(cache)} done or already English)")

    # 2. Translate
    if total:
        print(f"Loading {MODEL_NAME} (first time downloads ~2.5 GB)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model.eval()

        short_code = {nllb: short for short, nllb in LANGUAGES.values()}
        short_code["ceb_Latn"] = "ceb"
        done, start = 0, time.time()
        for src_code, texts in to_translate.items():
            texts.sort(key=len)  # similar lengths together = faster
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                pieces, owners = [], []
                for idx, t in enumerate(batch):
                    for p in split_pieces(t):
                        pieces.append(p)
                        owners.append(idx)
                translated = translate_pieces(model, tokenizer, pieces, src_code)
                joined = [[] for _ in batch]
                for idx, en in zip(owners, translated):
                    joined[idx].append(en)
                for t, parts in zip(batch, joined):
                    cache[t] = {"en": " ".join(parts), "lang": short_code[src_code]}
                save_cache(cache)

                done += len(batch)
                rate = done / (time.time() - start)
                left = (total - done) / rate / 60 if rate else 0
                print(f"  {done}/{total} translated  (~{left:.0f} min left)")

    # 3. Save
    df["language"] = df["text"].map(lambda t: cache[t]["lang"])
    df["text_en"] = df["text"].map(lambda t: cache[t]["en"])
    df["translated"] = ~df["language"].isin(["en", "none", "too_long"])

    # Remove spam: copy-pasted essays over MAX_TRANSLATE_CHARS characters.
    spam = df["language"] == "too_long"
    if spam.any():
        print(f"Removed {spam.sum()} spam comments (over {MAX_TRANSLATE_CHARS} characters)")
    df = df[~spam].reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} comments to {OUTPUT_FILE}")
    print("\nLanguages detected:")
    print(df["language"].value_counts().to_string())
    print(f"\nTranslated: {df['translated'].sum()}   Kept as is: {(~df['translated']).sum()}")


if __name__ == "__main__":
    main()
