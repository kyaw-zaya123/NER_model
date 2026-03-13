"""
=============================================================================
STEP 1: DATASET PREPARATION FOR MYANMAR NER WITH LOCATION SUB-CLASSIFICATION
=============================================================================
Myanmar Location Hierarchy:
  - COUNTRY       : နိုင်ငံ
  - STATE/REGION  : တိုင်းဒေသကြီး / ပြည်နယ် (e.g., စစ်ကိုင်းတိုင်းဒေသကြီး)
  - DISTRICT      : ခရိုင် (e.g., စစ်ကိုင်းခရိုင်)
  - TOWNSHIP      : မြို့နယ် (e.g., စစ်ကိုင်းမြို့နယ်)
  - CITY          : မြို့ (e.g., မန္တလေးမြို့)
  - VILLAGE       : ကျေးရွာ / ရွာ
  - WARD          : ရပ်ကွက်

Tag Schema: BIOES (Begin, Inside, Other, End, Single)
  B-LOC-COUNTRY, I-LOC-COUNTRY, E-LOC-COUNTRY, S-LOC-COUNTRY
  B-LOC-STATE,   I-LOC-STATE,   E-LOC-STATE,   S-LOC-STATE
  B-LOC-DISTRICT, ...
  B-LOC-TOWNSHIP, ...
  B-LOC-CITY, ...
  B-LOC-VILLAGE, ...
  B-LOC-WARD, ...
  S-lOC,...
  B-PER, I-PER, E-PER, S-PER
  B-ORG, I-ORG, E-ORG, S-ORG
  B-DATE, I-DATE, E-DATE, S-DATE
  B-TIME, I-TIME, E-TIME, S-TIME
  B-NUM, I-NUM, E-NUM, S-NUM
  O
"""

import re
import os
import random
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime


# 0.  Logging setup  (console + rotating file)
def setup_logging(log_dir: str = "logs", log_level: int = logging.DEBUG) -> logging.Logger:
    """
    Configure a logger that writes to both the console and a timestamped file.

    Args:
        log_dir   : directory where log files are stored (created if absent).
        log_level : minimum log level (default DEBUG captures everything).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(log_dir, f"dataset_preparation_{timestamp}.log")

    logger = logging.getLogger("myanmar_ner")
    logger.setLevel(log_level)

    fmt     = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("Logging initialised  →  %s", log_path)
    return logger

# Module-level logger 
log: logging.Logger = logging.getLogger("myanmar_ner")

# 1.  Myanmar location keyword taxonomy
LOC_SUFFIX_RULES: list[tuple[str, str]] = [
    ("နိုင်ငံ",                          "LOC-COUNTRY"),
    ("ပြည်ထောင်စု",           "LOC-COUNTRY"),
    ("သမ္မတနိုင်ငံ",           "LOC-COUNTRY"),
    ("နိုင်ငံတော်",            "LOC-COUNTRY"),
    ("တိုင်းဒေသကြီး",                   "LOC-STATE"),
    ("တိုင်း",                           "LOC-STATE"),
    ("ပြည်နယ်",                          "LOC-STATE"),
    ("ခရိုင်",                           "LOC-DISTRICT"),
    ("ကိုယ်ပိုင်အုပ်ချုပ်ခွင့်ရဒေသ",   "LOC-DISTRICT"),
    ("မြို့နယ်",                         "LOC-TOWNSHIP"),
    ("မြို့",                            "LOC-CITY"),
    ("ကျေးရွာအုပ်စု",                   "LOC-VILLAGE"),
    ("ကျေးရွာ",                          "LOC-VILLAGE"),
    ("ရွာ",                              "LOC-VILLAGE"),
    ("ရပ်ကွက်",                          "LOC-WARD"),
]

# Fast lookup dict  suffix → subtype
SUFFIX_TO_SUBTYPE: dict[str, str] = {rule[0]: rule[1] for rule in LOC_SUFFIX_RULES}
SUFFIX_ORDERED: list[str] = [rule[0] for rule in LOC_SUFFIX_RULES]


def classify_loc_span(tokens: list[str]) -> str:
    """
    Given the tokens forming a single LOC span, return the fine-grained
    location sub-type by inspecting the concatenated span and each token.

    Priority: first match in ``SUFFIX_ORDERED`` wins (longest/most-specific
    suffixes are listed first).

    Returns ``'LOC-GEN'`` when no known suffix is found.
    """
    full = "".join(tokens)
    for suffix in SUFFIX_ORDERED:
        if full.endswith(suffix):
            log.debug("classify_loc_span: '%s' → %s  (suffix='%s')",
                      full, SUFFIX_TO_SUBTYPE[suffix], suffix)
            return SUFFIX_TO_SUBTYPE[suffix]
        for tok in tokens:
            if tok == suffix:
                log.debug("classify_loc_span: '%s' → %s  (token match='%s')",
                          full, SUFFIX_TO_SUBTYPE[suffix], suffix)
                return SUFFIX_TO_SUBTYPE[suffix]

    log.debug("classify_loc_span: '%s' → LOC  (no suffix matched, remapped from LOC-GEN)", full)
    return "LOC"


# 2a.  Valid tag set for schema validation
VALID_PREFIXES  = {"B", "I", "E", "S", "O"}
VALID_ENTITY_TYPES = {
    "LOC", "LOC-COUNTRY", "LOC-STATE", "LOC-DISTRICT",
    "LOC-TOWNSHIP", "LOC-CITY", "LOC-VILLAGE", "LOC-WARD",
    "PER", "ORG", "DATE", "TIME", "NUM",
}

# Tags that are known typos / noise and can be auto-corrected
TAG_CORRECTIONS: dict[str, str] = {
    "0": "O",   # digit-zero written instead of letter-O
}

MAX_SENTENCE_TOKENS = 512   # transformer context window safety limit


def _clean_tag(tag: str, lineno: int) -> str:
    """
    Apply ``TAG_CORRECTIONS`` and warn on unrecognised tags.

    Args:
        tag    : raw NER tag from the CoNLL file.
        lineno : source line number (used in warnings).

    Returns:
        Corrected tag string.
    """
    if tag in TAG_CORRECTIONS:
        corrected = TAG_CORRECTIONS[tag]
        log.warning("Line %d: corrected tag %r → %r", lineno, tag, corrected)
        return corrected

    # Validate structure
    if tag != "O":
        parts = tag.split("-", 1)
        if parts[0] not in VALID_PREFIXES or (len(parts) == 1 and parts[0] != "O"):
            log.warning("Line %d: unrecognised tag %r — kept as-is", lineno, tag)

    return tag


def split_long_sentences(
    sentences: list[list[tuple]],
    max_tokens: int = MAX_SENTENCE_TOKENS,
) -> list[list[tuple]]:
    """
    Split sentences that exceed ``max_tokens`` at ``O``-tagged boundaries
    so no chunk overflows a transformer's context window.

    Splitting only happens at ``O`` tokens to avoid breaking entity spans.
    If an entity span itself exceeds ``max_tokens``, it is kept intact and a
    warning is emitted (truncation would corrupt the label).

    Args:
        sentences  : list of tagged sentences.
        max_tokens : maximum allowed tokens per sentence (default 512).

    Returns:
        New sentence list with long sentences replaced by shorter chunks.
    """
    result: list[list[tuple]] = []
    n_split = 0

    for sent in sentences:
        if len(sent) <= max_tokens:
            result.append(sent)
            continue

        # Split at O-tag boundaries
        chunks: list[list[tuple]] = []
        current: list[tuple] = []

        for token in sent:
            current.append(token)
            _, _, tag = token
            if len(current) >= max_tokens and tag == "O":
                chunks.append(current)
                current = []

        if current:
            chunks.append(current)

        # Merge any chunk that is still over the limit (entity spans too long)
        for chunk in chunks:
            if len(chunk) > max_tokens:
                log.warning(
                    "Chunk of %d tokens exceeds max_tokens=%d and cannot be "
                    "split safely (entity spans too long); kept intact.",
                    len(chunk), max_tokens,
                )
            result.append(chunk)

        n_split += 1

    if n_split:
        log.info(
            "split_long_sentences: %d sentences exceeded %d tokens and were split "
            "→ corpus grew from %d to %d sentences.",
            n_split, max_tokens, len(sentences), len(result),
        )
    return result


def validate_bioes(sentences: list[list[tuple]]) -> int:
    """
    Check BIOES tag sequence consistency and log all violations.

    Checks performed:

    * ``I``/``E`` token not preceded by ``B`` or ``I`` of the same type.
    * ``B`` span closed by a mismatched entity type.
    * Unclosed ``B`` span at sentence end.

    Args:
        sentences : list of tagged sentences.

    Returns:
        Total number of violations found (0 = clean).
    """
    violations = 0
    for s_idx, sent in enumerate(sentences):
        open_type: str | None = None
        for t_idx, (word, _, tag) in enumerate(sent):
            prefix = tag.split("-")[0] if "-" in tag else tag
            etype  = tag.split("-", 1)[1] if "-" in tag else None

            if prefix in ("I", "E"):
                if open_type is None or open_type != etype:
                    log.warning(
                        "BIOES violation — sentence %d token %d: %r tag '%s' "
                        "without matching B (open=%s)",
                        s_idx, t_idx, word, tag, open_type,
                    )
                    violations += 1
                if prefix == "E":
                    open_type = None

            elif prefix == "B":
                if open_type is not None:
                    log.warning(
                        "BIOES violation — sentence %d token %d: B tag '%s' "
                        "opened while '%s' span still open",
                        s_idx, t_idx, tag, open_type,
                    )
                    violations += 1
                open_type = etype

            elif prefix in ("S", "O"):
                if open_type is not None and prefix == "S":
                    log.warning(
                        "BIOES violation — sentence %d token %d: S tag '%s' "
                        "while B span ('%s') still open",
                        s_idx, t_idx, tag, open_type,
                    )
                    violations += 1
                open_type = None

        if open_type is not None:
            log.warning(
                "BIOES violation — sentence %d: unclosed B span '%s' at end of sentence",
                s_idx, open_type,
            )
            violations += 1

    log.info("BIOES validation: %d violation(s) found in %d sentences.", violations, len(sentences))
    return violations


# 2.  CoNLL reader / writer
def read_conll(path: str) -> list[list[tuple]]:
    """
    Read a CoNLL-formatted file.

    Supported line formats::

        word TAB pos TAB ner
        word TAB ner          (pos defaults to ``'n'``)

    Blank lines delimit sentences.

    Args:
        path : path to the CoNLL file.

    Returns:
        List of sentences; each sentence is a list of ``(word, pos, ner)``
        tuples.
    """
    log.info("Reading CoNLL file: %s", path)
    sentences: list[list[tuple]] = []
    sent: list[tuple] = []
    skipped = 0

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n\r")
            if line.strip() == "":
                if sent:
                    sentences.append(sent)
                    sent = []
            else:
                parts = line.split("\t")
                if len(parts) == 3:
                    word, pos, ner = parts
                    ner = _clean_tag(ner, lineno)
                    sent.append((word, pos, ner))
                elif len(parts) == 2:
                    word, ner = parts
                    ner = _clean_tag(ner, lineno)
                    sent.append((word, "n", ner))
                else:
                    log.warning("Line %d skipped (unexpected column count %d): %r",
                                lineno, len(parts), line)
                    skipped += 1

    if sent:         
        sentences.append(sent)

    log.info("  → %d sentences loaded  (%d lines skipped)", len(sentences), skipped)
    return sentences


def write_conll(sentences: list[list[tuple]], path: str) -> None:
    """
    Write sentences to CoNLL format (word TAB pos TAB ner, blank-line separated).

    Args:
        sentences : list of sentence token lists.
        path      : output file path (parent dirs created automatically).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for word, pos, ner in sent:
                f.write(f"{word}\t{pos}\t{ner}\n")
            f.write("\n")
    log.info("CoNLL written: %s  (%d sentences)", path, len(sentences))


# 3.  Span extraction helpers  (BIOES scheme)
def extract_spans(sentence: list[tuple]) -> list[tuple]:
    """
    Extract named-entity spans from a BIOES-tagged sentence.

    Args:
        sentence : list of ``(word, pos, ner)`` tuples.

    Returns:
        List of ``(start_idx, end_idx_inclusive, entity_type, tokens)`` tuples.
    """
    spans: list[tuple] = []
    i = 0
    while i < len(sentence):
        word, pos, tag = sentence[i]
        prefix = tag.split("-")[0] if "-" in tag else tag
        etype  = tag.split("-", 1)[1] if "-" in tag else None

        if prefix == "S" and etype:
            spans.append((i, i, etype, [word]))
            i += 1

        elif prefix == "B" and etype:
            span_tokens = [word]
            j = i + 1
            while j < len(sentence):
                _, _, next_tag = sentence[j]
                np = next_tag.split("-")[0] if "-" in next_tag else next_tag
                if np == "E":
                    span_tokens.append(sentence[j][0])
                    spans.append((i, j, etype, span_tokens))
                    i = j + 1
                    break
                elif np == "I":
                    span_tokens.append(sentence[j][0])
                    j += 1
                else:
                    # O, B, or S encountered before E: treat as unclosed span
                    log.warning(
                        "Unclosed B tag at token %d ('%s'), stopping at token %d "
                        "(tag '%s').",
                        i, word, j, next_tag,
                    )
                    i = j   # resume outer loop at the interrupting token
                    break
            else:
                log.warning("Unclosed B tag at token %d ('%s'), reached end of sentence.", i, word)
                i = j
        else:
            i += 1

    return spans


def rebuild_tags_with_spans(
    sentence: list[tuple],
    span_overrides: dict[int, str],
) -> list[tuple]:
    """
    Return a new sentence with selected token tags replaced.

    Args:
        sentence       : original ``(word, pos, ner)`` list.
        span_overrides : mapping ``{token_index: new_tag}``.

    Returns:
        New sentence list with overridden tags applied.
    """
    return [
        (word, pos, span_overrides.get(idx, old_tag))
        for idx, (word, pos, old_tag) in enumerate(sentence)
    ]


# 4.  Main transformation: flat LOC → fine-grained LOC sub-types
def transform_loc_tags(sentences: list[list[tuple]]) -> list[list[tuple]]:
    """
    Replace every flat ``LOC`` span with its fine-grained sub-type tag
    (e.g., ``LOC-CITY``, ``LOC-TOWNSHIP``) using BIOES notation.

    Non-LOC entity types (PER, ORG, DATE, …) are left unchanged.

    Args:
        sentences : list of BIOES-tagged sentences.

    Returns:
        New sentence list with refined LOC tags.
    """
    log.info("Applying fine-grained LOC sub-classification to %d sentences …", len(sentences))
    new_sentences: list[list[tuple]] = []
    subtype_counter: Counter = Counter()

    for sent in sentences:
        spans = extract_spans(sent)
        overrides: dict[int, str] = {}

        for start, end, etype, tokens in spans:
            if etype != "LOC":
                continue

            subtype = classify_loc_span(tokens)
            subtype_counter[subtype] += 1
            length = end - start + 1

            if length == 1:
                overrides[start] = f"S-{subtype}"
            else:
                for offset in range(length):
                    idx = start + offset
                    if offset == 0:
                        overrides[idx] = f"B-{subtype}"
                    elif offset == length - 1:
                        overrides[idx] = f"E-{subtype}"
                    else:
                        overrides[idx] = f"I-{subtype}"

        # Remap any pre-existing LOC-GEN tags (already in file) to plain LOC
        rebuilt = rebuild_tags_with_spans(sent, overrides)
        remapped = []
        for word, pos, tag in rebuilt:
            if tag.endswith("-LOC-GEN"):
                prefix = tag.split("-")[0]          # B / I / E / S
                tag = f"{prefix}-LOC"
                log.debug("Remapped *-LOC-GEN → %s for '%s'", tag, word)
            remapped.append((word, pos, tag))
        new_sentences.append(remapped)

    log.info("LOC sub-type distribution after transformation:")
    for subtype, cnt in sorted(subtype_counter.items()):
        log.info("  %-25s %6d spans", subtype, cnt)

    return new_sentences


# 5.  Dataset statistics
def dataset_stats(sentences: list[list[tuple]], label: str = "") -> None:
    """
    Log and print token-level tag counts and span-level entity type counts.

    Args:
        sentences : list of tagged sentences to analyse.
        label     : descriptive label shown in the report header.
    """
    tag_counter: Counter    = Counter()
    entity_counter: Counter = Counter()
    total_tokens = 0

    for sent in sentences:
        total_tokens += len(sent)
        for _, _, tag in sent:
            tag_counter[tag] += 1
        for _, _, etype, _ in extract_spans(sent):
            entity_counter[etype] += 1

    sep = "=" * 60
    lines = [
        sep,
        f"  Dataset : {label}",
        f"  Sentences : {len(sentences):,}",
        f"  Tokens    : {total_tokens:,}",
        "",
        "  Tag Distribution:",
    ]
    for tag, cnt in sorted(tag_counter.items()):
        lines.append(f"    {tag:<35} {cnt:>8,}")
    lines.append("")
    lines.append("  Entity Type Counts (span-level):")
    for etype, cnt in sorted(entity_counter.items()):
        lines.append(f"    {etype:<35} {cnt:>8,}")
    lines.append(sep)

    report = "\n".join(lines)
    print(report)        
    log.info("Statistics for '%s':\n%s", label, report)


# 6.  Train / val / test split
def _dominant_entity(sentence: list[tuple]) -> str:
    """Return the most frequent entity type in a sentence, or ``'O'`` if none."""
    counts: Counter = Counter()
    for _, _, tag in sentence:
        if tag != "O" and "-" in tag:
            counts[tag.split("-", 1)[1]] += 1
    return counts.most_common(1)[0][0] if counts else "O"


def split_dataset(
    sentences: list[list[tuple]],
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
) -> tuple[list, list, list]:
    """
    Stratified shuffle-split into train / val / test subsets.

    Sentences are grouped by their dominant entity type so that rare
    entity classes (e.g., ``LOC-WARD``, ``LOC-COUNTRY``) are represented
    proportionally in every split rather than accidentally concentrated in
    one partition.

    Args:
        sentences   : full sentence list.
        train_ratio : fraction for training   (default 0.80).
        val_ratio   : fraction for validation (default 0.10).
        seed        : random seed for reproducibility.

    Returns:
        ``(train_sentences, val_sentences, test_sentences)`` tuple.
    """
    random.seed(seed)

    # Group sentences by dominant entity type
    buckets: dict[str, list] = defaultdict(list)
    for sent in sentences:
        buckets[_dominant_entity(sent)].append(sent)

    train: list = []
    val:   list = []
    test:  list = []

    for entity_type, bucket in buckets.items():
        random.shuffle(bucket)
        n = len(bucket)
        t_end = int(n * train_ratio)
        v_end = t_end + int(n * val_ratio)
        train.extend(bucket[:t_end])
        val.extend(bucket[t_end:v_end])
        test.extend(bucket[v_end:])
        log.debug(
            "Stratified split — %-20s  total=%5d  train=%5d  val=%4d  test=%4d",
            entity_type, n, len(bucket[:t_end]), len(bucket[t_end:v_end]), len(bucket[v_end:]),
        )

    # Final shuffle within each split so entity types are interleaved
    for split in (train, val, test):
        random.shuffle(split)

    log.info(
        "Stratified dataset split (seed=%d, %.0f/%.0f/%.0f): "
        "train=%d  val=%d  test=%d",
        seed, train_ratio * 100, val_ratio * 100,
        (1 - train_ratio - val_ratio) * 100,
        len(train), len(val), len(test),
    )
    return train, val, test


# 7.  Vocabulary builder
def build_and_save_vocab(
    train_sentences: list[list[tuple]],
    output_dir: str,
) -> tuple[dict, dict, dict]:
    """
    Build word / POS / NER-tag vocabularies from the training set and save
    them as JSON files to ``output_dir``.

    Args:
        train_sentences : tokenised training sentences.
        output_dir      : directory for ``vocab.json``, ``pos_vocab.json``,
                          ``tag_vocab.json``.

    Returns:
        ``(vocab, pos_vocab, tag_vocab)`` dictionaries mapping items to indices.
    """
    log.info("Building vocabularies from %d training sentences …", len(train_sentences))

    vocab:     dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    pos_vocab: dict[str, int] = {"<PAD>": 0}
    tag_vocab: dict[str, int] = {"<PAD>": 0}

    for sent in train_sentences:
        for word, pos, tag in sent:
            if word not in vocab:
                vocab[word] = len(vocab)
            if pos not in pos_vocab:
                pos_vocab[pos] = len(pos_vocab)
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)

    os.makedirs(output_dir, exist_ok=True)

    def _save(obj: dict, fname: str) -> None:
        fpath = os.path.join(output_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        log.info("  Saved %s  (%d entries)", fpath, len(obj))

    _save(vocab,     "vocab.json")
    _save(pos_vocab, "pos_vocab.json")
    _save(tag_vocab, "tag_vocab.json")

    log.info(
        "Vocab summary: words=%d  pos_tags=%d  ner_tags=%d",
        len(vocab), len(pos_vocab), len(tag_vocab),
    )
    return vocab, pos_vocab, tag_vocab

# 8.  Main preparation pipeline
def prepare_dataset(
    output_dir:  str = "data/processed",
    single_file: str | None = None,
    train_path:  str | None = None,
    val_path:    str | None = None,
    test_path:   str | None = None,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
) -> tuple[dict, dict, dict]:
    """
    Full dataset preparation pipeline.

    Supply **either** ``single_file`` (auto-split 80/10/10) **or** all three
    of ``train_path``, ``val_path``, ``test_path``.

    Args:
        output_dir  : directory for processed CoNLL files and vocab JSONs.
        single_file : path to a single CoNLL file to split automatically.
        train_path  : path to a pre-split training CoNLL file.
        val_path    : path to a pre-split validation CoNLL file.
        test_path   : path to a pre-split test CoNLL file.
        train_ratio : fraction used for training when auto-splitting.
        val_ratio   : fraction used for validation when auto-splitting.
        seed        : random seed for the shuffle when auto-splitting.

    Returns:
        ``(vocab, pos_vocab, tag_vocab)`` dictionaries.

    Raises:
        ValueError : if neither ``single_file`` nor the three split paths are
                     provided.
    """
    log.info("=" * 60)
    log.info("Myanmar NER — Dataset Preparation Pipeline")
    log.info("=" * 60)

    if single_file:
        log.info("Mode: single-file auto-split  (%.0f/%.0f/%.0f)",
                 train_ratio * 100, val_ratio * 100, (1 - train_ratio - val_ratio) * 100)
        all_sents = read_conll(single_file)
        all_sents = split_long_sentences(all_sents)
        validate_bioes(all_sents)
        train_sents, val_sents, test_sents = split_dataset(
            all_sents, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
        )
    elif train_path and val_path and test_path:
        log.info("Mode: pre-split files")
        train_sents = read_conll(train_path)
        val_sents   = read_conll(val_path)
        test_sents  = read_conll(test_path)
        for name, split in (("train", train_sents), ("val", val_sents), ("test", test_sents)):
            split[:] = split_long_sentences(split)
            validate_bioes(split)
        log.info("Loaded — train: %d  val: %d  test: %d",
                 len(train_sents), len(val_sents), len(test_sents))
    else:
        raise ValueError(
            "Provide either 'single_file' or all of 'train_path', 'val_path', 'test_path'."
        )

    dataset_stats(train_sents, "Train (BEFORE LOC transformation)")

    log.info("Transforming LOC tags …")
    train_sents = transform_loc_tags(train_sents)
    val_sents   = transform_loc_tags(val_sents)
    test_sents  = transform_loc_tags(test_sents)

    dataset_stats(train_sents, "Train (AFTER LOC transformation)")

    os.makedirs(output_dir, exist_ok=True)
    write_conll(train_sents, os.path.join(output_dir, "train.conll"))
    write_conll(val_sents,   os.path.join(output_dir, "val.conll"))
    write_conll(test_sents,  os.path.join(output_dir, "test.conll"))
    log.info("All processed CoNLL files saved to: %s/", output_dir)

    # Build and save vocab 
    vocab, pos_vocab, tag_vocab = build_and_save_vocab(train_sents, output_dir)

    log.info("Pipeline complete.")
    return vocab, pos_vocab, tag_vocab


# 9.  Entry point
if __name__ == "__main__":
    log = setup_logging(log_dir="logs", log_level=logging.DEBUG)

    prepare_dataset(
        single_file = "corpus/train1_dedup.conll",
        output_dir  = "data/processed",
        train_ratio = 0.8,
        val_ratio   = 0.1,
        seed        = 42,
    )