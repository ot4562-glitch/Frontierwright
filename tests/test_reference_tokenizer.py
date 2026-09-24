import json
from pathlib import Path

from frontierwright.reference_tokenizer import (
    BASE_VOCAB_SIZE,
    decode_token_ids,
    encode_bytes,
    load_tokenizer_payload,
    train_byte_bpe,
    write_tokenizer_artifact,
)


def test_byte_bpe_is_deterministic_and_roundtrips() -> None:
    corpus = (b"banana bandana\n" * 20) + (b"frontierwright\n" * 10)

    first = train_byte_bpe(corpus, requested_vocab_size=280)
    second = train_byte_bpe(corpus, requested_vocab_size=280)

    assert first == second
    assert first
    assert first[0].token_id == BASE_VOCAB_SIZE

    payload = {
        "merges": [item.to_dict() for item in first],
    }
    sample = b"banana frontierwright bandana"
    encoded = encode_bytes(sample, payload)
    decoded = decode_token_ids(encoded, payload)

    assert decoded == sample
    assert len(encoded) < len(sample)


def test_byte_bpe_tie_breaks_pairs_lexicographically() -> None:
    # ab and ac occur equally often; (a,b) has the smaller pair IDs.
    merges = train_byte_bpe(b"abacabac", requested_vocab_size=257)

    assert len(merges) == 1
    assert merges[0].left == ord("a")
    assert merges[0].right == ord("b")


def test_tokenizer_artifact_is_content_deterministic(tmp_path: Path) -> None:
    corpus = b"hello hello hello world world\n" * 10

    first = write_tokenizer_artifact(
        tmp_path / "first",
        source_dataset_id="dataset-1",
        source_dataset_fingerprint="sha256:dataset",
        requested_vocab_size=272,
        max_training_bytes=4096,
        corpus=corpus,
    )
    second = write_tokenizer_artifact(
        tmp_path / "second",
        source_dataset_id="dataset-1",
        source_dataset_fingerprint="sha256:dataset",
        requested_vocab_size=272,
        max_training_bytes=4096,
        corpus=corpus,
    )

    assert first.artifact_id == second.artifact_id
    assert first.fingerprint == second.fingerprint
    assert first.vocab_size == second.vocab_size
    assert first.path.read_bytes() == second.path.read_bytes()

    payload = load_tokenizer_payload(first.path)
    assert payload["vocab_size"] == first.vocab_size
    assert payload["source_dataset_id"] == "dataset-1"
    assert json.loads(first.path.read_text(encoding="utf-8"))["merges"]
