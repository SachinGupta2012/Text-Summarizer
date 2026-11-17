"""Batched chat-template tokenization with completion-only labels."""

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Protocol

from summarizer_ml.fine_tuning.models import TokenizedTrainingExample
from summarizer_ml.training_data.models import ChatMessage, SftExample


class TrainingTokenizer(Protocol):
    """Tokenizer surface required for Phase 6 caching."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> object: ...

    def __call__(
        self,
        text: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> object: ...


def _messages(messages: tuple[ChatMessage, ...]) -> list[dict[str, str]]:
    return [{"role": item.role.value, "content": item.content} for item in messages]


def _render(
    tokenizer: TrainingTokenizer,
    messages: tuple[ChatMessage, ...],
    *,
    add_generation_prompt: bool,
) -> str:
    value = tokenizer.apply_chat_template(
        _messages(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(value, str) or not value:
        raise ValueError("Tokenizer returned invalid chat-template text.")
    return value


def _rows(raw: object, expected: int) -> tuple[tuple[int, ...], ...]:
    if isinstance(raw, Mapping):
        raw = raw.get("input_ids")
    tolist = getattr(raw, "tolist", None)
    if callable(tolist):
        raw = tolist()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != expected:
        raise ValueError("Tokenizer returned invalid token rows.")
    result: list[tuple[int, ...]] = []
    for raw_row in raw:
        if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
            raise ValueError("Tokenizer returned an invalid token row.")
        row: list[int] = []
        for value in raw_row:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("Tokenizer returned a non-integer token ID.")
            row.append(int(value))
        result.append(tuple(row))
    return tuple(result)


def tokenize_sft_batch(
    examples: Sequence[SftExample],
    tokenizer: TrainingTokenizer,
    max_sequence_tokens: int,
) -> tuple[TokenizedTrainingExample, ...]:
    """Tokenize once and mask every prompt token from the training loss."""

    if not examples:
        return ()
    prompt_texts = [
        _render(tokenizer, example.prompt, add_generation_prompt=True) for example in examples
    ]
    full_texts = [
        _render(
            tokenizer,
            (*example.prompt, *example.completion),
            add_generation_prompt=False,
        )
        for example in examples
    ]
    encoded = tokenizer(
        [*prompt_texts, *full_texts],
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    rows = _rows(encoded, len(examples) * 2)
    prompt_rows = rows[: len(examples)]
    full_rows = rows[len(examples) :]
    tokenized: list[TokenizedTrainingExample] = []
    for prompt_ids, full_ids in zip(prompt_rows, full_rows, strict=True):
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Generation prompt is not a prefix of the completed conversation.")
        if len(full_ids) > max_sequence_tokens:
            raise ValueError("Phase 6 example exceeds the configured sequence limit.")
        completion_ids = full_ids[len(prompt_ids) :]
        if not completion_ids:
            raise ValueError("Phase 6 example contains no completion tokens.")
        tokenized.append(
            TokenizedTrainingExample(
                input_ids=full_ids,
                attention_mask=(1,) * len(full_ids),
                labels=(-100,) * len(prompt_ids) + completion_ids,
                length=len(full_ids),
            )
        )
    return tuple(tokenized)

