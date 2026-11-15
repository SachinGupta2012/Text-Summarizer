"""Stable summarization prompts shared by training and inference."""

from summarizer_ml.preprocessing.models import Domain, ProcessedRecord, QueryType
from summarizer_ml.training_data.models import ChatMessage, ChatRole

SYSTEM_PROMPT = (
    "You are a precise document summarization assistant. Preserve important facts, entities, "
    "decisions, obligations, and conclusions. Do not invent information. Return only the summary."
)

_DOMAIN_LABELS = {
    Domain.GENERAL_ARTICLE: "article",
    Domain.LONG_REPORT: "report",
    Domain.LEGAL_DOCUMENT: "legal document",
    Domain.MEETING: "meeting transcript",
    Domain.CUSTOMER_SUPPORT: "customer-support conversation",
}


def build_prompt(record: ProcessedRecord, source_text: str) -> tuple[ChatMessage, ...]:
    """Build one versioned, domain-aware summarization prompt."""

    domain_label = _DOMAIN_LABELS[record.domain]
    instruction = f"Summarize the following {domain_label}."
    if record.query_type is QueryType.SPECIFIC_MEETING and record.query:
        instruction = (
            f"Summarize the meeting transcript specifically for this request:\n{record.query}"
        )
    elif record.query_type is QueryType.GENERAL_MEETING:
        instruction = (
            "Summarize the meeting transcript, including key topics, decisions, and action items."
        )
    elif record.query_type is QueryType.DIALOGUE:
        instruction = (
            "Summarize the customer-support conversation, including the customer's issue, "
            "actions taken, and final resolution."
        )

    title = record.metadata.get("title")
    title_block = f"\n\nTitle: {title}" if isinstance(title, str) and title.strip() else ""
    user_content = f"{instruction}{title_block}\n\n<source>\n{source_text}\n</source>"
    return (
        ChatMessage(role=ChatRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=ChatRole.USER, content=user_content),
    )
