"""분석 결과를 화면이나 평가에서 쓰기 좋은 형태로 정리합니다."""


DEFAULT_CONTEXT_CHAR_LIMIT = 700


def truncate_context(context: str, max_chars: int = DEFAULT_CONTEXT_CHAR_LIMIT) -> str:
    """검색 문서 조각을 화면에 보여줄 수 있는 길이로 줄입니다."""
    clean_context = context.strip()
    if max_chars <= 0:
        return ""
    if len(clean_context) <= max_chars:
        return clean_context

    suffix = "\n...(중략)"
    if max_chars <= len(suffix):
        return clean_context[:max_chars]

    return clean_context[: max_chars - len(suffix)].rstrip() + suffix


def format_context_preview(
    contexts: list[str],
    *,
    max_context_chars: int = DEFAULT_CONTEXT_CHAR_LIMIT,
) -> str:
    """테스트와 디버깅 화면에서 보여줄 검색 문서 조각 미리보기를 만듭니다."""
    if not contexts:
        return ""

    rendered_contexts = [
        f"[{index}] {truncate_context(context, max_context_chars)}"
        for index, context in enumerate(contexts, start=1)
    ]
    return "\n\n---\n\n".join(rendered_contexts)
