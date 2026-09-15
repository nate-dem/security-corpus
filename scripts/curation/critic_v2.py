"""Source-only second assessment of the bounded first batch.

The prompt addresses observed errors; this is development, not an independent
quality estimate. v4's labels, strict evidence parser and selection policy stay
unchanged. Synthetic demonstrations are authored here, not copied from controls.
"""

from bisect import bisect_right
import json

from . import rubric_v3, rubric_v4

VERSION = "content-critic-v2-literal-source"
QUALITY_CONCERNS = rubric_v4.QUALITY_CONCERNS
segments = rubric_v4.segments
schema = rubric_v4.schema
parse_response = rubric_v4.parse_response

SYSTEM = (
    rubric_v4.SYSTEM
    + """

This is a separate source-only quality assessment. No prior verdict is supplied.
Before deciding, check the literal source from beginning to end:
- Trace each procedure's required commands, arguments, option names and values.
  Is the actual operation present, or only a reference to a description, link,
  screenshot or absent example? Intact later paragraphs do not repair an earlier
  broken declaration. A continuation in the supplied context is not missing.
- Compare the mechanism with the stated impact. Reusing a login value and running
  machine instructions are distinct mechanisms. Similar vocabulary does not make
  different vulnerability classes interchangeable. Cite the specific mismatch.
- Assess substantive explanatory portions and substantial unrelated/support copy
  separately. A useful paragraph cannot make the entire passage unmixed.
- Do not invent a demand for depth: a short definition explaining operation, a
  complete UI procedure, or an abstract with an actual method and result can be
  substantive. If your rationale describes such content as present, do not call
  it limited merely for being short, introductory, an abstract or a manual.
- An attributed or explicitly qualified historical observation is not a universal
  security guarantee. Missing citations alone are not a technical error.

Keep rationale concise: name the actual lesson, then any specific defect. Output
only the schema. Do not speculate about hidden content or silently correct it.
"""
)


def examples():
    return [
        (
            "web",
            "A bounded FIFO queue blocks a producer when it is full. Removing one item frees a slot and allows a waiting producer to continue.",
            "computing",
            "substantive",
            "usable",
            {},
            "Explains the capacity condition that blocks and resumes a producer; shortness does not remove this mechanism.",
        ),
        (
            "web",
            "int open_gate(const char *label, int\nThe flags argument accepts these options:\nIf set, the request is asynchronous.\nThe routine returns zero on success.",
            "computing",
            "substantive",
            "partly_usable",
            {
                "damaged_or_missing_content": "The declaration is truncated and the described asynchronous flag has no name; the return-value sentence does not restore either."
            },
            "Explains return behavior but leaves the function declaration and required flag incomplete.",
        ),
        (
            "web",
            "A stolen one-use login token can be exchanged to sign into an account. Therefore stealing this token causes the operating system to execute arbitrary native instructions.",
            "security",
            "substantive",
            "usable",
            {
                "apparent_technical_error": "Exchanging a login token explains account access, not execution of native instructions; the conclusion substitutes a different mechanism."
            },
            "Describes account impersonation, then asserts a code-execution impact that does not follow from the stated mechanism.",
        ),
        (
            "youtube",
            "um the worker takes the mutex before changing the shared counter and releases it afterwards so another worker must wait rather than update it at the same time",
            "computing",
            "substantive",
            "usable",
            {},
            "The speech explains exclusion around a shared-counter update; filler words do not obscure the operation.",
        ),
    ]


def demonstration(example):
    kind, text, domain, substance, usability, concerns, reason = example
    value = {
        "content_form": "explanatory passage",
        "rationale": reason,
        "quality_checks": {
            k: {
                "status": "identified" if k in concerns else "not_identified",
                "evidence_ids": [1] if k in concerns else [],
                "reason": concerns.get(k, ""),
            }
            for k in QUALITY_CONCERNS
        },
        "domain_relevance": {"label": domain, "evidence_ids": [1]},
        "technical_substance": {
            "label": substance,
            "evidence_ids": [len(segments(text))],
        },
        "text_usability": {"label": usability, "evidence_ids": [1]},
        "text_language": "english",
        "mixed_content": "no",
    }
    return [
        {"role": "user", "content": rubric_v3.payload(text, kind)},
        {"role": "assistant", "content": json.dumps(value)},
    ]


def render(tokenizer, text, start, end, context_chars, kind):
    messages = [{"role": "system", "content": SYSTEM}]
    for example in examples():
        messages.extend(demonstration(example))
    messages.append(
        {
            "role": "user",
            "content": rubric_v3.payload(
                text[start:end],
                kind,
                text[max(0, start - context_chars) : start],
                text[end : end + context_chars],
            ),
        }
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return prompt, tokenizer.encode(prompt, add_special_tokens=False)


def make_spans(
    text,
    tokenizer,
    kind,
    *,
    focal_tokens=4096,
    context_chars=1024,
    max_model_len=8192,
    max_output_tokens=1024,
):
    if (
        kind not in {"web", "youtube"}
        or focal_tokens < 1
        or context_chars < 0
        or not 0 < max_output_tokens < max_model_len
    ):
        raise ValueError("Invalid source kind or inference budget")
    if not text.strip():
        return []
    ends = sorted(
        {
            end
            for _, end in tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            if end > 0
        }
        | {len(text)}
    )
    spans, start = [], 0
    while start < len(text):
        end = ends[min(bisect_right(ends, start) + focal_tokens - 1, len(ends) - 1)]
        if end < len(text):
            boundary = max(
                text.rfind("\n", max(start, end - 256), end),
                text.rfind(" ", max(start, end - 256), end),
            )
            if boundary > start:
                end = boundary + 1
        while True:
            prompt, ids = render(tokenizer, text, start, end, context_chars, kind)
            count = len(tokenizer.encode(text[start:end], add_special_tokens=False))
            if count <= focal_tokens and len(ids) + max_output_tokens <= max_model_len:
                break
            if end - start <= 1:
                raise ValueError("No room for source text within context budget")
            end = start + (end - start) // 2
        spans.append(
            {
                "start": start,
                "end": end,
                "focal_model_tokens": count,
                "prompt": prompt,
                "prompt_token_ids": ids,
                "response_schema": schema(text[start:end]),
            }
        )
        start = end
    return spans
