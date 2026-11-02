"""Source-only quality review, separate from the high-recall v3 screening prompt.

Same diagnostic labels/policy; no claim that another model call is independent
human validation. Quality evidence is generated before the categorical verdicts.
"""

from bisect import bisect_right
import json

from . import rubric_v3 as base

VERSION = "content-critic-v1"
QUALITY_CONCERNS = base.QUALITY_CONCERNS
segments = base.segments
parse_response = base.parse_response
SYSTEM = (
    base.SYSTEM
    + """

Your role in this call is the quality reviewer of a candidate passage. You see
only the source, not another model's verdict. Check it against these distinctions:
1. What SPECIFIC computing/security mechanism, procedure, tradeoff or analysis can
   a reader learn? Topic names, a list of risks, an event announcement, promised
   tutorials, company history, and general claims of importance do not explain
   the topic. Use limited or none when the mechanism is merely asserted.
2. Does the passage actually teach a computing/security subject? Physical access
   prevention (cameras, locks) is not cybersecurity without a digital mechanism.
   Scientific estimates, mathematical identities or engineering applications are
   not software instruction merely because a tool or algorithm is named. Actual
   explanation of code, programming semantics or implemented algorithms qualifies.
3. Read the supplied commands and comparisons literally. Do not silently restore
   lost paths, delimiters, formulas, code blocks, missing example values, or omitted
   steps. If two supposedly different example values are identical, or commands
   have been run together, flag the concrete lost distinction. A missing image
   only matters when the prose cannot supply the essential explanation.
4. For security benefits, look for the claimed causal mechanism. A record of file
   activity does not by itself explain access enforcement or theft prevention.
   Steps that adjust one OS subsystem need an actual connection to the problem
   being solved. Do not assume a detailed numbered list is technically sound.
5. Useful paragraphs mixed with substantial sales copy, unrelated material or a
   broken extraction require section review. Do not silently judge only the best
   paragraph and mark the entire supplied passage focused and usable.

Conversely, do not demand a complete textbook or externally checked proof from
all documents. Short, specific explanations, readable speech, ordinary wrappers,
and valid code are valuable. No identified concern means only that this pass
found no concrete concern. Do not invent problems to sound critical.

Write content_form and rationale FIRST. Rationale must name the concrete thing
learned, or explain what essential information is actually missing. Then write
quality_checks and the remaining labels. Your rationale must support those
labels; identifying only a topic or a promised benefit cannot justify substantive.
"""
)


def schema(focal):
    original = base.schema(focal)
    order = [
        "content_form",
        "rationale",
        "quality_checks",
        "domain_relevance",
        "technical_substance",
        "text_usability",
        "text_language",
        "mixed_content",
    ]
    return {
        **original,
        "properties": {k: original["properties"][k] for k in order},
        "required": order,
    }


def examples():
    # Authored contrastive examples; none is a development/evaluation source record.
    return [
        (
            "web",
            "A mutex allows at most one thread to execute the protected update at a time. The lock must cover the read-modify-write sequence, not just the assignment.",
            "computing",
            "substantive",
            "usable",
            "no",
            {},
            "Explains why locking the whole read-modify-write operation prevents competing updates.",
        ),
        (
            "web",
            "Join our cloud security seminar. Learn about identity, ransomware, zero trust and the latest threats. Register now for a safer future.",
            "mention_only",
            "none",
            "usable",
            "yes",
            {},
            "Advertises security topics but supplies no explanation of a mechanism, procedure or analysis.",
        ),
        (
            "web",
            "The experiment estimates battery charge from measured voltage using a numerical model in MATLAB. Results show lower prediction error than prior models.",
            "other",
            "limited",
            "usable",
            "no",
            {},
            "Reports a physical estimation result; names a computing tool without explaining its implementation or software operation.",
        ),
        (
            "web",
            "To connect, run:\n\nThe two examples differ in the hostname:\nserver.example\nserver.example",
            "computing",
            "limited",
            "partly_usable",
            "no",
            {
                "damaged_or_missing_content": "The connection command is missing and both purportedly different hostnames are identical."
            },
            "The procedure cannot be followed as written; the promised command and contrasting hostnames are absent.",
        ),
        (
            "youtube",
            "um the audit log records who opened the file and when so after an incident we can use that timeline to investigate access but recording events does not prevent someone opening the file",
            "security",
            "substantive",
            "usable",
            "no",
            {},
            "Explains the forensic use of an access timeline and explicitly distinguishes logging from prevention.",
        ),
        (
            "web",
            "Our plugin records every file-opening event on a ledger. This guarantees that nobody can steal your files. Buy the premium subscription today.",
            "security",
            "limited",
            "usable",
            "yes",
            {
                "unsupported_security_guarantee": "Recording file-opening events does not establish the asserted guarantee that theft is impossible."
            },
            "Describes event recording but asserts theft prevention without an enforcement mechanism, alongside a sales pitch.",
        ),
    ]


def render(tokenizer, text, start, end, context_chars, kind):
    messages = [{"role": "system", "content": SYSTEM}]
    for (
        source,
        example,
        domain,
        substance,
        usability,
        mixed,
        concerns,
        reason,
    ) in examples():
        response = {
            "content_form": "Candidate passage",
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
                "evidence_ids": [1] if substance == "substantive" else [],
            },
            "text_usability": {"label": usability, "evidence_ids": [1]},
            "text_language": "english",
            "mixed_content": mixed,
        }
        messages.extend(
            [
                {"role": "user", "content": base.payload(example, source)},
                {"role": "assistant", "content": json.dumps(response)},
            ]
        )
    messages.append(
        {
            "role": "user",
            "content": base.payload(
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
            focal_count = len(
                tokenizer.encode(text[start:end], add_special_tokens=False)
            )
            if (
                focal_count <= focal_tokens
                and len(ids) + max_output_tokens <= max_model_len
            ):
                break
            if end - start <= 1:
                raise ValueError("No room for source text within context budget")
            end = start + (end - start) // 2
        spans.append(
            {
                "start": start,
                "end": end,
                "focal_model_tokens": focal_count,
                "prompt": prompt,
                "prompt_token_ids": ids,
                "response_schema": schema(text[start:end]),
            }
        )
        start = end
    return spans
