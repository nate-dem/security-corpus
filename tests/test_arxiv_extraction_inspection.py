from scripts.release.inspect_arxiv_extraction import body_sections_are_only_includes, diagnose


def test_import_and_subfile_are_review_leads():
    text = "## Abstract\n\\import{sections/}{abstract.tex}\n## Body\n\\subfile{sections/intro}"
    result = diagnose(text)
    assert result["include_commands"] == [
        r"\import{sections/}{abstract.tex}", r"\subfile{sections/intro}",
    ]
    assert not result["no_section_after_abstract"]
    assert result["include_contexts"]


def test_abstract_only_is_not_confused_with_math():
    result = diagnose("## Abstract\nTraining with $x^2$.")
    assert result["no_section_after_abstract"]
    assert result["include_commands"] == []
    assert not diagnose("## Abstract\nSummary\n## Introduction\nBody")["no_section_after_abstract"]


def test_include_only_body_is_structural_not_a_length_filter():
    stub = "## Abstract\nUseful abstract.\n## Body\n\\subfile{intro}\n\\import{sections/}{results}\n"
    assert body_sections_are_only_includes(stub)
    assert not body_sections_are_only_includes(stub + "An actual result.\n")
    assert not body_sections_are_only_includes("## Body\n```latex\n\\input{example}\n```\n")
    assert not body_sections_are_only_includes("## Body\n\\input{missing\nActual prose\n}\n")
    assert not body_sections_are_only_includes("## Abstract\n\\input{abstract}\n")
