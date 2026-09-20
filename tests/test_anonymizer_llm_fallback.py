import json

from guardian_policy_agent.service import anonymizer


def test_llm_echo_falls_back_to_deterministic_anonymization(monkeypatch):
    original = "Contact Jane at jane@example.com or +61 400 123 456."
    monkeypatch.setattr(
        anonymizer.llm_io,
        "chat",
        lambda *_args, **_kwargs: json.dumps({"anonymized": original}),
    )

    result = anonymizer.anonymize_freetext_llm(original, level=3)

    assert result != original
    assert "jane@example.com" not in result
    assert "+61 400 123 456" not in result
    assert "[EMAIL]" in result
    assert "[PHONE]" in result


def test_llm_residual_identifier_falls_back(monkeypatch):
    original = "Email Jane at jane@example.com."
    monkeypatch.setattr(
        anonymizer.llm_io,
        "chat",
        lambda *_args, **_kwargs: json.dumps(
            {"anonymized": "Please contact Jane at jane@example.com."}
        ),
    )

    result = anonymizer.anonymize_freetext_llm(original, level=3)

    assert "jane@example.com" not in result
    assert "[EMAIL]" in result


def test_safe_llm_anonymization_is_preserved(monkeypatch):
    original = "Email Jane at jane@example.com."
    safe = "Email [PERSON] at [EMAIL]."
    monkeypatch.setattr(
        anonymizer.llm_io,
        "chat",
        lambda *_args, **_kwargs: json.dumps({"anonymized": safe}),
    )

    assert anonymizer.anonymize_freetext_llm(original, level=3) == safe
