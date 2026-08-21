"""End-to-end smoke test with a mock solver model and mock judges (no API keys)."""

from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from abca.abca import abca

# Test runs write here instead of Inspect's default log dir (`./logs`, resolved against the
# cwd), which is where real audit runs land. Without this, every `pytest` invocation left a
# permanent 2-sample mock log among the real ones. Anchored to the repo rather than the cwd so
# it lands in the same place however pytest is invoked; `logs/` is gitignored, so this is too.
TEST_LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "tests"


def _mock_graders():
    # Plenty of canned grades so each scored sample's two judge calls are covered.
    out = [ModelOutput.from_content("mockllm/model", "GRADE: GREEN\nREASON: ok") for _ in range(50)]
    return [
        get_model("mockllm/model", custom_outputs=list(out)),
        get_model("mockllm/model", custom_outputs=list(out)),
    ]


def test_abca_end_to_end() -> None:
    [log] = inspect_eval(
        tasks=abca(grader_models=_mock_graders()),
        limit=2,
        model="mockllm/model",
        log_dir=str(TEST_LOG_DIR),
    )
    assert log.status == "success"
    assert log.results is not None
    assert log.results.scores is not None
    assert len(log.results.scores) > 0
