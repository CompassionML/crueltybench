"""Tests for the data-collection policy applied to every call the eval makes.

Offline throughout: constructing a model client doesn't reach the network, and the routing
denials are replayed as the error objects OpenRouter actually returns (captured from a live
404) rather than provoked for real.
"""

import importlib

import pytest
from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import TaskState

from crueltybench import privacy
from crueltybench.crueltybench import TARGET_TEMPERATURE, adaptive_generate
from crueltybench.privacy import (
    GOOGLE_TIER_ENV,
    PROVIDER_POLICY,
    PROVIDER_ALLOWLIST_ENV,
    VERTEX_ENV,
    DataCollectionRefused,
    ModelAccessError,
    NoPrivateProviderError,
    RoutingNotAppliedError,
    check_data_policy,
    denial_message,
    denial_reason,
    enforce_privacy_routing,
    guarded_model_call,
    preflight,
    privacy_checked_model,
    provider_routing,
)
from crueltybench.scorer import _run_judges


@pytest.fixture(autouse=True)
def _shipped_config(monkeypatch: pytest.MonkeyPatch):
    """Default every test to the shipped config: no allowlist, no tier declaration."""
    for var in (PROVIDER_ALLOWLIST_ENV, GOOGLE_TIER_ENV, VERTEX_ENV):
        monkeypatch.delenv(var, raising=False)


class FakeStatusError(Exception):
    """Stand-in for the OpenAI client's status error: message text plus a parsed `.body`."""

    def __init__(self, message: str, body: object | None = None) -> None:
        super().__init__(message)
        self.body = body


# The real 404 payload OpenRouter returns when the data-collection filter empties the candidate
# list. Two independent tells — the human message and `failed_routing_step` — either of which is
# enough on its own, since only one survives some client wrappings.
DATA_POLICY_BODY = {
    "error": {
        "message": (
            "No endpoints found matching your data policy (Free model training). "
            "Configure: https://openrouter.ai/settings/privacy"
        ),
        "code": 404,
        "metadata": {
            "routing_funnel": [{"step": "Initial Endpoints", "endpoint_count": 1}],
            "failed_routing_step": "Filter by Data Policy",
        },
    }
}


def data_policy_error() -> FakeStatusError:
    return FakeStatusError(f"Error code: 404 - {DATA_POLICY_BODY}", DATA_POLICY_BODY)


class FakeOpenRouterError(Exception):
    """Stand-in for inspect_ai's `OpenRouterError`, raised when OpenRouter reports the failure in
    the body of a 200 rather than as an HTTP status.

    Two things about it matter here and neither is obvious: the payload lives on `.response`, not
    `.body`, and its `__str__` returns the **empty string** unless the error dict carries a
    `metadata` key (an upstream precedence bug). So on this path the message reaches us through
    `.response` alone.
    """

    def __init__(self, response: dict) -> None:
        super().__init__()
        self.response = response

    def __str__(self) -> str:
        return (
            f"Error {self.response['code']} - {self.response['message']}"
            if "metadata" in self.response
            else ""
        )


class TestRoutingPolicy:
    def test_denies_data_collection_and_fallbacks(self) -> None:
        # Both halves are load-bearing: without allow_fallbacks=False the deny is advisory.
        assert provider_routing() == {"data_collection": "deny", "allow_fallbacks": False}

    def test_no_allowlist_by_default(self) -> None:
        # The target model is whatever the user names, so a baked-in `only` would refuse most runs.
        assert "only" not in provider_routing()

    def test_allowlist_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PROVIDER_ALLOWLIST_ENV, "anthropic, openai ,google-vertex")
        routing_block = provider_routing()
        assert routing_block["only"] == ["anthropic", "openai", "google-vertex"]
        # Narrowing, not replacing: the data-collection filter still applies.
        assert routing_block["data_collection"] == "deny"

    def test_caller_routing_is_narrowed_not_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A caller's own `-M provider=...` survives; this eval's keys win where they overlap."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        class Built:
            def __init__(self) -> None:
                self.api = type("Api", (), {})()
                self.api.provider = {"order": ["anthropic"], "allow_fallbacks": True}

        built = Built()
        enforce_privacy_routing("openrouter/anthropic/claude-opus-5", built)  # type: ignore[arg-type]
        # `order` kept, `allow_fallbacks` overridden — routing can be tightened, never loosened.
        assert built.api.provider == {
            "order": ["anthropic"],
            "allow_fallbacks": False,
            "data_collection": "deny",
        }


class TestPrivacyRoutedModel:
    def test_routing_reaches_the_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        model = privacy_checked_model("openrouter/anthropic/claude-opus-5")
        # The provider block the request will actually carry, read back off the built client.
        assert model.api.provider == {"data_collection": "deny", "allow_fallbacks": False}

    def test_untouched_for_non_openrouter(self) -> None:
        model = privacy_checked_model("mockllm/model")
        assert getattr(model.api, "provider", None) is None

    def test_fails_loud_if_the_arg_stops_being_honoured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An Inspect upgrade that drops the `provider` arg must break the run, not the guarantee.

        Silently falling back to OpenRouter's default routing is the one failure mode a privacy
        control can't have, so the routing is verified on the built client rather than assumed.
        """

        class Unrouted:
            api = type("Api", (), {})()  # no `provider`: inspect_ai stopped honouring it

        monkeypatch.setattr(privacy, "get_model", lambda *a, **k: Unrouted())
        with pytest.raises(RoutingNotAppliedError, match="no `provider` routing attribute"):
            privacy_checked_model("openrouter/anthropic/claude-opus-5")


class TestDenialDetection:
    def test_data_policy_rejection(self) -> None:
        assert denial_reason(data_policy_error()) == "data_policy"

    def test_metadata_alone_is_enough(self) -> None:
        # Some wrappings keep only the parsed body; the `failed_routing_step` still identifies it.
        assert denial_reason(FakeStatusError("Error code: 404", DATA_POLICY_BODY)) == "data_policy"

    def test_empty_candidate_list_is_ambiguous_without_an_allowlist(self) -> None:
        # "No endpoints found for X" is also what a mistyped model id looks like, so claiming a
        # privacy denial here would be a misleading diagnostic.
        exc = FakeStatusError("Error code: 404 - No endpoints found for vendor/typo.")
        assert denial_reason(exc) is None

    def test_empty_candidate_list_is_the_allowlist_when_one_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PROVIDER_ALLOWLIST_ENV, "anthropic")
        exc = FakeStatusError("Error code: 404 - No endpoints found for openrouter/x/y.")
        assert denial_reason(exc) == "allowlist"

    def test_survives_an_error_that_stringifies_to_nothing(self) -> None:
        # The refusal this module exists to explain must not arrive as a blank error: without the
        # payload being read off `.response`, this denial is invisible and propagates unexplained.
        exc = FakeOpenRouterError(
            {
                "code": 404,
                "message": (
                    "No endpoints found matching your data policy (Zero data retention). "
                    "Configure: https://openrouter.ai/settings/privacy"
                ),
            }
        )
        assert str(exc) == "", "fixture must reproduce the empty-string stringification"
        assert denial_reason(exc) == "data_policy"

    def test_unrelated_failures_are_not_claimed(self) -> None:
        assert denial_reason(FakeStatusError("Error code: 401 - No auth credentials found")) is None
        assert denial_reason(TimeoutError("read timeout")) is None


class TestDenialMessage:
    def test_explains_the_refusal(self) -> None:
        msg = denial_message("openrouter/vendor/model", "data_policy")
        assert "openrouter/vendor/model" in msg
        assert "data_collection" in msg and "allow_fallbacks" in msg
        # The user's first question on seeing this is whether the prompts already went out.
        assert "Nothing leaked" in msg
        assert "before the prompt reached" in msg
        assert "https://openrouter.ai/settings/privacy" in msg

    def test_allowlist_refusal_names_the_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PROVIDER_ALLOWLIST_ENV, "anthropic")
        msg = denial_message("openrouter/vendor/model", "allowlist")
        assert PROVIDER_ALLOWLIST_ENV in msg
        assert "anthropic" in msg

    def test_never_offers_to_drop_the_policy(self) -> None:
        msg = denial_message("openrouter/vendor/model", "data_policy")
        assert "Dropping the policy is not one of the options." in msg


class TestPrivateRoute:
    @pytest.mark.asyncio
    async def test_converts_a_routing_denial(self) -> None:
        with pytest.raises(NoPrivateProviderError) as caught:
            async with guarded_model_call("openrouter/vendor/model"):
                raise data_policy_error()
        assert "Refusing to run openrouter/vendor/model" in str(caught.value)
        # The provider's own error is kept as the cause rather than swallowed.
        assert isinstance(caught.value.__cause__, FakeStatusError)

    @pytest.mark.asyncio
    async def test_passes_other_failures_through(self) -> None:
        # A rate limit or an outage must stay retryable/diagnosable as itself.
        with pytest.raises(FakeStatusError):
            async with guarded_model_call("openrouter/vendor/model"):
                raise FakeStatusError("Error code: 429 - rate limited")


class TestCallSitesAreCovered:
    """Every model the eval chooses is policy-checked; nothing reaches a provider unvetted."""

    @pytest.mark.asyncio
    async def test_target_is_preflighted_and_still_uses_inspects_own_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The solver must delegate to Inspect's `generate`, not replace it.

        Rebuilding the model from its id would drop `-M` args, `--model-base-url` and the eval's
        GenerateConfig, so the routing is pinned on the model Inspect already built and the
        generation itself is left alone.
        """
        # Stands in for the active model an eval would have set.
        monkeypatch.setenv("INSPECT_EVAL_MODEL", "mockllm/model")
        delegated: list[dict] = []

        async def fake_generate(state, **kwargs):
            delegated.append(kwargs)
            return state

        state = TaskState(
            model=ModelName("mockllm/model"),
            sample_id="fixture_harm_one",
            epoch=0,
            input="...",
            messages=[ChatMessageUser(content="...")],
        )
        solve = adaptive_generate(TARGET_TEMPERATURE, ["mockllm/model"])
        state = await solve(state, fake_generate)  # type: ignore[arg-type]
        assert delegated, "solver must call Inspect's generate rather than generating itself"
        # The verdicts ride along on the sample, so a log says which policy produced the numbers.
        assert state.metadata["privacy"]["target"]["model"] == "mockllm/model"
        assert state.metadata["privacy"]["judges"][0]["model"] == "mockllm/model"

    @pytest.mark.asyncio
    async def test_refuses_when_the_active_model_is_not_the_one_the_sample_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Vouch for one model and call another, and nothing downstream would notice.

        Both halves of the policy key off the sample's model id: `preflight` decides from it
        whether the provider may be sent benchmark material, and `enforce_privacy_routing` pins the
        OpenRouter deny-flag onto a live client. If the model Inspect actually generates with is a
        different one, the run would call an unvouched provider — or an OpenRouter client that
        never got the flag — while the log records `privacy: allowed` for every sample. It fails in
        the only direction that can't be detected afterwards, so it must stop the run.
        """
        monkeypatch.setenv("INSPECT_EVAL_MODEL", "mockllm/model")
        delegated: list[dict] = []

        async def fake_generate(state, **kwargs):
            delegated.append(kwargs)
            return state

        state = TaskState(
            model=ModelName("openrouter/vendor/some-other-model"),
            sample_id="fixture_harm_one",
            epoch=0,
            input="...",
            messages=[ChatMessageUser(content="...")],
        )
        solve = adaptive_generate(TARGET_TEMPERATURE, ["mockllm/model"])
        with pytest.raises(RuntimeError) as caught:
            await solve(state, fake_generate)  # type: ignore[arg-type]
        msg = str(caught.value)
        assert "mockllm/model" in msg and "openrouter/vendor/some-other-model" in msg
        assert not delegated, "must refuse before generating, not after"

    @pytest.mark.asyncio
    async def test_judges_are_policy_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Judge prompts carry the tier anchors, so this is the call site that matters most.
        seen: list[str] = []
        monkeypatch.setattr(
            "crueltybench.scorer.privacy_checked_model",
            lambda model_id, role="judge", **kw: (
                seen.append(model_id),
                privacy.get_model(model_id, **kw),
            )[1],
        )
        await _run_judges(["mockllm/model"], "grade this", ("green", "yellow", "red"))
        assert seen == ["mockllm/model"]


class TestDirectProviderPolicy:
    """Off OpenRouter there is no policy API, so the verdict comes from a maintained list."""

    def test_providers_whose_terms_exclude_training_are_allowed(self) -> None:
        for model_id in ("anthropic/claude-opus-5", "openai/gpt-5.6-sol", "bedrock/anything"):
            record = check_data_policy(model_id)
            assert record["verdict"] == "allowed"
            assert record["basis"]  # the reason is recorded, not just the verdict

    def test_unknown_provider_is_refused_not_assumed_safe(self) -> None:
        # Assuming safety for anything absent from the list is how the answer key leaks. The slug
        # is deliberately fictional: a real one can be added to PROVIDER_POLICY later and quietly
        # turn this into a test of nothing (which is how it failed once already).
        with pytest.raises(DataCollectionRefused, match="not in CrueltyBench's provider policy"):
            check_data_policy("notarealprovider/some-model")

    def test_providers_that_train_by_default_stay_refused(self) -> None:
        """These were read and found adverse, not merely unread.

        Recorded as a test so re-adding one takes new evidence and a deliberate edit here, rather
        than someone assuming the omission was an oversight. See the note under PROVIDER_POLICY.
        """
        for provider in ("deepseek", "moonshot", "mistral", "sambanova"):
            with pytest.raises(DataCollectionRefused):
                check_data_policy(f"{provider}/some-model")

    def test_every_entry_records_its_basis_and_review_date(self) -> None:
        # A verdict with no stated basis can't be re-checked, which is the whole point of the list.
        for name, policy in PROVIDER_POLICY.items():
            assert policy.basis, f"{name} has no basis"
            assert policy.reviewed, f"{name} has no review date"

    def test_refusal_names_the_stake_for_judges(self) -> None:
        with pytest.raises(DataCollectionRefused) as caught:
            check_data_policy("notarealprovider/some-model", "judge")
        # A judge leaks strictly more than a target, and the message should say so.
        assert "tier anchors" in str(caught.value)

    def test_google_needs_a_tier_declaration(self) -> None:
        with pytest.raises(DataCollectionRefused) as caught:
            check_data_policy("google/gemini-3.7-flash")
        msg = str(caught.value)
        assert GOOGLE_TIER_ENV in msg
        assert "free AI Studio" in msg  # says *why* the key alone isn't enough

    def test_declared_paid_tier_is_allowed_and_recorded(self, monkeypatch) -> None:
        monkeypatch.setenv(GOOGLE_TIER_ENV, "PAID")  # case-insensitive
        record = check_data_policy("google/gemini-3.7-flash")
        assert record["verdict"] == "allowed"
        # The declaration lands in the record so a published number carries the claim behind it.
        assert record["declared"] == "paid"

    def test_a_meaningless_declaration_is_rejected(self, monkeypatch) -> None:
        monkeypatch.setenv(GOOGLE_TIER_ENV, "yes")
        with pytest.raises(DataCollectionRefused, match="not one of"):
            check_data_policy("google/gemini-3.7-flash")

    def test_vertex_needs_no_declaration(self, monkeypatch) -> None:
        # Vertex never trains on inputs, and this env var is how google-genai is switched to it,
        # so its presence answers the tier question on its own.
        monkeypatch.setenv(VERTEX_ENV, "true")
        assert check_data_policy("google/gemini-3.7-flash")["verdict"] == "allowed"

    def test_mock_models_are_exempt(self) -> None:
        # Nothing leaves the machine, so the offline test suite needs no declarations.
        assert check_data_policy("mockllm/model")["verdict"] == "allowed"


class TestPreflight:
    def test_records_every_model_before_the_run(self) -> None:
        record = preflight("mockllm/model", ["mockllm/model"])
        assert record["target"]["model"] == "mockllm/model"
        assert len(record["judges"]) == 1
        # Stamped so a stale policy list is visible in the log rather than invisible.
        assert record["policy_reviewed"]

    def test_a_bad_judge_stops_the_run_not_just_a_bad_target(self) -> None:
        # Judges are checked too: a panel that leaks is worse than a target that does.
        with pytest.raises(DataCollectionRefused):
            preflight("mockllm/model", ["notarealprovider/some-model"])


class TestAccessFailures:
    """Anything wrong with the key should say which model, which role, and what to do."""

    def test_missing_key_is_explained(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ModelAccessError) as caught:
            # An id no other test has built, so get_model's memo cache can't mask the failure.
            privacy_checked_model("openrouter/vendor/never-constructed", "judge")
        msg = str(caught.value)
        assert "OPENROUTER_API_KEY" in msg
        assert "judge" in msg  # which seat failed, not just "a model"

    @pytest.mark.asyncio
    async def test_rejected_credentials_are_explained(self) -> None:
        with pytest.raises(ModelAccessError) as caught:
            async with guarded_model_call("anthropic/claude-opus-5", "judge"):
                raise FakeStatusError("Error code: 401 - invalid api key")
        assert "ANTHROPIC_API_KEY" in str(caught.value)

    @pytest.mark.asyncio
    async def test_out_of_credit_is_distinguished_from_a_bad_key(self) -> None:
        with pytest.raises(ModelAccessError) as caught:
            async with guarded_model_call("openrouter/x/y", "target"):
                raise FakeStatusError("Error code: 402 - Insufficient credits")
        assert "credit" in str(caught.value).lower()

    @pytest.mark.asyncio
    async def test_transient_failures_stay_transient(self) -> None:
        # A rate limit must reach Inspect as itself so it gets retried, not reported as a key fault.
        with pytest.raises(FakeStatusError):
            async with guarded_model_call("openrouter/x/y", "target"):
                raise FakeStatusError("Error code: 429 - rate limited")

    @pytest.mark.asyncio
    async def test_a_generation_id_is_not_mistaken_for_a_status_code(self) -> None:
        """An id that merely contains "403" must not be read as a rejected key.

        OpenRouter generation ids embed a Unix timestamp, so a bare digit match turns roughly one
        transient failure in forty into a confident "check your API key" — and buries the real
        fault, which here is an upstream 502.
        """
        body = {
            "error": {"code": 502, "message": "Upstream provider temporarily unavailable"},
            "id": "gen-1774031234-8sKqWm",
        }
        with pytest.raises(FakeStatusError):
            async with guarded_model_call("openrouter/x/y", "judge"):
                raise FakeStatusError("Provider returned error", body)
