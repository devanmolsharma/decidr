import pytest

from decidr import Backend, Client, DecisionError
from decidr.prefix import Candidate, build_prefix_messages, group_by_context, match_step


def _resp(token, alts):
    return {
        "content": token,
        "logprobs": [{"token": token, "logprob": alts.get(token, 0.0),
                      "top_logprobs": [{"token": t, "logprob": lp} for t, lp in alts.items()]}],
    }


class QueuedBackend(Backend):
    """Backend stubbed to a fixed sequence of responses, one per call, so a
    full multi-step prefix resolution can be driven without a live server.
    Records every call's messages for inspection. Responses are consumed
    by whichever request arrives first -- concurrent requests within one
    round don't have a fixed order, so tests only assert on the total
    call count and the resulting Decision, never on response-to-request
    pairing across a round."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.sent_messages = []
        self._lock = __import__("threading").Lock()

    def chat(self, model, messages, max_tokens=1):
        with self._lock:
            self.sent_messages.append(messages)
            if not self._responses:
                raise AssertionError("QueuedBackend ran out of queued responses")
            return self._responses.pop(0)


def QueuedFakeClient(responses, **kwargs):
    client = Client(model="test", backend=QueuedBackend(responses), cache=False, **kwargs)
    client.sent_bodies = client.backend.sent_messages
    return client


def test_no_collision_resolves_in_one_step():
    candidates = [Candidate(option_id="access", remaining="access"),
                  Candidate(option_id="billing", remaining="billing")]
    match_step(candidates, {"access": -0.1, "billing": -5.0, "shipping": -6.0})
    assert all(c.done for c in candidates)
    assert candidates[0].remaining == "" and candidates[0].logprob_sum == -0.1
    assert candidates[1].remaining == "" and candidates[1].logprob_sum == -5.0


def test_shared_prefix_stays_open_for_the_next_step():
    candidates = [Candidate(option_id="access_denied", remaining="access_denied"),
                  Candidate(option_id="access_revoked", remaining="access_revoked")]
    match_step(candidates, {"access": -0.01})
    assert not candidates[0].done and not candidates[1].done
    assert candidates[0].remaining == "_denied"
    assert candidates[1].remaining == "_revoked"
    assert candidates[0].logprob_sum == candidates[1].logprob_sum == -0.01
    assert candidates[0].consumed == candidates[1].consumed == "access"


def test_full_length_is_walked_when_no_competition_remains_early():
    c = Candidate(option_id="access_denied", remaining="access_denied")
    match_step([c], {"access": -0.01})
    assert c.remaining == "_denied" and not c.done
    match_step([c], {"_denied": -0.02, "_revoked": -0.5})
    assert c.done and c.remaining == ""
    assert c.logprob_sum == pytest.approx(-0.03)


def test_overshoot_is_not_treated_as_a_match():
    c = Candidate(option_id="no", remaining="no")
    match_step([c], {"none": -0.1, "not": -0.2})
    assert c.unscored_reason is not None
    assert c.remaining == "no"


def test_longest_valid_match_wins():
    c = Candidate(option_id="access_denied", remaining="_denied")
    match_step([c], {"_den": -1.0, "_denied": -0.05})
    assert c.done
    assert c.logprob_sum == -0.05


def test_no_matching_continuation_marks_unscored_with_spec_wording():
    c = Candidate(option_id="sales", remaining="sales")
    match_step([c], {"access": -0.1, "billing": -0.2})
    assert c.unscored_reason == 'no returned token matched the next part of "sales" ("sales" remaining)'
    assert c.logprob_sum == 0.0


def test_tokens_are_matched_exactly_as_returned_no_stripping():
    # A token with real leading whitespace must not be treated as if it
    # had none -- stripping it before matching could cause a false match
    # against a candidate's remaining text that the raw token wouldn't.
    c = Candidate(option_id="billing", remaining="billing")
    match_step([c], {" billing": -0.1})
    assert c.unscored_reason is not None  # " billing" does not prefix "billing"


def test_done_candidates_are_skipped_on_later_steps():
    c = Candidate(option_id="a", remaining="")
    match_step([c], {"a": -99.0})
    assert c.logprob_sum == 0.0


def test_group_by_context_batches_shared_prefixes_and_skips_done():
    done = Candidate(option_id="access", remaining="")
    open_a = Candidate(option_id="access_denied", remaining="_denied", consumed="access")
    open_b = Candidate(option_id="access_revoked", remaining="_revoked", consumed="access")
    open_c = Candidate(option_id="billing", remaining="billing", consumed="")

    groups = group_by_context([done, open_a, open_b, open_c])

    assert set(groups) == {"access", ""}
    assert {c.option_id for c in groups["access"]} == {"access_denied", "access_revoked"}
    assert {c.option_id for c in groups[""]} == {"billing"}


def test_group_by_context_empty_when_everything_is_done():
    candidates = [Candidate(option_id="a", remaining=""), Candidate(option_id="b", remaining="")]
    assert group_by_context(candidates) == {}


def test_build_prefix_messages_lists_real_ids_not_letters():
    row = {
        "id": "r1", "state": "some evidence", "question": "pick one",
        "options": [{"id": "access_denied", "description": "d1"}, {"id": "billing", "description": "d2"}],
    }
    messages = build_prefix_messages(row)
    assert len(messages) == 2
    assert "access_denied" in messages[1]["content"]
    assert "billing" in messages[1]["content"]
    assert "d1" not in messages[1]["content"]


def test_build_prefix_messages_continues_from_a_partial_answer():
    row = {"id": "r1", "state": "e", "question": "q", "options": [{"id": "access_denied", "description": "d"}]}
    messages = build_prefix_messages(row, prefix="access")
    assert messages[-1] == {"role": "assistant", "content": "access"}


def test_build_prefix_messages_accepts_structured_state():
    row = {"id": "r1", "state": {"foo": "bar"}, "question": "q", "options": [{"id": "a1", "description": "d"}]}
    messages = build_prefix_messages(row)
    assert '"foo"' in messages[1]["content"]


ROW = {
    "id": "access-1",
    "state": "The login failed with a 403 error.",
    "question": "Which category fits?",
    "options": [
        {"id": "access_denied", "description": "d"},
        {"id": "access_expired", "description": "d"},
        {"id": "access_revoked", "description": "d"},
    ],
}


def test_decide_prefix_resolves_a_real_collision_in_two_rounds():
    client = QueuedFakeClient([
        _resp("access", {"access": -0.01, "billing": -5.0}),
        _resp("_denied", {"_denied": -0.05, "_expired": -3.0, "_revoked": -1.5}),
    ])
    d = client._decide_prefix(ROW)
    assert d.mode == "prefix"
    assert d.choice == "access_denied"
    assert set(d.probabilities) == {"access_denied", "access_expired", "access_revoked"}
    assert len(client.sent_bodies) == 2


def test_decide_prefix_no_collision_is_one_request():
    row = {**ROW, "options": [{"id": "access", "description": "d"}, {"id": "billing", "description": "d"}]}
    client = QueuedFakeClient([_resp("access", {"access": -0.01, "billing": -2.0})])
    d = client.decide(row)
    assert d.choice == "access"
    assert len(client.sent_bodies) == 1


def test_stop_early_a_group_of_one_is_scored_without_a_request():
    # Three ids share "access_" in round 1. Round 2's response diverges
    # "denied" from the pack, leaving "expired" and "revoked" -- still a
    # group of 2, not yet stoppable. A third round would normally be
    # needed to separate those two, but calling _decide_prefix directly
    # on just "access_expired" vs "access_revoked" isolates the
    # stop-early case cleanly: once one of a two-candidate group
    # resolves and the other is still open, the survivor left alone in
    # its own group has nothing left to compete against and stops
    # without a further request.
    row = {
        "id": "r1", "state": "e", "question": "q",
        "options": [{"id": "access_denied", "description": "d"}, {"id": "billing_refund", "description": "d"}],
    }
    client = QueuedFakeClient([
        _resp("access", {"access": -0.01, "billing": -0.5}),
    ])
    d = client._decide_prefix(row)
    # Both candidates' very first token diverges them from each other
    # immediately (their consumed prefixes differ: "access" vs
    # "billing"), so after round 1 each is alone in its own group for
    # round 2 -- both stop early on their round-1 logprob, no second
    # request needed.
    assert d.choice == "access_denied"
    assert set(d.stopped_early) == {"access_denied", "billing_refund"}
    assert len(client.sent_bodies) == 1


def test_decide_prefix_reports_unscored_when_a_continuation_never_appears():
    from decidr import is_reliable

    client = QueuedFakeClient([
        _resp("access", {"access": -0.01}),
        _resp("_denied", {"_denied": -0.05}),
    ])
    d = client._decide_prefix(ROW)
    assert not is_reliable(d)
    assert set(d.unscored) == {"access_expired", "access_revoked"}
    assert d.choice == "access_denied"
    assert d.probabilities == {"access_denied": 1.0}


def test_decide_prefix_raises_when_nothing_resolves():
    client = QueuedFakeClient([_resp("something_else", {"something_else": -0.01})])
    with pytest.raises(DecisionError, match="none of the option ids|could not score"):
        client.decide(ROW)


HIERARCHICAL_ROW = {
    "id": "many", "state": "e", "question": "q",
    "options": (
        [{"id": f"a_opt{i}", "description": "d"} for i in range(10)]
        + [{"id": f"b_opt{i}", "description": "d"} for i in range(10)]
    ),
}


def test_decide_prefix_accepts_more_than_sixteen_options_when_hierarchical():
    client = QueuedFakeClient([
        _resp("a", {"a": -0.01, "b": -3.0}),
        _resp("opt0", {f"opt{i}": -float(i) - 1 for i in range(10)}),
    ], exhaustive=False)
    d = client.decide(HIERARCHICAL_ROW)
    assert d.choice == "a_opt0"
    assert len(d.probabilities) == 10
    assert len(d.eliminated) == 10
    assert len(client.sent_bodies) == 2


def test_decide_prefix_is_exhaustive_by_default():
    client = QueuedFakeClient([
        _resp("a", {"a": -0.01, "b": -3.0}),
        _resp("opt0", {f"opt{i}": -float(i) - 1 for i in range(10)}),
        _resp("opt0", {f"opt{i}": -float(i) - 1 for i in range(10)}),
    ])
    d = client.decide(HIERARCHICAL_ROW)
    assert d.choice == "a_opt0"
    assert len(d.probabilities) == 20
    assert d.eliminated == []
    assert len(client.sent_bodies) == 3


def test_flat_options_over_the_branch_cap_are_rejected():
    from decidr.core import MAX_BRANCHES_PER_LEVEL
    row = {
        "id": "many", "state": "e", "question": "q",
        "options": [{"id": f"opt{i}", "description": "d"} for i in range(MAX_BRANCHES_PER_LEVEL + 1)],
    }
    client = QueuedFakeClient([])
    with pytest.raises(DecisionError, match="branches, over the limit"):
        client.decide(row)


def test_decide_prefix_gives_up_after_max_depth_rather_than_looping_forever():
    from decidr.prefix import MAX_DEPTH
    # "aaaaaaaaaa" and "aaaaaaaaab" share a long common prefix, so they
    # stay grouped together and keep genuinely competing (never stopping
    # early) as long as both are still open. "billing" diverges
    # immediately and stops early on round 1's logprob. Every response
    # for the "a..." group only advances by one "a" -- a deliberately
    # pathological, non-terminating continuation for those two -- so
    # this only stops because the depth cap fires for them, not because
    # they resolve or stop early; "billing" still comes back scored.
    row = {**ROW, "options": [
        {"id": "aaaaaaaaaa", "description": "d"},
        {"id": "aaaaaaaaab", "description": "d"},
        {"id": "billing", "description": "d"},
    ]}
    responses = [_resp("a", {"a": -0.1, "billing": -0.1}) for _ in range(MAX_DEPTH + 2)]
    client = QueuedFakeClient(responses)
    d = client.decide(row)
    assert "aaaaaaaaaa" in d.unscored
    assert "aaaaaaaaab" in d.unscored
    assert d.choice == "billing"
    assert len(client.sent_bodies) <= MAX_DEPTH + 1


def test_build_tree_flat_ids_become_one_level_children():
    from decidr.prefix import build_tree
    opts = [{"id": "access", "description": "d1"}, {"id": "billing", "description": "d2"}]
    root = build_tree(opts)
    assert set(root.children) == {"access", "billing"}
    assert root.children["access"].options == [opts[0]]
    assert len(root.children["access"].children) == 0


def test_build_tree_shared_first_segment_becomes_one_child_with_a_subtree():
    from decidr.prefix import build_tree
    opts = [
        {"id": "access_denied", "description": "d1"},
        {"id": "access_expired", "description": "d2"},
        {"id": "billing_refund", "description": "d3"},
    ]
    root = build_tree(opts)
    assert set(root.children) == {"access", "billing"}
    access_node = root.children["access"]
    assert {o["id"] for o in access_node.options} == {"access_denied", "access_expired"}
    assert set(access_node.children) == {"denied", "expired"}
    billing_node = root.children["billing"]
    assert set(billing_node.children) == {"refund"}


def test_child_summaries_shows_the_remaining_path_and_description():
    from decidr.prefix import build_tree
    opts = [
        {"id": "access_denied", "description": "Login rejected"},
        {"id": "access_expired", "description": "Token expired"},
    ]
    root = build_tree(opts)
    access_node = root.children["access"]
    summaries = access_node.child_summaries()
    assert summaries["denied"] == "denied: Login rejected"
    assert summaries["expired"] == "expired: Token expired"


def test_child_summaries_at_the_root_uses_the_full_id_for_leaves():
    from decidr.prefix import build_tree
    opts = [{"id": "access", "description": "Account help"}, {"id": "billing", "description": "Payment help"}]
    root = build_tree(opts)
    summaries = root.child_summaries()
    assert summaries["access"] == "access: Account help"
    assert summaries["billing"] == "billing: Payment help"


def test_no_logprobs_at_all_fails_the_whole_race_not_just_one_candidate():
    client = QueuedFakeClient([{"content": "access", "logprobs": []}])
    with pytest.raises(DecisionError, match="could not score any option"):
        client.decide(ROW)
