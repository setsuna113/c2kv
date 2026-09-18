from generality.prefix_replay import ReplayedEnvironment


def test_current_payload_does_not_open_bfcl_slot_again():
    class Env:
        def __init__(self):
            self.next_payload_calls = 0

        def next_payload(self):
            self.next_payload_calls += 1
            raise AssertionError("current_payload reopened the pending BFCL slot")

    env = Env()
    pending = {"decision_key": "turn-1/step-2", "messages": []}
    replay = ReplayedEnvironment(
        env=env,
        bindings=None,
        task={},
        ground_truth=[],
        turn_index=1,
        prefix_payload_sha256="sha",
        replayed_assistant_messages=1,
        replayed_tool_messages=1,
        previous_turn_valid=True,
        pending_payload=pending,
    )

    assert replay.current_payload() == pending
    assert env.next_payload_calls == 0
