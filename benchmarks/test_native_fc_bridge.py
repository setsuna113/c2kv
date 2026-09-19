"""Exercise the adapter bridge without importing the optional BFCL installation."""
import ast
from pathlib import Path
from types import SimpleNamespace


def load_function(path, name, namespace=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {} if namespace is None else namespace
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


class Message(SimpleNamespace):
    def model_dump(self):
        return vars(self).copy()

    @classmethod
    def model_validate(cls, values):
        return cls(**values)


def response(content, identifier="request_1"):
    return SimpleNamespace(id=identifier, choices=[SimpleNamespace(
        message=Message(content=content, tool_calls=None), finish_reason="stop")])


def test_explicit_native_calls_preserved_and_ids_scoped():
    normalize = load_function(Path(__file__).parent / "adapters/bfcl_adapter.py",
                              "normalize_native_calls")
    text = 'Checking. <tool_call>{"name":"lookup","arguments":{"city":"X"}}</tool_call>'
    first = normalize(response(text)).choices[0]
    second = normalize(response(text, "request_2")).choices[0]
    assert first.message.content == "Checking."
    assert first.finish_reason == "tool_calls"
    assert first.message.tool_calls[0]["function"] == {
        "name": "lookup", "arguments": '{"city":"X"}'}
    assert first.message.tool_calls[0]["id"] != second.message.tool_calls[0]["id"]


def test_invalid_native_arguments_are_not_repaired():
    normalize = load_function(Path(__file__).parent / "adapters/bfcl_adapter.py",
                              "normalize_native_calls")
    for text in ['<tool_call>{"name":"lookup","arguments":[]}</tool_call>',
                 '<tool_call>{"name":"lookup","arguments":{}}',
                 '<tool_call>{"name":"lookup","arguments":{"x":1,"x":2}}</tool_call>']:
        item = normalize(response(text)).choices[0]
        assert item.message.content == text
        assert item.message.tool_calls is None
        assert item.finish_reason == "stop"


def test_chat_lock_covers_forwarding():
    import threading
    lock = threading.Lock()
    dispatch = load_function(Path(__file__).parent / "proxy.py", "do_POST",
                             {"_EPISODE_REQUEST_LOCK": lock})
    handler = SimpleNamespace(_is_chat=lambda: True,
                              _do_POST_serialized=lambda: lock.locked())
    assert dispatch(handler) is True
    assert not lock.locked()
