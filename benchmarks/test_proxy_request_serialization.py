"""The proxy must finish an episode request before switching episode state."""

import ast
import threading
from pathlib import Path
from types import SimpleNamespace


def _post_handler():
    source = Path(__file__).with_name("proxy.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ProxyHandler"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "do_POST"
    )
    namespace = {"STATE": SimpleNamespace(request_lock=threading.Lock())}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["do_POST"]


def test_episode_switch_waits_for_inflight_chat_request():
    post = _post_handler()
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def first_request():
        first_started.set()
        assert release_first.wait(timeout=2)

    first = threading.Thread(
        target=post, args=(SimpleNamespace(_handle_post=first_request),)
    )
    second = threading.Thread(
        target=post,
        args=(SimpleNamespace(_handle_post=second_started.set),),
    )

    first.start()
    assert first_started.wait(timeout=2)
    second.start()
    try:
        assert not second_started.wait(timeout=0.1)
    finally:
        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)
    assert second_started.is_set()
    assert not first.is_alive() and not second.is_alive()
