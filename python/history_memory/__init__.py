"""Shared observable-history contracts for C2KV training and runtime views."""

from .events import EventRecord, EventStore, Message, build_events

__all__ = ["EventRecord", "EventStore", "Message", "build_events"]
