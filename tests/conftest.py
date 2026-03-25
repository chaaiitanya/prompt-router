"""
tests/conftest.py — Shared pytest fixtures and configuration.

CONCEPT: conftest.py
  pytest automatically loads conftest.py before any tests in the same
  directory or subdirectories. Fixtures defined here are available to
  all test files without needing to import them.
"""

import pytest
from app.models import Message


# ── Common message factories ───────────────────────────────────────────────────

@pytest.fixture
def user_message():
    """A single user message for simple test cases."""
    return [Message(role="user", content="What is Python?")]


@pytest.fixture
def system_user_messages():
    """A system + user turn for multi-message test cases."""
    return [
        Message(role="system", content="You are a concise assistant."),
        Message(role="user", content="Explain recursion briefly."),
    ]
