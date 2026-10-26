"""Errors from routed backends must name the path visible to the agent."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langgraph.store.memory import InMemoryStore

from deepagents.backends import CompositeBackend, FilesystemBackend, StoreBackend
from deepagents.backends.protocol import BackendProtocol, ReadResult


@pytest.fixture(params=["store", "filesystem"])
def backend(request, tmp_path: Path):
    default = StoreBackend(store=InMemoryStore(), namespace=lambda _: ("default",))
    if request.param == "filesystem":
        root = tmp_path / "routed"
        root.mkdir()
        routed = FilesystemBackend(root_dir=root, virtual_mode=True)
    else:
        routed = StoreBackend(store=InMemoryStore(), namespace=lambda _: ("routed",))
    return CompositeBackend(default=default, routes={"/memories/": routed})


@pytest.mark.parametrize("operation", ["read", "edit", "delete"])
def test_routed_file_errors_use_original_path(backend, operation):
    path = "/memories/missing.txt"
    args = {"read": (), "edit": ("old", "new"), "delete": ()}[operation]

    result = getattr(backend, operation)(path, *args)

    assert result.error is not None
    assert f"'{path}'" in result.error
    assert "'/missing.txt'" not in result.error


@pytest.mark.parametrize("operation", ["read", "edit", "delete"])
async def test_async_routed_file_errors_use_original_path(backend, operation):
    path = "/memories/missing.txt"
    args = {"read": (), "edit": ("old", "new"), "delete": ()}[operation]

    result = await getattr(backend, f"a{operation}")(path, *args)

    assert result.error is not None
    assert f"'{path}'" in result.error
    assert "'/missing.txt'" not in result.error


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_routed_listing_error_uses_original_path(tmp_path: Path, asynchronous):
    routed = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    backend = CompositeBackend(default=routed, routes={"/memories/": routed})

    result = await backend.als("/memories/missing") if asynchronous else backend.ls("/memories/missing")

    assert result.error is not None
    assert "'/memories/missing'" in result.error


def test_default_backend_error_is_unchanged(backend):
    assert backend.read("/missing.txt").error == backend.default.read("/missing.txt").error


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_routed_write_error_uses_original_path(tmp_path: Path, asynchronous):
    (tmp_path / "blocked").write_text("not a directory", encoding="utf-8")
    routed = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    backend = CompositeBackend(default=routed, routes={"/memories/": routed})
    path = "/memories/blocked/file.txt"

    result = await backend.awrite(path, "new") if asynchronous else backend.write(path, "new")

    assert result.error is not None
    assert f"'{path}'" in result.error


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("File '/missing.txt' not found", "File '/memories/missing.txt' not found"),
        ("File '/missing.txt.bak' not found", "File '/missing.txt.bak' not found"),
        ("Unquoted /missing.txt in custom error", "Unquoted /missing.txt in custom error"),
        ("Permission denied", "Permission denied"),
    ],
)
def test_remapping_only_changes_exact_quoted_path(error, expected):
    routed = MagicMock(spec=BackendProtocol)
    routed.read.return_value = ReadResult(error=error, start_line=1, end_line=1, total_lines=3)
    backend = CompositeBackend(default=routed, routes={"/memories/": routed})

    result = backend.read("/memories/missing.txt")

    assert result.error == expected
    assert result.total_lines == 3
