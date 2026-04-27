"""Shared file safety rules used by both tools and ACP shims."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# ACOS-HERMES: identity / config files inside HERMES_HOME that the agent
# must not self-modify. See HARDENING_PLAN.md Patch 4.
_HERMES_IDENTITY_FILES = (
    "SOUL.md",
    "HERMES.md",
    "config.yaml",
    "cli-config.yaml",
)

# ACOS-HERMES: file extensions that signal key material — denied anywhere.
_DENIED_KEY_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".ppk")

# ACOS-HERMES: env-file basenames that are public templates (no secrets).
# Excluded from the .env block so dev workflows still work.
_ENV_TEMPLATE_BASENAMES = frozenset({
    ".env.example",
    ".env.sample",
    ".env.template",
    ".env.dist",
    ".env.defaults",
})


def _hermes_home_path() -> Path:
    """Resolve the active HERMES_HOME (profile-aware) without circular imports."""
    try:
        from hermes_constants import get_hermes_home  # local import to avoid cycles
        return get_hermes_home()
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def build_write_denied_paths(home: str) -> set[str]:
    """Return exact sensitive paths that must never be written."""
    hermes_home = _hermes_home_path()
    paths = [
        os.path.join(home, ".ssh", "authorized_keys"),
        os.path.join(home, ".ssh", "id_rsa"),
        os.path.join(home, ".ssh", "id_ed25519"),
        os.path.join(home, ".ssh", "config"),
        str(hermes_home / ".env"),
        os.path.join(home, ".bashrc"),
        os.path.join(home, ".zshrc"),
        os.path.join(home, ".zshenv"),  # ACOS-HERMES: holds user API keys
        os.path.join(home, ".profile"),
        os.path.join(home, ".bash_profile"),
        os.path.join(home, ".zprofile"),
        os.path.join(home, ".netrc"),
        os.path.join(home, ".pgpass"),
        os.path.join(home, ".npmrc"),
        os.path.join(home, ".pypirc"),
        "/etc/sudoers",
        "/etc/passwd",
        "/etc/shadow",
    ]
    # ACOS-HERMES: identity files inside HERMES_HOME (no self-modification)
    paths.extend(str(hermes_home / fname) for fname in _HERMES_IDENTITY_FILES)
    return {os.path.realpath(p) for p in paths}


def build_write_denied_prefixes(home: str) -> list[str]:
    """Return sensitive directory prefixes that must never be written."""
    return [
        os.path.realpath(p) + os.sep
        for p in [
            os.path.join(home, ".ssh"),
            os.path.join(home, ".aws"),
            os.path.join(home, ".gnupg"),
            os.path.join(home, ".kube"),
            "/etc/sudoers.d",
            "/etc/systemd",
            "/etc/hermes",  # ACOS-HERMES: systemd EnvironmentFile location on VPS
            os.path.join(home, ".docker"),
            os.path.join(home, ".azure"),
            os.path.join(home, ".config", "gh"),
        ]
    ]


def _is_denied_by_basename(path: str) -> bool:
    """ACOS-HERMES: deny by basename pattern regardless of containing directory.

    Catches key material (.pem/.key/.p12/.pfx/.ppk) and .env files
    (.env, .env.local, .env.production, ...) wherever they live, while
    allowing well-known templates (.env.example, .env.sample, ...).
    """
    basename = os.path.basename(path).lower()

    # Key material anywhere
    if basename.endswith(_DENIED_KEY_SUFFIXES):
        return True

    # .env templates (allowlist) — must precede the .env catch-all
    if basename in _ENV_TEMPLATE_BASENAMES:
        return False

    # Dotted .env variants: .env, .env.local, .env.production, ...
    if basename == ".env" or basename.startswith(".env."):
        return True

    # Suffix .env on non-dotfiles (e.g. secrets.env, prod.env)
    if basename.endswith(".env") and not basename.startswith("."):
        return True

    return False


def get_safe_write_root() -> Optional[str]:
    """Return the resolved HERMES_WRITE_SAFE_ROOT path, or None if unset."""
    root = os.getenv("HERMES_WRITE_SAFE_ROOT", "")
    if not root:
        return None
    try:
        return os.path.realpath(os.path.expanduser(root))
    except Exception:
        return None


def is_write_denied(path: str) -> bool:
    """Return True if path is blocked by the write denylist or safe root."""
    home = os.path.realpath(os.path.expanduser("~"))
    resolved = os.path.realpath(os.path.expanduser(str(path)))

    if resolved in build_write_denied_paths(home):
        return True
    for prefix in build_write_denied_prefixes(home):
        if resolved.startswith(prefix):
            return True

    # ACOS-HERMES: basename-based deny (key files, env files anywhere)
    if _is_denied_by_basename(resolved):
        return True

    safe_root = get_safe_write_root()
    if safe_root and not (resolved == safe_root or resolved.startswith(safe_root + os.sep)):
        return True

    return False


def get_read_block_error(path: str) -> Optional[str]:
    """Return an error message when a read targets internal Hermes cache files."""
    resolved = Path(path).expanduser().resolve()
    hermes_home = _hermes_home_path().resolve()
    blocked_dirs = [
        hermes_home / "skills" / ".hub" / "index-cache",
        hermes_home / "skills" / ".hub",
    ]
    for blocked in blocked_dirs:
        try:
            resolved.relative_to(blocked)
        except ValueError:
            continue
        return (
            f"Access denied: {path} is an internal Hermes cache file "
            "and cannot be read directly to prevent prompt injection. "
            "Use the skills_list or skill_view tools instead."
        )
    return None
