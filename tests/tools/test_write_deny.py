"""Tests for _is_write_denied() — verifies deny list blocks sensitive paths on all platforms."""

import os
import pytest
from pathlib import Path

from tools.file_operations import _is_write_denied


class TestWriteDenyExactPaths:
    def test_etc_shadow(self):
        assert _is_write_denied("/etc/shadow") is True

    def test_etc_passwd(self):
        assert _is_write_denied("/etc/passwd") is True

    def test_etc_sudoers(self):
        assert _is_write_denied("/etc/sudoers") is True

    def test_ssh_authorized_keys(self):
        assert _is_write_denied("~/.ssh/authorized_keys") is True

    def test_ssh_id_rsa(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_rsa")
        assert _is_write_denied(path) is True

    def test_ssh_id_ed25519(self):
        path = os.path.join(str(Path.home()), ".ssh", "id_ed25519")
        assert _is_write_denied(path) is True

    def test_netrc(self):
        path = os.path.join(str(Path.home()), ".netrc")
        assert _is_write_denied(path) is True

    def test_hermes_env(self):
        # ``.env`` under the active HERMES_HOME (profile-aware, not just
        # ``~/.hermes``) must be write-denied. The hermetic test conftest
        # points HERMES_HOME at a tempdir — resolve via get_hermes_home()
        # to match the denylist.
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / ".env")
        assert _is_write_denied(path) is True

    def test_shell_profiles(self):
        home = str(Path.home())
        for name in [".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile"]:
            assert _is_write_denied(os.path.join(home, name)) is True, f"{name} should be denied"

    def test_package_manager_configs(self):
        home = str(Path.home())
        for name in [".npmrc", ".pypirc", ".pgpass"]:
            assert _is_write_denied(os.path.join(home, name)) is True, f"{name} should be denied"


class TestWriteDenyPrefixes:
    def test_ssh_prefix(self):
        path = os.path.join(str(Path.home()), ".ssh", "some_key")
        assert _is_write_denied(path) is True

    def test_aws_prefix(self):
        path = os.path.join(str(Path.home()), ".aws", "credentials")
        assert _is_write_denied(path) is True

    def test_gnupg_prefix(self):
        path = os.path.join(str(Path.home()), ".gnupg", "secring.gpg")
        assert _is_write_denied(path) is True

    def test_kube_prefix(self):
        path = os.path.join(str(Path.home()), ".kube", "config")
        assert _is_write_denied(path) is True

    def test_sudoers_d_prefix(self):
        assert _is_write_denied("/etc/sudoers.d/custom") is True

    def test_systemd_prefix(self):
        assert _is_write_denied("/etc/systemd/system/evil.service") is True


class TestWriteAllowed:
    def test_tmp_file(self):
        assert _is_write_denied("/tmp/safe_file.txt") is False

    def test_project_file(self):
        assert _is_write_denied("/home/user/project/main.py") is False

    def test_env_example_template(self):
        # ACOS-HERMES: .env.example is a public template, not a secret
        assert _is_write_denied("/tmp/project/.env.example") is False

    def test_env_sample_template(self):
        assert _is_write_denied("/tmp/project/.env.sample") is False


class TestAcosHermesIdentityFiles:
    """ACOS-HERMES Patch 4: identity files in HERMES_HOME are write-denied."""

    def test_hermes_config_yaml_is_denied(self):
        # ACOS-HERMES policy: AH must not self-modify its config.
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / "config.yaml")
        assert _is_write_denied(path) is True

    def test_hermes_cli_config_yaml_is_denied(self):
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / "cli-config.yaml")
        assert _is_write_denied(path) is True

    def test_hermes_soul_md_is_denied(self):
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / "SOUL.md")
        assert _is_write_denied(path) is True

    def test_hermes_md_is_denied(self):
        from hermes_constants import get_hermes_home
        path = str(get_hermes_home() / "HERMES.md")
        assert _is_write_denied(path) is True


class TestAcosHermesAdditionalDeny:
    """ACOS-HERMES Patch 4: additional shell rc and VPS env paths."""

    def test_zshenv_is_denied(self):
        # User-global env vars file containing API keys
        path = os.path.join(str(Path.home()), ".zshenv")
        assert _is_write_denied(path) is True

    def test_etc_hermes_prefix_is_denied(self):
        # systemd EnvironmentFile location on the ACOS Hermes VPS
        assert _is_write_denied("/etc/hermes/env.list") is True


class TestAcosHermesBasenameDeny:
    """ACOS-HERMES Patch 4: basename-pattern deny (key files, env files)."""

    def test_pem_anywhere_is_denied(self):
        assert _is_write_denied("/tmp/foo.pem") is True

    def test_key_anywhere_is_denied(self):
        assert _is_write_denied("/tmp/private.key") is True

    def test_p12_anywhere_is_denied(self):
        assert _is_write_denied("/tmp/cert.p12") is True

    def test_pfx_anywhere_is_denied(self):
        assert _is_write_denied("/tmp/cert.pfx") is True

    def test_ppk_anywhere_is_denied(self):
        assert _is_write_denied("/tmp/putty.ppk") is True

    def test_dotenv_anywhere_is_denied(self):
        assert _is_write_denied("/tmp/project/.env") is True

    def test_dotenv_local_is_denied(self):
        assert _is_write_denied("/tmp/project/.env.local") is True

    def test_dotenv_production_is_denied(self):
        assert _is_write_denied("/tmp/project/.env.production") is True

    def test_secrets_env_suffix_is_denied(self):
        # Non-dotfile .env (e.g. secrets.env, prod.env)
        assert _is_write_denied("/tmp/secrets.env") is True
