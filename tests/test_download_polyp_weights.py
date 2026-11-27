import hashlib
import subprocess

import pytest

import download_polyp_weights as setup


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    files = {}
    checkpoints = []
    for index, original in enumerate(setup.CHECKPOINTS):
        path = tmp_path / f"cache with spaces {index}" / original["filename"]
        path.parent.mkdir()
        path.write_bytes(f"test checkpoint {index}".encode())
        files[original["repo_id"]] = path
        checkpoints.append(dict(original, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    monkeypatch.setattr(setup, "CHECKPOINTS", tuple(checkpoints))
    return files


@pytest.mark.parametrize("token", [None, "test-access-token"])
def test_download_pins_auth_and_sourceable_paths(tmp_path, monkeypatch, artifacts, token):
    calls = []
    monkeypatch.setattr(setup, "hf_token", lambda: token)

    def download(**kwargs):
        calls.append(kwargs)
        return str(artifacts[kwargs["repo_id"]])

    monkeypatch.setattr(setup, "hf_hub_download", download)
    env_file = tmp_path / "configuration" / "polyp.env"
    configuration = setup.download_weights(env_file, cache_dir=tmp_path / "hub", local_files_only=True)
    for call, checkpoint in zip(calls, setup.CHECKPOINTS, strict=True):
        assert call == dict(repo_id=checkpoint["repo_id"], filename=checkpoint["filename"],
                            revision=checkpoint["revision"], cache_dir=str(tmp_path / "hub"),
                            token=token, local_files_only=True)
    assert "test-access-token" not in env_file.read_text()
    for name, value in configuration.items():
        result = subprocess.run(["bash", "-c", 'source "$1"; printenv "$2"', "bash", str(env_file), name],
                                text=True, capture_output=True, check=True)
        assert result.stdout.strip() == value


@pytest.mark.parametrize("failure", ["corrupt", "no_access"])
def test_failure_preserves_existing_configuration(tmp_path, monkeypatch, artifacts, failure):
    def download(**kwargs):
        if kwargs["repo_id"] == setup.CHECKPOINTS[1]["repo_id"]:
            if failure == "no_access":
                raise PermissionError("private repository")
            artifacts[kwargs["repo_id"]].write_bytes(b"corrupted checkpoint")
        return str(artifacts[kwargs["repo_id"]])

    monkeypatch.setattr(setup, "hf_hub_download", download)
    env_file = tmp_path / "existing.env"
    env_file.write_text("existing configuration\n")
    with pytest.raises((ValueError, PermissionError)):
        setup.download_weights(env_file)
    assert env_file.read_text() == "existing configuration\n"


def test_generated_yolo_path_preserves_hub_snapshot_filename(tmp_path, monkeypatch, artifacts):
    repo = setup.CHECKPOINTS[1]["repo_id"]
    blob = tmp_path / "extensionless-blob"
    artifacts[repo].rename(blob)
    snapshot = tmp_path / "yolo_polyp_finetuned.pt"
    snapshot.symlink_to(blob)
    artifacts[repo] = snapshot
    monkeypatch.setattr(setup, "hf_hub_download", lambda **kwargs: str(artifacts[kwargs["repo_id"]]))
    config = setup.download_weights(tmp_path / "polyp.env")
    assert config["IQUANA_YOLO_POLYP_CHECKPOINT"] == str(snapshot)
