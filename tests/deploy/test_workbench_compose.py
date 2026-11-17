# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("ngc_api_key", ["validation-only", None])
def test_workbench_can_load_compose_using_its_project_name(ngc_api_key: str | None):
    if shutil.which("docker") is None:
        pytest.skip("docker is required to validate Workbench Compose loading")

    version = subprocess.run(["docker", "compose", "version"], capture_output=True, check=False, text=True)
    if version.returncode != 0:
        pytest.skip("docker compose is required to validate Workbench Compose loading")

    spec = yaml.safe_load((REPO_ROOT / ".project" / "spec.yaml").read_text(encoding="utf-8"))
    env = os.environ.copy()
    if ngc_api_key is None:
        env.pop("NGC_API_KEY", None)
    else:
        env["NGC_API_KEY"] = ngc_api_key

    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            str(REPO_ROOT / "deploy" / ".env.example"),
            "--project-name",
            spec["meta"]["name"],
            "-f",
            str(REPO_ROOT / spec["environment"]["compose_file_path"]),
            "config",
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )

    if ngc_api_key is None:
        assert result.returncode != 0
        assert "Set NGC_API_KEY" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        project = json.loads(result.stdout)
        assert project["name"] == spec["meta"]["name"]
        assert project["services"]["embedder"]["environment"]["NGC_API_KEY"] == ngc_api_key
        backend_env = project["services"]["deep-researcher-agent"]["environment"]
        profile = yaml.safe_load(
            (REPO_ROOT / "configs" / "config_web_onprem_documents.yml").read_text(encoding="utf-8")
        )
        knowledge_config = profile["functions"]["knowledge_search"]
        with OpenAI(
            base_url=knowledge_config["embed_base_url"],
            api_key=backend_env.get("DEEP_RESEARCHER_EMBED_API_KEY") or backend_env.get("NVIDIA_API_KEY", ""),
        ) as client:
            assert client.base_url.host == "embedder"
