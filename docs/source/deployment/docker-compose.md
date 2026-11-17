<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Docker Compose

Docker Compose is the recommended way to run the full Deep Researcher Agent blueprint stack (backend, frontend, and database) without managing individual processes.

## Prerequisites

- Docker Engine and Docker Compose v2.
- API keys for the models and tools you plan to use (refer to [Installation -- API Key Setup](../get-started/installation.md#api-key-setup)).
- Ports `3000`, `8000`, and `5432` available on your host.
- Enough disk space for Docker volumes and cached model artifacts.

## Files and Directories

The Docker Compose setup uses these files:

| File | Purpose |
|------|---------|
| `compose.yaml` | Workbench entry point for the on-prem stack |
| `deploy/compose/docker-compose.yaml` | Standard stack (LlamaIndex or FRAG backend) |
| `deploy/compose/docker-compose.onprem.yaml` | Optional DGX Spark / GB10 model-serving overlay |
| `deploy/compose/docker-compose.workbench.yaml` | Workbench-only settings: no backend `.env` loading and frontend proxy support |
| `deploy/.env` | Environment variables for all services |
| `deploy/.env.example` | Template with all available variables |
| `deploy/compose/init-db.sql` | PostgreSQL initialization script |
| `configs/config_web_default_llamaindex.yml` | LlamaIndex backend config (default) |
| `configs/config_web_frag.yml` | Foundational RAG backend config |

## Environment Setup

Copy the example environment file and edit it:

```bash
cp deploy/.env.example deploy/.env
```

The sections below explain each group of variables.

### API keys

| Variable | Required | Description |
|----------|----------|-------------|
| `NVIDIA_API_KEY` | Hosted profiles only | NVIDIA API key for hosted NIM inference. The on-prem profile does not use it. |
| `NGC_API_KEY` | On-prem overlay | NGC API key for pulling the embedding NIM and model artifacts. |
| `TAVILY_API_KEY` | Conditional | Web search provider key (required if using `tavily_web_search`). |
| `EXA_API_KEY` | Conditional | Web search provider key (required if using `exa_web_search`). |
| `NIMBLE_API_KEY` | Conditional | Web search provider key (required if using `nimble_web_search`). |
| `SERPER_API_KEY` | No | Google Scholar paper search key (optional). |

### API keys (optional)

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required only if your config uses OpenAI models directly. |
| `JINA_API_KEY` | Required only if you enable the evaluation suite. |
| `WANDB_API_KEY` | Required only if you enable experiment tracking. |

### Application Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `production` | Application environment (`development` or `production`). The compose files default to `production`; `deploy/.env.example` sets `development`. |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

### Backend Configuration

Set `BACKEND_CONFIG` to select which workflow config the backend loads at startup. The compose stacks mount `configs/` into the container at `/app/configs`.

| Config path | Description |
|-------------|-------------|
| `/app/configs/config_web_default_llamaindex.yml` | Default -- LlamaIndex backend (no external RAG required). |
| `/app/configs/config_web_onprem_documents.yml` | On-premises, document-only profile used by the DGX Spark overlay. |
| `/app/configs/config_web_frag.yml` | Foundational RAG mode (requires a running RAG Blueprint). |

Example in `deploy/.env`:

```bash
BACKEND_CONFIG=/app/configs/config_web_default_llamaindex.yml
```

### Database Settings

Choose one of the following database configurations.

**PostgreSQL (recommended for all deployments):**

```bash
NAT_JOB_STORE_DB_URL=postgresql+asyncpg://deep_researcher:deep_researcher_dev@postgres:5432/deep_researcher_jobs  # pragma: allowlist secret
DEEP_RESEARCHER_CHECKPOINT_DB=postgresql://deep_researcher:deep_researcher_dev@postgres:5432/deep_researcher_checkpoints  # pragma: allowlist secret
DEEP_RESEARCHER_SUMMARY_DB=postgresql+psycopg://deep_researcher:deep_researcher_dev@postgres:5432/deep_researcher_jobs  # pragma: allowlist secret
```

These are the default values used by the compose stack when the variables are unset.

**SQLite (development only):**

```bash
NAT_JOB_STORE_DB_URL=sqlite+aiosqlite:///./data/jobs.db
DEEP_RESEARCHER_CHECKPOINT_DB=/app/data/checkpoints.db
# DEEP_RESEARCHER_SUMMARY_DB defaults to sqlite+aiosqlite:///./summaries.db
```

When using SQLite, you can optionally remove the `depends_on` block for the `deep-researcher-agent` service since the `postgres` container is no longer needed.

### Artifact Storage

Artifact metadata always uses the job database. Artifact bytes use SQL BLOB storage
by default. The Compose stack does not provision an AWS S3 bucket; create the bucket
externally, then set:

```bash
DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER=s3
DEEP_RESEARCHER_ARTIFACT_S3_BUCKET=YOUR_BUCKET_NAME
DEEP_RESEARCHER_ARTIFACT_S3_REGION=us-west-2
DEEP_RESEARCHER_ARTIFACT_S3_PREFIX=artifacts/v1
```

The checked-in Compose stack also does not deploy MinIO. To run a local MinIO server
with Docker:

```bash
export MINIO_CONTAINER=deep-researcher-minio
export MINIO_ROOT_USER=YOUR_ACCESS_KEY
export MINIO_ROOT_PASSWORD=YOUR_SECRET_KEY
export DEEP_RESEARCHER_ARTIFACT_S3_BUCKET=YOUR_BUCKET_NAME

docker pull minio/minio
docker run --detach --name "$MINIO_CONTAINER" \
  --publish 9000:9000 --publish 9001:9001 \
  --env MINIO_ROOT_USER --env MINIO_ROOT_PASSWORD \
  --volume deep-researcher-minio-data:/data \
  minio/minio server /data --console-address ":9001"
```

Open the MinIO console at `http://localhost:9001` and create the bucket named by
`DEEP_RESEARCHER_ARTIFACT_S3_BUCKET`. Then add the same bucket and credentials to `deploy/.env`.
With Docker Desktop, the backend container reaches the host through
`host.docker.internal`:

```bash
DEEP_RESEARCHER_ARTIFACT_BLOB_PROVIDER=s3
DEEP_RESEARCHER_ARTIFACT_S3_BUCKET=YOUR_BUCKET_NAME
DEEP_RESEARCHER_ARTIFACT_S3_ENDPOINT_URL=http://host.docker.internal:9000
AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY
```

For another environment, replace the endpoint with an address reachable from the
`deep-researcher-agent` container.

The bucket is required when the provider is `s3`. Endpoint, region, and prefix are
optional; leave the endpoint unset for AWS S3, and the prefix defaults to
`artifacts/v1`. Configure credentials through workload identity, deployment secrets, or
the standard AWS credential chain. When the provider is `s3`, artifact bytes are stored
in the configured bucket and SQL stores artifact metadata only.

The access-key example above is for local MinIO development only. For production,
use workload identity or a short-lived role, restrict bucket access to the Deep Researcher Agent worker
identity and configured prefix, enable Block Public Access and TLS-only access, and
enable storage-layer encryption such as SSE-KMS. Deep Researcher Agent API ownership checks do not
protect direct bucket access, and Deep Researcher Agent does not application-encrypt artifact blob bytes.
See [Production Considerations](./production.md#s3-security-responsibility).

### Frontend Runtime Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://deep-researcher-agent:8000` | Backend API URL as seen from the frontend container. |

### Dask Worker Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DASK_NWORKERS` | `1` | Number of Dask workers for background job processing. |
| `DASK_NTHREADS` | `4` | Number of threads per Dask worker. |
| `DASK_DISTRIBUTED__LOGGING__DISTRIBUTED` | `warning` | Dask log level (reduce noise). |

## Standard Stack

### Build and Run Locally

From the repository root:

```bash
cd deploy/compose
docker compose --env-file ../.env -f docker-compose.yaml up -d --build
```

This starts three services:

| Service | Container name | Port | Description |
|---------|---------------|------|-------------|
| `deep-researcher-agent` | `deep-researcher-agent` | 8000 | Backend API server with embedded Dask cluster |
| `frontend` | `deep-researcher-blueprint-ui` | 3000 | [Next.js](https://nextjs.org/) web UI |
| `postgres` | `deep-researcher-postgres` | 5432 | PostgreSQL database |

Open [http://localhost:3000](http://localhost:3000) to access the web UI.

### DGX Spark / GB10 with local models

The on-prem overlay adds a Qwen3.5-35B-A3B vLLM service and a NeMo Retriever
Nemotron 3.5 Embed NIM. The local profile registers only the internal-document
source; it does not configure web or paper search. Model services are reachable
only on the Compose network and are not published on host ports.

The first profile uses Nemotron 3.5 Embed rather than BGE-M3 because the current
LlamaIndex adapter uses NVIDIA's embeddings client, and this embedding NIM is
[validated for DGX Spark / GB10](https://docs.nvidia.com/nim/nemo-retriever/embedding/latest/support-matrix.html).
BGE-M3 remains a candidate for a later comparison once a compatible local
serving path is validated.

This profile uses the current workflow-config runtime and is an initial on-prem
test step; it does not complete the planned migration away from NeMo Agent
Toolkit. Stop other GPU model services on the DGX Spark before testing so the
models do not compete for its unified memory.

#### AI Workbench with a managed secret

The committed `.project/spec.yaml` selects the root `compose.yaml`, which merges
the standard stack with the on-prem and Workbench overlays. Use Docker Compose
2.24.4 or newer for `include` and `!reset` support.

Workbench passes `meta.name` from `.project/spec.yaml` as the Compose project
name, so it must contain only lowercase letters, digits, hyphens, and
underscores, starting with a letter or digit. A top-level Compose `name` does
not override an explicitly supplied project name. If updating an existing
clone from a mixed-case name, stop the project container before pulling the
change from a host terminal, then close and reopen the project in Workbench.
Do not rename the repository directory or edit Workbench's runtime folders.
Workbench may prompt you to configure the sensitive variable again after a
project-name change; its value is not part of the Git update.

1. In **Project Container > Environment Variables**, configure `NGC_API_KEY`
   as **Sensitive**, then restart the project container. Workbench stores only
   the variable name and description in Git, not the value.
2. Configure **Settings > Integrations > NGC** to authenticate container-image
   pulls. This is separate from the project secret used by the embedding NIM
   to download model artifacts. The key must permit the required NGC Catalog
   and Private Registry resources.
3. Stop other GPU model services before starting this stack.
4. Open **Environment > Compose** and use **Start**. Workbench passes its
   sensitive variables to Compose for interpolation and displays service logs.
   The frontend enables Workbench proxy access.

This path does not require `deploy/.env`; the Workbench overlay disables the
backend's environment-file loading. Leave the key out of `.env` and
`variables.env`. Optional tuning values from this page must instead be added
as Workbench environment variables. Both model services explicitly share GPU 0;
the Workbench project container requests no GPU.

A project secret is not exported into an unrelated SSH session. Use Workbench's
Compose controls, not the Docker tasks below, when relying on a Workbench
secret. Do not mount the host Docker socket into the project container.
Never share resolved Compose output or `docker inspect` output: the NIM receives
the key as a runtime environment variable even though Workbench keeps its value
out of Git.

For Workbench setup details, see NVIDIA's
[runtime settings](https://docs.nvidia.com/ai-workbench/how-to/environments/runtime-configuration)
and [multi-container guide](https://docs.nvidia.com/ai-workbench/how-to/environments/multi-container).

#### Standalone Docker Compose

The following commands are for a host terminal outside the Workbench-managed
Compose environment.

1. Add an NGC API key to `deploy/.env` as `NGC_API_KEY`, then log in to NGC
   without printing the key:

   ```bash
   NGC_API_KEY="$(sed -n 's/^NGC_API_KEY=//p' deploy/.env | tail -n 1)"
   test -n "$NGC_API_KEY" || { echo "NGC_API_KEY is empty" >&2; exit 1; }
   printf '%s' "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
   unset NGC_API_KEY
   ```

2. From `deploy/compose`, validate and start the overlay:

   ```bash
   docker compose --env-file ../.env -f docker-compose.yaml -f docker-compose.onprem.yaml config --quiet
   docker compose --env-file ../.env -f docker-compose.yaml -f docker-compose.onprem.yaml up -d --build
   ```

3. Follow model startup and download progress:

   ```bash
   docker compose --env-file ../.env -f docker-compose.yaml -f docker-compose.onprem.yaml logs -f qwen embedder
   ```

The first startup downloads Qwen weights from Hugging Face and embedding NIM
artifacts from NGC. The vLLM service defaults to a 32K context, two concurrent
sequences, and 68% GPU-memory utilization to leave headroom for the embedding
NIM and the application. Tune `DEEP_RESEARCHER_VLLM_MAX_MODEL_LEN` or
`DEEP_RESEARCHER_VLLM_GPU_MEMORY_UTILIZATION` in `deploy/.env` only if startup
or workload testing shows that adjustment is needed.

### VS Code tasks

When the repository is open as a VS Code workspace on a Docker-enabled host, use
**Terminal > Run Task...** or `Ctrl+Shift+P` → **Tasks: Run Task** to select an
on-prem task. **On-prem: Start** validates the application config and Compose
files before starting the stack. **On-prem: Follow model logs** streams Qwen and
embedding-NIM logs until you stop the task; **On-prem: Stop** tears down the
containers and network while keeping the model and document volumes.

The tasks require `uv`, Docker Compose, and `deploy/.env` with `NGC_API_KEY` set.
Compose validation uses `config --quiet` so it does not print the resolved
configuration, which includes the NGC key.
These are standalone host tasks, not Workbench project-container tasks. A VS
Code terminal attached through Workbench runs inside the project container;
use Workbench's Compose controls for that setup instead.

### Use Pre-Built NGC Images

To skip the local build and pull pre-built images from the NGC container registry:

```bash
# Log in to the container registry
docker login nvcr.io

# Run with pre-built images (no --build flag)
cd deploy/compose
BACKEND_IMAGE=nvcr.io/nvidia/blueprint/aiq-agent:2.2.0 \
FRONTEND_IMAGE=nvcr.io/nvidia/blueprint/aiq-frontend:2.2.0 \
docker compose --env-file ../.env -f docker-compose.yaml up -d
```

You can also add the image variables to `deploy/.env` instead of passing them on the command line:

```bash
BACKEND_IMAGE=nvcr.io/nvidia/blueprint/aiq-agent:2.2.0
FRONTEND_IMAGE=nvcr.io/nvidia/blueprint/aiq-frontend:2.2.0
```

### Release Build

To build the production (release) image instead of the development image:

```bash
cd deploy/compose
BUILD_TARGET=release docker compose --env-file ../.env -f docker-compose.yaml up -d --build
```

The release image excludes the CLI and debug UI. Refer to [Docker Build System](./docker-build.md) for details on build targets.

## Port Configuration

If the default ports conflict with other services, override them in `deploy/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Backend API host port |
| `FRONTEND_PORT` | `3000` | Frontend UI host port |

```bash
PORT=8100 docker compose --env-file ../.env -f docker-compose.yaml up -d
```

**Note:** The backend API always runs on port 8000 inside the container. The `PORT` variable only changes the host port mapping.

Common conflicts:
- The NVIDIA RAG Blueprint `page-elements` service uses ports 8000--8002. Set `PORT=8100` to avoid this.
- Other development servers may occupy ports 8000, 8080, or 3000.

## Foundational RAG (FRAG) Integration

If you switch the backend to `configs/config_web_frag.yml`, you must run a compatible RAG server and ingest server separately and set these variables in `deploy/.env`:

```bash
RAG_SERVER_URL=http://rag-server:8081/v1
RAG_INGEST_URL=http://ingestor-server:8082/v1
```

Deploy the RAG services using the NVIDIA RAG Blueprint Docker guides:

- [Get Started With the NVIDIA RAG Blueprint (self-hosted)](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-self-hosted.md)
- [Deploy NVIDIA RAG Blueprint with Docker (NVIDIA-hosted models)](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/deploy-docker-nvidia-hosted.md)

### Cross-Stack Networking

When Deep Researcher Agent and RAG are deployed as separate Docker Compose stacks, the Deep Researcher Agent backend cannot resolve RAG service names (`rag-server`, `ingestor-server`) because the containers are on different Docker networks.

Connect the Deep Researcher Agent backend container to the RAG network after both stacks are running:

```bash
docker network connect nvidia-rag deep-researcher-agent
```

Then use the RAG service names directly in `deploy/.env`:

```bash
RAG_SERVER_URL=http://rag-server:8081/v1
RAG_INGEST_URL=http://ingestor-server:8082/v1
```

This must be re-run if the `deep-researcher-agent` container is recreated (for example, after `docker compose down && up`).

## Database Setup

### PostgreSQL (default)

The compose stack includes a PostgreSQL 16 container (`postgres:16-alpine`). On first startup with a fresh volume, the `init-db.sql` script runs automatically and:

1. Creates the `deep_researcher_checkpoints` database (the `deep_researcher_jobs` database is created by the `POSTGRES_DB` environment variable).
2. Grants permissions to the `deep-researcher` user.
3. Creates the `job_info` table in `deep_researcher_jobs` with performance indices.

Tables created automatically by the application at runtime:
- `job_events` -- created by `event_store.py` using SQLAlchemy.
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) checkpoint tables -- created by `AsyncPostgresSaver`.
- `summaries` -- created by `summary_store.py` if not present.

The PostgreSQL healthcheck verifies both `deep_researcher_jobs` and `deep_researcher_checkpoints` databases are ready before the backend starts.

### SQLite Alternative

For lightweight development without PostgreSQL, configure SQLite connection strings in `deploy/.env` (refer to [Database Settings](#database-settings) above). No `init-db.sql` is needed -- the application creates SQLite files on demand.

## Volume Mounts

| Volume | Mount point | Purpose |
|--------|-------------|---------|
| `../../configs` (bind mount, read-only) | `/app/configs` | Workflow configuration files |
| `deep-researcher-data` (named volume) | `/app/data` | LlamaIndex persistence (ChromaDB), SQLite databases |
| `postgres-data` (named volume) | `/var/lib/postgresql/data` | PostgreSQL data directory |

## Stopping and Cleanup

```bash
cd deploy/compose

# Stop containers (preserves data volumes)
docker compose --env-file ../.env -f docker-compose.yaml down

# Stop and remove volumes (deletes all database data)
docker compose --env-file ../.env -f docker-compose.yaml down -v
```

## Troubleshooting

### Check Container Logs

```bash
docker logs deep-researcher-agent -f
docker logs deep-researcher-blueprint-ui -f
docker logs deep-researcher-postgres -f
```

### Verify the Backend Is Healthy

```bash
curl http://localhost:8000/health
```

### Connect to the Database

```bash
docker exec -it deep-researcher-postgres psql -U deep_researcher -d deep_researcher_jobs
```

### Rebuild Without Cache

If containers fail to start or you suspect stale build layers:

```bash
cd deploy/compose
docker compose --env-file ../.env -f docker-compose.yaml down
docker compose --env-file ../.env -f docker-compose.yaml build --no-cache
docker compose --env-file ../.env -f docker-compose.yaml up -d
```

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Backend fails to start | Missing API keys in `deploy/.env` | Verify `NVIDIA_API_KEY` and at least one search key are set. |
| Frontend shows connection error | Backend not yet ready | Wait for the backend healthcheck to pass; check `docker logs deep-researcher-agent`. |
| Port already in use | Another service occupies 3000, 8000, or 5432 | Override with `PORT` or `FRONTEND_PORT` variables. |
| Database connection refused | PostgreSQL not healthy | Check `docker logs deep-researcher-postgres`; verify `init-db.sql` ran correctly. |
| FRAG mode fails to connect to RAG | Separate Docker networks | Run `docker network connect nvidia-rag deep-researcher-agent`. |
