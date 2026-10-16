# A clean, repeatable machine for Project 1: same Python, same locked packages,
# every time. The data is NEVER copied into the image -- it is mounted
# read-only when the container runs (`make docker-check`), and the container
# runs with no network.
FROM ghcr.io/astral-sh/uv:0.12.6-python3.11-trixie-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies first, so code changes don't reinstall everything.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra dev --no-install-project

COPY . .
RUN uv sync --frozen --extra dev \
    && useradd --create-home runner \
    && chown -R runner /app
USER runner

CMD ["scripts/docker_check.sh"]
