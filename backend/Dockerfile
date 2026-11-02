# Use Python 3.11 slim image
FROM python:3.11.17-slim-trixie

# Install system dependencies including ffmpeg and unzip (for deno).
# Noto provides fallback glyphs for multilingual captions (including Hindi)
# and colour emoji. libass/fontconfig selects them when a chosen font lacks glyphs.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    curl \
    unzip \
    fonts-noto-core \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    fontconfig \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

# Install deno (required by yt-dlp for YouTube JS challenge solving)
RUN curl -fsSL https://deno.land/install.sh | sh
ENV DENO_DIR=/root/.deno
ENV PATH="/root/.deno/bin:${PATH}"

# Optional: Claude Code and Codex CLIs for LLM=claude-code:* / codex:*.
# docker-compose.llm-cli.yml turns this on and mounts the host's sign-ins.
# The CLIs run as llmcli, matching the host user that owns ~/.claude and
# ~/.codex, so refreshed credentials never become root-owned on the host.
ARG INSTALL_LLM_CLIS=false
ARG LLM_CLI_UID=1000
ARG LLM_CLI_GID=1000
ARG CLAUDE_CODE_VERSION=2.1.316
ARG CODEX_VERSION=0.166.0
RUN if [ "$INSTALL_LLM_CLIS" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends nodejs npm \
      && rm -rf /var/lib/apt/lists/* \
      && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" "@openai/codex@${CODEX_VERSION}" \
      && npm cache clean --force \
      && groupadd -o -g "$LLM_CLI_GID" llmcli \
      && useradd -o -u "$LLM_CLI_UID" -g "$LLM_CLI_GID" -d /llm-home -M -s /usr/sbin/nologin llmcli \
      && mkdir -p /llm-home && chown llmcli:llmcli /llm-home; \
    fi

# Install uv package manager
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy pyproject.toml and uv.lock for dependency installation
COPY pyproject.toml uv.lock* ./

# New self-hosted setups can select CPU wheels without changing legacy builds.
ARG BACKEND_CPU_ONLY=false
RUN uv venv .venv && \
    if [ "$BACKEND_CPU_ONLY" = "true" ]; then \
      uv sync --frozen --extra cpu; \
    else \
      uv sync --frozen --extra cuda; \
    fi

# yt-dlp and its PO-token plugin are installed from uv.lock above.

# Copy source code
COPY src/ ./src/

# Copy fonts and transitions directories
COPY fonts/ ./fonts/
COPY transitions/ ./transitions/

# Create necessary directories for video processing
RUN mkdir -p /app/uploads /app/clips /app/logs /tmp

# Expose the port FastAPI will run on
EXPOSE 8000

# Activate virtual environment and start the application
CMD [".venv/bin/uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
