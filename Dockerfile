FROM python:3.11-slim

RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /app
ENV TZ=Asia/Shanghai

# COPY local Docker CLI binary and install it
COPY build/docker-29.0.1.tgz /tmp/docker-29.0.1.tgz
RUN tar -xzf /tmp/docker-29.0.1.tgz --strip-components=1 -C /usr/local/bin/ docker/docker && \
    rm /tmp/docker-29.0.1.tgz && \
    chmod +x /usr/local/bin/docker && \
    /usr/local/bin/docker --version

# Install uv to a shared, world-readable location
ADD https://astral.sh/uv/0.11.6/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    rm -rf /root/.local/bin /uv-installer.sh

# Install dependencies from pyproject.toml
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_HTTP_TIMEOUT=1200 \
    UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" \
    uv sync --frozen --no-install-project

# Create a non-root user with fixed UID/GID
RUN useradd -m -u 1008 appuser && chown -R appuser:appuser /app /app/.venv

# Create necessary directories with correct permissions
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Set PATH explicitly so /usr/local/bin/docker is found for appuser
ENV PATH="/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin"

# Switch to non-root user for security
USER appuser

# Expose the service port (documentation only)
EXPOSE 8013