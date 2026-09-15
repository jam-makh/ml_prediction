# Matches the Python already used locally (3.13), so the notebook and this
# image are the same interpreter. "slim" rather than "alpine": Alpine uses musl
# instead of glibc, which means no manylinux wheels -- numpy, scipy and
# scikit-learn would all be compiled from source, turning a one-minute build
# into a twenty-minute one for a smaller image nobody ships anywhere.
FROM python:3.13-slim

# No .pyc files: the source directory is bind-mounted from Windows, and cache
# written by root inside the container litters the host checkout.
ENV PYTHONDONTWRITEBYTECODE=1
# Unbuffered stdout, so training progress appears as it happens rather than
# arriving in a lump when the process exits.
ENV PYTHONUNBUFFERED=1

# libpq5 is the runtime PostgreSQL client library psycopg2-binary links
# against. Only the runtime lib is needed -- the -binary wheel ships its own
# compiled extension, so there is no libpq-dev and no build-essential here.
# Removing the apt lists in the same layer keeps them out of the image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies are copied and installed BEFORE the application code. Docker
# caches layers in order, so editing a file in src/ does not invalidate this
# step -- pip only reruns when requirements.txt itself changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as a non-root user with a fixed UID. The container writes trained models
# into a bind-mounted host directory; as root those files come back owned by
# root and are awkward to delete from the host side.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app
USER appuser

# Makes `python -m src.train` and `from src.data import ...` resolve against
# the bind-mounted /app rather than depending on the current directory.
ENV PYTHONPATH=/app

# Source is bind-mounted by compose, so it is deliberately not COPYied here:
# an edit on the host takes effect on the next run with no rebuild.
CMD ["python", "-m", "src.train"]
