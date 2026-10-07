# syntax=docker/dockerfile:1
# Airflow 3.2.0 on Python 3.13 (Debian bookworm), the same interpreter as the laptop.
FROM apache/airflow:3.2.0-python3.13

# Java 17 runtime for pyspark's JVM, the same version as the laptop's Temurin 17.
# Headless JRE only: no compiler or GUI libraries, and apt lists are dropped to keep the layer small.
USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
USER airflow

# Each group is its own layer, ordered from most stable to most often changed.
# apache-airflow is pinned in every group so pip can never move Airflow's own dependencies.
# The cache mount keeps downloads outside the image, so a repeated build does not fetch them again.
RUN --mount=type=cache,target=/tmp/.cache/pip,uid=50000,gid=0 \
    pip install "apache-airflow==${AIRFLOW_VERSION}" \
    SQLAlchemy==2.0.52 \
    psycopg2-binary==2.9.13 \
    pydantic==2.13.5 \
    PyYAML==6.0.3 \
    python-dotenv==1.2.3 \
    loguru==0.7.3

RUN --mount=type=cache,target=/tmp/.cache/pip,uid=50000,gid=0 \
    pip install "apache-airflow==${AIRFLOW_VERSION}" \
    numpy==2.5.3 \
    pandas==3.0.5 \
    scipy==1.18.1

RUN --mount=type=cache,target=/tmp/.cache/pip,uid=50000,gid=0 \
    pip install "apache-airflow==${AIRFLOW_VERSION}" \
    pyspark==4.2.0

# xgboost-cpu is the same library without the CUDA/NCCL libraries a CPU-only machine never uses.
RUN --mount=type=cache,target=/tmp/.cache/pip,uid=50000,gid=0 \
    pip install "apache-airflow==${AIRFLOW_VERSION}" \
    scikit-learn==1.9.0 \
    xgboost-cpu==3.4.1 \
    optuna==5.0.0 \
    joblib==1.6.0

RUN --mount=type=cache,target=/tmp/.cache/pip,uid=50000,gid=0 \
    pip install "apache-airflow==${AIRFLOW_VERSION}" \
    matplotlib==3.11.1

# Fails the build if any installed package's declared requirements are broken.
RUN pip check

# src/ is bind-mounted under /opt/ml so `import src` resolves; dag_utils/ rides in dags/, which Airflow puts on sys.path.
ENV PYTHONPATH=/opt/ml
