# ----------------------------------------------------------------------
# LEDGERX AIRFLOW DOCKERFILE (FINAL)
# Includes: joblib, mlflow, sklearn, catboost, shap, matplotlib, pandas
# ----------------------------------------------------------------------

FROM apache/airflow:2.9.3-python3.12

# ----------------------------------------------------------------------
# Install system-level dependencies
# ----------------------------------------------------------------------
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
        ghostscript \
        git \
        build-essential \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------
# Switch to airflow for pip installs
# ----------------------------------------------------------------------
USER airflow

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt


# ----------------------------------------------------------------------
# Project Folder Copy (for DAGs, src, reports, tests)
# ----------------------------------------------------------------------
USER root

RUN mkdir -p /opt/airflow/{dags,src,data,reports,tests}

COPY ./dags /opt/airflow/dags/
COPY ./src /opt/airflow/src/
COPY ./reports /opt/airflow/reports/
COPY ./tests /opt/airflow/tests/

# Startup script
COPY ./start_ledgerx.sh /opt/airflow/start_ledgerx.sh

# Convert CRLF → LF to avoid bash errors
RUN sed -i 's/\r$//' /opt/airflow/start_ledgerx.sh && \
    chmod +x /opt/airflow/start_ledgerx.sh

# Permissions
RUN chown -R airflow:0 /opt/airflow

# ----------------------------------------------------------------------
# Environment Variables
# ----------------------------------------------------------------------
ENV AIRFLOW_HOME=/opt/airflow \
    PYTHONPATH="/opt/airflow:${PYTHONPATH}"

WORKDIR /opt/airflow

# ----------------------------------------------------------------------
# Default CMD
# ----------------------------------------------------------------------
CMD ["bash", "/opt/airflow/start_ledgerx.sh", "webserver"]
