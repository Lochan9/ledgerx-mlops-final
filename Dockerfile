# ---------- Base Image ----------
FROM apache/airflow:2.9.3-python3.12

# ---------- Switch to root for system-level installs ----------
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
        poppler-utils \
        ghostscript \
        git && \
    rm -rf /var/lib/apt/lists/*

# ---------- Switch to airflow for pip ----------
USER airflow

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /requirements.txt

# ---------- Project Setup ----------
USER root

RUN mkdir -p /opt/airflow/{dags,src,data,tests,reports}

COPY ./dags /opt/airflow/dags/
COPY ./src /opt/airflow/src/
COPY ./tests /opt/airflow/tests/
COPY ./reports /opt/airflow/reports/

# COPY THE SCRIPT (clean LF file)
COPY ./start_ledgerx.sh /opt/airflow/start_ledgerx.sh

# Force convert CRLF → LF inside Linux image
RUN sed -i 's/\r$//' /opt/airflow/start_ledgerx.sh && \
    chmod +x /opt/airflow/start_ledgerx.sh


# Permissions
RUN chmod +x /opt/airflow/start_ledgerx.sh && \
    chown -R airflow:0 /opt/airflow

# ---------- Environment Variables ----------
ENV AIRFLOW_HOME=/opt/airflow \
    PYTHONPATH="/opt/airflow:${PYTHONPATH}"

WORKDIR /opt/airflow

# ---------- Default CMD ----------
CMD ["bash", "/opt/airflow/start_ledgerx.sh", "webserver"]
