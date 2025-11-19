#!/bin/bash
set -e

ROLE="$1"   # webserver or scheduler

echo "Booting LedgerX Airflow Environment for role: ${ROLE}"

echo "Initializing Airflow DB..."
airflow db upgrade

echo "Ensuring admin user exists..."
if ! airflow users list | grep -w "admin" >/dev/null 2>&1; then
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --password admin \
        --role Admin \
        --email admin@example.com
    echo "Admin user created"
else
    echo "Admin user already exists"
fi

echo "Configuring Git + DVC..."
git config --global user.name "LedgerX Pipeline"
git config --global user.email "pipeline@ledgerx.local"

if [ "$ROLE" = "webserver" ]; then
    echo "Starting Airflow WEB SERVER..."
    exec airflow webserver

elif [ "$ROLE" = "scheduler" ]; then
    echo "Starting Airflow SCHEDULER..."
    exec airflow scheduler

else
    echo "Unknown role: ${ROLE}. Use 'webserver' or 'scheduler'."
    exit 1
fi
