#!/bin/sh
# Creates the MLflow and test databases next to the application database (runs once on an empty volume).
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
CREATE DATABASE mlflow;
CREATE DATABASE treadops_test;
SQL
