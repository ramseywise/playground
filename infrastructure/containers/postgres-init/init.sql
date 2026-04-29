-- Creates application databases on first Postgres startup.
-- This script runs only when the data directory is empty (fresh volume).
SELECT 'CREATE DATABASE va_support_rag'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'va_support_rag')\gexec
