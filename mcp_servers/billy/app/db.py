"""SQLite persistence layer for the Billy stub server."""

import os
import sqlite3
from contextlib import contextmanager

_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    type              TEXT NOT NULL,
    country           TEXT NOT NULL DEFAULT 'DK',
    street            TEXT,
    city              TEXT,
    zipcode           TEXT,
    phone             TEXT,
    email             TEXT,
    contact_person_id TEXT,
    registration_no   TEXT,
    is_customer       INTEGER NOT NULL DEFAULT 1,
    is_supplier       INTEGER NOT NULL DEFAULT 0,
    created_time      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    product_no  TEXT,
    unit        TEXT NOT NULL DEFAULT 'pcs',
    is_archived INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS product_prices (
    id         TEXT PRIMARY KEY,
    product_id TEXT NOT NULL REFERENCES products(id),
    unit_price REAL NOT NULL,
    currency   TEXT NOT NULL DEFAULT 'DKK'
);

CREATE TABLE IF NOT EXISTS invoices (
    id               TEXT PRIMARY KEY,
    invoice_no       TEXT NOT NULL,
    contact_id       TEXT NOT NULL,
    customer_name    TEXT,
    entry_date       TEXT NOT NULL,
    due_date         TEXT NOT NULL,
    state            TEXT NOT NULL DEFAULT 'draft',
    sent_state       TEXT NOT NULL DEFAULT 'unsent',
    amount           REAL NOT NULL DEFAULT 0,
    tax              REAL NOT NULL DEFAULT 0,
    gross_amount     REAL NOT NULL DEFAULT 0,
    currency         TEXT NOT NULL DEFAULT 'DKK',
    exchange_rate    REAL NOT NULL DEFAULT 1.0,
    balance          REAL NOT NULL DEFAULT 0,
    is_paid          INTEGER NOT NULL DEFAULT 0,
    payment_terms    TEXT,
    tax_mode         TEXT NOT NULL DEFAULT 'excl',
    approved_time    TEXT,
    created_time     TEXT NOT NULL,
    download_url     TEXT,
    contact_message  TEXT,
    line_description TEXT
);

CREATE TABLE IF NOT EXISTS invoice_lines (
    id          TEXT PRIMARY KEY,
    invoice_id  TEXT NOT NULL REFERENCES invoices(id),
    product_id  TEXT,
    description TEXT,
    quantity    REAL NOT NULL DEFAULT 1,
    unit_price  REAL NOT NULL DEFAULT 0,
    unit        TEXT NOT NULL DEFAULT 'pcs',
    amount      REAL NOT NULL DEFAULT 0,
    tax         REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS quotes (
    id           TEXT PRIMARY KEY,
    quote_no     TEXT NOT NULL,
    contact_id   TEXT NOT NULL,
    customer_name TEXT,
    entry_date   TEXT NOT NULL,
    expiry_date  TEXT NOT NULL,
    state        TEXT NOT NULL DEFAULT 'open',
    amount       REAL NOT NULL DEFAULT 0,
    tax          REAL NOT NULL DEFAULT 0,
    gross_amount REAL NOT NULL DEFAULT 0,
    currency     TEXT NOT NULL DEFAULT 'DKK',
    created_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quote_lines (
    id          TEXT PRIMARY KEY,
    quote_id    TEXT NOT NULL REFERENCES quotes(id),
    product_id  TEXT,
    description TEXT,
    quantity    REAL NOT NULL DEFAULT 1,
    unit_price  REAL NOT NULL DEFAULT 0,
    amount      REAL NOT NULL DEFAULT 0,
    tax         REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS expenses (
    id           TEXT PRIMARY KEY,
    vendor       TEXT NOT NULL,
    amount       REAL NOT NULL,
    tax          REAL NOT NULL DEFAULT 0,
    gross_amount REAL NOT NULL,
    currency     TEXT NOT NULL DEFAULT 'DKK',
    date         TEXT NOT NULL,
    category     TEXT NOT NULL DEFAULT 'other',
    is_fixed     INTEGER NOT NULL DEFAULT 0,
    description  TEXT,
    contact_id   TEXT,
    status       TEXT NOT NULL DEFAULT 'approved',
    created_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bank_accounts (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    bank_name    TEXT NOT NULL DEFAULT 'Nordea Bank',
    currency     TEXT NOT NULL DEFAULT 'DKK',
    balance      REAL NOT NULL DEFAULT 0,
    account_no   TEXT,
    created_time TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bank_transactions (
    id                 TEXT PRIMARY KEY,
    account_id         TEXT NOT NULL REFERENCES bank_accounts(id),
    date               TEXT NOT NULL,
    description        TEXT NOT NULL,
    amount             REAL NOT NULL,
    type               TEXT NOT NULL DEFAULT 'credit',
    matched_invoice_id TEXT,
    created_time       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""

_SEED = """
INSERT OR IGNORE INTO customers VALUES
('cus_001','Acme A/S','company','DK','Vesterbrogade 1','København V','1620',
 '+45 70 10 20 30','kontakt@acme.dk','cp_001','12345678',1,0,'2023-01-15T09:00:00Z'),
('cus_002','Nordisk Tech ApS','company','DK','Nørrebrogade 42','København N','2200',
 '+45 33 44 55 66','info@nordisktech.dk','cp_002','87654321',1,0,'2023-03-20T11:30:00Z'),
('cus_003','Lars Hansen','person','DK','Åboulevard 15','Aarhus C','8000',
 '+45 42 33 21 10','lars@hansen.dk','cp_003',NULL,1,0,'2024-06-01T08:00:00Z');

INSERT OR IGNORE INTO products VALUES
('prod_001','Konsulentydelser','Timebaseret konsulentbistand','SRV-001','hours',0),
('prod_002','Softwarelicens','Månedlig softwarelicens','LIC-001','pcs',0),
('prod_003','Support & Vedligehold','Månedlig supportaftale','SUP-001','pcs',0),
('prod_004','Uddannelse','Kursus og oplæring (pr. dag)','TRN-001','days',0),
('prod_005','Rejseomkostninger','Viderefakturering af rejseomkostninger','EXP-001','pcs',1);

INSERT OR IGNORE INTO product_prices VALUES
('price_001a','prod_001',1000.00,'DKK'),
('price_002a','prod_002',5000.00,'DKK'),
('price_003a','prod_003',2500.00,'DKK'),
('price_004a','prod_004',8000.00,'DKK'),
('price_005a','prod_005',0.00,'DKK');

INSERT OR IGNORE INTO invoices VALUES
('inv_001','2024-001','cus_001','Acme A/S','2024-01-15','2024-01-22',
 'approved','sent',10000.00,2500.00,12500.00,'DKK',1.0,0.00,1,'net 7 days','excl',
 '2024-01-15T10:00:00Z','2024-01-15T09:55:00Z',
 'https://app.billy.dk/invoices/inv_001/download',NULL,'Konsulentydelser januar'),
('inv_002','2024-002','cus_002','Nordisk Tech ApS','2024-02-01','2024-02-08',
 'approved','sent',5000.00,1250.00,6250.00,'DKK',1.0,6250.00,0,'net 7 days','excl',
 '2024-02-01T09:00:00Z','2024-02-01T08:50:00Z',
 'https://app.billy.dk/invoices/inv_002/download',NULL,'Softwarelicens februar'),
('inv_003','2024-003','cus_001','Acme A/S','2024-03-01','2024-03-08',
 'draft','unsent',3000.00,750.00,3750.00,'DKK',1.0,3750.00,0,'net 7 days','excl',
 NULL,'2024-03-01T14:00:00Z',NULL,NULL,'Support marts');

INSERT OR IGNORE INTO invoice_lines VALUES
('line_001a','inv_001','prod_001','Konsulentydelser',10,1000.00,'hours',10000.00,2500.00),
('line_002a','inv_002','prod_002','Softwarelicens',1,5000.00,'pcs',5000.00,1250.00),
('line_003a','inv_003','prod_001','Support',3,1000.00,'hours',3000.00,750.00);

INSERT OR IGNORE INTO quotes VALUES
('quo_001','Q-2024-001','cus_001','Acme A/S','2024-03-01','2024-03-31','open',
 20000.00,5000.00,25000.00,'DKK','2024-03-01T09:00:00Z'),
('quo_002','Q-2024-002','cus_002','Nordisk Tech ApS','2024-04-01','2024-04-30','accepted',
 15000.00,3750.00,18750.00,'DKK','2024-04-01T10:00:00Z'),
('quo_003','Q-2025-001','cus_001','Acme A/S','2025-01-10','2025-02-10','declined',
 8000.00,2000.00,10000.00,'DKK','2025-01-10T08:30:00Z');

INSERT OR IGNORE INTO quote_lines VALUES
('qline_001a','quo_001','prod_001','Konsulentydelser Q1',20,1000.00,20000.00,5000.00),
('qline_002a','quo_002','prod_002','Softwarelicens Q2',3,5000.00,15000.00,3750.00),
('qline_003a','quo_003','prod_004','Uddannelsesdage',1,8000.00,8000.00,2000.00);

INSERT OR IGNORE INTO expenses VALUES
('exp_001','Lejemål kontor',12500.00,0.00,12500.00,'DKK','2024-01-01','rent',1,'Månedlig husleje',NULL,'approved','2024-01-01T09:00:00Z'),
('exp_002','AWS Cloud Hosting',2300.00,575.00,2875.00,'DKK','2024-01-15','software',0,'Cloud infrastructure',NULL,'approved','2024-01-15T10:00:00Z'),
('exp_003','Teamfrokost',850.00,212.50,1062.50,'DKK','2024-01-20','meals',0,'Teamfrokost januar',NULL,'approved','2024-01-20T12:00:00Z'),
('exp_004','Google Ads',5000.00,0.00,5000.00,'DKK','2024-02-01','marketing',0,'Online annoncering',NULL,'approved','2024-02-01T08:00:00Z'),
('exp_005','Dinero regnskabssoftware',1500.00,375.00,1875.00,'DKK','2024-02-01','software',1,'Månedlig licens',NULL,'approved','2024-02-01T09:00:00Z'),
('exp_006','Lejemål kontor',12500.00,0.00,12500.00,'DKK','2024-02-01','rent',1,'Månedlig husleje',NULL,'approved','2024-02-01T09:00:00Z'),
('exp_007','Kontorinteriør',3200.00,800.00,4000.00,'DKK','2024-03-10','office',0,'Nye stole og borde',NULL,'approved','2024-03-10T14:00:00Z'),
('exp_008','Freelancer - grafisk design',8000.00,2000.00,10000.00,'DKK','2024-03-15','professional_services',0,'Logodesign projekt',NULL,'approved','2024-03-15T10:00:00Z');

INSERT OR IGNORE INTO bank_accounts VALUES
('bank_001','Driftskonto','Nordea Bank','DKK',125000.00,'DK3000102030405060','2023-01-01T09:00:00Z'),
('bank_002','Skattekonto','Nordea Bank','DKK',35000.00,'DK3000102030405061','2023-01-01T09:00:00Z');

INSERT OR IGNORE INTO bank_transactions VALUES
('txn_001','bank_001','2024-01-22','Acme A/S - Faktura 2024-001',12500.00,'credit','inv_001','2024-01-22T14:00:00Z'),
('txn_002','bank_001','2024-01-31','Leje jan 2024',12500.00,'debit',NULL,'2024-01-31T09:00:00Z'),
('txn_003','bank_001','2024-02-15','AWS Cloud jan 2024',2875.00,'debit',NULL,'2024-02-15T10:00:00Z'),
('txn_004','bank_001','2024-02-20','Nordisk Tech ApS - indbetaling',6250.00,'credit',NULL,'2024-02-20T11:00:00Z'),
('txn_005','bank_001','2024-03-01','Google Ads feb 2024',5000.00,'debit',NULL,'2024-03-01T08:00:00Z'),
('txn_006','bank_001','2024-03-10','Kontorinventar',4000.00,'debit',NULL,'2024-03-10T14:00:00Z');

INSERT OR IGNORE INTO counters VALUES
('customer',4),
('invoice',4),
('product',6),
('quote',4),
('expense',9),
('bank_transaction',7);
"""


def _db_path() -> str:
    return os.getenv("BILLY_DB", "billy.db")


@contextmanager
def get_conn():
    """Yield a WAL-mode SQLite connection; commit on success, rollback on error."""
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def next_id(conn: sqlite3.Connection, counter_name: str) -> int:
    """Return the current counter value and increment it atomically."""
    val = conn.execute(
        "SELECT value FROM counters WHERE name = ?", (counter_name,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE counters SET value = value + 1 WHERE name = ?", (counter_name,)
    )
    return val


def init_db() -> None:
    """Create schema and seed initial data. Safe to call multiple times."""
    conn = sqlite3.connect(_db_path())
    try:
        conn.executescript(_SCHEMA)
        conn.executescript(_SEED)
    finally:
        conn.close()


# Initialise on first import so any process that imports a tool gets a ready DB.
init_db()
