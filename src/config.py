"""Global constants shared by every module."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
LOCKBOX_DIR = DATA / "lockbox"          # only written/read in Section 8
EXP = ROOT / "experiments"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"

# Lockbox: the most recent 12 months before "today" (2026-09-26) are sealed.
LOCKBOX_START = pd.Timestamp("2025-09-26 00:00", tz="UTC")
LOCKBOX_END = pd.Timestamp("2026-09-26 00:00", tz="UTC")
DEV_END = LOCKBOX_START                 # exclusive
OOS_START = pd.Timestamp("2018-01-01", tz="UTC")

START_CAPITAL_INR = 10_000.0
TARGET_MONTHLY_INR = 100_000.0

for p in (RAW, PROC, EXP / "configs", FIG):
    p.mkdir(parents=True, exist_ok=True)
