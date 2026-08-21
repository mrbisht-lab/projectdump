#!/usr/bin/env python3

"""
SmartStore
AI-Powered Intelligent Storage Optimizer for Linux

Single-file prototype.

Features:
    - Filesystem scanning
    - Storage statistics
    - File feature extraction
    - Intelligent cleanup scoring
    - Duplicate detection using SHA-256
    - Storage growth forecasting
    - Risk-aware recommendations
    - Dry-run cleanup
    - Quarantine instead of immediate deletion
    - Undo/quarantine restore
    - SQLite history
    - Optional scikit-learn model
    - CLI interface

Install optional ML dependency:
    pip install scikit-learn

Examples:
    python smartstore.py scan ~/Downloads
    python smartstore.py analyze ~/Downloads
    python smartstore.py recommend ~/Downloads
    python smartstore.py duplicates ~/Downloads
    python smartstore.py forecast
    python smartstore.py clean ~/Downloads --dry-run
    python smartstore.py clean ~/Downloads --confidence 95
    python smartstore.py history
    python smartstore.py undo

IMPORTANT:
    This is a prototype. Run it on user-owned directories first.
    Never run filesystem-cleaning software as root until thoroughly tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sqlite3
import statistics
import sys
import time

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "SmartStore"
VERSION = "0.1.0"

HOME = Path.home()
DATA_DIR = HOME / ".smartstore"
QUARANTINE_DIR = DATA_DIR / "quarantine"
DATABASE = DATA_DIR / "smartstore.db"

MAX_FILE_SIZE_FOR_HASH = 10 * 1024 * 1024 * 1024  # 10 GB

DEFAULT_MIN_SIZE_MB = 10

# Paths that should never be cleaned automatically.
PROTECTED_SYSTEM_PATHS = {
    Path("/"),
    Path("/boot"),
    Path("/etc"),
    Path("/bin"),
    Path("/sbin"),
    Path("/usr"),
    Path("/lib"),
    Path("/lib64"),
    Path("/opt"),
    Path("/root"),
    Path("/sys"),
    Path("/proc"),
    Path("/dev"),
    Path("/run"),
    Path("/var/lib"),
}

# File extensions which often represent temporary/cache data.
TEMP_EXTENSIONS = {
    ".tmp",
    ".temp",
    ".cache",
    ".bak",
    ".old",
    ".swp",
    ".part",
}

# Common cache directory names.
CACHE_NAMES = {
    ".cache",
    "cache",
    "caches",
    "__pycache__",
    ".thumbnails",
    "thumbnails",
}

# Common log extensions.
LOG_EXTENSIONS = {
    ".log",
    ".logs",
}


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class FileRecord:
    path: str
    size: int
    modified: float
    accessed: float
    extension: str
    age_days: float
    unused_days: float
    is_cache: bool
    is_temp: bool
    is_log: bool
    is_protected: bool
    is_hidden: bool


@dataclass
class Recommendation:
    path: str
    action: str
    confidence: float
    risk: str
    score: float
    size: int
    reason: str


# ============================================================
# GENERAL UTILITIES
# ============================================================

def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)


def now() -> float:
    return time.time()


def human_size(size: int | float) -> str:
    value = float(size)

    units = ["B", "KB", "MB", "GB", "TB", "PB"]

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} EB"


def human_time(seconds: float) -> str:
    seconds = max(0, seconds)

    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)

    if days:
        return f"{days}d {hours}h"

    if hours:
        return f"{hours}h {minutes}m"

    return f"{minutes}m"


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def safe_resolve(path: Path) -> Optional[Path]:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return None


# ============================================================
# SAFETY ENGINE
# ============================================================

class SafetyEngine:
    """
    Determines whether a file is eligible for recommendation.

    The AI/recommendation layer is NOT allowed to override this layer.
    """

    @staticmethod
    def is_protected(path: Path) -> bool:
        resolved = safe_resolve(path)

        if resolved is None:
            return True

        # Never touch SmartStore's own quarantine/database.
        if DATA_DIR in resolved.parents or resolved == DATA_DIR:
            return True

        # Never clean protected system directories.
        for protected in PROTECTED_SYSTEM_PATHS:
            if resolved == protected:
                return True

            try:
                if protected in resolved.parents:
                    return True
            except Exception:
                return True

        return False

    @staticmethod
    def is_user_data(path: Path) -> bool:
        resolved = safe_resolve(path)

        if resolved is None:
            return True

        important = {
            "Documents",
            "Pictures",
            "Videos",
            "Music",
            "Desktop",
            "Projects",
        }

        return any(part in important for part in resolved.parts)

    @staticmethod
    def risk_level(
        path: Path,
        is_cache: bool,
        is_temp: bool,
        is_log: bool,
    ) -> str:

        if SafetyEngine.is_protected(path):
            return "CRITICAL"

        if SafetyEngine.is_user_data(path):
            return "HIGH"

        if is_cache or is_temp:
            return "LOW"

        if is_log:
            return "MEDIUM"

        return "MEDIUM"


# ============================================================
# DATABASE
# ============================================================

class Database:
    def __init__(self, database: Path = DATABASE):
        ensure_directories()

        self.connection = sqlite3.connect(database)
        self.connection.row_factory = sqlite3.Row

        self.initialize()

    def initialize(self):
        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                path TEXT NOT NULL,
                total_size INTEGER NOT NULL,
                file_count INTEGER NOT NULL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                original_path TEXT NOT NULL,
                quarantine_path TEXT,
                action TEXT NOT NULL,
                size INTEGER NOT NULL,
                reason TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                path TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                user_action TEXT NOT NULL
            )
            """
        )

        self.connection.commit()

    def record_scan(
        self,
        path: str,
        total_size: int,
        file_count: int,
    ):
        self.connection.execute(
            """
            INSERT INTO scans
            (timestamp, path, total_size, file_count)
            VALUES (?, ?, ?, ?)
            """,
            (now(), path, total_size, file_count),
        )

        self.connection.commit()

    def record_action(
        self,
        original_path: str,
        quarantine_path: str,
        action: str,
        size: int,
        reason: str,
    ):
        self.connection.execute(
            """
            INSERT INTO actions
            (
                timestamp,
                original_path,
                quarantine_path,
                action,
                size,
                reason
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                now(),
                original_path,
                quarantine_path,
                action,
                size,
                reason,
            ),
        )

        self.connection.commit()

    def record_feedback(
        self,
        path: str,
        recommendation: str,
        user_action: str,
    ):
        self.connection.execute(
            """
            INSERT INTO feedback
            (
                timestamp,
                path,
                recommendation,
                user_action
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                now(),
                path,
                recommendation,
                user_action,
            ),
        )

        self.connection.commit()

    def get_recent_scans(self, limit: int = 30):
        cursor = self.connection.execute(
            """
            SELECT *
            FROM scans
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )

        return cursor.fetchall()

    def get_actions(self, limit: int = 30):
        cursor = self.connection.execute(
            """
            SELECT *
            FROM actions
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )

        return cursor.fetchall()


# ============================================================
# FILE SCANNER
# ============================================================

class FileScanner:

    def __init__(
        self,
        minimum_size_mb: int = DEFAULT_MIN_SIZE_MB,
    ):
        self.minimum_size = minimum_size_mb * 1024 * 1024

    def scan(self, root: Path) -> list[FileRecord]:

        root = root.expanduser()

        if not root.exists():
            raise FileNotFoundError(f"Path does not exist: {root}")

        if not root.is_dir():
            raise ValueError(f"Not a directory: {root}")

        records = []

        current_time = now()

        for directory, dirnames, filenames in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):

            directory_path = Path(directory)

            # Don't enter SmartStore's own data.
            dirnames[:] = [
                d
                for d in dirnames
                if (directory_path / d) != DATA_DIR
            ]

            for filename in filenames:

                path = directory_path / filename

                try:
                    if path.is_symlink():
                        continue

                    stat = path.stat()

                    if stat.st_size < self.minimum_size:
                        continue

                    age_days = (
                        current_time - stat.st_mtime
                    ) / 86400

                    unused_days = (
                        current_time - stat.st_atime
                    ) / 86400

                    extension = path.suffix.lower()

                    is_cache = self.detect_cache(path)
                    is_temp = extension in TEMP_EXTENSIONS
                    is_log = extension in LOG_EXTENSIONS

                    protected = SafetyEngine.is_protected(path)

                    record = FileRecord(
                        path=str(path),
                        size=stat.st_size,
                        modified=stat.st_mtime,
                        accessed=stat.st_atime,
                        extension=extension,
                        age_days=max(0, age_days),
                        unused_days=max(0, unused_days),
                        is_cache=is_cache,
                        is_temp=is_temp,
                        is_log=is_log,
                        is_protected=protected,
                        is_hidden=filename.startswith("."),
                    )

                    records.append(record)

                except (
                    PermissionError,
                    FileNotFoundError,
                    OSError,
                ):
                    continue

        return records

    @staticmethod
    def detect_cache(path: Path) -> bool:
        return any(
            part.lower() in CACHE_NAMES
            for part in path.parts
        )


# ============================================================
# DUPLICATE DETECTION
# ============================================================

class DuplicateDetector:

    @staticmethod
    def hash_file(
        path: Path,
        chunk_size: int = 1024 * 1024,
    ) -> Optional[str]:

        try:
            hasher = hashlib.sha256()

            with path.open("rb") as file:
                while True:
                    chunk = file.read(chunk_size)

                    if not chunk:
                        break

                    hasher.update(chunk)

            return hasher.hexdigest()

        except (
            PermissionError,
            FileNotFoundError,
            OSError,
        ):
            return None

    def find_duplicates(
        self,
        records: list[FileRecord],
    ) -> dict[str, list[str]]:

        # First group by size.
        size_groups: dict[int, list[FileRecord]] = {}

        for record in records:
            size_groups.setdefault(
                record.size,
                [],
            ).append(record)

        duplicates: dict[str, list[str]] = {}

        for size, group in size_groups.items():

            if len(group) < 2:
                continue

            if size > MAX_FILE_SIZE_FOR_HASH:
                continue

            hashes: dict[str, list[str]] = {}

            for record in group:

                digest = self.hash_file(
                    Path(record.path)
                )

                if digest is None:
                    continue

                hashes.setdefault(
                    digest,
                    [],
                ).append(record.path)

            for digest, paths in hashes.items():

                if len(paths) > 1:
                    duplicates[digest] = paths

        return duplicates


# ============================================================
# STORAGE FORECASTER
# ============================================================

class StorageForecaster:

    def __init__(self, database: Database):
        self.database = database

    def forecast(
        self,
        days_ahead: int = 30,
    ) -> Optional[dict]:

        scans = list(
            reversed(
                self.database.get_recent_scans(30)
            )
        )

        if len(scans) < 2:
            return None

        timestamps = [
            row["timestamp"]
            for row in scans
        ]

        sizes = [
            row["total_size"]
            for row in scans
        ]

        # Simple linear regression.
        t0 = timestamps[0]

        x = [
            (t - t0) / 86400
            for t in timestamps
        ]

        y = sizes

        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)

        denominator = sum(
            (value - x_mean) ** 2
            for value in x
        )

        if denominator == 0:
            return None

        slope = sum(
            (x[i] - x_mean)
            * (y[i] - y_mean)
            for i in range(len(x))
        ) / denominator

        intercept = y_mean - slope * x_mean

        future_x = max(x) + days_ahead

        predicted_size = (
            slope * future_x
            + intercept
        )

        current_size = sizes[-1]

        growth_per_day = slope

        return {
            "current_size": current_size,
            "predicted_size": predicted_size,
            "growth_per_day": growth_per_day,
            "days_ahead": days_ahead,
        }


# ============================================================
# AI / INTELLIGENT RECOMMENDATION ENGINE
# ============================================================

class RecommendationEngine:
    """
    Lightweight intelligent scoring engine.

    This version works without external AI services.

    If scikit-learn is installed, a RandomForest model can
    optionally be trained later using SmartStore feedback data.

    The current scoring model combines:
        - size
        - age
        - unused period
        - cache status
        - temporary-file status
        - log status
        - protected status
        - user-data risk
    """

    def __init__(self):
        self.ml_model = None

        try:
            from sklearn.ensemble import RandomForestClassifier

            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
            )

        except ImportError:
            self.ml_model = None

    @staticmethod
    def normalize_size(size: int) -> float:

        # Log scaling prevents huge files from completely dominating.
        return min(
            1.0,
            math.log10(
                max(size, 1)
            ) / 12,
        )

    @staticmethod
    def calculate_score(
        record: FileRecord,
    ) -> float:

        if record.is_protected:
            return 0.0

        score = 0.0

        # Large files provide greater storage benefit.
        score += (
            RecommendationEngine.normalize_size(
                record.size
            ) * 0.20
        )

        # Old files.
        age_score = min(
            1.0,
            record.age_days / 365,
        )

        score += age_score * 0.15

        # Long unused period.
        unused_score = min(
            1.0,
            record.unused_days / 180,
        )

        score += unused_score * 0.25

        # Cache files.
        if record.is_cache:
            score += 0.30

        # Temporary files.
        if record.is_temp:
            score += 0.20

        # Logs.
        if record.is_log:
            score += 0.12

        # Hidden files are slightly more conservative.
        if record.is_hidden:
            score -= 0.03

        # User data gets a large penalty.
        if SafetyEngine.is_user_data(
            Path(record.path)
        ):
            score -= 0.45

        return max(
            0.0,
            min(1.0, score),
        )

    def recommend(
        self,
        record: FileRecord,
    ) -> Recommendation:

        path = Path(record.path)

        score = self.calculate_score(record)

        risk = SafetyEngine.risk_level(
            path,
            record.is_cache,
            record.is_temp,
            record.is_log,
        )

        if record.is_protected:

            return Recommendation(
                path=record.path,
                action="KEEP",
                confidence=1.0,
                risk="CRITICAL",
                score=0.0,
                size=record.size,
                reason="Protected system location",
            )

        if risk == "HIGH":

            return Recommendation(
                path=record.path,
                action="REVIEW",
                confidence=0.90,
                risk=risk,
                score=score,
                size=record.size,
                reason=(
                    "Potentially important user data; "
                    "manual review required"
                ),
            )

        if score >= 0.85:

            reason_parts = []

            if record.is_cache:
                reason_parts.append(
                    "application/cache directory"
                )

            if record.is_temp:
                reason_parts.append(
                    "temporary file"
                )

            if record.unused_days > 30:
                reason_parts.append(
                    f"unused for {record.unused_days:.0f} days"
                )

            reason = "; ".join(reason_parts)

            if not reason:
                reason = "high cleanup score"

            confidence = min(
                0.99,
                0.75 + score * 0.25,
            )

            return Recommendation(
                path=record.path,
                action="CLEAN",
                confidence=confidence,
                risk=risk,
                score=score,
                size=record.size,
                reason=reason,
            )

        if score >= 0.55:

            return Recommendation(
                path=record.path,
                action="REVIEW",
                confidence=0.70 + score * 0.2,
                risk=risk,
                score=score,
                size=record.size,
                reason=(
                    "Moderate cleanup potential; "
                    "review before removal"
                ),
            )

        return Recommendation(
            path=record.path,
            action="KEEP",
            confidence=1.0 - score,
            risk=risk,
            score=score,
            size=record.size,
            reason="No strong cleanup signal",
        )


# ============================================================
# QUARANTINE / CLEANUP
# ============================================================

class CleanupManager:

    def __init__(self, database: Database):
        self.database = database

        ensure_directories()

    def quarantine(
        self,
        recommendation: Recommendation,
        dry_run: bool = True,
    ) -> bool:

        source = Path(recommendation.path)

        if not source.exists():
            print(
                f"[SKIP] File no longer exists: {source}"
            )
            return False

        if SafetyEngine.is_protected(source):
            print(
                f"[BLOCKED] Protected path: {source}"
            )
            return False

        if recommendation.risk in {
            "HIGH",
            "CRITICAL",
        }:
            print(
                f"[BLOCKED] High-risk path: {source}"
            )
            return False

        if dry_run:

            print(
                f"[DRY-RUN] Would quarantine: "
                f"{source}"
            )

            return True

        # Generate unique destination.
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        safe_name = (
            source.name
            + "_"
            + timestamp
            + "_"
            + hashlib.sha1(
                str(source).encode()
            ).hexdigest()[:8]
        )

        destination = QUARANTINE_DIR / safe_name

        try:

            shutil.move(
                str(source),
                str(destination),
            )

            self.database.record_action(
                original_path=str(source),
                quarantine_path=str(destination),
                action="QUARANTINE",
                size=recommendation.size,
                reason=recommendation.reason,
            )

            print(
                f"[QUARANTINED] "
                f"{source} -> {destination}"
            )

            return True

        except (
            PermissionError,
            OSError,
            shutil.Error,
        ) as exc:

            print(
                f"[ERROR] Could not quarantine "
                f"{source}: {exc}"
            )

            return False

    def undo_last(self) -> bool:

        actions = self.database.get_actions(1)

        if not actions:
            print("No actions to undo.")
            return False

        action = actions[0]

        source = Path(
            action["quarantine_path"]
        )

        destination = Path(
            action["original_path"]
        )

        if not source.exists():
            print(
                "Quarantine file no longer exists."
            )
            return False

        if destination.exists():
            print(
                "Original location already exists. "
                "Refusing to overwrite."
            )
            return False

        try:

            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.move(
                str(source),
                str(destination),
            )

            print(
                f"[RESTORED] {destination}"
            )

            return True

        except OSError as exc:

            print(
                f"[ERROR] Restore failed: {exc}"
            )

            return False


# ============================================================
# SYSTEM INFORMATION
# ============================================================

def get_disk_usage(path: Path):

    try:
        usage = shutil.disk_usage(path)

        return {
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "percent": (
                usage.used / usage.total
                if usage.total
                else 0
            ),
        }

    except OSError:
        return None


# ============================================================
# COMMAND: SCAN
# ============================================================

def command_scan(args):

    root = Path(args.path).expanduser()

    scanner = FileScanner(
        minimum_size_mb=args.min_size
    )

    database = Database()

    print()
    print("=" * 70)
    print("SMARTSTORE - STORAGE SCAN")
    print("=" * 70)
    print()

    usage = get_disk_usage(root)

    if usage:

        print(
            f"Disk total : "
            f"{human_size(usage['total'])}"
        )

        print(
            f"Disk used  : "
            f"{human_size(usage['used'])}"
        )

        print(
            f"Disk free  : "
            f"{human_size(usage['free'])}"
        )

        print(
            f"Usage      : "
            f"{usage['percent'] * 100:.1f}%"
        )

        print()

    print(f"Scanning: {root}")

    records = scanner.scan(root)

    total = sum(
        record.size
        for record in records
    )

    database.record_scan(
        path=str(root),
        total_size=total,
        file_count=len(records),
    )

    print()
    print(
        f"Large files found : {len(records):,}"
    )

    print(
        f"Total size        : {human_size(total)}"
    )

    print()

    records.sort(
        key=lambda record: record.size,
        reverse=True,
    )

    for record in records[:args.limit]:

        print(
            f"{human_size(record.size):>12} "
            f"{record.path}"
        )


# ============================================================
# COMMAND: ANALYZE
# ============================================================

def command_analyze(args):

    root = Path(args.path).expanduser()

    scanner = FileScanner(
        minimum_size_mb=args.min_size
    )

    print()
    print("=" * 70)
    print("SMARTSTORE - ANALYSIS")
    print("=" * 70)
    print()

    records = scanner.scan(root)

    engine = RecommendationEngine()

    recommendations = [
        engine.recommend(record)
        for record in records
    ]

    recommendations.sort(
        key=lambda r: r.score,
        reverse=True,
    )

    clean = [
        r for r in recommendations
        if r.action == "CLEAN"
    ]

    review = [
        r for r in recommendations
        if r.action == "REVIEW"
    ]

    keep = [
        r for r in recommendations
        if r.action == "KEEP"
    ]

    cleanable = sum(
        r.size
        for r in clean
    )

    print(
        f"Files analyzed : {len(records):,}"
    )

    print(
        f"Clean candidates: {len(clean):,}"
    )

    print(
        f"Review candidates: {len(review):,}"
    )

    print(
        f"Potential recovery: "
        f"{human_size(cleanable)}"
    )

    print()

    print("TOP RECOMMENDATIONS")
    print("-" * 70)

    for recommendation in recommendations[
        :args.limit
    ]:

        print(
            f"{recommendation.action:7} "
            f"{recommendation.confidence * 100:5.1f}% "
            f"{recommendation.risk:8} "
            f"{human_size(recommendation.size):>12} "
            f"{recommendation.path}"
        )

        print(
            f"         {recommendation.reason}"
        )

    print()

    print(
        f"Keep: {len(keep):,} | "
        f"Review: {len(review):,} | "
        f"Clean: {len(clean):,}"
    )


# ============================================================
# COMMAND: RECOMMEND
# ============================================================

def command_recommend(args):

    root = Path(args.path).expanduser()

    scanner = FileScanner(
        minimum_size_mb=args.min_size
    )

    records = scanner.scan(root)

    engine = RecommendationEngine()

    recommendations = [
        engine.recommend(record)
        for record in records
    ]

    recommendations = [
        r
        for r in recommendations
        if r.action in {
            "CLEAN",
            "REVIEW",
        }
    ]

    recommendations.sort(
        key=lambda r: (
            r.action != "CLEAN",
            -r.score,
            -r.size,
        )
    )

    print()
    print("=" * 70)
    print("SMARTSTORE - AI RECOMMENDATIONS")
    print("=" * 70)
    print()

    total = 0

    for recommendation in recommendations[
        :args.limit
    ]:

        print(
            f"[{recommendation.action}] "
            f"{recommendation.confidence * 100:.1f}% "
            f"{human_size(recommendation.size)}"
        )

        print(
            f"Path: {recommendation.path}"
        )

        print(
            f"Risk: {recommendation.risk}"
        )

        print(
            f"Score: {recommendation.score:.3f}"
        )

        print(
            f"Why: {recommendation.reason}"
        )

        print("-" * 70)

        if recommendation.action == "CLEAN":
            total += recommendation.size

    print(
        f"\nPotential safe recovery: "
        f"{human_size(total)}"
    )


# ============================================================
# COMMAND: DUPLICATES
# ============================================================

def command_duplicates(args):

    root = Path(args.path).expanduser()

    scanner = FileScanner(
        minimum_size_mb=args.min_size
    )

    print()
    print("=" * 70)
    print("SMARTSTORE - DUPLICATE DETECTION")
    print("=" * 70)
    print()

    print(
        "Scanning files..."
    )

    records = scanner.scan(root)

    print(
        f"Candidate files: {len(records):,}"
    )

    detector = DuplicateDetector()

    duplicates = detector.find_duplicates(
        records
    )

    if not duplicates:

        print(
            "\nNo duplicates found."
        )

        return

    total_duplicate_space = 0

    for index, (digest, paths) in enumerate(
        duplicates.items(),
        start=1,
    ):

        print()
        print(
            f"Duplicate group #{index}"
        )

        print(
            f"SHA-256: {digest}"
        )

        group_size = 0

        for path in paths:

            try:
                size = Path(path).stat().st_size
            except OSError:
                size = 0

            group_size += size

            print(
                f"  {human_size(size):>12} "
                f"{path}"
            )

        # If there are N identical copies,
        # N-1 copies are theoretically removable.
        if len(paths) > 1:
            total_duplicate_space += (
                group_size
                - group_size / len(paths)
            )

    print()
    print(
        f"Potential duplicate recovery: "
        f"{human_size(total_duplicate_space)}"
    )


# ============================================================
# COMMAND: FORECAST
# ============================================================

def command_forecast(args):

    database = Database()

    forecaster = StorageForecaster(
        database
    )

    result = forecaster.forecast(
        days_ahead=args.days
    )

    print()
    print("=" * 70)
    print("SMARTSTORE - STORAGE FORECAST")
    print("=" * 70)
    print()

    if result is None:

        print(
            "Not enough historical scan data."
        )

        print(
            "Run 'scan' periodically to "
            "build forecasting data."
        )

        return

    current = result["current_size"]
    predicted = result["predicted_size"]
    growth = result["growth_per_day"]

    print(
        f"Current tracked usage: "
        f"{human_size(current)}"
    )

    print(
        f"Estimated daily growth: "
        f"{human_size(max(0, growth))}/day"
    )

    print(
        f"Predicted usage in "
        f"{args.days} days: "
        f"{human_size(predicted)}"
    )

    if growth > 0:

        print()
        print(
            "Storage is growing."
        )

    elif growth < 0:

        print()
        print(
            "Storage usage is decreasing."
        )

    else:

        print()
        print(
            "Storage usage appears stable."
        )


# ============================================================
# COMMAND: CLEAN
# ============================================================

def command_clean(args):

    root = Path(args.path).expanduser()

    scanner = FileScanner(
        minimum_size_mb=args.min_size
    )

    records = scanner.scan(root)

    engine = RecommendationEngine()

    recommendations = [
        engine.recommend(record)
        for record in records
    ]

    candidates = [
        r
        for r in recommendations
        if (
            r.action == "CLEAN"
            and r.confidence * 100 >= args.confidence
        )
    ]

    candidates.sort(
        key=lambda r: r.size,
        reverse=True,
    )

    print()
    print("=" * 70)
    print("SMARTSTORE - CLEANUP")
    print("=" * 70)
    print()

    print(
        f"Confidence threshold: "
        f"{args.confidence:.1f}%"
    )

    print(
        f"Candidates: {len(candidates):,}"
    )

    total = sum(
        r.size
        for r in candidates
    )

    print(
        f"Potential recovery: "
        f"{human_size(total)}"
    )

    print()

    if not candidates:

        print(
            "No candidates meet the threshold."
        )

        return

    if args.dry_run:

        print(
            "DRY-RUN MODE: No files will be moved."
        )

        print()

        for recommendation in candidates:

            print(
                f"[DRY-RUN] "
                f"{human_size(recommendation.size):>12} "
                f"{recommendation.path}"
            )

        return

    print(
        "WARNING: Files will be moved to quarantine."
    )

    print(
        "They will NOT be permanently deleted."
    )

    print()

    confirmation = input(
        "Type CLEAN to continue: "
    ).strip()

    if confirmation != "CLEAN":

        print(
            "Cleanup cancelled."
        )

        return

    manager = CleanupManager(
        Database()
    )

    successful = 0

    for recommendation in candidates:

        if manager.quarantine(
            recommendation,
            dry_run=False,
        ):
            successful += 1

    print()
    print(
        f"Quarantined: {successful:,} files"
    )


# ============================================================
# COMMAND: UNDO
# ============================================================

def command_undo(args):

    manager = CleanupManager(
        Database()
    )

    manager.undo_last()


# ============================================================
# COMMAND: HISTORY
# ============================================================

def command_history(args):

    database = Database()

    print()
    print("=" * 70)
    print("SMARTSTORE - HISTORY")
    print("=" * 70)
    print()

    actions = database.get_actions(
        args.limit
    )

    if not actions:

        print("No cleanup history.")
        return

    for action in actions:

        timestamp = datetime.fromtimestamp(
            action["timestamp"]
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        print(
            f"{timestamp} | "
            f"{action['action']}"
        )

        print(
            f"  Original: "
            f"{action['original_path']}"
        )

        print(
            f"  Quarantine: "
            f"{action['quarantine_path']}"
        )

        print(
            f"  Size: "
            f"{human_size(action['size'])}"
        )

        print(
            f"  Reason: "
            f"{action['reason']}"
        )

        print("-" * 70)


# ============================================================
# COMMAND: STATUS
# ============================================================

def command_status(args):

    path = Path(args.path).expanduser()

    usage = get_disk_usage(path)

    print()
    print("=" * 70)
    print("SMARTSTORE - SYSTEM STATUS")
    print("=" * 70)
    print()

    print(
        f"Platform: {sys.platform}"
    )

    print(
        f"Python: {sys.version.split()[0]}"
    )

    print(
        f"Home: {HOME}"
    )

    print(
        f"SmartStore data: {DATA_DIR}"
    )

    print()

    if usage:

        print(
            f"Total: "
            f"{human_size(usage['total'])}"
        )

        print(
            f"Used: "
            f"{human_size(usage['used'])}"
        )

        print(
            f"Free: "
            f"{human_size(usage['free'])}"
        )

        print(
            f"Usage: "
            f"{usage['percent'] * 100:.1f}%"
        )

    print()

    print(
        f"Quarantine: "
        f"{QUARANTINE_DIR}"
    )

    print(
        f"Database: "
        f"{DATABASE}"
    )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        prog="smartstore",
        description=(
            "AI-powered intelligent "
            "storage optimizer for Linux"
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"{APP_NAME} {VERSION}",
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    # --------------------------------------------------------
    # scan
    # --------------------------------------------------------

    scan = subparsers.add_parser(
        "scan",
        help="Scan a directory",
    )

    scan.add_argument(
        "path",
        nargs="?",
        default=str(HOME),
    )

    scan.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_SIZE_MB,
        help="Minimum file size in MB",
    )

    scan.add_argument(
        "--limit",
        type=int,
        default=30,
    )

    scan.set_defaults(
        function=command_scan
    )

    # --------------------------------------------------------
    # analyze
    # --------------------------------------------------------

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze storage usage",
    )

    analyze.add_argument(
        "path",
        nargs="?",
        default=str(HOME),
    )

    analyze.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_SIZE_MB,
    )

    analyze.add_argument(
        "--limit",
        type=int,
        default=30,
    )

    analyze.set_defaults(
        function=command_analyze
    )

    # --------------------------------------------------------
    # recommend
    # --------------------------------------------------------

    recommend = subparsers.add_parser(
        "recommend",
        help="Generate intelligent cleanup recommendations",
    )

    recommend.add_argument(
        "path",
        nargs="?",
        default=str(HOME),
    )

    recommend.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_SIZE_MB,
    )

    recommend.add_argument(
        "--limit",
        type=int,
        default=30,
    )

    recommend.set_defaults(
        function=command_recommend
    )

    # --------------------------------------------------------
    # duplicates
    # --------------------------------------------------------

    duplicates = subparsers.add_parser(
        "duplicates",
        help="Find duplicate files",
    )

    duplicates.add_argument(
        "path",
        nargs="?",
        default=str(HOME),
    )

    duplicates.add_argument(
        "--min-size",
        type=int,
        default=1,
    )

    duplicates.set_defaults(
        function=command_duplicates
    )

    # --------------------------------------------------------
    # forecast
    # --------------------------------------------------------

    forecast = subparsers.add_parser(
        "forecast",
        help="Forecast storage growth",
    )

    forecast.add_argument(
        "--days",
        type=int,
        default=30,
    )

    forecast.set_defaults(
        function=command_forecast
    )

    # --------------------------------------------------------
    # clean
    # --------------------------------------------------------

    clean = subparsers.add_parser(
        "clean",
        help="Safely quarantine high-confidence candidates",
    )

    clean.add_argument(
        "path",
        nargs="?",
        default=str(HOME),
    )

    clean.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_SIZE_MB,
    )

    clean.add_argument(
        "--confidence",
        type=float,
        default=95.0,
        help="Minimum recommendation confidence",
    )

    clean.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned",
    )

    clean.set_defaults(
        function=command_clean
    )

    # --------------------------------------------------------
    # undo
    # --------------------------------------------------------

    undo = subparsers.add_parser(
        "undo",
        help="Restore the most recently quarantined file",
    )

    undo.set_defaults(
        function=command_undo
    )

    # --------------------------------------------------------
    # history
    # --------------------------------------------------------

    history = subparsers.add_parser(
        "history",
        help="Show cleanup history",
    )

    history.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    history.set_defaults(
        function=command_history
    )

    # --------------------------------------------------------
    # status
    # --------------------------------------------------------

    status = subparsers.add_parser(
        "status",
        help="Show SmartStore/system status",
    )

    status.add_argument(
        "path",
        nargs="?",
        default="/",
    )

    status.set_defaults(
        function=command_status
    )

    return parser


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()

    parser = build_parser()

    args = parser.parse_args()

    if not args.command:

        parser.print_help()

        print()
        print(
            "Example:"
        )

        print(
            "  python smartstore.py "
            "recommend ~/Downloads"
        )

        return

    try:

        args.function(args)

    except KeyboardInterrupt:

        print(
            "\nOperation cancelled."
        )

    except PermissionError as exc:

        print(
            f"\nPermission denied: {exc}"
        )

        print(
            "Try a user-owned directory "
            "rather than running as root."
        )

    except Exception as exc:

        print(
            f"\nError: {exc}"
        )


if __name__ == "__main__":
    main()
