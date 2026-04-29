"""
API router for the arrival classifier.

Provides endpoints to:
- Browse parquet files in the data directory
- Load and classify query arrival data from parquet files

Data is aggregated server-side into minute buckets to optimize performance
for large workloads.
"""

import os
import re
from collections import defaultdict
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import autoslo.filesystem.path_utils as pu
from autoslo.forecasting.arrival_classifier import ArrivalClassifier
from autoslo.workload_definition.query import Query, QueryTextId

router = APIRouter()

# Global cache for loaded workloads (stores queries and bucket size by file path)
_workload_cache: dict[str, tuple[list[Query], int]] = {}  # (queries, bucket_size)

# Target number of buckets for visualization (aim for ~500-1000 points)
TARGET_BUCKETS = 800
MIN_BUCKET_SIZE_S = 10  # Minimum 10 seconds
MAX_BUCKET_SIZE_S = 3600  # Maximum 1 hour for auto-computed


class FileInfo(BaseModel):
    """Schema for file/directory info."""

    name: str
    path: str  # Relative path from data directory
    is_dir: bool
    children: list["FileInfo"] | None = None


class ClassificationResult(BaseModel):
    """Schema for classification results."""

    template_id: int
    classification: str  # "windowed", "normal", "ad-hoc", "unclassified"


class ArrivalPoint(BaseModel):
    """Schema for a single arrival point in the time series."""

    query_id: str
    start_time_s: float
    template_id: int
    classification: str


class BucketData(BaseModel):
    """Schema for aggregated bucket data."""

    bucket_start_s: float
    count: int


class TemplateBuckets(BaseModel):
    """Schema for a template's aggregated buckets."""

    template_id: int
    buckets: list[BucketData]
    total_queries: int


class LoadFileResponse(BaseModel):
    """Response schema for loading a file (aggregated)."""

    file_path: str
    num_queries: int
    num_templates: int
    bucket_size_s: int
    aggregate_buckets: list[BucketData]
    template_ids: list[int]


class ClassifierResponse(BaseModel):
    """Response schema for the classifier endpoint."""

    classifications: list[ClassificationResult]
    summary: dict[str, Any]
    aggregate_buckets: list[BucketData]
    template_buckets: dict[str, TemplateBuckets]  # keyed by template_id as string


def _extract_template_id(template_str: str) -> int:
    """Extract numeric template ID from various string formats.

    Handles formats like:
    - "1" -> 1
    - "template_1" -> 1
    - "1_0" -> 1
    - "template_1_0" -> 1

    Returns the first number found in the string.
    """
    # Find all numbers in the string
    numbers = re.findall(r'\d+', str(template_str))
    if numbers:
        return int(numbers[0])
    # Fallback: hash the string to get a consistent ID
    return abs(hash(template_str)) % 10000


def _compute_bucket_size(min_time: float, max_time: float) -> int:
    """Compute adaptive bucket size based on workload duration.

    Targets approximately TARGET_BUCKETS data points for smooth visualization.
    """
    duration_s = max_time - min_time
    if duration_s <= 0:
        return MIN_BUCKET_SIZE_S

    # Calculate bucket size to get ~TARGET_BUCKETS buckets
    bucket_size = int(duration_s / TARGET_BUCKETS)

    # Clamp to reasonable range
    bucket_size = max(MIN_BUCKET_SIZE_S, min(MAX_BUCKET_SIZE_S, bucket_size))

    # Round to nice intervals (1min, 5min, 10min, 15min, 30min, 1hr)
    nice_intervals = [60, 300, 600, 900, 1800, 3600]
    for interval in nice_intervals:
        if bucket_size <= interval:
            return interval
    return MAX_BUCKET_SIZE_S


def _aggregate_to_buckets(
    queries: list[Query], bucket_size_s: int | None = None
) -> tuple[list[BucketData], dict[int, list[BucketData]], int]:
    """Aggregate queries into time buckets.

    Args:
        queries: List of Query objects
        bucket_size_s: Override bucket size (if None, auto-compute)

    Returns:
        Tuple of (aggregate_buckets, template_buckets_dict, bucket_size_s)
        - aggregate_buckets: Total arrivals per bucket
        - template_buckets_dict: Dict of template_id -> buckets for that template
        - bucket_size_s: The bucket size used
    """
    if not queries:
        return [], {}, MIN_BUCKET_SIZE_S

    # Find time range
    min_time = min(q.rel_start_time_s for q in queries)
    max_time = max(q.rel_start_time_s for q in queries)

    # Compute bucket size if not provided
    if bucket_size_s is None:
        bucket_size_s = _compute_bucket_size(min_time, max_time)

    # Create bucket starts
    bucket_starts = []
    current = (min_time // bucket_size_s) * bucket_size_s
    while current <= max_time:
        bucket_starts.append(current)
        current += bucket_size_s

    # Initialize bucket counts
    aggregate_counts: dict[float, int] = {b: 0 for b in bucket_starts}
    template_counts: dict[int, dict[float, int]] = defaultdict(
        lambda: {b: 0 for b in bucket_starts}
    )

    # Count queries per bucket
    for query in queries:
        bucket = (query.rel_start_time_s // bucket_size_s) * bucket_size_s
        if bucket in aggregate_counts:
            aggregate_counts[bucket] += 1
            template_id = int(query.query_text_id.template_id)
            template_counts[template_id][bucket] += 1

    # Convert to response format
    aggregate_buckets = [
        BucketData(bucket_start_s=b, count=c)
        for b, c in sorted(aggregate_counts.items())
    ]

    template_buckets_dict: dict[int, list[BucketData]] = {}
    for tid, counts in template_counts.items():
        template_buckets_dict[tid] = [
            BucketData(bucket_start_s=b, count=c)
            for b, c in sorted(counts.items())
        ]

    return aggregate_buckets, template_buckets_dict, bucket_size_s


def _get_data_dir() -> str:
    """Get the data directory path."""
    return pu.get_data_path()


def _scan_parquet_files(
    base_path: str, rel_path: str = "", max_depth: int = 4
) -> list[FileInfo]:
    """Recursively scan for parquet files in the given directory."""
    full_path = os.path.join(base_path, rel_path) if rel_path else base_path

    if not os.path.exists(full_path) or not os.path.isdir(full_path):
        return []

    items = []
    try:
        entries = sorted(os.listdir(full_path))
    except PermissionError:
        return []

    for entry in entries:
        entry_rel_path = os.path.join(rel_path, entry) if rel_path else entry
        entry_full_path = os.path.join(full_path, entry)

        if os.path.isdir(entry_full_path):
            if max_depth > 0:
                # Check if directory contains any parquet files
                children = _scan_parquet_files(base_path, entry_rel_path, max_depth - 1)
                if children:  # Only include directories with parquet files
                    items.append(
                        FileInfo(
                            name=entry,
                            path=entry_rel_path,
                            is_dir=True,
                            children=children,
                        )
                    )
        elif entry.endswith(".parquet"):
            items.append(
                FileInfo(
                    name=entry,
                    path=entry_rel_path,
                    is_dir=False,
                    children=None,
                )
            )

    return items


def _parse_queries_from_parquet(file_path: str) -> list[Query]:
    """Parse query arrivals from a parquet file.

    Expected columns (based on data_schemas.yml):
    - query_id: unique identifier
    - rel_start_time_s: relative start time in seconds
    - query_template: template identifier (e.g., "1" or "template_1")
    - query_num_within_template: query number within template (optional)

    Also supports columns from chunk_traces schema:
    - chunk_id: chunk identifier (optional)
    """
    df = pd.read_parquet(file_path)

    # Normalize column names (lowercase, strip whitespace)
    df.columns = df.columns.str.lower().str.strip()

    # Find the time column
    time_col = None
    for col in ["rel_start_time_s", "start_time_s", "timestamp"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        raise ValueError(
            f"Parquet file must have 'rel_start_time_s', 'start_time_s', or 'timestamp' column. "
            f"Found columns: {list(df.columns)}"
        )

    # Find the template column
    template_col = None
    for col in ["query_template", "template_id", "template"]:
        if col in df.columns:
            template_col = col
            break

    if template_col is None:
        raise ValueError(
            f"Parquet file must have 'query_template' or 'template_id' column. "
            f"Found columns: {list(df.columns)}"
        )

    # Find query_id column (optional)
    query_id_col = None
    for col in ["query_id", "id"]:
        if col in df.columns:
            query_id_col = col
            break

    # Find query_num_within_template (optional)
    query_num_col = None
    if "query_num_within_template" in df.columns:
        query_num_col = "query_num_within_template"

    queries = []
    for idx, row in df.iterrows():
        # Get query_id
        if query_id_col:
            query_id = str(row[query_id_col])
        else:
            query_id = f"q_{idx}"

        # Get start time
        start_time = float(row[time_col])

        # Get template (format as "template_querynum" for Query.template_id compatibility)
        template_raw = str(row[template_col])

        # Extract template number - handle various formats
        if "_" in template_raw:
            # Already in "X_Y" format
            template_str = template_raw
        else:
            # Get query num within template if available
            query_num = 0
            if query_num_col and pd.notna(row[query_num_col]):
                query_num = int(row[query_num_col])
            template_str = f"{template_raw}_{query_num}"

        query = Query(
            query_id=query_id,
            query_text_id=QueryTextId(f"unknown#{template_str.replace('_', '#')}"),
            rel_start_time_s=start_time,
        )
        queries.append(query)

    return queries


def _run_classification(queries: list[Query]) -> dict[str, Any]:
    """Run the arrival classifier and return aggregated results."""
    classifier = ArrivalClassifier(queries)
    classifier.classify_arrivals()

    # Get the classification results
    template_classifications = classifier._template_classification
    template_details = classifier._template_details

    # Build template query counts for summary
    template_query_counts: dict[int, int] = defaultdict(int)
    for query in queries:
        tid = int(query.query_text_id.template_id)
        template_query_counts[tid] += 1

    classifications = [
        ClassificationResult(template_id=tid, classification=cls)
        for tid, cls in template_classifications.items()
    ]

    # Build summary
    classification_counts: dict[str, int] = defaultdict(int)
    query_counts: dict[str, int] = defaultdict(int)

    for tid, cls in template_classifications.items():
        classification_counts[cls] += 1
        query_counts[cls] += template_query_counts.get(tid, 0)

    total_templates = len(template_classifications)
    total_queries = len(queries)

    by_class: dict[str, dict[str, Any]] = {}
    for cls in classification_counts:
        # Get templates in this class with their details
        templates_in_class = []
        for tid, tcls in template_classifications.items():
            if tcls == cls:
                template_info = {
                    "template_id": tid,
                    "num_queries": template_query_counts.get(tid, 0),
                }
                # Add detailed parameters from detector
                if tid in template_details:
                    details = template_details[tid]
                    # For windowed templates
                    if cls == "windowed":
                        if details.get("period_s"):
                            template_info["period_s"] = details["period_s"]
                        if details.get("active_length_s"):
                            template_info["active_length_s"] = details["active_length_s"]
                        if details.get("idle_ratio"):
                            template_info["idle_ratio"] = details["idle_ratio"]
                    # For normal/ad-hoc templates
                    if details.get("num_unique_days"):
                        template_info["num_unique_days"] = details["num_unique_days"]
                    if details.get("has_weekday_seasonality"):
                        template_info["has_weekday_seasonality"] = details["has_weekday_seasonality"]
                    if details.get("has_weekend_seasonality"):
                        template_info["has_weekend_seasonality"] = details["has_weekend_seasonality"]
                    if details.get("has_weekly_seasonality"):
                        template_info["has_weekly_seasonality"] = details["has_weekly_seasonality"]
                templates_in_class.append(template_info)
        
        # Sort templates by query count descending
        templates_in_class.sort(key=lambda x: x["num_queries"], reverse=True)
        
        by_class[cls] = {
            "num_templates": classification_counts[cls],
            "pct_templates": (
                classification_counts[cls] / total_templates * 100
                if total_templates > 0
                else 0
            ),
            "num_queries": query_counts.get(cls, 0),
            "pct_queries": (
                query_counts.get(cls, 0) / total_queries * 100 if total_queries > 0 else 0
            ),
            "templates": templates_in_class,
        }

    summary: dict[str, Any] = {
        "total_templates": total_templates,
        "total_queries": total_queries,
        "by_class": by_class,
    }

    # Build aggregated bucket data
    aggregate_buckets, template_buckets_dict, _ = _aggregate_to_buckets(queries)

    # Convert template_buckets to response format with classification info
    template_buckets: dict[str, TemplateBuckets] = {}
    for tid, buckets in template_buckets_dict.items():
        template_buckets[str(tid)] = TemplateBuckets(
            template_id=tid,
            buckets=buckets,
            total_queries=template_query_counts.get(tid, 0),
        )

    return {
        "classifications": classifications,
        "summary": summary,
        "aggregate_buckets": aggregate_buckets,
        "template_buckets": template_buckets,
    }


@router.get("/classifier/files", response_model=list[FileInfo])
async def list_data_files():
    """
    List all parquet files available in the data directory.

    Returns a tree structure of directories and parquet files that can be
    used by the UI to allow users to select files for classification.
    """
    data_dir = _get_data_dir()
    if not os.path.exists(data_dir):
        return []

    files = _scan_parquet_files(data_dir, max_depth=4)
    return files


@router.get("/classifier/load")
async def load_parquet_file(path: str, bucket_size_s: int | None = None) -> LoadFileResponse:
    """
    Load a parquet file and return aggregated arrival data for visualization.

    Parameters:
        path: Relative path to the parquet file within the data directory.
        bucket_size_s: Optional bucket size in seconds. If not provided, auto-computed.

    Returns:
        Aggregated bucket data for efficient visualization.
    """
    global _workload_cache

    data_dir = _get_data_dir()
    full_path = os.path.join(data_dir, path)

    # Security check: ensure path doesn't escape data directory
    real_data_dir = os.path.realpath(data_dir)
    real_full_path = os.path.realpath(full_path)
    if not real_full_path.startswith(real_data_dir):
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if not full_path.endswith(".parquet"):
        raise HTTPException(status_code=400, detail="Only parquet files are supported")

    try:        
        queries = _parse_queries_from_parquet(full_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse parquet file: {str(e)}")

    if len(queries) == 0:
        raise HTTPException(status_code=400, detail="No queries found in file")

    # Get unique template IDs
    template_ids = sorted(set(int(q.query_text_id.template_id) for q in queries))

    # Aggregate to buckets (with optional custom bucket size)
    aggregate_buckets, _, bucket_size = _aggregate_to_buckets(queries, bucket_size_s)

    # Cache queries and bucket size for later use
    _workload_cache[path] = (queries, bucket_size)

    return LoadFileResponse(
        file_path=path,
        num_queries=len(queries),
        num_templates=len(template_ids),
        bucket_size_s=bucket_size,
        aggregate_buckets=aggregate_buckets,
        template_ids=template_ids,
    )


@router.post("/classifier/classify-file", response_model=ClassifierResponse)
async def classify_file(path: str):
    """
    Classify arrival patterns for a loaded parquet file.

    Uses cached queries if available, otherwise loads the file.

    Parameters:
        path: Relative path to the parquet file within the data directory.

    Returns:
        Classifications and aggregated bucket data.
    """
    global _workload_cache

    # Check cache first
    if path in _workload_cache:
        queries, _ = _workload_cache[path]
    else:
        # Load the file
        data_dir = _get_data_dir()
        full_path = os.path.join(data_dir, path)

        # Security check
        real_data_dir = os.path.realpath(data_dir)
        real_full_path = os.path.realpath(full_path)
        if not real_full_path.startswith(real_data_dir):
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        if not full_path.endswith(".parquet"):
            raise HTTPException(status_code=400, detail="Only parquet files are supported")

        try:
            queries = _parse_queries_from_parquet(full_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse parquet file: {str(e)}")

        if len(queries) == 0:
            raise HTTPException(status_code=400, detail="No queries found in file")

        # Cache for later (compute bucket size)
        _, _, bucket_size = _aggregate_to_buckets(queries)
        _workload_cache[path] = (queries, bucket_size)

    result = _run_classification(queries)
    return ClassifierResponse(**result)


@router.get("/classifier/template-buckets/{template_id}")
async def get_template_buckets(template_id: int, path: str) -> TemplateBuckets:
    """
    Get aggregated bucket data for a specific template.

    Parameters:
        template_id: The template ID to get buckets for.
        path: The file path (must be loaded first).

    Returns:
        Bucket data for the specified template.
    """
    global _workload_cache

    if path not in _workload_cache:
        raise HTTPException(
            status_code=400,
            detail="File not loaded. Call /classifier/load first."
        )

    queries, bucket_size_s = _workload_cache[path]

    # Filter queries for this template
    template_queries = [
        q for q in queries
        if int(q.query_text_id.template_id) == template_id
    ]

    if not template_queries:
        raise HTTPException(
            status_code=404,
            detail=f"No queries found for template {template_id}"
        )

    # Aggregate just this template's queries but use the full time range
    min_time = min(q.rel_start_time_s for q in queries)
    max_time = max(q.rel_start_time_s for q in queries)

    bucket_starts = []
    current = (min_time // bucket_size_s) * bucket_size_s
    while current <= max_time:
        bucket_starts.append(current)
        current += bucket_size_s

    counts: dict[float, int] = {b: 0 for b in bucket_starts}
    for q in template_queries:
        bucket = (q.rel_start_time_s // bucket_size_s) * bucket_size_s
        if bucket in counts:
            counts[bucket] += 1

    buckets = [
        BucketData(bucket_start_s=b, count=c)
        for b, c in sorted(counts.items())
    ]

    return TemplateBuckets(
        template_id=template_id,
        buckets=buckets,
        total_queries=len(template_queries),
    )


# Keep classify-arrivals for backward compat but update to new response format
@router.post("/classifier/classify-arrivals", response_model=ClassifierResponse)
async def classify_arrivals(arrivals: list[ArrivalPoint]):
    """
    Classify already-loaded arrivals.

    This endpoint is useful when the arrivals have already been loaded via
    the /classifier/load endpoint and the user wants to classify them
    without reloading the file.

    Parameters:
        arrivals: List of arrival points to classify.

    Returns:
        The arrivals with their classifications and a summary.
    """
    if len(arrivals) == 0:
        raise HTTPException(status_code=400, detail="No arrivals provided")

    # Convert ArrivalPoints to Query objects
    queries = []
    for a in arrivals:
        query = Query(
            query_id=a.query_id,
            query_text_id=QueryTextId(f"unknown#{a.template_id}#0"),
            rel_start_time_s=a.start_time_s,
        )
        queries.append(query)

    result = _run_classification(queries)
    return ClassifierResponse(**result)
