# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

"""Performance optimization configuration for data cache operations."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations in data cache operations."""

    # Compression settings - optimized for speed
    compression_level: int = 9
    use_pigz: bool = True

    # Parallel processing settings - maximize parallelism
    max_workers: int = 16
    enable_progress_monitoring: bool = False

    # Memory optimization settings - larger chunks for speed
    chunk_size_mb: int = 512
    enable_memory_mapping: bool = True

    # Cache settings - disable compression for speed
    enable_compression_caching: bool = True
    cache_cleanup_threshold_gb: float = 500.0

    @classmethod
    def from_env(cls) -> "PerformanceConfig":
        """Create configuration from environment variables."""
        return cls(
            compression_level=int(os.environ.get("CACHE_COMPRESSION_LEVEL", "0")),
            use_pigz=os.environ.get("CACHE_USE_PIGZ", "false").lower() == "true",
            max_workers=int(os.environ.get("CACHE_MAX_WORKERS", "8")),
            enable_progress_monitoring=os.environ.get("CACHE_PROGRESS_MONITORING", "false").lower() == "true",
            chunk_size_mb=int(os.environ.get("CACHE_CHUNK_SIZE_MB", "256")),
            enable_memory_mapping=os.environ.get("CACHE_MEMORY_MAPPING", "true").lower() == "true",
            enable_compression_caching=os.environ.get("CACHE_COMPRESSION_CACHING", "false").lower() == "true",
            cache_cleanup_threshold_gb=float(os.environ.get("CACHE_CLEANUP_THRESHOLD_GB", "100.0")),
        )

    def get_cache_dir(self, base_dir: Path) -> Path:
        """Get the cache directory with performance optimizations applied."""
        cache_dir = base_dir / "performance_optimized_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


# Global performance configuration
PERF_CONFIG = PerformanceConfig.from_env()
