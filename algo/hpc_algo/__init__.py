"""Independent processing module for hidden-patterns-combat.

Основной публичный интерфейс — :func:`hpc_algo.api.analyze_source`, который
принимает путь к Excel-источнику и конфигурацию, а возвращает структурированный
:class:`hpc_algo.schema.AnalysisResult`.

Модуль строго не знает ни про HTTP, ни про базу данных, ни про UI. Он пригоден
для вызова из backend API, CLI, тестов и notebook.
"""

from hpc_algo.api import AnalyzeConfig, analyze_source, preflight_mapping
from hpc_algo.schema import (
    AnalysisResult,
    AnalysisStatus,
    AuditReport,
    BaselineReport,
    ChartData,
    ColumnDetectionReport,
    ColumnMappingConfig,
    HiddenGroup,
    HiddenGroupCandidate,
    HMMParameters,
    HMMResult,
    HMMTrajectory,
    SheetMapping,
    SourceMetadata,
    TimeStats,
    WarningItem,
)

__all__ = [
    "analyze_source",
    "preflight_mapping",
    "AnalyzeConfig",
    "AnalysisResult",
    "AnalysisStatus",
    "AuditReport",
    "BaselineReport",
    "ChartData",
    "ColumnDetectionReport",
    "ColumnMappingConfig",
    "HiddenGroup",
    "HiddenGroupCandidate",
    "HMMParameters",
    "HMMResult",
    "HMMTrajectory",
    "SheetMapping",
    "SourceMetadata",
    "TimeStats",
    "WarningItem",
]

__version__ = "0.3.0"
