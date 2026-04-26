"""Pydantic-схемы для результата работы processing module.

Схема — единственный контракт между :mod:`hpc_algo`, backend и frontend.
Любая информация, которую видит пользователь, должна быть представима в этой
модели. Сам алгоритм не знает, как она будет отрисована.

Ключевые предметные инварианты (см. docs/agent_context/DOMAIN_SPEC.md):

* observations = ``ЗАП``;
* hidden states — элементы соревновательной деятельности
  (маневрирование / КФВ / ВУП);
* статус анализа обязан честно отражать, что алгоритм реально успел
  восстановить, и никогда не объявлять диагностику выполненной без оснований.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalysisStatus(str, Enum):
    """Честный статус анализа.

    * ``audit_only`` — удалось выполнить только аудит файла; ни одна
      колоночная группа не распознана уверенно.
    * ``baseline_only`` — выполнен аудит и базовые распределения по
      кандидатам ЗАП; полноценная HMM невозможна.
    * ``needs_column_mapping`` — структура Excel не позволяет надёжно
      определить группы, требуется ручное сопоставление колонок.
    * ``hmm_ready`` — HMM обучена, прошла все guard-ы и sanity-check'и.
    * ``hmm_low_signal`` — HMM обучена, но ЗАП-сигнал в данных
      разрежённый: эмиссии состояний почти идентичны, и часть
      Viterbi-траектории фактически отражает приор переходов, а не
      наблюдения. Результат публикуется, но в UI идут явные
      предупреждения о низкой надёжности интерпретации.
    * ``failed`` — анализ не удался (см. ``errors``).
    """

    AUDIT_ONLY = "audit_only"
    BASELINE_ONLY = "baseline_only"
    NEEDS_COLUMN_MAPPING = "needs_column_mapping"
    HMM_READY = "hmm_ready"
    HMM_LOW_SIGNAL = "hmm_low_signal"
    FAILED = "failed"


class HiddenGroup(str, Enum):
    """Предметные группы скрытых состояний и служебных колонок.

    Сами по себе hidden states не рассчитываются на этапе baseline — это
    только кандидаты на сопоставление, см. :class:`HiddenGroupCandidate`.
    """

    MANEUVERING = "маневрирование"
    KFV = "КФВ"
    VUP = "ВУП"
    ZAP = "ЗАП"
    TIME = "time"
    ATHLETE = "athlete"
    EPISODE = "episode"
    BOUT = "bout"
    WEIGHT = "weight"


class WarningSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class WarningItem(BaseModel):
    """Честное предупреждение для пользователя."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="Короткий машинный код предупреждения.")
    message: str = Field(..., description="Человекочитаемое сообщение (ru).")
    severity: WarningSeverity = WarningSeverity.WARNING
    context: dict[str, Any] = Field(default_factory=dict)


class SourceMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    size_bytes: int
    sha256: str | None = None
    sheet_count: int = 0
    sheet_names: list[str] = Field(default_factory=list)


class ColumnInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_ratio: float
    unique_count: int
    sample_values: list[Any] = Field(default_factory=list)


class SheetAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    n_rows: int
    n_cols: int
    columns: list[ColumnInfo]
    preview: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Первые N строк листа в виде списка словарей.",
    )
    suspicious: list[str] = Field(
        default_factory=list,
        description="Человекочитаемые сообщения о подозрительных наблюдениях.",
    )


class AuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sheets: list[SheetAudit]
    total_rows: int
    total_cells: int
    overall_null_ratio: float


class HiddenGroupCandidate(BaseModel):
    """Кандидат на сопоставление группы колонок с предметной сущностью."""

    model_config = ConfigDict(extra="forbid")

    group: HiddenGroup
    sheet: str
    column: str
    score: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    sample_values: list[Any] = Field(default_factory=list)


class ColumnDetectionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[HiddenGroupCandidate] = Field(default_factory=list)
    detected_groups: list[HiddenGroup] = Field(default_factory=list)
    missing_groups: list[HiddenGroup] = Field(default_factory=list)
    assumptions: list[str] = Field(
        default_factory=list,
        description="Явно зафиксированные предположения о структуре данных.",
    )


class ChartData(BaseModel):
    """Chart-ready структура, не зависящая от конкретной библиотеки графиков."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    kind: str = Field(
        ...,
        description="Тип графика: bar, hbar, heatmap, missing, hist, line.",
    )
    x: list[Any] = Field(default_factory=list)
    y: list[Any] = Field(default_factory=list)
    series: list[dict[str, Any]] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class TimeStats(BaseModel):
    """Базовые числовые характеристики колонки времени."""

    model_config = ConfigDict(extra="forbid")

    count: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    null_count: int = 0


class BaselineReport(BaseModel):
    """Всё, что можно честно посчитать без полноценной HMM.

    До применения column mapping заполняется только ``zap_value_counts``
    для уверенных ЗАП-кандидатов. После применения mapping — дополнительно
    распределения по всем четырём предметным группам и time-статистики.
    """

    model_config = ConfigDict(extra="forbid")

    zap_value_counts: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="sheet.column -> {value: count}. Для уверенных ЗАП-кандидатов.",
    )
    missing_values_per_column: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="sheet -> {column: null_count}.",
    )

    hidden_group_value_counts: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict,
        description=(
            "group -> sheet.column -> {value: count}. Заполняется только"
            " после применения подтверждённого column_mapping."
        ),
    )
    hidden_group_totals: dict[str, int] = Field(
        default_factory=dict,
        description="group -> суммарное число наблюдений по всем листам/колонкам.",
    )

    # --- ЗАП-specific: TASK_SPEC_003_1 ---
    zap_column_kinds: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "sheet.column -> один из 'categorical' / 'binary' / 'count' / 'empty'."
        ),
    )
    zap_events: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "sheet.column -> число эпизодов, где значение > 0 (binary/count) "
            "или непустое категориальное значение."
        ),
    )
    zap_total_triggers: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "sheet.column -> сумма целочисленных значений "
            "(binary/count). Для categorical равно числу непустых строк."
        ),
    )
    zap_events_by_channel: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "channel_label -> суммарное число событий по каналу. "
            "channel_label — атомарный уровень flatten-имени колонки "
            "(например 'Удержание', 'На руку', 'ЗАП-Р')."
        ),
    )

    time_statistics: dict[str, TimeStats] = Field(
        default_factory=dict,
        description="sheet.column -> TimeStats. Только для колонок роли `time`.",
    )
    episodes_per_sheet: dict[str, int] = Field(
        default_factory=dict,
        description="sheet -> число эпизодов, восстановленных по mapping (role=episode).",
    )
    empty_data_rows_per_sheet: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "sheet -> число полностью пустых строк после применения header_rows."
            " Заполняется только в mapping-ветке."
        ),
    )
    notes: list[str] = Field(default_factory=list)


class AthleteEpisodeStats(BaseModel):
    """Описательная сводка одного спортсмена для тренерской вкладки.

    Считается только в mapping-ветке при наличии ролей `athlete` и
    `episode`. ``episode_count`` — число уникальных эпизодов, в которых
    встретился этот спортсмен. Уникальность ключа эпизода учитывает имя
    листа, чтобы совпадающие номера эпизодов в разных весовых категориях
    не схлопывались между собой.
    """

    model_config = ConfigDict(extra="forbid")

    athlete: str
    episode_count: int = Field(..., ge=0)


class TrainerAthleteSummary(BaseModel):
    """Контейнер тренерской вкладки.

    Отдельная модель, чтобы добавить агрегаты (общее число спортсменов /
    эпизодов) и явные ``notes`` без раздувания корневого результата.
    """

    model_config = ConfigDict(extra="forbid")

    athletes: list[AthleteEpisodeStats] = Field(default_factory=list)
    total_athletes: int = 0
    total_episodes: int = 0
    notes: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Полный результат работы ``analyze_source``.

    Любая наблюдаемая в UI информация строится поверх этой модели.
    """

    model_config = ConfigDict(extra="forbid")

    status: AnalysisStatus
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    algo_version: str = "0.3.0"

    source_metadata: SourceMetadata
    data_audit: AuditReport
    detected_columns: ColumnDetectionReport
    basic_statistics: BaselineReport = Field(default_factory=BaselineReport)
    applied_mapping: ColumnMappingConfig | None = Field(
        default=None,
        description=(
            "Column mapping, который фактически использовался в этом анализе."
            " None означает, что использовался только эвристический путь."
        ),
    )

    charts: list[ChartData] = Field(default_factory=list)
    warnings: list[WarningItem] = Field(default_factory=list)
    errors: list[WarningItem] = Field(default_factory=list)
    trainer_athlete_summary: TrainerAthleteSummary | None = Field(
        default=None,
        description=(
            "Описательная сводка по спортсменам для тренерской вкладки."
            " Заполняется только в mapping-ветке при наличии ролей athlete"
            " и episode хотя бы на одном листе."
        ),
    )

    report: str = Field(
        default="",
        description="Короткий текстовый отчёт (ru) с честной формулировкой итога.",
    )

    hmm: HMMResult | None = Field(
        default=None,
        description=(
            "Результат HMM-ветки (TASK_SPEC_004). Заполняется при"
            " status == hmm_ready или hmm_low_signal; при любом другом"
            " статусе остаётся None."
        ),
    )


# ---------------------------------------------------------------------------
# Column mapping (TASK_SPEC_003)
# ---------------------------------------------------------------------------


class SheetMapping(BaseModel):
    """Пользовательское сопоставление колонок одного листа.

    Ключи ``roles`` — предметные роли (ЗАП / маневрирование / КФВ / ВУП /
    time / athlete / episode / bout / weight). Значения — список
    flatten-имён колонок, принятых на этом листе после учёта
    многострочного заголовка (см. :func:`hpc_algo.mapping.flatten_columns`).
    """

    model_config = ConfigDict(extra="forbid")

    header_rows: list[int] = Field(
        default_factory=lambda: [0],
        description="Строки (0-indexed) в листе, составляющие заголовок.",
    )
    data_start_row: int | None = Field(
        default=None,
        description=(
            "Первая строка данных. Если None, используется max(header_rows)+1."
        ),
    )
    roles: dict[HiddenGroup, list[str]] = Field(
        default_factory=dict,
        description="role -> [flatten column name, ...]",
    )


class ColumnMappingConfig(BaseModel):
    """Config, сохраняемый в backend и переиспользуемый при повторных анализах."""

    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    sheets: dict[str, SheetMapping] = Field(default_factory=dict)

    def roles_for(self, sheet_name: str) -> dict[HiddenGroup, list[str]]:
        sheet = self.sheets.get(sheet_name)
        return sheet.roles if sheet else {}

    def is_empty(self) -> bool:
        return not any(sm.roles for sm in self.sheets.values())


# ---------------------------------------------------------------------------
# HMM (TASK_SPEC_004)
# ---------------------------------------------------------------------------


class HMMParameters(BaseModel):
    """Параметры обученной HMM.

    * ``state_labels`` — строго предметные (``маневрирование`` / ``КФВ`` / ``ВУП``);
    * ``observation_labels`` — алфавит наблюдений, построенный из ЗАП-каналов.
    """

    model_config = ConfigDict(extra="forbid")

    n_states: int
    n_observations: int
    state_labels: list[str]
    observation_labels: list[str]
    initial_distribution: list[float]
    transition_matrix: list[list[float]]
    emission_matrix: list[list[float]]
    random_seed: int
    n_iter: int
    converged: bool
    log_likelihood: float
    variant: str = Field(
        default="basic_3state",
        description="Какая вариация HMM была использована: basic_3state или detailed_7state.",
    )
    observation_emission: str = Field(
        default="categorical",
        description="Тип эмиссии: categorical (по одному токену на шаг) или bernoulli (вектор каналов).",
    )
    bic: float | None = Field(
        default=None,
        description="Bayesian Information Criterion; ниже — лучше при сравнении вариантов.",
    )


class HMMTrajectory(BaseModel):
    """Viterbi-путь для одного эпизода (последовательности).

    Дополнительно несёт честные индикаторы качества интерпретации:

    * ``has_zap`` — был ли в серии хотя бы один не-noop токен. Если
      False, траектория восстановлена по приору переходов, и в UI
      такие эпизоды помечаются явно;
    * ``state_posterior`` — γ_t (T × K), per-step posterior из
      forward-backward; ``None`` означает, что posterior не считался
      (например, для bernoulli-ветки на этапах, где ещё не реализован
      вывод γ);
    * ``confidence`` — средняя по шагам максимальная вероятность
      состояния (mean_t max_k γ_t,k); если posterior не доступен,
      также ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    sheet: str
    episode_index: int
    episode_key: str = Field(
        default="",
        description=(
            "Композитный ключ серии (athlete + bout): нужен для UI и"
            " join'ов с baseline."
        ),
    )
    length: int
    observation_tokens: list[str]
    state_path: list[str]
    log_likelihood: float
    has_zap: bool = Field(
        default=True,
        description=(
            "Есть ли в наблюдениях хотя бы один не-noop токен. False -"
            " серия использовалась только для отчёта; в обучении HMM"
            " она не участвовала."
        ),
    )
    state_posterior: list[list[float]] | None = Field(
        default=None,
        description=(
            "Per-step posterior γ_t размера T × K (rows нормированы"
            " в 1). None, если backend не считает forward-backward."
        ),
    )
    confidence: float | None = Field(
        default=None,
        description=(
            "Средняя уверенность Viterbi-пути: mean_t(max_k γ_t,k)."
            " None, если posterior отсутствует."
        ),
    )


class VariantAttempt(BaseModel):
    """Попытка обучить вариант HMM (basic_3state / detailed_7state).

    Используется для прозрачности выбора в UI: пользователь видит, что
    было опробовано, что отбраковано и почему.
    """

    model_config = ConfigDict(extra="forbid")

    variant: str = Field(..., description="basic_3state или detailed_7state.")
    status: str = Field(
        ...,
        description=(
            "Один из: applied | rejected_by_guard | rejected_by_sanity |"
            " rejected_by_bic | fit_failed."
        ),
    )
    reason: str = ""
    bic: float | None = None
    log_likelihood: float | None = None
    n_states_used: int | None = None
    n_states: int | None = None


class HMMResult(BaseModel):
    """Итог HMM-ветки. Попадает в :attr:`AnalysisResult.hmm` при
    ``status == hmm_ready`` или ``status == hmm_low_signal``."""

    model_config = ConfigDict(extra="forbid")

    parameters: HMMParameters
    trajectories: list[HMMTrajectory] = Field(default_factory=list)
    state_distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Доля времени, проведённого в каждом скрытом состоянии (по всем эпизодам).",
    )
    sanity: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Результаты sanity-check'ов матрицы переходов и полносвязности "
            "эмиссионной матрицы."
        ),
    )
    interpretation: str = ""
    average_confidence: float | None = Field(
        default=None,
        description=(
            "Средняя по эпизодам confidence Viterbi-пути."
            " None, если posterior не считался."
        ),
    )
    training_excluded_episodes: int = Field(
        default=0,
        description=(
            "Сколько серий не использовалось при обучении (например, серий"
            " только из noop-токенов). Они попадают в trajectories с"
            " has_zap=False, но в fit не участвовали."
        ),
    )
    no_zap_trajectories: int = Field(
        default=0,
        description=(
            "Число траекторий без ни одного не-noop наблюдения. Полезно"
            " для UI, чтобы выделить блок «эпизоды без ZAP»."
        ),
    )
    tried_variants: list[VariantAttempt] = Field(
        default_factory=list,
        description=(
            "Лог попыток вариантов HMM: что обучали, что отбраковали"
            " и по какой причине. Применённый вариант помечен"
            " status=='applied'."
        ),
    )
