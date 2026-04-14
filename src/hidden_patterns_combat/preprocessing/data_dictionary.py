from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DictionaryEntry:
    original_header: str
    normalized_field: str
    logical_group: str
    description: str


@dataclass(frozen=True)
class BlockRule:
    block: str
    tokens: tuple[str, ...]


class DataDictionary:
    """Mapping registry for `original excel header -> normalized field -> logical group`.

    Primary behavior:
    - exact mapping from JSON dictionary;
    - token-based fallback for unseen columns.
    """

    def __init__(self, entries: list[DictionaryEntry]):
        self.entries = entries
        self._by_header = {e.original_header.lower(): e for e in entries}
        self.canonical_metadata_map: dict[str, str] = {
            "_sheet": "metadata__sheet",
            "source_sheet": "metadata__sheet",
            "фио борца": "metadata__athlete_name",
            "athlete_name": "metadata__athlete_name",
            "episode_id": "metadata__episode_id",
            "episode_order": "metadata__episode_order",
            "episode_duration": "metadata__episode_duration",
            "pause_duration": "metadata__pause_duration",
            "баллы": "outcomes__score",
            "observed_result": "outcomes__score",
        }

        # Order matters: tactical blocks should be checked before metadata.
        # Real headers often include phrase "в эпизоде", which must not force
        # maneuvering/KFV/VUP columns into metadata.
        self.block_rules: tuple[BlockRule, ...] = (
            BlockRule("maneuvering", ("стойка", "маневр", "maneuver")),
            BlockRule("kfv", ("кфв", "контакты физического взаимодействия", "захват", "обхват", "прихват", "хват", "упор")),
            BlockRule("vup", ("вуп", "выведение", "vup")),
            BlockRule("outcomes", ("балл", "результ", "score", "outcome", "заверша", "брос", "удерж", "болев")),
            BlockRule(
                "metadata",
                (
                    "фио",
                    "борца",
                    "athlete",
                    "fighter",
                    "технико-тактический эпизод",
                    "эпизод",
                    "episode",
                    "время",
                    "duration",
                    "пауз",
                    "pause",
                    "sheet",
                ),
            ),
        )
        self.required_blocks: tuple[str, ...] = ("metadata", "maneuvering", "kfv", "vup", "outcomes")

    @classmethod
    def from_json(cls, path: str | Path) -> "DataDictionary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [DictionaryEntry(**row) for row in payload.get("entries", [])]
        return cls(entries)

    @classmethod
    def default(cls) -> "DataDictionary":
        here = Path(__file__).resolve().parent
        path = here / "resources" / "data_dictionary_v1.json"
        return cls.from_json(path)

    def lookup(self, original_header: str) -> DictionaryEntry | None:
        return self._by_header.get(original_header.lower())

    def infer_block(self, column_name: str) -> str:
        low = column_name.lower()
        for prefix in ("metadata__", "maneuvering__", "kfv__", "vup__", "outcomes__", "other__"):
            if low.startswith(prefix):
                return prefix.replace("__", "")
        for rule in self.block_rules:
            if any(token in low for token in rule.tokens):
                return rule.block
        return "other"

    def columns_for_group(self, columns: list[str], group: str) -> list[str]:
        exact = [c for c in columns if (self.lookup(c) and self.lookup(c).logical_group == group)]
        if exact:
            return exact
        return [c for c in columns if self.infer_block(c) == group]
