# Data Dictionary (Current Dataset, MVP)

Словарь фиксирует связь:
`original excel header -> normalized field name -> logical group -> description`.

## Где хранится машинный словарь
- JSON: `src/hidden_patterns_combat/preprocessing/resources/data_dictionary_v1.json`
- Python API: `hidden_patterns_combat.preprocessing.data_dictionary.DataDictionary`

## Логические группы
- `metadata` — идентификация эпизода и контекст листа.
- `maneuvering` — бинарные индикаторы стойки/маневрирования.
- `kfv` — бинарные индикаторы контактов физического взаимодействия.
- `vup` — бинарные индикаторы выведения из устойчивого положения.
- `outcomes` — баллы и завершающие атакующие действия.
- `other` — временная зона для нераспознанных колонок.

## Примеры mapping
| original excel header | normalized field name | logical group | description |
|---|---|---|---|
| `фио борца` | `metadata__athlete_name` | `metadata` | ФИО спортсмена |
| `технико-тактический эпизод` | `metadata__episode_attr_01` | `metadata` | Атрибут эпизода (1) |
| `баллы` | `outcomes__score` | `outcomes` | Судейская оценка/балл эпизода |
| `стойка и маневрирование самбиста (основные в эпизоде)` | `maneuvering__indicator_01` | `maneuvering` | Индикатор маневрирования №01 |
| `контакты физического взаимодействия (захваты, обхваты, прихваты, хваты, упоры)_7` | `kfv__indicator_07` | `kfv` | Индикатор КФВ №07 |
| `выведение соперника из устойчивого положения (при выполнении n или n1)_3` | `vup__indicator_03` | `vup` | Индикатор ВУП №03 |
| `завершающие атаку приемы (n)_2` | `outcomes__finish_action_02` | `outcomes` | Завершающее действие №02 |

## Полный набор текущего файла
В `data_dictionary_v1.json` описаны 70 записей (включая `_sheet`) для текущего формата Excel (`episodes.xlsx`).

## Интеграция в pipeline
1. **Preprocessing**: `transform_raw_to_tidy(...)` сначала делает exact lookup по JSON-словарю, затем fallback на token-rules.
2. **Feature engineering**: `encode_features(...)` использует dictionary-группы для отбора `maneuvering/kfv/vup` колонок; если колонка не найдена — fallback на токены.
3. **Валидация**: `validate_tidy_structure(...)` проверяет присутствие обязательных блоков из словаря.
