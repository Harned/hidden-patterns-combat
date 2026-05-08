# TASK_SPEC_008_EMISSIONS_CV

> **Status: under review — depends on resolution of Conflict C1**
>
> Multivariate Bernoulli-эмиссии и k-fold CV имеют смысл только
> внутри HMM-ветки (`TASK_SPEC_004/005`). Поскольку HMM отложена
> до окончания Уровня 1 (`TASK_SPEC_011/012/013`), этот файл
> также отложен. До решения по C1 новые работы по эмиссиям и CV
> не начинать.

## Текущая задача

1. Дать пользователю **альтернативу Categorical-эмиссиям**: multivariate
   Bernoulli по каналам ЗАП. Это соответствует реальной кодировке, где
   наблюдение на шаге эпизода — это набор 0/1 по нескольким каналам
   одновременно (удержание + болевой приём могут сосуществовать в
   одном эпизоде).
2. Добавить утилиту **cross-validation** для HMM: k-fold по весовым
   категориям (либо по листам) с оценкой средней log-likelihood на
   hold-out. Это честный способ сравнить варианты модели без оверфита.

## Мотивация

Текущая `CategoricalHMM` считает каждый токен как один символ. Если у
эпизода одновременно есть «Удержание» и «На руку», они идут двумя
отдельными токенами подряд, и модель видит их как последовательность.
Это достаточно для первого приближения, но теряет информацию о
ко-встречаемости. Bernoulli-эмиссия моделирует наблюдение как вектор
флагов по каналам.

## Что должно появиться

1. **algo**:
   * Новый тип эмиссии `observation_emission`:
     `categorical` (как сейчас) или `bernoulli`.
   * Функции сборки Bernoulli-последовательностей и реализация EM
     (без `hmmlearn`, т.к. он не поддерживает multivariate Bernoulli
     в общем виде — пишем свой минимальный EM).
   * `AnalyzeConfig.observation_emission: Literal[...]` + проброс.
   * Обновлённый `HMMParameters` с полем `observation_emission`;
     для Bernoulli — `emission_matrix: n_states × n_channels`
     (вероятности по каналам).
2. **Cross-validation helper**:
   * `hpc_algo.cv.cross_validate(analyze_fn, frames, config, k=5)` —
     делает k-fold по sheet-именам, возвращает mean/std
     held-out log-likelihood.
   * CLI: `hpc-algo cross-validate <xlsx> <mapping.json>`.
3. **algo tests**:
   * юнит-тест Bernoulli-EM на игрушечной выборке (известный π/A/B
     → сгенерированные данные → восстановленные параметры близки к
     истинным);
   * инварианты: observations остаются ЗАП, состояния из доменного
     списка;
   * тест CV возвращает осмысленный средний LL для плотной фикстуры.

## Что НЕ делает этот шаг

* Не меняет UI (выбор эмиссии появится позже или только через API).
* Не реализует Poisson / HSMM.
* Не меняет формат `AnalysisResult`, кроме поля
  `HMMParameters.observation_emission`.

## Definition of Done

1. `AnalyzeConfig(observation_emission="bernoulli")` на плотной
   synthetic-фикстуре возвращает `status = hmm_ready` с
   `observation_emission == "bernoulli"` и ненулевыми эмиссиями по
   ожидаемым каналам.
2. `hpc_algo.cv.cross_validate` на плотной фикстуре возвращает
   mean/std LL, которые можно сравнить между `categorical` и
   `bernoulli` (и между `basic` / `detailed`).
3. Все прежние инварианты сохранены. Тесты зелёные.
4. README обновлён, ограничения (реализация EM без SciPy-зависимостей)
   явно зафиксированы.
