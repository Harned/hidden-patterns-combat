#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Устойчивая HMM-модель для выявления скрытых экономических состояний
с ограниченным объемом данных (10-20 наблюдений)
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Загрузка и подготовка экономических данных
    """
    # Источник данных
    data = {
        'Год': [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        'Оборот_млрд': [750, 650, 550, 490, 550, 600, 650, 680, 700, 720, 680],
        'Сальдо_млрд': [150, 50, -30, -50, 20, 30, 80, 100, 120, 70, 20],
        'Нефтегаз_доля': [65, 68, 70, 68, 65, 60, 55, 50, 52, 55, 60]
    }
    
    df = pd.DataFrame(data)
    print("Данные загружены:")
    print(df)
    print(f"\nКоличество наблюдений: {len(df)}")
    
    # Подготовка признаков для моделирования
    features = ['Оборот_млрд', 'Сальдо_млрд', 'Нефтегаз_доля']
    X = df[features].values
    
    # Нормализация данных для улучшения обучения
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return df, X, X_scaled, scaler


def run_hmm_multistart(X, n_runs=20):
    """
    Запуск HMM с мультистартом для обеспечения устойчивости
    """
    print(f"\nЗапуск HMM с мультистартом ({n_runs} запусков)...")
    
    results = []
    
    for i in range(n_runs):
        # Создание модели с заданными ограничениями
        model = hmm.GaussianHMM(
            n_components=3,
            covariance_type="spherical", 
            n_iter=100,
            random_state=i
        )
        
        try:
            # Обучение модели
            model.fit(X)
            
            # Получение log-likelihood
            log_likelihood = model.score(X)
            
            # Сохранение результатов
            results.append({
                'model': model,
                'log_likelihood': log_likelihood,
                'transmat': model.transmat_,
                'means': model.means_,
                'covars': model.covars_,
                'random_state': i
            })
            
            print(f"Запуск {i}: log-likelihood = {log_likelihood:.2f}")
            
        except Exception as e:
            print(f"Ошибка в запуске {i}: {str(e)}")
            continue
    
    if not results:
        raise ValueError("Не удалось обучить ни одну модель")
    
    return results


def validate_best_model(results):
    """
    Валидация лучшей модели по log-likelihood
    """
    # Находим лучшую модель
    best_result = max(results, key=lambda x: x['log_likelihood'])
    
    # Статистика по всем запускам
    log_likelihoods = [res['log_likelihood'] for res in results]
    
    stats = {
        'mean_log_likelihood': np.mean(log_likelihoods),
        'std_log_likelihood': np.std(log_likelihoods),
        'min_log_likelihood': np.min(log_likelihoods),
        'max_log_likelihood': np.max(log_likelihoods)
    }
    
    print("\n--- Статистика устойчивости ---")
    print(f"Среднее log-likelihood: {stats['mean_log_likelihood']:.2f}")
    print(f"Std log-likelihood: {stats['std_log_likelihood']:.2f}")
    print(f"Min log-likelihood: {stats['min_log_likelihood']:.2f}")
    print(f"Max log-likelihood: {stats['max_log_likelihood']:.2f}")
    
    # Проверка на нестабильность
    if stats['std_log_likelihood'] > 0.2 * abs(stats['mean_log_likelihood']):
        print("⚠️ Модель нестабильна. Рекомендуется увеличить объём данных.")
    
    return best_result, stats


def interpret_states(means, states_sequence, df):
    """
    Интерпретация скрытых состояний на основе средних значений признаков
    """
    # Вычисление средних значений признаков для каждого состояния
    state_means = {}
    for state in range(3):
        state_mask = (states_sequence == state)
        if np.sum(state_mask) > 0:
            state_means[state] = means[state]
        else:
            state_means[state] = np.mean(means, axis=0)
    
    # Определение меток состояний
    labels = {}
    for state in range(3):
        mean_turnover = state_means[state][0]  # Оборот
        mean_balance = state_means[state][1]   # Сальдо
        
        if mean_turnover > np.mean([state_means[s][0] for s in range(3)]) and mean_balance > 0:
            labels[state] = "Рост"
        elif mean_turnover < np.mean([state_means[s][0] for s in range(3)]) and mean_balance < 0:
            labels[state] = "Кризис"
        else:
            labels[state] = "Стагнация"
    
    # Проверка на пустые состояния
    unique, counts = np.unique(states_sequence, return_counts=True)
    total_obs = len(states_sequence)
    
    for state, count in zip(unique, counts):
        percentage = 100 * count / total_obs
        if percentage < 5:
            print(f"⚠️ Пустое состояние {state}: {percentage:.1f}% наблюдений")
    
    return labels


def visualize_results(df, X, best_model, results, labels):
    """
    Визуализация результатов
    """
    # Предсказание последовательности состояний для лучшей модели
    states_sequence = best_model.predict(X)
    
    # 1. График BIC/AIC для n=2,3,4 (подтверждение выбора n=3)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Подсчет BIC/AIC для разных чисел компонент
    n_components_range = [2, 3, 4]
    bics = []
    aics = []
    
    for n_comp in n_components_range:
        model_temp = hmm.GaussianHMM(n_components=n_comp, covariance_type="spherical", random_state=0)
        try:
            model_temp.fit(X)
            log_likelihood = model_temp.score(X)
            n_params = (n_comp - 1) + n_comp * X.shape[1] + n_comp  # параметры: начальные вероятности + переходы + средние + дисперсии
            
            bic = -2 * log_likelihood + n_params * np.log(X.shape[0])  # BIC
            aic = -2 * log_likelihood + 2 * n_params                   # AIC
            
            bics.append(bic)
            aics.append(aic)
        except:
            bics.append(np.inf)
            aics.append(np.inf)
    
    axes[0, 0].plot(n_components_range, bics, marker='o', label='BIC')
    axes[0, 0].plot(n_components_range, aics, marker='s', label='AIC')
    axes[0, 0].set_title('BIC/AIC для разного числа компонент')
    axes[0, 0].set_xlabel('Число компонент')
    axes[0, 0].set_ylabel('Значение')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Boxplot log-likelihood по 20 запускам
    log_likelihoods = [res['log_likelihood'] for res in results]
    axes[0, 1].boxplot(log_likelihoods)
    axes[0, 1].set_title('Распределение log-likelihood по запускам')
    axes[0, 1].set_ylabel('Log-likelihood')
    axes[0, 1].grid(True)
    
    # 3. Временной ряд с цветовой разметкой состояний
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    for i in range(3):
        mask = states_sequence == i
        if np.any(mask):
            axes[1, 0].scatter(df['Год'][mask], df['Оборот_млрд'][mask], 
                              c=colors[i], label=f'{labels[i]} (сост. {i})', 
                              alpha=0.7, s=100)
    
    axes[1, 0].set_title('Временной ряд с цветовой разметкой состояний')
    axes[1, 0].set_xlabel('Год')
    axes[1, 0].set_ylabel('Оборот (млрд)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Heatmap матрицы переходов
    transmat = best_model.transmat_
    im = axes[1, 1].imshow(transmat, cmap='Blues', aspect='auto')
    axes[1, 1].set_title('Матрица переходов между состояниями')
    axes[1, 1].set_xlabel('Состояние назначения')
    axes[1, 1].set_ylabel('Состояние источника')
    
    # Добавляем числовые значения на ячейках
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            axes[1, 1].text(j, i, f'{transmat[i, j]:.2f}', 
                           ha="center", va="center", color="black")
    
    # Добавляем подписи к осям
    tick_labels = [f'{labels[i]} ({i})' for i in range(3)]
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_xticklabels(tick_labels, rotation=45)
    axes[1, 1].set_yticklabels(tick_labels)
    
    plt.tight_layout()
    plt.show()


def print_results_table(df, states_sequence, labels):
    """
    Вывод таблицы с присвоенными состояниями
    """
    # Добавляем состояния и метки в DataFrame
    df_results = df.copy()
    df_results['Состояние'] = states_sequence
    df_results['Метка_состояния'] = [labels[state] for state in states_sequence]
    
    print("\n--- Таблица: Год | Состояние | Метка состояния ---")
    print(df_results[['Год', 'Состояние', 'Метка_состояния']].to_string(index=False))
    
    return df_results


def print_model_parameters(best_result):
    """
    Вывод параметров лучшей модели
    """
    print("\n--- Матрица переходов (округлено до 2 знаков) ---")
    transmat = best_result['transmat']
    for i, row in enumerate(transmat):
        rounded_row = [f"{val:.2f}" for val in row]
        print(f"Состояние {i}: [{', '.join(rounded_row)}]")
    
    print("\n--- Средние значения признаков для каждого состояния ---")
    means = best_result['means']
    feature_names = ["Оборот_млрд", "Сальдо_млрд", "Нефтегаз_доля"]
    
    for state_idx, state_means in enumerate(means):
        print(f"Состояние {state_idx}:")
        for feat_idx, (feature_name, mean_val) in enumerate(zip(feature_names, state_means)):
            print(f"  {feature_name}: {mean_val:.2f}")
        print()


def main():
    """
    Основная функция выполнения анализа
    """
    print("=== Устойчивая HMM-модель для выявления экономических состояний ===")
    
    # Загрузка данных
    df, X, X_scaled, scaler = load_data()
    
    # Запуск HMM с мультистартом
    results = run_hmm_multistart(X_scaled, n_runs=20)
    
    # Валидация лучшей модели
    best_result, stats = validate_best_model(results)
    
    # Предсказание состояний лучшей моделью
    best_model = best_result['model']
    states_sequence = best_model.predict(X_scaled)
    
    # Интерпретация состояний
    labels = interpret_states(best_result['means'], states_sequence, df)
    
    # Вывод параметров модели
    print_model_parameters(best_result)
    
    # Вывод таблицы результатов
    df_results = print_results_table(df, states_sequence, labels)
    
    # Визуализация результатов
    visualize_results(df, X_scaled, best_model, results, labels)
    
    print("\n=== Анализ завершен ===")


if __name__ == "__main__":
    main()