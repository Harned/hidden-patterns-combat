"""
Расширенная демонстрация HMM анализа экономических данных с валидацией модели
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from economic_hmm_analysis import EconomicHMMAnalyzer

def extended_validation_demo():
    """
    Демонстрация расширенной валидации HMM модели
    """
    print("=== РАСШИРЕННАЯ ВАЛИДАЦИЯ HMM МОДЕЛИ ===\n")
    
    # Загружаем данные
    analyzer = EconomicHMMAnalyzer()
    data = analyzer.load_sample_data()
    X = analyzer.preprocess_data(data)
    
    # 1. Расширенный подбор состояний
    print("1. РАСШИРЕННЫЙ ПОДБОР ЧИСЛА СОСТОЯНИЙ")
    print("----------------------------------------")
    n_states_range = range(2, 8 + 1)  # от 2 до 8
    bic_scores = []
    aic_scores = []
    log_likelihoods = []
    
    for n_states in n_states_range:
        model_temp = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
            tol=1e-6
        )
        
        try:
            model_temp.fit(X)
            log_likelihood = model_temp.score(X)
            log_likelihoods.append(log_likelihood)
            
            # Число параметров
            n_features = X.shape[1]
            n_params = (n_states * (n_states - 1) +  # переходы
                       (n_states - 1) +              # начальные вероятности
                       n_states * n_features +       # means
                       n_states * n_features * (n_features + 1) // 2)  # covariances
            
            # BIC и AIC
            bic = -2 * log_likelihood + n_params * np.log(X.shape[0])
            aic = -2 * log_likelihood + 2 * n_params
            
            bic_scores.append(bic)
            aic_scores.append(aic)
            
            print(f"n_states={n_states:2d}: BIC={bic:8.2f}, AIC={aic:8.2f}, logL={log_likelihood:8.2f}, params={n_params:3d}")
            
        except Exception as e:
            print(f"n_states={n_states:2d}: Ошибка - {str(e)}")
            bic_scores.append(np.inf)
            aic_scores.append(np.inf)
            log_likelihoods.append(-np.inf)
    
    # Построение графиков
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # BIC и AIC
    axes[0,0].plot(n_states_range, bic_scores, marker='o', label='BIC', linewidth=2)
    axes[0,0].plot(n_states_range, aic_scores, marker='s', label='AIC', linewidth=2)
    axes[0,0].set_xlabel('Число скрытых состояний')
    axes[0,0].set_ylabel('Информационный критерий')
    axes[0,0].set_title('BIC и AIC для разных чисел состояний')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Log-likelihood
    axes[0,1].plot(n_states_range, log_likelihoods, marker='v', label='Log-likelihood', linewidth=2, color='green')
    axes[0,1].set_xlabel('Число скрытых состояний')
    axes[0,1].set_ylabel('Log-likelihood')
    axes[0,1].set_title('Log-likelihood для разных чисел состояний')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Проверка "локтя" (elbow point) для BIC
    # Находим точку с максимальной кривизной
    try:
        bic_diff = np.diff(bic_scores)
        bic_diff2 = np.diff(bic_diff)
        elbow_idx = np.argmax(bic_diff2) + 1 if len(bic_diff2) > 0 else 0
        if elbow_idx < len(n_states_range):
            axes[0,0].axvline(x=n_states_range[elbow_idx], color='red', linestyle='--', alpha=0.7, label=f'Elbow point: {n_states_range[elbow_idx]}')
            axes[0,0].legend()
    except:
        pass
    
    # Проверка на переобучение - если критерии продолжают падать
    if len(bic_scores) > 2 and bic_scores[-1] < bic_scores[0]:
        print("\nПРЕДУПРЕЖДЕНИЕ: Критерии продолжают падать - возможен риск переобучения при большем числе состояний")
    
    # 2. Устойчивость к инициализации
    print("\n2. УСТОЙЧИВОСТЬ К ИНИЦИАЛИЗАЦИИ")
    print("------------------------------------")
    n_runs = 10
    run_log_likelihoods = []
    
    for run in range(n_runs):
        model_temp = hmm.GaussianHMM(
            n_components=3,  # используем 3 состояния для стабильности
            covariance_type="full",
            n_iter=1000,
            random_state=run,
            tol=1e-6
        )
        
        try:
            model_temp.fit(X)
            run_log_likelihoods.append(model_temp.score(X))
        except Exception as e:
            run_log_likelihoods.append(-np.inf)
            print(f"Run {run}: Ошибка - {str(e)}")
    
    run_log_likelihoods = np.array(run_log_likelihoods)
    valid_runs = run_log_likelihoods[run_log_likelihoods != -np.inf]
    
    if len(valid_runs) > 0:
        mean_logl = np.mean(valid_runs)
        std_logl = np.std(valid_runs)
        cv = (std_logl / abs(mean_logl)) * 100 if mean_logl != 0 else 0
        
        print(f"Средний log-likelihood: {mean_logl:.4f}")
        print(f"Стандартное отклонение: {std_logl:.4f}")
        print(f"Коэффициент вариации: {cv:.2f}%")
        
        if cv > 10:
            print("ПРЕДУПРЕЖДЕНИЕ: Модель нестабильна (коэффициент вариации > 10%)")
        else:
            print("Модель стабильна")
        
        # График стабильности
        axes[1,0].plot(range(len(valid_runs)), valid_runs, marker='o', linewidth=2)
        axes[1,0].axhline(y=mean_logl, color='r', linestyle='--', label=f'Среднее: {mean_logl:.2f}')
        axes[1,0].set_xlabel('Номер запуска')
        axes[1,0].set_ylabel('Log-likelihood')
        axes[1,0].set_title('Устойчивость к инициализации (10 запусков)')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
    
    # 3. Темпоральная валидация
    print("\n3. ТЕМПОРАЛЬНАЯ ВАЛИДАЦИЯ")
    print("-------------------------------")
    split_point = int(0.8 * len(X))
    
    X_train = X[:split_point]
    X_test = X[split_point:]
    
    # Обучаем модель на тренировочных данных
    model_train = hmm.GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
        tol=1e-6
    )
    
    try:
        model_train.fit(X_train)
        train_score = model_train.score(X_train)
        test_score = model_train.score(X_test)
        print(f"Log-likelihood на train: {train_score:.4f}")
        print(f"Log-likelihood на test:  {test_score:.4f}")
        print(f"Разница (train-test):    {train_score - test_score:.4f}")
        
        if train_score - test_score > 10:
            print("ПРЕДУПРЕЖДЕНИЕ: Сильное переобучение (разница > 10)")
        
        # Baseline - случайное угадывание (оценка с помощью модели, обученной на всем наборе)
        baseline_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
        baseline_model.fit(X)
        baseline_test_score = baseline_model.score(X_test)
        print(f"Baseline на test:        {baseline_test_score:.4f}")
        
        if test_score > baseline_test_score:
            print("МОДЕЛЬ ЛУЧШЕ БАЗОВОЙ!")
        else:
            print("БАЗОВАЯ МОДЕЛЬ ЛУЧШЕ - возможна проблема с обобщающей способностью")
            
    except Exception as e:
        print(f"Ошибка при темпоральной валидации: {str(e)}")
    
    # 4. Сопоставление с событиями
    print("\n4. СОПОСТАВЛЕНИЕ С ИЗВЕСТНЫМИ СОБЫТИЯМИ")
    print("------------------------------------------")
    events = {
        "2014": "Санкции (Крым)", 
        "2020": "COVID + обвал нефти", 
        "2022": "Геополитический шок"
    }
    
    # Обучаем финальную модель
    analyzer.fit_model(X, n_hidden_states=3)
    state_sequence = analyzer.decode_states(X)
    
    print("Дата | Состояние | Событие")
    print("-----|----------|---------")
    
    for i, year in enumerate(data['Год']):
        year_str = str(year)
        event = events.get(year_str, "")
        state = state_sequence[i]
        print(f"{year_str} | Состояние {state}  | {event}")
    
    # Найдем совпадения смены состояний и событий
    for event_year, event_desc in events.items():
        event_idx = data[data['Год'] == int(event_year)].index
        if len(event_idx) > 0:
            event_idx = event_idx[0]
            # Проверим, были ли изменения состояния рядом с этим годом
            if event_idx > 0:
                prev_state = state_sequence[event_idx-1]
                curr_state = state_sequence[event_idx]
                if prev_state != curr_state:
                    print(f"  -> Смена состояния {prev_state}→{curr_state} совпадает с '{event_desc}'")
    
    # 5. Визуализация
    print("\n5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("---------------------------")
    
    # Временной ряд с цветовой разметкой состояний
    years = data['Год'].values
    features = ['Внешнеторговый оборот (млрд $)', 'Сальдо торговли (млрд $)', 'Доля нефтегазовых доходов (%)']
    
    for i, feature in enumerate(features):
        ax = axes[1,1] if i == 0 else plt.subplots(figsize=(12, 5))[1]
        scatter = ax.scatter(years, data[feature], c=state_sequence, cmap='viridis', s=100, edgecolors='black')
        ax.set_xlabel('Год')
        ax.set_ylabel(feature)
        ax.set_title(f'{feature} с пометкой скрытых состояний')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Скрытое состояние')
        
        if i == 0:  # Показываем только первую в главном окне
            continue
    
    plt.tight_layout()
    plt.show()
    
    # Heatmap матрицы переходов
    if analyzer.transition_matrix is not None:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            analyzer.transition_matrix,
            annot=True,
            fmt='.3f',
            xticklabels=[f"Состояние {i}" for i in range(len(analyzer.transition_matrix))],
            yticklabels=[f"Состояние {i}" for i in range(len(analyzer.transition_matrix))],
            cmap='Blues',
            cbar_kws={'label': 'Вероятность перехода'}
        )
        plt.title('Матрица переходов между скрытыми состояниями')
        plt.xlabel('Состояние в момент t+1')
        plt.ylabel('Состояние в момент t')
        plt.tight_layout()
        plt.show()
    
    # Столбчатая диаграмма средних значений признаков по состояниям
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Подготавливаем данные для визуализации
    data_with_states = data.copy()
    data_with_states['Скрытое состояние'] = state_sequence
    
    state_means = {}
    for state in range(3):
        state_data = data_with_states[data_with_states['Скрытое состояние'] == state]
        if len(state_data) > 0:
            state_means[state] = {
                'Внешнеторг. оборот': state_data['Внешнеторговый оборот (млрд $)'].mean(),
                'Сальдо торговли': state_data['Сальдо торговли (млрд $)'].mean(),
                'Доля нефтегаз.': state_data['Доля нефтегазовых доходов (%)'].mean()
            }
    
    # Создаем столбчатую диаграмму
    x = np.arange(len(features))  # количество признаков
    width = 0.25  # ширина столбцов
    
    if state_means:
        for i, state in enumerate(state_means.keys()):
            values = [state_means[state][f] for f in features]
            ax.bar(x + i*width, values, width, label=f'Состояние {state}', alpha=0.8)
    
    ax.set_xlabel('Признаки')
    ax.set_ylabel('Среднее значение')
    ax.set_title('Средние значения признаков по скрытым состояниям')
    ax.set_xticks(x + width * len(state_means.keys()) / 2)
    ax.set_xticklabels([f.split(' (')[0] for f in features])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== РАСШИРЕННЫЙ АНАЛИЗ ЗАВЕРШЕН ===")


if __name__ == "__main__":
    extended_validation_demo()