"""
Скрытая Марковская модель для выявления скрытых факторов, влияющих на доходы от торговли

Задача: выявление скрытых экономических режимов, влияющих на торговые поступления
- Наблюдаемые данные: временные ряды внешнеторгового оборота, сальдо торговли, доли нефтегазовых доходов
- Скрытые состояния: санкционное давление, ценовой шок, нормальный режим, структурная перестройка
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Установка параметров matplotlib для более красивого вывода
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EconomicHMMAnalyzer:
    """
    Класс для анализа экономических данных с помощью скрытых Марковских моделей
    """
    
    def __init__(self, n_hidden_states=None):
        """
        Инициализация анализатора
        
        Args:
            n_hidden_states: Количество скрытых состояний (если None, будет выбрано автоматически)
        """
        self.n_hidden_states = n_hidden_states
        self.model = None
        self.scaler = StandardScaler()
        self.transition_matrix = None
        self.state_sequence = None
        self.state_probabilities = None
        self.fitted = False
        
    def load_sample_data(self):
        """
        Загрузка примера данных для демонстрации работы модели
        """
        # Создание синтетических данных, имитирующих реальные экономические показатели
        years = np.arange(2014, 2025)
        
        # Синтетические данные, приближенные к реальным значениям внешней торговли России
        external_trade = np.array([
            750, 650, 550, 500, 520, 550, 650, 700, 730, 680, 620  # млрд $
        ])
        
        # Сальдо торговли (может быть как положительным, так и отрицательным)
        trade_balance = np.array([
            150, 50, -30, -50, -20, 30, 80, 100, 120, 70, 20  # млрд $
        ])
        
        # Доля нефтегазовых доходов (в процентах)
        oil_gas_ratio = np.array([
            65, 68, 70, 68, 65, 60, 55, 50, 52, 55, 60  # %
        ])
        
        # Создание DataFrame
        data = pd.DataFrame({
            'Год': years,
            'Внешнеторговый оборот (млрд $)': external_trade,
            'Сальдо торговли (млрд $)': trade_balance,
            'Доля нефтегазовых доходов (%)': oil_gas_ratio
        })
        
        return data
    
    def preprocess_data(self, data, columns=['Внешнеторговый оборот (млрд $)', 'Сальдо торговли (млрд $)', 'Доля нефтегазовых доходов (%)']):
        """
        Предобработка данных: нормализация и подготовка для HMM
        
        Args:
            data: исходные данные (DataFrame)
            columns: список колонок для использования
            
        Returns:
            X: подготовленные данные для модели
        """
        print("Начинаем предобработку данных...")
        
        # Выбор колонок и преобразование в numpy массив
        X = data[columns].values.astype(float)
        
        # Нормализация данных
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Размер данных после предобработки: {X_scaled.shape}")
        print(f"Колонки: {columns}")
        
        return X_scaled
    
    def fit_model(self, X, n_hidden_states=None):
        """
        Обучение HMM модели
        
        Args:
            X: подготовленные данные
            n_hidden_states: количество скрытых состояний (если None, используется значение из init)
        """
        if n_hidden_states is not None:
            self.n_hidden_states = n_hidden_states
        
        if self.n_hidden_states is None:
            print("Количество скрытых состояний не задано, будет выбрано автоматически")
            self.n_hidden_states = 4  # По умолчанию используем 4 состояния
        
        print(f"Обучение модели с {self.n_hidden_states} скрытыми состояниями...")
        
        # Создание и обучение модели HMM с гауссовыми эмиссиями
        self.model = hmm.GaussianHMM(
            n_components=self.n_hidden_states,
            covariance_type="full",
            n_iter=1000,
            random_state=42,
            tol=1e-6
        )
        
        # Обучение модели (алгоритм Баума-Уэлча)
        self.model.fit(X)
        
        # Предсказание последовательности скрытых состояний (алгоритм Витерби)
        self.state_sequence = self.model.predict(X)
        
        # Получение вероятностей состояний
        self.state_probabilities = self.model.predict_proba(X)
        
        # Сохранение матрицы переходов
        self.transition_matrix = self.model.transmat_
        
        self.fitted = True
        print(f"Модель успешно обучена! Логарифм правдоподобия: {self.model.score(X):.4f}")
    
    def find_optimal_states(self, X, max_states=6):
        """
        Подбор оптимального числа скрытых состояний с использованием BIC и AIC
        
        Args:
            X: подготовленные данные
            max_states: максимальное число состояний для проверки
            
        Returns:
            optimal_n_states: оптимальное количество состояний
        """
        print(f"Поиск оптимального числа скрытых состояний (max={max_states})...")
        
        n_states_range = range(2, max_states + 1)
        bic_scores = []
        aic_scores = []
        
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
                
                # Вычисляем log-likelihood
                log_likelihood = model_temp.score(X)
                
                # Число параметров модели для BIC и AIC
                # Для HMM с n_states состояниями:
                # - переходная матрица: n_states*(n_states-1) параметров
                # - начальные вероятности: n_states-1 параметров  
                # - эмиссионные параметры (гауссовы): n_states * (n_features + n_features*(n_features+1)/2)
                # где n_features - размерность наблюдений
                
                n_features = X.shape[1]
                n_params = (n_states * (n_states - 1) +  # переходы
                           (n_states - 1) +              # начальные вероятности
                           n_states * n_features +       # means
                           n_states * n_features * (n_features + 1) // 2)  # covariances
                
                # BIC = -2*logL + k*log(N) где k - число параметров, N - число наблюдений
                bic = -2 * log_likelihood + n_params * np.log(X.shape[0])
                # AIC = -2*logL + 2*k
                aic = -2 * log_likelihood + 2 * n_params
                
                bic_scores.append(bic)
                aic_scores.append(aic)
                
            except Exception as e:
                print(f"Ошибка при обучении модели с {n_states} состояниями: {str(e)}")
                bic_scores.append(np.inf)
                aic_scores.append(np.inf)
        
        # Находим минимальные значения (лучшие модели)
        optimal_bic_idx = np.argmin(bic_scores)
        optimal_aic_idx = np.argmin(aic_scores)
        
        optimal_n_states_bic = n_states_range[optimal_bic_idx]
        optimal_n_states_aic = n_states_range[optimal_aic_idx]
        
        print(f"Оптимальное количество состояний по BIC: {optimal_n_states_bic}")
        print(f"Оптимальное количество состояний по AIC: {optimal_n_states_aic}")
        
        # Используем BIC для выбора (обычно более консервативен)
        self.n_hidden_states = optimal_n_states_bic
        
        # Построение графика
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(n_states_range, bic_scores, marker='o', label='BIC', linewidth=2)
        ax.plot(n_states_range, aic_scores, marker='s', label='AIC', linewidth=2)
        ax.axvline(x=optimal_n_states_bic, color='red', linestyle='--', alpha=0.7, label=f'BIC оптимум: {optimal_n_states_bic}')
        ax.set_xlabel('Число скрытых состояний')
        ax.set_ylabel('Информационный критерий')
        ax.set_title('Выбор оптимального числа скрытых состояний\n(BIC и AIC)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return self.n_hidden_states
    
    def decode_states(self, X):
        """
        Декодирование последовательности скрытых состояний (уже выполнено в fit_model)
        """
        if not self.fitted:
            raise ValueError("Модель не обучена! Сначала вызовите метод fit_model.")
        
        return self.state_sequence
    
    def visualize_results(self, data, state_names=None):
        """
        Визуализация результатов
        
        Args:
            data: исходные данные
            state_names: названия скрытых состояний
        """
        if not self.fitted:
            raise ValueError("Модель не обучена! Сначала вызовите метод fit_model.")
        
        if state_names is None:
            state_names = [f"Состояние {i}" for i in range(self.n_hidden_states)]
        
        # Создание графиков
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        years = data['Год'].values
        
        # 1. Временной ряд внешнеторгового оборота с цветовой разметкой состояний
        ax1 = axes[0]
        scatter = ax1.scatter(years, data['Внешнеторговый оборот (млрд $)'], 
                            c=self.state_sequence, cmap='viridis', s=100, edgecolors='black')
        ax1.set_xlabel('Год')
        ax1.set_ylabel('Внешнеторговый оборот (млрд $)')
        ax1.set_title('Внешнеторговый оборот с пометкой скрытых состояний')
        ax1.grid(True, alpha=0.3)
        
        # Добавляем легенду
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Скрытое состояние')
        
        # 2. Временной ряд сальдо торговли с цветовой разметкой состояний
        ax2 = axes[1]
        scatter2 = ax2.scatter(years, data['Сальдо торговли (млрд $)'], 
                             c=self.state_sequence, cmap='viridis', s=100, edgecolors='black')
        ax2.set_xlabel('Год')
        ax2.set_ylabel('Сальдо торговли (млрд $)')
        ax2.set_title('Сальдо торговли с пометкой скрытых состояний')
        ax2.grid(True, alpha=0.3)
        
        # Добавляем легенду
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Скрытое состояние')
        
        # 3. Временной ряд доли нефтегазовых доходов с цветовой разметкой состояний
        ax3 = axes[2]
        scatter3 = ax3.scatter(years, data['Доля нефтегазовых доходов (%)'], 
                             c=self.state_sequence, cmap='viridis', s=100, edgecolors='black')
        ax3.set_xlabel('Год')
        ax3.set_ylabel('Доля нефтегазовых доходов (%)')
        ax3.set_title('Доля нефтегазовых доходов с пометкой скрытых состояний')
        ax3.grid(True, alpha=0.3)
        
        # Добавляем легенду
        cbar3 = plt.colorbar(scatter3, ax=ax3)
        cbar3.set_label('Скрытое состояние')
        
        plt.tight_layout()
        plt.show()
        
        # Визуализация матрицы переходов
        self.visualize_transition_matrix(state_names)
    
    def visualize_transition_matrix(self, state_names=None):
        """
        Визуализация матрицы переходов
        
        Args:
            state_names: названия скрытых состояний
        """
        if not self.fitted or self.transition_matrix is None:
            raise ValueError("Модель не обучена или матрица переходов отсутствует!")
        
        if state_names is None:
            state_names = [f"Состояние {i}" for i in range(self.n_hidden_states)]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            self.transition_matrix,
            annot=True,
            fmt='.3f',
            xticklabels=state_names,
            yticklabels=state_names,
            cmap='Blues',
            cbar_kws={'label': 'Вероятность перехода'}
        )
        plt.title('Матрица переходов между скрытыми состояниями')
        plt.xlabel('Состояние в момент t+1')
        plt.ylabel('Состояние в момент t')
        plt.tight_layout()
        plt.show()
    
    def interpret_states(self, data):
        """
        Интерпретация скрытых состояний
        
        Args:
            data: исходные данные
            
        Returns:
            interpretation: словарь с интерпретацией состояний
        """
        if not self.fitted:
            raise ValueError("Модель не обучена! Сначала вызовите метод fit_model.")
        
        print("\\n=== ИНТЕРПРЕТАЦИЯ СКРЫТЫХ СОСТОЯНИЙ ===")
        
        # Добавляем предсказанные состояния к данным
        data_with_states = data.copy()
        data_with_states['Скрытое состояние'] = self.state_sequence
        
        interpretations = {}
        
        for state in range(self.n_hidden_states):
            state_data = data_with_states[data_with_states['Скрытое состояние'] == state]
            
            if len(state_data) == 0:
                print(f"Состояние {state}: Не было наблюдено в последовательности")
                continue
            
            print(f"\\n--- Состояние {state} ({len(state_data)} наблюдений) ---")
            
            # Средние значения характеристик для этого состояния
            avg_trade = state_data['Внешнеторговый оборот (млрд $)'].mean()
            avg_balance = state_data['Сальдо торговли (млрд $)'].mean()
            avg_oil_gas = state_data['Доля нефтегазовых доходов (%)'].mean()
            
            print(f"Средний внешнеторговый оборот: {avg_trade:.2f} млрд $")
            print(f"Среднее сальдо торговли: {avg_balance:.2f} млрд $")
            print(f"Средняя доля нефтегазовых доходов: {avg_oil_gas:.2f}%")
            
            # Определение типа состояния на основе характеристик
            state_type = self._classify_state(avg_trade, avg_balance, avg_oil_gas)
            print(f"Интерпретация: {state_type}")
            
            interpretations[state] = {
                'description': state_type,
                'avg_trade': avg_trade,
                'avg_balance': avg_balance,
                'avg_oil_gas': avg_oil_gas,
                'years_in_state': state_data['Год'].tolist()
            }
        
        return interpretations
    
    def _classify_state(self, avg_trade, avg_balance, avg_oil_gas):
        """
        Классификация типа состояния на основе средних значений
        
        Args:
            avg_trade: средний внешнеторговый оборот
            avg_balance: среднее сальдо торговли
            avg_oil_gas: средняя доля нефтегазовых доходов
            
        Returns:
            type_str: строковое описание типа состояния
        """
        # Нормализуем характеристики для сравнения
        # Примерные пороги для интерпретации (на основе данных)
        high_trade = avg_trade > 600  # высокий внешнеторговый оборот
        positive_balance = avg_balance > 0  # положительное сальдо
        high_oil_gas_ratio = avg_oil_gas > 60  # высокая доля нефтегазовых доходов
        
        if high_oil_gas_ratio and high_trade and positive_balance:
            return "Нормальный режим (стабильные высокие доходы)"
        elif high_oil_gas_ratio and not high_trade:
            return "Санкционное давление (низкий оборот, зависимость от энергоэкспорта)"
        elif not high_oil_gas_ratio and avg_balance < -20:
            return "Ценовой шок (негативное сальдо, снижение зависимости от нефтегаза)"
        elif high_trade and not positive_balance:
            return "Структурная перестройка (высокий оборот, но дефицит)"
        else:
            return "Переходное состояние (смешанные характеристики)"
    
    def print_summary(self, data):
        """
        Вывод сводки результатов
        
        Args:
            data: исходные данные
        """
        if not self.fitted:
            raise ValueError("Модель не обучена! Сначала вызовите метод fit_model.")
        
        print("\\n=== СВОДКА РЕЗУЛЬТАТОВ ===")
        print(f"Количество скрытых состояний: {self.n_hidden_states}")
        column_names = ['Внешнеторговый оборот (млрд $)', 'Сальдо торговли (млрд $)', 'Доля нефтегазовых доходов (%)']
        print(f"Логарифм правдоподобия: {self.model.score(data[column_names].values.astype(float)):.4f}")
        
        print(f"\\nПоследовательность скрытых состояний:")
        for year, state in zip(data['Год'], self.state_sequence):
            print(f"{year}: Состояние {state}")
        
        print(f"\\nМатрица переходов:")
        for i in range(self.n_hidden_states):
            row = self.transition_matrix[i]
            print(f"Из состояния {i}: {row}")

        print(f"\\nВероятности нахождения в каждом состоянии:")
        for i in range(len(self.state_probabilities)):
            print(f"{data.iloc[i]['Год']}: {self.state_probabilities[i]}")
