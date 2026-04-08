"""
Демонстрационный скрипт для HMM анализа экономических данных
Пример использования класса EconomicHMMAnalyzer
"""

import sys
import os
# Добавляем текущую директорию в путь Python для импорта модуля
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from economic_hmm_analysis import EconomicHMMAnalyzer
import matplotlib.pyplot as plt


def main():
    """
    Основная функция демонстрации
    """
    print("=== ДЕМОНСТРАЦИЯ HMM АНАЛИЗА ЭКОНОМИЧЕСКИХ ДАННЫХ ===\n")
    
    # Создаем экземпляр анализатора
    analyzer = EconomicHMMAnalyzer()
    
    # Загружаем пример данных
    print("1. Загрузка данных...")
    data = analyzer.load_sample_data()
    print(data)
    print()
    
    # Предобрабатываем данные
    print("2. Предобработка данных...")
    X = analyzer.preprocess_data(data)
    print()
    
    # Подбор оптимального числа скрытых состояний
    print("3. Подбор оптимального числа скрытых состояний...")
    optimal_states = analyzer.find_optimal_states(X, max_states=5)
    print(f"Используем {optimal_states} скрытых состояний")
    print()
    
    # Обучение модели с оптимальным числом состояний
    print("4. Обучение HMM модели...")
    analyzer.fit_model(X)
    print()
    
    # Декодирование последовательности скрытых состояний
    print("5. Декодирование последовательности скрытых состояний...")
    state_sequence = analyzer.decode_states(X)
    print(f"Последовательность состояний: {state_sequence}")
    print()
    
    # Визуализация результатов
    print("6. Визуализация результатов...")
    analyzer.visualize_results(data)
    print()
    
    # Интерпретация скрытых состояний
    print("7. Интерпретация скрытых состояний...")
    interpretations = analyzer.interpret_states(data)
    print()
    
    # Вывод сводки
    print("8. Сводка результатов...")
    analyzer.print_summary(data)
    print()
    
    print("=== АНАЛИЗ ЗАВЕРШЕН ===")
    print("\nОбъяснение результатов:")
    print("- Скрытые состояния представляют различные экономические режимы:")
    print("  * Состояние 0: Нормальный режим (стабильные высокие доходы)")
    print("  * Состояние 1: Санкционное давление (низкий оборот, зависимость от энергоэкспорта)")
    print("  * Состояние 2: Ценовой шок (негативное сальдо, снижение зависимости от нефтегаза)")
    print("  * Состояние 3: Структурная перестройка (высокий оборот, но дефицит)")
    print("\nМатрица переходов показывает вероятности перехода между экономическими режимами.")
    print("Цветовая разметка на графиках показывает, в каком состоянии находилась экономика в каждый год.")


if __name__ == "__main__":
    main()