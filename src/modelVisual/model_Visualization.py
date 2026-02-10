
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

##################### Результаты модели (Табличное представление) #################################################
def print_model_results(r2_test, r2_train, MAE_test, MSE_test, MAPE_test):
    """Вывод результатов модели"""
    
    # Определяем цветовую кодировку для R²
    def get_r2_color(r2):
        if r2 > 0.9:
            return "🟢"  # зеленый
        elif r2 > 0.7:
            return "🟡"  # желтый
        elif r2 > 0.5:
            return "🟠"  # оранжевый
        else:
            return "🔴"  # красный
    
    # Определяем цвет для MAPE
    def get_mape_color(mape):
        if mape < 10:
            return "🟢"
        elif mape < 20:
            return "🟡"
        elif mape < 50:
            return "🟠"
        else:
            return "🔴"
    
    print("\n" + "="*40)
    print("           РЕЗУЛЬТАТЫ ЛИНЕЙНОЙ РЕГРЕССИИ")
    #print("📈"*40)
    
    print(f"\n{'='*60}")
    print(f"{'МЕТРИКА':<25} {'ЗНАЧЕНИЕ':<15} {'ОЦЕНКА':<20}")
    print(f"{'='*60}")
    
    print(f"{'R² (тест)':<25} {r2_test:<15.4f} {get_r2_color(r2_test):<2} {'Высокое' if r2_test > 0.7 else 'Среднее' if r2_test > 0.5 else 'Низкое'}")
    print(f"{'R² (обучение)':<25} {r2_train:<15.4f} {get_r2_color(r2_train):<2} {'Высокое' if r2_train > 0.7 else 'Среднее' if r2_train > 0.5 else 'Низкое'}")
    print(f"{'Разница R²':<25} {(r2_train - r2_test):<15.4f} {'✓' if abs(r2_train - r2_test) < 0.1 else '⚠'} {'Нет переобуч.' if abs(r2_train - r2_test) < 0.1 else 'Возм. переобуч.'}")
    print(f"{'-'*60}")
    print(f"{'MAE (тест)':<25} {MAE_test:<15.4f} {'↓ Лучше'}")
    print(f"{'RMSE (тест)':<25} {MSE_test:<15.4f} {'↓ Лучше'}")
    print(f"{'MAPE (тест)':<25} {MAPE_test:<15.2f}% {get_mape_color(MAPE_test):<2} {'Точно' if MAPE_test < 10 else 'Приемлемо' if MAPE_test < 20 else 'Неточно'}")
    print(f"{'='*60}")



#################### Результаты модели (Графическое представление) #################################################
def analyze_residuals(y_test, predict,r2_test):
    """Анализ и визуализация остатков модели"""
    
    # Преобразуем все в numpy arrays для надежности
    if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
        y_test_array = y_test.values.flatten()
    else:
        y_test_array = np.array(y_test).flatten()
        
    if isinstance(predict, pd.Series) or isinstance(predict, pd.DataFrame):
        predict_array = predict.values.flatten()
    else:
        predict_array = np.array(predict).flatten()
    
    # Вычисляем остатки
    residuals = y_test_array - predict_array
    
    # Создаем графики
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Гистограмма остатков
    axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue', density=True)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    
    # Добавляем нормальное распределение для сравнения
    from scipy import stats
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_pdf = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    axes[0].plot(x, normal_pdf, 'r-', linewidth=2, alpha=0.7, label='Normal dist')
    
    axes[0].set_xlabel('Ошибка (residuals)')
    axes[0].set_ylabel('Плотность вероятности')
    axes[0].set_title(f'Распределение ошибок\nMean: {np.mean(residuals):.6f}, STD: {np.std(residuals):.6f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Q-Q plot для проверки нормальности
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot остатков')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Фактические vs Предсказанные значения
    axes[2].scatter(y_test_array, predict_array, alpha=0.6, s=30, 
                   c=residuals, cmap='coolwarm', edgecolors='black', linewidth=0.5)
    
    # Линия идеального прогноза
    min_val = min(y_test_array.min(), predict_array.min())
    max_val = max(y_test_array.max(), predict_array.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Идеальная линия')
    
    axes[2].set_xlabel('Фактические значения')
    axes[2].set_ylabel('Предсказанные значения')
    axes[2].set_title(f'Фактические vs Предсказанные\nR² = {r2_test:.4f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Добавляем colorbar для остатков
    plt.colorbar(axes[2].collections[0], ax=axes[2], label='Величина ошибки')
    
    plt.tight_layout()
    plt.show()
    
    # Выводим статистику
    print_stats(residuals)
    
    return residuals

def print_stats(residuals):
    """Вывод статистики остатков"""
    print(f"\n📊 СТАТИСТИКА ОШИБОК:")
    print(f"{'='*50}")
    print(f"{'Метрика':<25} {'Значение':<20}")
    print(f"{'='*50}")
    print(f"{'Средняя ошибка':<25} {np.mean(residuals):<20.6f}")
    print(f"{'Медианная ошибка':<25} {np.median(residuals):<20.6f}")
    print(f"{'Стандартное отклонение':<25} {np.std(residuals):<20.6f}")
    print(f"{'Средняя абсолютная ошибка':<25} {np.mean(np.abs(residuals)):<20.6f}")
    print(f"{'Min ошибка':<25} {np.min(residuals):<20.6f}")
    print(f"{'Max ошибка':<25} {np.max(residuals):<20.6f}")
    print(f"{'Диапазон (95%)':<25} [{np.percentile(residuals, 2.5):.6f}, {np.percentile(residuals, 97.5):.6f}]")
    print(f"{'Skewness':<25} {stats.skew(residuals):<20.6f}")
    print(f"{'Kurtosis':<25} {stats.kurtosis(residuals):<20.6f}")
    print(f"{'='*50}")
    
    # Проверка на нормальность (тест Шапиро-Уилка)
    if len(residuals) <= 5000:  # Shapiro-Wilk работает до 5000 наблюдений
        
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(f"{'Shapiro-Wilk test':<25} p-value: {shapiro_p:.6f}")
        if shapiro_p > 0.05:
            print("✅ Ошибки распределены нормально (p > 0.05)")
        else:
            print("⚠ Ошибки не распределены нормально (p ≤ 0.05)")

#################################################################################
def simple_residuals_analysis(y_test, predict, r2_score=None):
    """Простой анализ остатков без зависимостей от scipy"""
    
    # Конвертируем в numpy arrays
    y_test_np = np.array(y_test).flatten()
    predict_np = np.array(predict).flatten()
    residuals_np = y_test_np - predict_np
    
    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Гистограмма остатков
    axes[0, 0].hist(residuals_np, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].axvline(x=np.mean(residuals_np), color='green', linestyle='-', linewidth=2, alpha=0.5, label=f'Mean: {np.mean(residuals_np):.4f}')
    axes[0, 0].set_xlabel('Ошибки (residuals)')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].set_title('Распределение ошибок')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Boxplot остатков
    axes[0, 1].boxplot(residuals_np, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue'))
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_ylabel('Величина ошибки')
    axes[0, 1].set_title('Boxplot ошибок')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Добавляем выбросы
    q1 = np.percentile(residuals_np, 25)
    q3 = np.percentile(residuals_np, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = residuals_np[(residuals_np < lower_bound) | (residuals_np > upper_bound)]
    axes[0, 1].text(1.1, upper_bound, f'Выбросы: {len(outliers)}', 
                   verticalalignment='center')
    
    # 3. Фактические vs Предсказанные
    axes[1, 0].scatter(y_test_np, predict_np, alpha=0.6, s=20)
    # Идеальная линия
    min_val = min(y_test_np.min(), predict_np.min())
    max_val = max(y_test_np.max(), predict_np.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='Идеальная линия')
    axes[1, 0].set_xlabel('Фактические значения')
    axes[1, 0].set_ylabel('Предсказанные значения')
    title_text = 'Фактические vs Предсказанные'
    if r2_score is not None:
        title_text += f' (R² = {r2_score:.4f})'
    axes[1, 0].set_title(title_text)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Ошибки по порядку наблюдений
    axes[1, 1].plot(residuals_np, 'o-', alpha=0.6, markersize=3)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].fill_between(range(len(residuals_np)), 
                           residuals_np, 0, 
                           where=(residuals_np >= 0), 
                           alpha=0.3, color='green', label='Положительные ошибки')
    axes[1, 1].fill_between(range(len(residuals_np)), 
                           residuals_np, 0, 
                           where=(residuals_np < 0), 
                           alpha=0.3, color='red', label='Отрицательные ошибки')
    axes[1, 1].set_xlabel('Номер наблюдения')
    axes[1, 1].set_ylabel('Величина ошибки')
    axes[1, 1].set_title('Ошибки по порядку наблюдений')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод статистики
    print_residuals_stats(residuals_np)
    
    return residuals_np

def print_residuals_stats(residuals):
    """Вывод статистики остатков"""
    print(f"\n📊 СТАТИСТИКА ОШИБОК:")
    print(f"{'='*60}")
    print(f"{'Метрика':<30} {'Значение':<20} {'Интерпретация':<20}")
    print(f"{'='*60}")
    
    # Основные статистики
    mean_err = np.mean(residuals)
    median_err = np.median(residuals)
    std_err = np.std(residuals)
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    
    # Процент ошибок в пределах ±1σ, ±2σ, ±3σ
    within_1sigma = np.mean(np.abs(residuals) <= std_err) * 100
    within_2sigma = np.mean(np.abs(residuals) <= 2*std_err) * 100
    within_3sigma = np.mean(np.abs(residuals) <= 3*std_err) * 100
    
    print(f"{'Средняя ошибка':<30} {mean_err:<20.6f} {'Близко к 0 ✓' if abs(mean_err) < 0.01 else '⚠ Проверить bias'}")
    print(f"{'Медианная ошибка':<30} {median_err:<20.6f} {'Близко к 0 ✓' if abs(median_err) < 0.01 else ''}")
    print(f"{'Стандартное отклонение':<30} {std_err:<20.6f} {'Малое ✓' if std_err < 0.1 else ''}")
    print(f"{'Средняя абсолютная ошибка':<30} {mae:<20.6f} {'Малая ✓' if mae < 0.1 else ''}")
    print(f"{'RMSE':<30} {rmse:<20.6f} {'Малая ✓' if rmse < 0.1 else ''}")
    print(f"{'Min ошибка':<30} {np.min(residuals):<20.6f}")
    print(f"{'Max ошибка':<30} {np.max(residuals):<20.6f}")
    print(f"{'Размах ошибок':<30} {np.ptp(residuals):<20.6f}")
    print(f"{'Процентиль 2.5%':<30} {np.percentile(residuals, 2.5):<20.6f}")
    print(f"{'Процентиль 97.5%':<30} {np.percentile(residuals, 97.5):<20.6f}")
    print(f"{'='*60}")
    
    print(f"\n📈 РАСПРЕДЕЛЕНИЕ ОШИБОК:")
    print(f"{'='*60}")
    print(f"В пределах ±1σ (±{std_err:.4f}): {within_1sigma:.1f}% {'✓ 68% нормального' if abs(within_1sigma - 68.3) < 5 else ''}")
    print(f"В пределах ±2σ (±{2*std_err:.4f}): {within_2sigma:.1f}% {'✓ 95% нормального' if abs(within_2sigma - 95.4) < 5 else ''}")
    print(f"В пределах ±3σ (±{3*std_err:.4f}): {within_3sigma:.1f}% {'✓ 99.7% нормального' if abs(within_3sigma - 99.7) < 5 else ''}")
    
    # Проверка на симметричность
    positive_errors = residuals[residuals > 0]
    negative_errors = residuals[residuals < 0]
    symmetry_ratio = len(positive_errors) / len(negative_errors) if len(negative_errors) > 0 else np.inf
    print(f"\nСимметричность ошибок:")
    print(f"  Положительных ошибок: {len(positive_errors)} ({len(positive_errors)/len(residuals)*100:.1f}%)")
    print(f"  Отрицательных ошибок: {len(negative_errors)} ({len(negative_errors)/len(residuals)*100:.1f}%)")
    print(f"  Соотношение +/-: {symmetry_ratio:.2f} {'✓ Сбалансировано' if 0.8 < symmetry_ratio < 1.2 else '⚠ Несбалансировано'}")
    
    print(f"{'='*60}")

###########################################################################################
def short_residuals_analysis(y_test, predict, r2_test):
    # Быстрая проверка остатков
    y_test_np = np.array(y_test).flatten()
    predict_np = np.array(predict).flatten()
    residuals_np = y_test_np - predict_np

    # Просто 2 графика
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Гистограмма
    ax1.hist(residuals_np, bins=30, alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--')
    ax1.set_xlabel('Ошибки')
    ax1.set_ylabel('Частота')
    ax1.set_title(f'Ошибки модели\nMean: {np.mean(residuals_np):.4f}, Std: {np.std(residuals_np):.4f}')
    ax1.grid(True, alpha=0.3)

    # 2. Scatter plot
    ax2.scatter(y_test_np, predict_np, alpha=0.5, s=10)
    lims = [min(y_test_np.min(), predict_np.min()), max(y_test_np.max(), predict_np.max())]
    ax2.plot(lims, lims, 'r--')
    ax2.set_xlabel('Фактические значения')
    ax2.set_ylabel('Предсказанные значения')
    ax2.set_title(f'Actual vs Predicted (R²={r2_test:.4f})')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

##############################################################################################
def calculate_r2(y_true, y_pred):
    """Вычисление R-squared"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return r2

def calculate_mape(y_true, y_pred):
    """Вычисление MAPE (в %) с проверкой на нули"""
    # Проверяем, что нет нулевых значений
    if np.any(y_true == 0):
        # Альтернатива: заменить нули на очень маленькое число или использовать другой подход
        print("Внимание: есть нулевые значения в y_true!")
        # Исключаем нули из расчета
        mask = y_true != 0
        if np.sum(mask) == 0:
            return np.nan
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def plot_predict(y_test, predict):
    plt.figure(figsize=(13, 6))

    # Данные
    x = np.arange(len(y_test))
    y_actual = np.array(y_test).flatten()
    y_pred = np.array(predict).flatten()

    mse = np.mean((y_actual - y_pred) ** 2)
    rmse = np.sqrt(mse)
    r2 = calculate_r2(y_actual, y_pred)
    MAE = np.mean(np.abs(y_actual - y_pred))
    MAPE = calculate_mape(y_actual, y_pred)

    # Рисуем основную область
    plt.fill_between(x, y_actual, y_pred, alpha=0.1, color='gray', label='Разница')

    # Кривые
    plt.plot(x, y_actual, 'o-', color='#1f77b4', linewidth=1.5, markersize=4, 
            alpha=0.8, label='Факт (y_test)', markevery=10)
    plt.plot(x, y_pred, 's-', color='#ff7f0e', linewidth=1.5, markersize=4, 
            alpha=0.8, label='Прогноз (predict)', markevery=10, linestyle='--')

    # Выделяем точку с максимальной ошибкой
    max_err_idx = np.argmax(np.abs(y_actual - y_pred))
    plt.plot(max_err_idx, y_actual[max_err_idx], 'ro', markersize=10, alpha=0.7, 
            label=f'Макс. ошибка (idx={max_err_idx})')
    plt.plot(max_err_idx, y_pred[max_err_idx], 'ro', markersize=10, alpha=0.7)

    # Настройки
    plt.xlabel('Индекс наблюдения', fontsize=11)
    plt.ylabel('Значение', fontsize=11)
    plt.title(f'Фактические vs Предсказанные значения\n'
            f'R² = {r2:.4f} | MAE = {MAE:.4f} | RMSE = {rmse:.4f} | MAPE = {MAPE:.1f}%', 
            fontsize=13, fontweight='bold')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10)
    plt.grid(True, alpha=0.3, linestyle=':')

    # Добавляем подпись внизу
    plt.figtext(0.5, 0.01, f'Всего наблюдений: {len(y_test)} | '
                f'Средняя ошибка: {np.mean(y_actual - y_pred):.4f} | '
                f'Std ошибки: {np.std(y_actual - y_pred):.4f}', 
                ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def plot_predict_detal(y_test, predict):
    # Подготовка данных
    x_indices = np.arange(len(y_test))
    y_test_np = np.array(y_test).flatten()
    predict_np = np.array(predict).flatten()

    # Создаем фигуру с двумя областями
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Полный график
    ax1.plot(x_indices, y_test_np, 'b-', linewidth=1.5, alpha=0.7, label='y_test')
    ax1.plot(x_indices, predict_np, 'r-', linewidth=1.5, alpha=0.7, label='predict')
    ax1.set_xlabel('Номер отсчета', fontsize=11)
    ax1.set_ylabel('Значение', fontsize=11)
    ax1.set_title('Полное сравнение фактических и предсказанных значений', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Zoom на часть графика (первые 50 точек)
    zoom_size = min(50, len(y_test_np))
    ax2.plot(x_indices[:zoom_size], y_test_np[:zoom_size], 'b-', linewidth=2, alpha=0.8, 
            marker='o', markersize=4, label='y_test')
    ax2.plot(x_indices[:zoom_size], predict_np[:zoom_size], 'r--', linewidth=2, alpha=0.8, 
            marker='s', markersize=4, label='predict')

    # Показываем ошибки стрелками
    for i in range(zoom_size):
        if i % 5 == 0:  # Каждую 5-ю точку для читаемости
            ax2.annotate('', xy=(i, predict_np[i]), xytext=(i, y_test_np[i]),
                        arrowprops=dict(arrowstyle='<->', color='gray', alpha=0.5, lw=1))

    ax2.set_xlabel(f'Номер отсчета (первые {zoom_size} точек)', fontsize=11)
    ax2.set_ylabel('Значение', fontsize=11)
    ax2.set_title(f'Детальный вид (первые {zoom_size} точек)', fontsize=13)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()    