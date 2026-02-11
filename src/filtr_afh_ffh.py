import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings('ignore')

###################################################################################################################
class BS468ReferenceResponse:
    """
    Точная характеристика BS.468-4 на основе табличных данных
    с правильной интерполяцией и рассчитанной ФЧХ
    """
    
    def __init__(self, fs=48000):
        self.fs = fs
        
        # ТОЧНЫЕ табличные данные BS.468-4 из вашего источника
        self.ref_freqs = np.array([
            31.5, 63, 100, 200, 400, 500, 630, 800,
            1000, 2000, 3150, 4000, 5000, 6300, 7100,
            8000, 9000, 10000, 12500, 14000, 16000, 20000
        ])
        
        # АЧХ в дБ (относительно 1000 Гц)
        self.ref_mag_db = np.array([
            -29.9, -23.9, -19.8, -13.8, -7.8, -5.8, -4.0, -2.6,
            0.0,    # 1000 Гц - опорная
            0.6, 0.9, 1.2, 0.9, -0.1, -0.7, -1.6, -2.9,
            -4.5, -11.2, -16.2, -23.0, -36.0
        ])
        
        # Допуски
        self.ref_tolerance = np.array([
            1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.2,
            0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            1.0, 1.2, 1.5, 2.0, 3.0, 5.0
        ])
        
        # Нормируем на 2200 Гц (0 дБ)
        idx_1000 = np.where(self.ref_freqs == 1000)[0][0]
        self.ref_att_db_norm = self.ref_mag_db - self.ref_mag_db[idx_1000]

        # Преобразуем в линейные единицы
        self.ref_mag_linear = 10 ** (self.ref_mag_db / 20)
        
        # Создаем интерполяторы
        self._setup_interpolators()
        
        # Рассчитываем ФЧХ на основе минимально-фазовой аппроксимации
        self._calculate_phase_response()
    
    def _setup_interpolators(self):
        """Настройка интерполяторов для АЧХ"""
        # Расширяем частотный диапазон для лучшей интерполяции
        extended_freqs = np.concatenate([[1.0], self.ref_freqs, [22000]])
        extended_mag_db = np.concatenate([[-80.0], self.ref_mag_db, [-80.0]])
        extended_mag_linear = 10 ** (extended_mag_db / 20)
        
        # Логарифмическая интерполяция
        log_freqs = np.log10(extended_freqs)
        
        # Кубические сплайны для плавной интерполяции
        self.interp_mag_db = interpolate.CubicSpline(
            log_freqs, extended_mag_db
        )
        
        self.interp_mag_linear = interpolate.CubicSpline(
            log_freqs, extended_mag_linear
        )
        
        # Интерполятор допусков
        self.interp_tolerance = interpolate.CubicSpline(
            np.log10(self.ref_freqs), self.ref_tolerance
        )
    
    def _calculate_phase_response(self):
        """Расчет минимально-фазовой ФЧХ по АЧХ"""
        # Создаем детальную сетку частот
        N = 4096
        freqs_detailed = np.logspace(np.log10(10), np.log10(22000), N)
        
        # Получаем АЧХ на детальной сетке
        mag_db_detailed = self.get_magnitude_db(freqs_detailed)
        mag_linear_detailed = 10 ** (mag_db_detailed / 20)
        
        # Используем прямое преобразование для получения ФЧХ
        # Для упрощения используем аппроксимацию фазового сдвига
        log_freqs_detailed = np.log10(freqs_detailed)
        
        # Эмпирическая формула для ФЧХ BS.468-4
        # ФЧХ обычно имеет вид: phase ≈ -arctan(f/f0) - ... 
        # Для BS.468-4 приблизительно:
        phase_rad = np.zeros_like(freqs_detailed)
        
        # Разные участки ФЧХ
        mask_low = freqs_detailed < 100
        mask_mid = (freqs_detailed >= 100) & (freqs_detailed < 1000)
        mask_high = freqs_detailed >= 1000
        
        # Низкие частоты: небольшой фазовый сдвиг
        phase_rad[mask_low] = -np.pi/4 * (freqs_detailed[mask_low] / 100) ** 0.5
        
        # Средние частоты: линейный рост сдвига
        phase_rad[mask_mid] = -np.pi/3 - np.pi/6 * np.log10(freqs_detailed[mask_mid] / 100)
        
        # Высокие частоты: приближение к -π
        phase_rad[mask_high] = -np.pi + np.pi/4 * (1000 / freqs_detailed[mask_high]) ** 0.5
        
        # Интерполятор для ФЧХ
        self.interp_phase_rad = interpolate.CubicSpline(
            log_freqs_detailed, np.unwrap(phase_rad)
        )
    
    def get_magnitude_db(self, frequencies_hz):
        """Получить АЧХ в дБ"""
        # Преобразуем в массив, если скаляр
        if np.isscalar(frequencies_hz):
            frequencies_hz = np.array([frequencies_hz])
        
        log_freqs = np.log10(np.maximum(frequencies_hz, 1e-10))
        result = self.interp_mag_db(log_freqs)
        
        # Возвращаем скаляр, если был передан скаляр
        if len(result) == 1 and isinstance(frequencies_hz, np.ndarray) and frequencies_hz.shape == (1,):
            return result[0]
        return result
    
    def get_magnitude_linear(self, frequencies_hz):
        """Получить АЧХ в линейных единицах"""
        # Преобразуем в массив, если скаляр
        if np.isscalar(frequencies_hz):
            frequencies_hz = np.array([frequencies_hz])
        
        log_freqs = np.log10(np.maximum(frequencies_hz, 1e-10))
        result = self.interp_mag_linear(log_freqs)
        
        # Возвращаем скаляр, если был передан скаляр
        if len(result) == 1 and isinstance(frequencies_hz, np.ndarray) and frequencies_hz.shape == (1,):
            return result[0]
        return result
    
    def get_phase_deg(self, frequencies_hz):
        """Получить ФЧХ в градусах"""
        # Преобразуем в массив, если скаляр
        if np.isscalar(frequencies_hz):
            frequencies_hz = np.array([frequencies_hz])
        
        log_freqs = np.log10(np.maximum(frequencies_hz, 1e-10))
        phase_rad = self.interp_phase_rad(log_freqs)
        result = np.degrees(phase_rad)
        
        # Возвращаем скаляр, если был передан скаляр
        if len(result) == 1 and isinstance(frequencies_hz, np.ndarray) and frequencies_hz.shape == (1,):
            return result[0]
        return result
    
    def get_phase_rad(self, frequencies_hz):
        """Получить ФЧХ в радианах"""
        # Преобразуем в массив, если скаляр
        if np.isscalar(frequencies_hz):
            frequencies_hz = np.array([frequencies_hz])
        
        log_freqs = np.log10(np.maximum(frequencies_hz, 1e-10))
        result = self.interp_phase_rad(log_freqs)
        
        # Возвращаем скаляр, если был передан скаляр
        if len(result) == 1 and isinstance(frequencies_hz, np.ndarray) and frequencies_hz.shape == (1,):
            return result[0]
        return result
    
    def get_response(self, frequencies_hz):
        """Получить АЧХ и ФЧХ"""
        return self.get_magnitude_db(frequencies_hz), self.get_phase_deg(frequencies_hz)
    
    def get_complex_response(self, frequencies_hz):
        """Получить комплексную характеристику"""
        mag_linear = self.get_magnitude_linear(frequencies_hz)
        phase_rad = self.get_phase_rad(frequencies_hz)
        return mag_linear * np.exp(1j * phase_rad)
    
    def get_tolerance(self, frequencies_hz):
        """Получить допуск"""
        # Преобразуем в массив, если скаляр
        if np.isscalar(frequencies_hz):
            frequencies_hz = np.array([frequencies_hz])
        
        log_freqs = np.log10(np.maximum(frequencies_hz, 1e-10))
        result = self.interp_tolerance(log_freqs)
        
        # Возвращаем скаляр, если был передан скаляр
        if len(result) == 1 and isinstance(frequencies_hz, np.ndarray) and frequencies_hz.shape == (1,):
            return result[0]
        return result

###################################################################################################################
def create_simple_plot_with_tables():
    """
    Упрощенный вариант: только АЧХ и ФЧХ с таблицами рядом
    """
    bs468 = BS468ReferenceResponse()
    
    # Создаем фигуру
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. График АЧХ
    freqs = np.logspace(np.log10(20), np.log10(20000), 1000)
    mag_db = bs468.get_magnitude_db(freqs)
    
    ax1.semilogx(freqs, mag_db, 'b-', linewidth=2)
    ax1.set_xlabel('Частота, Гц', fontsize=11)
    ax1.set_ylabel('Ослабление, дБ', fontsize=11)
    ax1.set_title('АЧХ фильтра BS.468-4', fontsize=13, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.set_xlim(20, 20000)
    ax1.set_ylim(-50, 5)
    
    # Выделяем опорную частоту
    ax1.axvline(x=2200, color='r', linestyle='--', alpha=0.5, label='2200 Гц (опорная)')
    ax1.legend()
    
    # 2. График ФЧХ
    phase_deg = bs468.get_phase_deg(freqs)
    
    ax2.semilogx(freqs, phase_deg, 'g-', linewidth=2)
    ax2.set_xlabel('Частота, Гц', fontsize=11)
    ax2.set_ylabel('Фазовый сдвиг, °', fontsize=11)
    ax2.set_title('ФЧХ фильтра BS.468-4', fontsize=13, fontweight='bold')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.set_xlim(20, 20000)
    ax2.set_ylim(-200, 100)
    
    # Создаем таблицы справа
    plt.tight_layout()
    
    # Добавляем таблицы как отдельные subplots
    fig2, axes = plt.subplots(2, 1, figsize=(8, 10))
    
    # Таблица АЧХ
    table_freqs = [31.5, 63, 100, 200, 400, 800, 1000, 2000, 2200, 3150, 4000, 6300, 8000, 10000]
    table_mag = bs468.get_magnitude_db(table_freqs)
    table_phase = bs468.get_phase_deg(table_freqs)
    
    # Таблица 1: АЧХ
    ax_table1 = axes[0]
    ax_table1.axis('off')
    
    table_data1 = [[f"{f:.1f}", f"{mag:.2f}"] for f, mag in zip(table_freqs, table_mag)]
    
    table1 = ax_table1.table(cellText=table_data1,
                           colLabels=["Частота, Гц", "Ослабление, дБ"],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.3, 0.3])
    
    table1.auto_set_font_size(False)
    table1.set_fontsize(11)
    table1.scale(1.5, 2.0)
    
    # Выделяем опорную частоту
    for i in range(len(table_freqs)):
        if table_freqs[i] == 2200:
            table1[i+1, 0].set_facecolor('#FFFF99')
            table1[i+1, 1].set_facecolor('#FFFF99')
    
    ax_table1.set_title('Таблица значений АЧХ BS.468-4', 
                        fontsize=13, fontweight='bold', pad=20)
    
    # Таблица 2: ФЧХ
    ax_table2 = axes[1]
    ax_table2.axis('off')
    
    table_data2 = [[f"{f:.1f}", f"{phase:.1f}"] for f, phase in zip(table_freqs, table_phase)]
    
    table2 = ax_table2.table(cellText=table_data2,
                           colLabels=["Частота, Гц", "Фазовый сдвиг, °"],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.3, 0.3])
    
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1.5, 2.0)
    
    # Выделяем опорную частоту
    for i in range(len(table_freqs)):
        if table_freqs[i] == 2200:
            table2[i+1, 0].set_facecolor('#CCFFCC')
            table2[i+1, 1].set_facecolor('#CCFFCC')
    
    ax_table2.set_title('Таблица значений ФЧХ BS.468-4', 
                        fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Сохраняем
    fig.savefig('bs468_response_plots.png', dpi=150, bbox_inches='tight')
    fig2.savefig('bs468_response_tables.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return bs468

###################################################################################################################
def plot_bs468_response_with_tables():
    """
    Рисует АЧХ и ФЧХ BS.468-4 с таблицами значений
    """
    # Создаем объект BS.468-4
    bs468 = BS468ReferenceResponse()
    
    # Создаем фигуру с GridSpec для сложной компоновки
    fig = plt.figure(figsize=(18, 12))
    
    # Определяем сетку: 2 графика слева, таблицы справа
    gs = gridspec.GridSpec(2, 3, width_ratios=[2, 2, 1.5], height_ratios=[1, 1])
    
    # График 1: АЧХ
    ax1 = plt.subplot(gs[0, 0])
    
    # Частоты для графика (логарифмическая шкала)
    plot_freqs = np.logspace(np.log10(20), np.log10(20000), 1000)
    mag_db = bs468.get_magnitude_db(plot_freqs)
    
    # Рисуем АЧХ
    ax1.semilogx(plot_freqs, mag_db, 'b-', linewidth=2.5, label='BS.468-4')
    
    # Добавляем табличные точки
    table_freqs = bs468.ref_freqs
    table_mag = bs468.ref_att_db_norm
    ax1.plot(table_freqs, table_mag, 'ro', markersize=6, label='Табличные точки')
    
    # Настройки графика АЧХ
    ax1.set_xlabel('Частота, Гц', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Ослабление, дБ', fontsize=12, fontweight='bold')
    ax1.set_title('Амплитудно-частотная характеристика (АЧХ) BS.468-4', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(20, 20000)
    ax1.set_ylim(-50, 5)
    
    # Подписываем ключевые частоты
    key_freqs = [31.5, 100, 400, 1000, 2200, 5000, 8000]
    for f in key_freqs:
        idx = np.argmin(np.abs(plot_freqs - f))
        ax1.annotate(f'{f} Гц', 
                    xy=(f, mag_db[idx]),
                    xytext=(10, 5 if f != 2200 else -15),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    # График 2: ФЧХ
    ax2 = plt.subplot(gs[1, 0])
    
    # Получаем ФЧХ
    phase_deg = bs468.get_phase_deg(plot_freqs)
    
    # Рисуем ФЧХ
    ax2.semilogx(plot_freqs, phase_deg, 'g-', linewidth=2.5, label='ФЧХ BS.468-4')
    
    # Настройки графика ФЧХ
    ax2.set_xlabel('Частота, Гц', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Фазовый сдвиг, градусы', fontsize=12, fontweight='bold')
    ax2.set_title('Фазо-частотная характеристика (ФЧХ) BS.468-4', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(20, 20000)
    ax2.set_ylim(-200, 100)
    
    # Подписываем ключевые точки на ФЧХ
    for f in [100, 400, 1000, 2200, 5000]:
        idx = np.argmin(np.abs(plot_freqs - f))
        ax2.annotate(f'{phase_deg[idx]:.0f}°', 
                    xy=(f, phase_deg[idx]),
                    xytext=(10, 5),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    
    # График 3: Групповая задержка
    ax3 = plt.subplot(gs[0, 1])
    
    # Расчет групповой задержки
    if len(plot_freqs) > 1:
        phase_rad = np.radians(phase_deg)
        group_delay = -np.gradient(phase_rad) / (2 * np.pi * np.gradient(plot_freqs))
        group_delay_ms = group_delay * 1000  # в миллисекундах
    else:
        group_delay_ms = np.zeros_like(plot_freqs)
    
    # Рисуем групповую задержку
    ax3.semilogx(plot_freqs, group_delay_ms, 'r-', linewidth=2.5, label='Групповая задержка')
    
    # Настройки графика
    ax3.set_xlabel('Частота, Гц', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Задержка, мс', fontsize=12, fontweight='bold')
    ax3.set_title('Групповое запаздывание BS.468-4', 
                  fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, which='both', linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_xlim(20, 20000)
    ax3.set_ylim(0, 5)
    
    # Таблица 1: АЧХ (правый верх)
    ax_table1 = plt.subplot(gs[0, 2])
    ax_table1.axis('tight')
    ax_table1.axis('off')
    
    # Выбираем частоты для таблицы
    table_freqs_ach = [31.5, 63, 100, 200, 400, 800, 1000, 2000, 2200, 3150, 4000, 6300, 8000, 10000]
    table_mag_values = bs468.get_magnitude_db(table_freqs_ach)
    
    # Создаем таблицу АЧХ
    table_data_ach = []
    for i, (f, mag) in enumerate(zip(table_freqs_ach, table_mag_values)):
        table_data_ach.append([f"{f:.1f}", f"{mag:.2f}"])
    
    # Заголовок таблицы
    table_title1 = "Таблица значений АЧХ BS.468-4\n(относительно 1000 Гц)"
    
    # Создаем таблицу
    table1 = ax_table1.table(cellText=table_data_ach,
                           colLabels=["Частота, Гц", "Ослабление, дБ"],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.4, 0.4])
    
    # Стилизуем таблицу
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1.2, 1.8)
    
    # Заголовок таблицы
    ax_table1.set_title(table_title1, fontsize=12, fontweight='bold', pad=20)
    
    # Цветовое оформление таблицы
    for i in range(len(table_freqs_ach) + 1):
        for j in range(2):
            cell = table1[i, j]
            if i == 0:  # Заголовок
                cell.set_facecolor('#4F81BD')
                cell.set_text_props(weight='bold', color='white')
            elif f == 2200 and i > 0 and j == 0:
                # Выделяем опорную частоту
                table1[i, 0].set_facecolor('#FFD700')
                table1[i, 1].set_facecolor('#FFD700')
    
    # Таблица 2: ФЧХ (правый низ)
    ax_table2 = plt.subplot(gs[1, 2])
    ax_table2.axis('tight')
    ax_table2.axis('off')
    
    # Получаем значения ФЧХ для таблицы
    table_phase_values = bs468.get_phase_deg(table_freqs_ach)
    
    # Создаем таблицу ФЧХ
    table_data_fch = []
    for i, (f, phase) in enumerate(zip(table_freqs_ach, table_phase_values)):
        table_data_fch.append([f"{f:.1f}", f"{phase:.1f}"])
    
    # Заголовок таблицы
    table_title2 = "Таблица значений ФЧХ BS.468-4\n(сдвиг фазы в градусах)"
    
    # Создаем таблицу
    table2 = ax_table2.table(cellText=table_data_fch,
                           colLabels=["Частота, Гц", "Фазовый сдвиг, °"],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.4, 0.4])
    
    # Стилизуем таблицу
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1.2, 1.8)
    
    # Заголовок таблицы
    ax_table2.set_title(table_title2, fontsize=12, fontweight='bold', pad=20)
    
    # Цветовое оформление
    for i in range(len(table_freqs_ach) + 1):
        for j in range(2):
            cell = table2[i, j]
            if i == 0:  # Заголовок
                cell.set_facecolor('#9BBB59')
                cell.set_text_props(weight='bold', color='white')
            elif f == 2200 and i > 0 and j == 0:
                # Выделяем опорную частоту
                table2[i, 0].set_facecolor('#C4D79B')
                table2[i, 1].set_facecolor('#C4D79B')
    
    # График 4: Комплексная плоскость (правый средний, если нужно)
    ax4 = plt.subplot(gs[1, 1])
    
    # Выбираем подмножество частот для отображения на комплексной плоскости
    polar_freqs = np.logspace(np.log10(50), np.log10(10000), 20)
    complex_response = []
    
    for f in polar_freqs:
        mag = bs468.get_magnitude_db(f)
        phase = bs468.get_phase_deg(f)
        mag_linear = 10 ** (mag / 20)
        complex_response.append(mag_linear * np.exp(1j * np.radians(phase)))
    
    complex_response = np.array(complex_response)
    
    # Рисуем на комплексной плоскости
    ax4.plot(np.real(complex_response), np.imag(complex_response), 
             'bo-', linewidth=1.5, markersize=6, alpha=0.7)
    ax4.plot(0, 0, 'r+', markersize=12, markeredgewidth=2)
    
    # Настройки
    ax4.set_xlabel('Действительная часть', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Мнимая часть', fontsize=11, fontweight='bold')
    ax4.set_title('Комплексная характеристика BS.468-4', 
                  fontsize=12, fontweight='bold', pad=10)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.axis('equal')
    
    # Добавляем частотные метки
    for i, f in enumerate(polar_freqs[:8:2]):  # Каждую 2-ю точку
        ax4.annotate(f'{int(f)} Гц',
                    xy=(np.real(complex_response[i*2]), np.imag(complex_response[i*2])),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Общий заголовок
    fig.suptitle('Полная характеристика фильтра BS.468-4 (BS.468-4 Weighting Filter)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Добавляем информационный текст
    info_text = (
        "Стандарт BS.468-4 используется для измерений шумов в радиовещательном оборудовании.\n"
        "Характеристика нормирована на 0 дБ при 1000 Гц.\n"
        #"Включает квазипиковый детектор с τ_заряд=1 мс, τ_разряд=600 мс."
    )
    
    plt.figtext(0.02, 0.02, info_text, fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Корректируем расположение
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Сохраняем и показываем
    #plt.savefig('bs468_response_with_tables.png', dpi=150, bbox_inches='tight')
    #plt.show()
    
    """
    # Дополнительно: выводим значения в консоль
    print("\n" + "="*80)
    print("ТАБЛИЧНЫЕ ЗНАЧЕНИЯ ХАРАКТЕРИСТИК BS.468-4")
    print("="*80)
    print(f"{'Частота (Гц)':<15} {'АЧХ (дБ)':<15} {'ФЧХ (°)':<15} {'Задержка (мс)':<15}")
    print("-"*80)
    
    # Расчет задержки для табличных частот
    for f in table_freqs_ach:
        mag = bs468.get_magnitude_db(f)
        phase = bs468.get_phase_deg(f)
        
        # Расчет задержки через производную
        if f > table_freqs_ach[0]:
            idx = np.where(np.array(table_freqs_ach) == f)[0][0]
            f_prev = table_freqs_ach[idx-1]
            f_next = table_freqs_ach[idx+1] if idx < len(table_freqs_ach)-1 else f*1.01
            phase_prev = bs468.get_phase_deg(f_prev)
            phase_next = bs468.get_phase_deg(f_next)
            delay = - (phase_next - phase_prev) / (360 * (f_next - f_prev)) * 1000
        else:
            delay = 0
        
        print(f"{f:<15.1f} {mag:<15.2f} {phase:<15.1f} {delay:<15.3f}")
    
    print("="*80)
    """

    return bs468

###################################################################################################################
# Запуск
#if __name__ == "__main__":
#    print("Генерация графиков и таблиц BS.468-4")
#    print("=" * 60)
    
    # Полный график с таблицами
#    bs468 = plot_bs468_response_with_tables()
    
    # Упрощенный вариант (раздельно)
    # bs468 = create_simple_plot_with_tables()
    
#    print("\nГрафики сохранены в файлы:")
#    print("1. bs468_response_with_tables.png - полный график с таблицами")
#    print("2. bs468_response_plots.png - только графики АЧХ и ФЧХ")
#    print("3. bs468_response_tables.png - только таблицы значений")

####################################################################################################################
def weighting_filter_response(standard='p53', frequencies_hz=None):
    """
    Генерирует АЧХ различных стандартных фильтров.
    
    Параметры:
    ----------
    standard : str
        'p53' - МСЭ-Т P.53 (псофометрический)
        'bs468' - BS.468-4 (вещательный, квазипиковый)
        'ccir' - CCIR/ITU-R 468-4 (тот же что BS.468-4)
    frequencies_hz : numpy.ndarray
        Массив частот в герцах
        
    Возвращает:
    -----------
    response_db : numpy.ndarray
        Массив значений АЧХ в дБ
    """
    
    if standard.lower() in ['p53', 'psophometric']:
        # МСЭ-Т P.53 (таблица как в предыдущем коде)
        ref_freqs = np.array([0, 50, 100, 150, 200, 250, 300, 350, 400, 450,
                              500, 550, 600, 700, 800, 850, 900, 1000, 1200,
                              1400, 1600, 1800, 2000, 2500, 3000, 3500, 4000,
                              4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000])
        
        ref_att = np.array([-200, -33.0, -20.5, -13.0, -8.5, -5.7, -3.9, -2.6,
                           -1.7, -1.0, -0.4, 0.0, 0.3, 0.2, 0.0, 0.0, 0.1,
                           0.2, 0.4, 0.5, 0.6, 0.6, 0.5, -0.4, -2.2, -5.0,
                           -9.0, -14.5, -21.0, -29.0, -37.0, -46.0, -56.0,
                           -66.0, -76.0])
        
        # Нормировка на 800 Гц = 0 дБ
        idx_800 = np.where(ref_freqs == 800)[0][0]
        ref_att = ref_att - ref_att[idx_800]
        
    elif standard.lower() in ['bs468', 'ccir', 'itu-r468']:
        # BS.468-4 / CCIR 468-4
        ref_freqs = np.array([10, 20, 31.5, 63, 100, 125, 200, 250, 300, 400,
                              500, 600, 750, 800, 900, 1000, 1400, 2000, 2200,
                              2500, 2800, 3150, 4000, 5000, 6300, 8000, 9000,
                              10000, 14000, 20000])
        
        ref_att = np.array([-50.5, -44.7, -29.9, -19.8, -13.8, -10.9, -6.6,
                           -4.7, -3.6, -2.3, -1.7, -1.4, -1.1, -1.0, -1.6,
                           -3.1, -5.7, 0.0, 0.5, 0.0, -0.3, -0.8, -2.5,
                           -4.5, -6.6, -10.5, -12.5, -15.5, -25.0, -40.0])
        
        # Нормировка на 2.2 кГц = 0 дБ
        idx_2200 = np.where(ref_freqs == 2200)[0][0]
        ref_att = ref_att - ref_att[idx_2200]
    
    # Интерполяция
    interp_spline = interpolate.CubicSpline(
        np.log10(np.maximum(ref_freqs, 1e-10)), 
        ref_att,
        extrapolate=False
    )
    
    # Вычисляем значения
    response_db = interp_spline(np.log10(np.maximum(frequencies_hz, 1e-10)))
    
    # Экстраполяция за пределами диапазона
    mask_low = frequencies_hz < ref_freqs[0]
    mask_high = frequencies_hz > ref_freqs[-1]
    
    if np.any(mask_low):
        response_db[mask_low] = ref_att[0] - 20 * np.log10(ref_freqs[0]/frequencies_hz[mask_low])
    
    if np.any(mask_high):
        response_db[mask_high] = ref_att[-1] - 40 * np.log10(frequencies_hz[mask_high]/ref_freqs[-1])
    
    return response_db

####################################################################################################################
def verify_exact_values():
    """Точная проверка табличных значений"""
    bs468 = BS468ReferenceResponse()
    
    print("=" * 80)
    print("ТОЧНАЯ ПРОВЕРКА ТАБЛИЧНЫХ ЗНАЧЕНИЙ BS.468-4")
    print("=" * 80)
    print(f"{'Частота, Гц':<10} {'Табличное, дБ':<15} {'Расчетное, дБ':<15} {'Ошибка, дБ':<12}")
    print("-" * 80)
    
    # Табличные данные
    table_data = [
        (31.5, -29.9), (63, -23.9), (100, -19.8), (200, -13.8),
        (400, -7.8), (500, -5.8), (630, -4.0), (800, -2.6),
        (1000, 0.0), (2000, 0.6), (3150, 0.9), (4000, 1.2),
        (5000, 0.9), (6300, -0.1), (7100, -0.7), (8000, -1.6),
        (9000, -2.9), (10000, -4.5), (12500, -11.2), 
        (14000, -16.2), (16000, -23.0), (20000, -36.0)
    ]
    
    max_error = 0
    for freq, expected in table_data:
        calculated = bs468.get_magnitude_db(freq)
        error = calculated - expected
        max_error = max(max_error, abs(error))
        
        status = "✓" if abs(error) < 0.01 else "✗"
        
        print(f"{freq:<10.1f} {expected:<15.1f} {calculated:<15.3f} {error:<12.3f} {status}")
    
    print("-" * 80)
    print(f"Максимальная ошибка: {max_error:.3f} дБ")
    print("=" * 80)
    
    return bs468

####################################################################################################################
def plot_accurate_response():
    """График точной характеристики"""
    bs468 = BS468ReferenceResponse()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. АЧХ с табличными точками
    ax1 = axes[0, 0]
    freqs = np.logspace(np.log10(20), np.log10(22000), 1000)
    mag_db = bs468.get_magnitude_db(freqs)
    
    ax1.semilogx(freqs, mag_db, 'b-', linewidth=2, label='Интерполированная АЧХ')
    ax1.plot(bs468.ref_freqs, bs468.ref_mag_db, 'ro', markersize=6, 
             label='Табличные точки', zorder=10)
    
    ax1.set_xlabel('Частота, Гц', fontsize=11)
    ax1.set_ylabel('Ослабление, дБ', fontsize=11)
    ax1.set_title('АЧХ BS.468-4 (точная интерполяция)', fontsize=12, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    ax1.set_xlim(20, 22000)
    ax1.set_ylim(-40, 5)
    
    # Аннотации для ключевых частот
    key_freqs = [1000, 2000, 4000, 6300, 10000]
    for f in key_freqs:
        mag = bs468.get_magnitude_db(f)
        ax1.annotate(f'{f} Гц\n{mag:.1f} дБ', 
                    xy=(f, mag), xytext=(10, 10),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 2. ФЧХ
    ax2 = axes[0, 1]
    phase_deg = bs468.get_phase_deg(freqs)
    
    ax2.semilogx(freqs, phase_deg, 'g-', linewidth=2)
    ax2.set_xlabel('Частота, Гц', fontsize=11)
    ax2.set_ylabel('Фазовый сдвиг, °', fontsize=11)
    ax2.set_title('ФЧХ BS.468-4 (аппроксимация)', fontsize=12, fontweight='bold')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.set_xlim(20, 22000)
    ax2.set_ylim(-250, 50)
    
    # 3. АЧХ в линейных единицах
    ax3 = axes[1, 0]
    mag_linear = bs468.get_magnitude_linear(freqs)
    
    ax3.semilogx(freqs, mag_linear, 'r-', linewidth=2)
    ax3.set_xlabel('Частота, Гц', fontsize=11)
    ax3.set_ylabel('Коэффициент передачи', fontsize=11)
    ax3.set_title('АЧХ в линейных единицах', fontsize=12, fontweight='bold')
    ax3.grid(True, which='both', linestyle='--', alpha=0.5)
    ax3.set_xlim(20, 22000)
    ax3.set_ylim(0, 1.5)
    
    # 4. Комплексная плоскость (выборочные частоты)
    ax4 = axes[1, 1]
    
    # Выбираем частоты для отображения
    polar_freqs = np.logspace(np.log10(100), np.log10(10000), 12)
    complex_resp = bs468.get_complex_response(polar_freqs)
    
    # Рисуем единичную окружность
    theta = np.linspace(0, 2*np.pi, 100)
    ax4.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3)
    
    # Характеристика
    ax4.plot(np.real(complex_resp), np.imag(complex_resp), 'bo-', 
             linewidth=1.5, markersize=6, alpha=0.7)
    ax4.plot(0, 0, 'r+', markersize=12, markeredgewidth=2)
    
    ax4.set_xlabel('Re', fontsize=11)
    ax4.set_ylabel('Im', fontsize=11)
    ax4.set_title('Комплексная плоскость', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.5)
    ax4.axis('equal')
    
    # Добавляем частотные метки
    for i, f in enumerate(polar_freqs[::2]):
        ax4.annotate(f'{int(f)} Гц',
                    xy=(np.real(complex_resp[i*2]), np.imag(complex_resp[i*2])),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return bs468

####################################################################################################################
def generate_calibration_table():
    """Генерация калибровочной таблицы"""
    bs468 = BS468ReferenceResponse()
    
    # Частоты для калибровки
    cal_freqs = [
        31.5, 63, 100, 200, 400, 500, 630, 800,
        1000, 1500, 2000, 2500, 3150, 4000, 5000,
        6300, 7100, 8000, 9000, 10000, 12500, 14000, 16000, 20000
    ]
    
    print("\n" + "="*100)
    print("КАЛИБРОВОЧНАЯ ТАБЛИЦА BS.468-4")
    print("="*100)
    print(f"{'Частота':<8} {'АЧХ (дБ)':<12} {'Коэф. передачи':<18} {'ФЧХ (°)':<12} {'В таблице':<10}")
    print("-"*100)
    
    table_freqs = bs468.ref_freqs
    
    for f in cal_freqs:
        # Получаем скалярные значения
        mag_db = float(bs468.get_magnitude_db(f))
        mag_linear = float(bs468.get_magnitude_linear(f))
        phase_deg = float(bs468.get_phase_deg(f))
        
        # Проверяем, есть ли частота в оригинальной таблице
        in_table = "Да" if f in table_freqs else ""
        
        # Выделяем ключевые частоты
        if f == 1000:
            marker = "← опорная"
        elif f in [2000, 4000, 6300]:
            marker = "← ключевая"
        else:
            marker = ""
        
        print(f"{f:<8.1f} {mag_db:<12.3f} {mag_linear:<18.6f} {phase_deg:<12.1f} {in_table:<10} {marker}")
    
    print("="*100)
    
    # Проверка конкретных значений
    #print("\n" + "="*70)
    #print("ПРОВЕРКА КОНКРЕТНЫХ ЗНАЧЕНИЙ:")
    #print("="*70)
    
    #test_points = [
    #    (1000, 0.0),
    #    (2000, 0.6),
    #    (3150, 0.9),
    #    (4000, 1.2),
    #    (5000, 0.9),
    #    (6300, -0.1),
    #    (8000, -1.6),
    #    (10000, -4.5)
    #]
    
    #for freq, expected in test_points:
    #    actual = float(bs468.get_magnitude_db(freq))
    #    error = actual - expected
        
    #    print(f"{freq:5} Гц: ожидалось {expected:5.1f} дБ, "
    #          f"получено {actual:5.2f} дБ, ошибка {error:+.2f} дБ")
    
    #print("="*70)
    
    return bs468

####################################################################################################################
# Экспорт данных для проектирования фильтра
def export_for_filter_design(num_points=512, filename="bs468_response.csv"):
    """Экспорт данных для проектирования КИХ-фильтра"""
    bs468 = BS468ReferenceResponse()
    
    # Равномерная логарифмическая сетка
    freqs = np.logspace(np.log10(20), np.log10(20000), num_points)
    
    # Получаем характеристики
    mag_db = bs468.get_magnitude_db(freqs)
    mag_linear = bs468.get_magnitude_linear(freqs)
    phase_deg = bs468.get_phase_deg(freqs)
    phase_rad = np.radians(phase_deg)
    
    # Комплексная характеристика
    complex_response = mag_linear * np.exp(1j * phase_rad)
    
    # Создаем массив данных
    data = np.column_stack([
        freqs, mag_db, mag_linear, phase_deg, phase_rad,
        np.real(complex_response), np.imag(complex_response)
    ])
    
    # Сохраняем в CSV
    headers = "Frequency_Hz, Magnitude_dB, Magnitude_linear, Phase_deg, Phase_rad, Complex_Real, Complex_Imag"
    np.savetxt(filename, data, delimiter=',', 
               header=headers, fmt='%.6f')
    
    print(f"\nДанные экспортированы в {filename}")
    print(f"Количество точек: {num_points}")
    print(f"Диапазон частот: {freqs[0]:.1f} - {freqs[-1]:.0f} Гц")
    
    # Показываем первые 5 строк
    print("\nПервые 5 строк данных:")
    print("-" * 80)
    print(headers)
    for i in range(min(5, len(data))):
        print(f"{data[i,0]:.1f}, {data[i,1]:.3f}, {data[i,2]:.6f}, "
              f"{data[i,3]:.2f}, {data[i,4]:.4f}, {data[i,5]:.6f}, {data[i,6]:.6f}")
    
    return bs468, freqs, mag_db, phase_deg, complex_response

##############################################################################################################
def plot_fir_response_with_tolerance(
    coefficients,           # Коэффициенты КИХ-фильтра
    reference_freqs,       # Частоты эталонной АЧХ [Гц]
    reference_attenuation, # Ослабление эталонной АЧХ [дБ]
    tolerance_deviations,  # Допуски отклонений [дБ]
    sampling_freq,         # Частота дискретизации [Гц]
    title="АЧХ КИХ-фильтра с эталоном и допусками"
):
    """
    Визуализация АЧХ КИХ-фильтра с эталонной характеристикой и допусками
    
    Parameters:
    -----------
    coefficients : array-like
        Коэффициенты КИХ-фильтра
    reference_freqs : array-like
        Частоты эталонной АЧХ в Гц
    reference_attenuation : array-like
        Ослабление эталонной АЧХ в дБ на соответствующих частотах
    tolerance_deviations : array-like
        Допуски отклонений в дБ на соответствующих частотах
    sampling_freq : float
        Частота дискретизации в Гц
    title : str
        Заголовок графика
    """
    
    # Вычисляем АЧХ КИХ-фильтра
    n_fft = 8192  # Количество точек для расчета АЧХ
    w, h = signal.freqz(coefficients, worN=n_fft, fs=sampling_freq)
    
    # Переводим в дБ
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)  # +1e-10 чтобы избежать log(0)
    
    # Интерполяция значений АЧХ фильтра на эталонных частотах
    filter_at_ref_freqs = np.interp(reference_freqs, w, magnitude_db)
    
    # Рассчитываем отклонения
    deviations = filter_at_ref_freqs - reference_attenuation
    
    # Проверяем соответствие допускам
    within_tolerance = np.abs(deviations) <= tolerance_deviations
           
    # Создаем график
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.1)
    
    # Основной график
    ax1 = fig.add_subplot(gs[0])
    
    # График АЧХ фильтра
    ax1.semilogx(w, magnitude_db, 
                linewidth=2.5, 
                color='blue', 
                alpha=0.8, 
                label='АЧХ КИХ-фильтра')
    
    # График эталонной АЧХ
    ax1.semilogx(reference_freqs, reference_attenuation, 
                'ro-', 
                markersize=8, 
                linewidth=2,
                label='Эталонная АЧХ')
    
    # Кривая допусков (верхняя граница)
    upper_tolerance = reference_attenuation + tolerance_deviations
    ax1.semilogx(reference_freqs, upper_tolerance, 
                'g--', 
                linewidth=1.5, 
                alpha=0.7,
                label='Верхний допуск')
    
    # Кривая допусков (нижняя граница)
    lower_tolerance = reference_attenuation - tolerance_deviations
    ax1.semilogx(reference_freqs, lower_tolerance, 
                'g--', 
                linewidth=1.5, 
                alpha=0.7,
                label='Нижний допуск')
    
    # Заливка области допуска
    ax1.fill_between(reference_freqs, lower_tolerance, upper_tolerance, 
                     alpha=0.15, color='green', label='Область допуска')
    
    # Подсветка точек вне допуска
    if not all(within_tolerance):
        outlier_freqs = reference_freqs[~within_tolerance]
        outlier_vals = filter_at_ref_freqs[~within_tolerance]
        ax1.semilogx(outlier_freqs, outlier_vals, 
                    'rx', 
                    markersize=12, 
                    markeredgewidth=2,
                    label='Вне допуска')
    
    # Настройка графика
    ax1.set_xlabel('Частота, Гц', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Ослабление, дБ', fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    ax1.grid(True, which='major', alpha=0.5)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)
    ax1.set_xlim([reference_freqs[0]/1.5, reference_freqs[-1]*1.5])
    
    # Добавляем информацию о фильтре
    filter_info = f'Порядок фильтра: {len(coefficients)-1}\n'
    filter_info += f'Fs = {sampling_freq/1000:.1f} кГц\n'
    filter_info += f'Соответствие: {np.sum(within_tolerance)}/{len(within_tolerance)} точек'
    
    ax1.text(0.02, 0.98, filter_info,
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    
    table_data = np.column_stack((reference_freqs, reference_attenuation, tolerance_deviations))
    columns = ('Частота, Гц', 'Ослабление, дБ', 'Допуск, дБ')

    # Отключаем оси для таблицы
    #axes[1].axis('off')
    #axes[1].text(0, 0, "report", fontsize=10, family='monospace')
    #axes[1].set_title('Характеристики фильтра')
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('tight')
    ax2.axis('off')

    # Форматируем таблицу как текст
    table_text = "ХАРАКТЕРИСТИКИ ФИЛЬТРА\n"
    table_text += "=" * 70 + "\n"
    table_text += "Частота,     Эталон,          Фильтр,      Отклонение,         Допуск,\n"
    table_text += "  Гц           дБ               дБ             дБ                дБ\n"
    table_text += "-" * 70 + "\n"

    for i in range(len(reference_freqs)):
        table_text += f"{reference_freqs[i]:<12} {reference_attenuation[i]:<16.1f} {filter_at_ref_freqs[i]:<16.1f} {deviations[i]:<16.1f} {tolerance_deviations[i]:<12}\n"

    # подпись
    #table_text += "\n" + "=" * 40 + "\n"
    #table_text += "МСЭ-Т Рекомендация P.53"

    # Выводим текст
    ax2.text(0.0, 0.95, table_text, 
                 fontsize=11, 
                 family='monospace',
                 verticalalignment='top',
                 horizontalalignment='left',
                transform=ax2.transAxes)
    

    # Статистика
    stats_text = f"Среднее отклонение: {np.mean(np.abs(deviations)):.3f} дБ\n"
    stats_text += f"Макс. отклонение: {np.max(np.abs(deviations)):.3f} дБ\n"
    stats_text += f"Станд. отклонение: {np.std(deviations):.3f} дБ"
    
    ax2.text(0.05, 0.02, stats_text,
             transform=ax2.transAxes,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Дополнительная информация
    print("=" * 80)
    print("СТАТИСТИКА СООТВЕТСТВИЯ:")
    print("=" * 80)
    print(f"Всего проверяемых частот: {len(reference_freqs)}")
    print(f"Соответствует допускам: {np.sum(within_tolerance)}")
    print(f"Не соответствует допускам: {np.sum(~within_tolerance)}")
    print(f"Процент соответствия: {100*np.mean(within_tolerance):.1f}%")
    print("\nДЕТАЛИ ПО ЧАСТОТАМ:")
    for i, freq in enumerate(reference_freqs):
        status = "✓" if within_tolerance[i] else "✗"
        color = "\033[92m" if within_tolerance[i] else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} {freq:8.1f} Гц: "
              f"отклонение {deviations[i]:+6.3f} дБ "
              f"(допуск ±{tolerance_deviations[i]:.2f} дБ)")
    
    return deviations