import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz

def visualize_fir_response(input_freq_hz, fs, fir_coeffs, duration_sec=0.1, show_freq_response=False):
    """
    Визуализирует входной и выходной сигналы КИХ-фильтра на заданной частоте
    
    Parameters:
    -----------
    input_freq_hz : float
        Частота входного синусоидального сигнала в Герцах
    fs : float
        Частота дискретизации в Герцах
    fir_coeffs : array-like
        Коэффициенты КИХ-фильтра
    duration_sec : float
        Длительность сигнала в секундах
    show_freq_response : bool
        Если True, показывает АЧХ фильтра рядом с временными графиками
    """
    
    # 1. Параметры сигнала
    N = len(fir_coeffs)  # Порядок фильтра
    n_samples = int(fs * duration_sec)
    t = np.arange(n_samples) / fs  # Временная ось в секундах
    
    # 2. Генерация входного сигнала
    omega = 2 * np.pi * input_freq_hz  # Угловая частота в радианах/секунду
    x = np.sin(omega * t)  # Входной синусоидальный сигнал
    
    # 3. Применение КИХ-фильтра
    y = lfilter(fir_coeffs, 1.0, x)  # Фильтрация
    
    # 4. Вычисление АЧХ на частоте входного сигнала
    # Преобразуем частоту в Герцах в нормированную частоту
    norm_freq = input_freq_hz / (fs/2) * np.pi  # ω в рад/отсчет
    
    # Вычисляем комплексный коэффициент передачи на этой частоте
    H_at_freq = 0j
    for k, hk in enumerate(fir_coeffs):
        H_at_freq += hk * np.exp(-1j * norm_freq * k)
    
    gain = np.abs(H_at_freq)  # Амплитудный коэффициент
    phase_shift = np.angle(H_at_freq)  # Фазовый сдвиг в радианах
    
    # 5. Создание идеального выходного сигнала (для сравнения)
    y_ideal = gain * np.sin(omega * t + phase_shift)
    
    # 6. Визуализация
    if show_freq_response:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    
    # График 1: Входной и выходной сигналы
    ax1.plot(t * 1000, x, 'b-', linewidth=2, alpha=0.7, label=f'Входной сигнал: {input_freq_hz} Гц')
    ax1.plot(t * 1000, y, 'r-', linewidth=2, label=f'Выходной сигнал (КИХ)')
    ax1.plot(t * 1000, y_ideal, 'g--', linewidth=1.5, alpha=0.7, label='Идеальный выход')
    ax1.set_xlabel('Время, мс')
    ax1.set_ylabel('Амплитуда')
    ax1.set_title(f'КИХ-фильтр: входной и выходной сигналы на частоте {input_freq_hz} Гц')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Фрагмент сигнала (для детального рассмотрения)
    start_idx = max(0, n_samples//2 - 50)
    end_idx = min(n_samples, start_idx + 100)
    
    ax2.plot(t[start_idx:end_idx] * 1000, x[start_idx:end_idx], 'b-', linewidth=2, alpha=0.7, marker='o', markersize=4)
    ax2.plot(t[start_idx:end_idx] * 1000, y[start_idx:end_idx], 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Время, мс')
    ax2.set_ylabel('Амплитуда')
    ax2.set_title('Детальный вид (фрагмент)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(['Вход', 'Выход'])
    
    # График 3: Импульсная характеристика фильтра
    ax3.stem(np.arange(N), fir_coeffs, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
    ax3.set_xlabel('Отсчет (k)')
    ax3.set_ylabel('Коэффициент h[k]')
    ax3.set_title(f'Импульсная характеристика КИХ-фильтра (N={N})')
    ax3.grid(True, alpha=0.3)
    
    # Добавляем информацию о коэффициентах
    coeffs_text = "Коэффициенты: " + ", ".join([f"{c:.3f}" for c in fir_coeffs])
    ax3.text(0.02, 0.98, coeffs_text, transform=ax3.transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # График 4: АЧХ фильтра (если нужно)
    if show_freq_response:
        # Вычисляем полную АЧХ
        w, H = freqz(fir_coeffs, worN=2000)
        freqs_hz = w / np.pi * (fs/2)
        
        ax4.plot(freqs_hz, 20 * np.log10(np.abs(H) + 1e-10), 'b-', linewidth=2)
        ax4.axvline(x=input_freq_hz, color='r', linestyle='--', linewidth=2, 
                   label=f'Частота сигнала: {input_freq_hz} Гц')
        
        # Показываем усиление на этой частоте
        gain_db = 20 * np.log10(gain + 1e-10)
        ax4.plot(input_freq_hz, gain_db, 'ro', markersize=10)
        ax4.text(input_freq_hz + 0.02*fs/2, gain_db, 
                f'Усиление: {gain:.3f} ({gain_db:.1f} дБ)', 
                verticalalignment='center')
        
        ax4.set_xlabel('Частота, Гц')
        ax4.set_ylabel('Усиление, дБ')
        ax4.set_title('АЧХ КИХ-фильтра')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_xlim([0, fs/2])
    
    plt.tight_layout()
    
    # 7. Вывод информации в консоль
    print("=" * 60)
    print(f"ПАРАМЕТРЫ ФИЛЬТРАЦИИ:")
    print(f"  Частота входного сигнала: {input_freq_hz} Гц")
    print(f"  Частота дискретизации: {fs} Гц")
    print(f"  Порядок фильтра: {N}")
    print(f"  Нормированная частота: {norm_freq:.3f} рад/отсчет")
    print(f"  Коэффициент передачи на этой частоте: {gain:.4f}")
    print(f"  Коэффициент передачи в дБ: {20*np.log10(gain+1e-10):.1f} дБ")
    print(f"  Фазовый сдвиг: {phase_shift:.3f} рад ({np.degrees(phase_shift):.1f}°)")
    print(f"  Задержка группы: {(N-1)/2:.1f} отсчетов")
    
    # Проверяем линейность фазы (для симметричных фильтров)
    if np.allclose(fir_coeffs, fir_coeffs[::-1]):
        print(f"  Фаза линейная (фильтр симметричный)")
    print("=" * 60)
    
    plt.show()
    
    return x, y, y_ideal, gain, phase_shift


# ============================================================================
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ============================================================================

def example_lowpass_filter():
    """Пример 1: НЧ-фильтр"""
    print("\n" + "="*60)
    print("ПРИМЕР 1: НИЗКОЧАСТОТНЫЙ ФИЛЬТР (усредняющий)")
    print("="*60)
    
    # Коэффициенты простого усредняющего фильтра (НЧ-характеристика)
    fir_coeffs = [0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1]
    
    # Параметры
    fs = 1000  # Частота дискретизации 1 кГц
    input_freq = 50  # Частота сигнала 50 Гц (в полосе пропускания)
    
    visualize_fir_response(input_freq, fs, fir_coeffs, duration_sec=0.05, show_freq_response=True)


def example_bandpass_filter():
    """Пример 2: Полосовой фильтр"""
    print("\n" + "="*60)
    print("ПРИМЕР 2: ПОЛОСОВОЙ ФИЛЬТР")
    print("="*60)
    
    # Коэффициенты полосового фильтра (центр ~150 Гц при fs=1000)
    fir_coeffs = [0.05, -0.1, 0.15, 0.7, 0.15, -0.1, 0.05]
    
    fs = 1000
    # Тестируем на частоте в полосе пропускания
    input_freq = 150
    
    visualize_fir_response(input_freq, fs, fir_coeffs, duration_sec=0.05, show_freq_response=True)


def example_high_frequency():
    """Пример 3: Высокочастотный сигнал на НЧ-фильтре"""
    print("\n" + "="*60)
    print("ПРИМЕР 3: ВЫСОКОЧАСТОТНЫЙ СИГНАЛ НА НЧ-ФИЛЬТРЕ")
    print("="*60)
    
    # Коэффициенты НЧ-фильтра
    fir_coeffs = [0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1]
    
    fs = 1000
    input_freq = 400  # Высокая частота (должна подавляться)
    
    visualize_fir_response(input_freq, fs, fir_coeffs, duration_sec=0.03)


def custom_example():
    """Пример 4: Пользовательские параметры"""
    print("\n" + "="*60)
    print("ПРИМЕР 4: ПОЛЬЗОВАТЕЛЬСКИЙ ФИЛЬТР")
    print("="*60)
    
    # Можно ввести свои коэффициенты
    # Например, фильтр с резонансом на 100 Гц при fs=1000
    #fir_coeffs = [0.02, 0.05, 0.12, 0.2, 0.25, 0.2, 0.12, 0.05, 0.02]
    fir_coeffs = [-0.17929851,-0.15620984,0.20098885,0.3777962,0.20098885,-0.15620984,-0.17929851]

    
    fs = float(input("Введите частоту дискретизации (Гц) [по умолчанию 1000]: ") or 1000)
    input_freq = float(input("Введите частоту сигнала (Гц) [по умолчанию 100]: ") or 100)
    
    visualize_fir_response(input_freq, fs, fir_coeffs, duration_sec=0.05, show_freq_response=True)


# ============================================================================
# ЗАПУСК ПРИМЕРОВ
# ============================================================================

if __name__ == "__main__":
    print("ВИЗУАЛИЗАЦИЯ РАБОТЫ КИХ-ФИЛЬТРА")
    print("="*60)
    
    while True:
        print("\nВыберите пример:")
        print("1. Низкочастотный фильтр (50 Гц)")
        print("2. Полосовой фильтр (150 Гц)")
        print("3. Высокочастотный сигнал на НЧ-фильтре (400 Гц)")
        print("4. Пользовательские параметры")
        print("0. Выход")
        
        choice = input("Ваш выбор: ").strip()
        
        if choice == '1':
            example_lowpass_filter()
        elif choice == '2':
            example_bandpass_filter()
        elif choice == '3':
            example_high_frequency()
        elif choice == '4':
            custom_example()
        elif choice == '0':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")