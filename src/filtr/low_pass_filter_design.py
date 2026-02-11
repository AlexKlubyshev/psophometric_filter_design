import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Анализ фильтра
def analyze_filter(coeffs, fs=48000, title=""):
    """Анализ характеристик фильтра"""
    # 1. АЧХ
    w, H = signal.freqz(coeffs, worN=8192)
    freqs = w / np.pi * (fs/2)  # Частоты в Гц
    H_mag = np.abs(H)
    H_db = 20 * np.log10(np.maximum(H_mag, 1e-10))
    
    # 2. Импульсная характеристика
    impulse_response = coeffs
    
    # 3. Групповая задержка
    H_phase = np.unwrap(np.angle(H))
    group_delay = -np.diff(H_phase) / np.diff(w)
    freqs_gd = (w[:-1] / np.pi + w[1:] / np.pi) / 2 * (fs/2)
    
    # 4. Находим частоту среза (-3 dB)
    max_gain = np.max(H_mag)
    cutoff_level = max_gain / np.sqrt(2)  # -3 dB
    idx_cutoff = np.where(H_mag <= cutoff_level)[0]
    f_cutoff = freqs[idx_cutoff[0]] if len(idx_cutoff) > 0 else 0
    
    # 5. Пульсации в полосе пропускания
    idx_passband = np.where(freqs <= 20000)[0]
    ripple_passband = np.max(H_db[idx_passband]) - np.min(H_db[idx_passband])
    
    # 6. Подавление в полосе задерживания
    idx_stopband = np.where(freqs >= 22000)[0]
    attenuation = -np.min(H_db[idx_stopband]) if len(idx_stopband) > 0 else 0
    
    print(f"\nХарактеристики фильтра '{title}':")
    print(f"  Частота среза (-3 dB): {f_cutoff:.1f} Гц")
    print(f"  Пульсации в полосе пропускания: {ripple_passband:.2f} dB")
    print(f"  Подавление в полосе задерживания: {attenuation:.1f} dB")
    print(f"  Групповая задержка (средняя): {np.mean(group_delay):.2f} отсчетов")
    
    return {
        'freqs': freqs,
        'H_db': H_db,
        'H_mag': H_mag,
        'impulse': impulse_response,
        'group_delay': group_delay,
        'freqs_gd': freqs_gd,
        'f_cutoff': f_cutoff,
        'ripple': ripple_passband,
        'attenuation': attenuation
    }

def design_lowpass_fir_128_coeffs(fs=48000, fpass=18300, fstop=21000):
    """
    Проектирование КИХ ФНЧ фильтра с 128 коэффициентами
    
    Parameters:
    -----------
    fs : float
        Частота дискретизации (Гц)
    fpass : float
        Частота конца полосы пропускания (Гц)
    fstop : float
        Частота начала полосы задерживания (Гц)
    
    Returns:
    --------
    coeffs : array
        128 коэффициентов фильтра
    """
    # Параметры
    N = 128  # Количество коэффициентов
    numtaps = N
    
    # Нормированные частоты (относительно fs/2)
    f_pass_norm = fpass / (fs/2)  # 20000/24000 = 0.8333
    f_stop_norm = fstop / (fs/2)  # 22000/24000 = 0.9167
    
    print(f"Параметры фильтра:")
    print(f"  Порядок фильтра: {N-1}")
    print(f"  Частота дискретизации: {fs} Гц")
    print(f"  Полоса пропускания: 0-{fpass} Гц")
    print(f"  Полоса задерживания: {fstop}-{fs/2} Гц")
    print(f"  Нормированная f_pass: {f_pass_norm:.4f}")
    print(f"  Нормированная f_stop: {f_stop_norm:.4f}")
    
    # Вариант 1: Метод Паркса-Маклеллана (Remez) - оптимальный КИХ
    try:
        # Границы полос [0, f_pass, f_stop, 1]
        bands = [0, f_pass_norm, f_stop_norm, 1.0]
        desired = [1, 0]  # Желаемый отклик в полосах
        weight = [1, 100]  # Веса ошибок
        
        coeffs = signal.remez(
            numtaps=numtaps,
            bands=bands,
            desired=desired,
            weight=weight,
            fs=2,  # т.к. нормированная частота от 0 до 1
            maxiter=100
        )
        method = "Remez (Паркса-Маклеллана)"
        
    except Exception as e:
        print(f"Remez не сработал: {e}")
        # Вариант 2: Фильтр с окном Кайзера
        # Расчет параметра beta для окна Кайзера
        transition_width = f_stop_norm - f_pass_norm
        attenuation_db = 60  # Подавление в полосе задерживания
        beta = signal.kaiser_beta(attenuation_db)
        
        coeffs = signal.firwin(
            numtaps=numtaps,
            cutoff=f_pass_norm,
            window=('kaiser', beta),
            scale=True,
            fs=2
        )
        method = "Окно Кайзера"
    
    return coeffs, method

def create_low_pass_filter(fs=48000, fpass=18300, fstop=21000):
    
    # Проектируем фильтр
    coeffs_rem, method = design_lowpass_fir_128_coeffs(fs, fpass, fstop)
    print(f"\nМетод проектирования: {method}")
    print(f"Количество коэффициентов: {len(coeffs_rem)}")
    #print(f"Первые 10 коэффициентов: {coeffs_rem[:10]}")
    #print(f"Последние 10 коэффициентов: {coeffs_rem[-10:]}")
    #print(f"Сумма коэффициентов: {np.sum(coeffs_rem):.6f}")

    # Анализируем
    analysis = analyze_filter(coeffs_rem, title=method)

    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. АЧХ в линейном масштабе
    axes[0, 0].plot(analysis['freqs'], analysis['H_mag'], 'b-', linewidth=2)
    axes[0, 0].axvline(20000, color='r', linestyle='--', alpha=0.7, label='20 кГц')
    axes[0, 0].axvline(22000, color='g', linestyle='--', alpha=0.7, label='22 кГц')
    axes[0, 0].axhline(1/np.sqrt(2), color='k', linestyle=':', alpha=0.5, label='-3 dB')
    axes[0, 0].set_xlabel('Частота, Гц')
    axes[0, 0].set_ylabel('Коэффициент передачи')
    axes[0, 0].set_title('АЧХ фильтра (линейный масштаб)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 24000])

    # 2. АЧХ в логарифмическом масштабе
    axes[0, 1].plot(analysis['freqs'], analysis['H_db'], 'b-', linewidth=2)
    axes[0, 1].axvline(20000, color='r', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(22000, color='g', linestyle='--', alpha=0.7)
    axes[0, 1].axhline(-3, color='k', linestyle=':', alpha=0.5)
    axes[0, 1].set_xlabel('Частота, Гц')
    axes[0, 1].set_ylabel('Коэффициент передачи, dB')
    axes[0, 1].set_title('АЧХ фильтра (логарифмический масштаб)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 24000])
    axes[0, 1].set_ylim([-80, 5])

    # 3. Детальный вид полосы пропускания
    idx_detail = np.where(analysis['freqs'] <= 25000)[0]
    axes[0, 2].plot(analysis['freqs'][idx_detail], analysis['H_db'][idx_detail], 'b-', linewidth=2)
    axes[0, 2].axvline(20000, color='r', linestyle='--', alpha=0.7)
    axes[0, 2].axvline(22000, color='g', linestyle='--', alpha=0.7)
    axes[0, 2].axhline(-3, color='k', linestyle=':', alpha=0.5)
    axes[0, 2].set_xlabel('Частота, Гц')
    axes[0, 2].set_ylabel('Коэффициент передачи, dB')
    axes[0, 2].set_title('Детальный вид (0-25 кГц)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim([0, 25000])
    axes[0, 2].set_ylim([-60, 5])

    # 4. Импульсная характеристика
    axes[1, 0].stem(range(len(coeffs_rem)), coeffs_rem, linefmt='b-', markerfmt='bo', basefmt=' ')
    axes[1, 0].set_xlabel('Отсчет')
    axes[1, 0].set_ylabel('Амплитуда')
    axes[1, 0].set_title(f'Импульсная характеристика (N={len(coeffs_rem)})')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Симметрия коэффициентов
    N = len(coeffs_rem)
    if N % 2 == 1:
        center = N // 2
        left = coeffs_rem[:center]
        right = coeffs_rem[:center:-1]
    else:
        left = coeffs_rem[:N//2]
        right = coeffs_rem[:N//2-1:-1]

    axes[1, 1].plot(left, 'bo-', label='Левая половина', alpha=0.7)
    axes[1, 1].plot(right, 'ro-', label='Правая половина', alpha=0.7)
    axes[1, 1].set_xlabel('Индекс относительно центра')
    axes[1, 1].set_ylabel('Амплитуда')
    axes[1, 1].set_title('Симметрия коэффициентов')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Групповая задержка
    axes[1, 2].plot(analysis['freqs_gd'], analysis['group_delay'], 'b-', linewidth=2)
    axes[1, 2].set_xlabel('Частота, Гц')
    axes[1, 2].set_ylabel('Групповая задержка, отсчеты')
    axes[1, 2].set_title(f'Групповая задержка (средняя: {np.mean(analysis["group_delay"]):.2f})')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim([0, 24000])

    plt.tight_layout()
    plt.show()

    return coeffs_rem

# Сохранение коэффициентов в файл
def save_coefficients(coeffs, filename="fir_coefficients_128.txt"):
    """Сохранение коэффициентов в текстовый файл"""
    with open(filename, 'w') as f:
        f.write(f"// КИХ ФНЧ фильтр 128 коэффициентов\n")
        f.write(f"// Частота дискретизации: 48000 Гц\n")
        f.write(f"// Полоса пропускания: 0-20000 Гц\n")
        f.write(f"// Полоса задерживания: 22000-24000 Гц\n")
        f.write(f"// Метод проектирования: {method}\n")
        f.write(f"// Сумма коэффициентов: {np.sum(coeffs):.10f}\n\n")
        
        f.write("float fir_coefficients[128] = {\n")
        for i, coeff in enumerate(coeffs):
            f.write(f"    {coeff:.10f}")
            if i < len(coeffs) - 1:
                f.write(",")
            if (i + 1) % 4 == 0:
                f.write("\n")
        f.write("\n};\n")
    
    print(f"\nКоэффициенты сохранены в файл: {filename}")
    print(f"Формат: float массив на C/C++")

#save_coefficients(coeffs_rem, "lowpass_20khz_128_coeffs.txt")    

##################################################################################
def analyze_phase_problem(coeff_psophometric, coeff_lowpass, fs=48000):
    """
    Анализ фазовых искажений при объединении фильтров
    """
    print("="*60)
    print("АНАЛИЗ ФАЗОВЫХ ИСКАЖЕНИЙ")
    print("="*60)
    
    # 1. Частотные характеристики
    n_fft = 8192
    w, H_psoph = signal.freqz(coeff_psophometric, worN=n_fft, fs=fs)
    w, H_lp = signal.freqz(coeff_lowpass, worN=n_fft, fs=fs)
    w, H_combined = signal.freqz(np.convolve(coeff_psophometric, coeff_lowpass), 
                                worN=n_fft, fs=fs)
    
    # 2. Фазовые характеристики
    phase_psoph = np.unwrap(np.angle(H_psoph))
    phase_lp = np.unwrap(np.angle(H_lp))
    phase_combined = np.unwrap(np.angle(H_combined))
    
    # 3. Групповая задержка
    group_delay_psoph = -np.diff(phase_psoph) / np.diff(w)
    group_delay_lp = -np.diff(phase_lp) / np.diff(w)
    group_delay_combined = -np.diff(phase_combined) / np.diff(w)
    
    # 4. Фазовый сдвиг, внесенный ФНЧ
    phase_shift = phase_lp - phase_lp[0]  # Относительный фазовый сдвиг
    
    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # АЧХ
    axes[0, 0].plot(w, 20*np.log10(np.abs(H_psoph)), 'b-', label='Псофометрический')
    axes[0, 0].plot(w, 20*np.log10(np.abs(H_lp)), 'g-', label='ФНЧ')
    axes[0, 0].plot(w, 20*np.log10(np.abs(H_combined)), 'r-', label='Комбинированный')
    axes[0, 0].set_xlabel('Частота, Гц')
    axes[0, 0].set_ylabel('АЧХ, dB')
    axes[0, 0].set_title('Амплитудно-частотные характеристики')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, fs/2])
    axes[0, 0].set_ylim([-60, 5])
    
    # ФЧХ
    axes[0, 1].plot(w, phase_psoph, 'b-', label='Псофометрический')
    axes[0, 1].plot(w, phase_lp, 'g-', label='ФНЧ')
    axes[0, 1].plot(w, phase_combined, 'r-', label='Комбинированный')
    axes[0, 1].set_xlabel('Частота, Гц')
    axes[0, 1].set_ylabel('Фаза, радианы')
    axes[0, 1].set_title('Фазо-частотные характеристики')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, fs/2])
    
    # Фазовый сдвиг ФНЧ
    axes[0, 2].plot(w[:-1], np.diff(phase_lp), 'g-', linewidth=2)
    axes[0, 2].set_xlabel('Частота, Гц')
    axes[0, 2].set_ylabel('ΔФаза/Δf')
    axes[0, 2].set_title('Производная фазы ФНЧ (нелинейность)')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim([0, fs/2])
    
    # Групповая задержка
    axes[1, 0].plot(w[:-1], group_delay_psoph, 'b-', label='Псофометрический')
    axes[1, 0].plot(w[:-1], group_delay_lp, 'g-', label='ФНЧ')
    axes[1, 0].plot(w[:-1], group_delay_combined, 'r-', label='Комбинированный')
    axes[1, 0].set_xlabel('Частота, Гц')
    axes[1, 0].set_ylabel('Групповая задержка, отсчеты')
    axes[1, 0].set_title('Групповая задержка')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, fs/2])
    
    # Фазовая ошибка в полосе пропускания
    freq_mask = w < 4000  # Полоса до 4 кГц (важная для псофометрического)
    if np.any(freq_mask):
        ideal_phase = -w[freq_mask] * (len(coeff_psophometric)-1)/2 / fs  # Линейная фаза
        
        axes[1, 1].plot(w[freq_mask], phase_psoph[freq_mask], 'b-', label='Псофометрический')
        axes[1, 1].plot(w[freq_mask], ideal_phase, 'k--', label='Идеальная линейная')
        axes[1, 1].plot(w[freq_mask], phase_combined[freq_mask], 'r-', label='Комбинированный')
        axes[1, 1].set_xlabel('Частота, Гц')
        axes[1, 1].set_ylabel('Фаза, радианы')
        axes[1, 1].set_title('Фаза в полосе пропускания (0-4 кГц)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Корреляция во временной области
    # Создаем тестовый сигнал
    t = np.arange(0, 1000) / fs
    test_signal = np.sin(2*np.pi*1000*t) + 0.5*np.sin(2*np.pi*3000*t)
    
    # Применяем фильтры
    y_psoph = signal.lfilter(coeff_psophometric, 1, test_signal)
    y_combined = signal.lfilter(np.convolve(coeff_psophometric, coeff_lowpass), 1, test_signal)
    
    # Временная задержка
    correlation = np.correlate(y_psoph, y_combined, mode='same')
    delay_idx = np.argmax(np.abs(correlation)) - len(y_psoph)//2
    
    axes[1, 2].plot(test_signal[:200], 'k-', alpha=0.5, label='Исходный')
    axes[1, 2].plot(y_psoph[:200], 'b-', label='Псофометрический')
    axes[1, 2].plot(y_combined[:200], 'r-', label='Комбинированный')
    axes[1, 2].set_xlabel('Отсчет')
    axes[1, 2].set_ylabel('Амплитуда')
    axes[1, 2].set_title(f'Временная задержка: {delay_idx} отсчетов')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'phase_shift': phase_shift,
        'group_delay': group_delay_combined,
        'time_delay': delay_idx
    }