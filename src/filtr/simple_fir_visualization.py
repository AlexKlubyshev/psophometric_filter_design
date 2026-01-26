import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def simple_fir_visualization(freq_hz, fs, coeffs):
    """
    Простая визуализация работы КИХ-фильтра
    """
    # Генерация сигнала
    duration = 0.02  # 20 мс
    t = np.arange(0, duration, 1/fs)
    x = np.sin(2 * np.pi * freq_hz * t)
    
    # Фильтрация
    y = lfilter(coeffs, 1.0, x)
    
    # Расчет коэффициента передачи
    norm_freq = freq_hz / (fs/2) * np.pi
    H = sum(hk * np.exp(-1j * norm_freq * k) for k, hk in enumerate(coeffs))
    gain = np.abs(H)
    phase = np.angle(H)
    
    # Построение графиков
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # Временные графики
    ax1.plot(t*1000, x, 'b-', label=f'Вход: {freq_hz} Гц', linewidth=2)
    ax1.plot(t*1000, y, 'r-', label='Выход КИХ', linewidth=2)
    ax1.plot(t*1000, gain*np.sin(2*np.pi*freq_hz*t + phase), 'g--', 
             label='Идеальный выход', alpha=0.7)
    ax1.set_xlabel('Время (мс)')
    ax1.set_ylabel('Амплитуда')
    ax1.set_title(f'КИХ-фильтр: f={freq_hz} Гц, усиление={gain:.3f}, фаза={np.degrees(phase):.1f}°')
    ax1.legend()
    ax1.grid(True)
    
    # Импульсная характеристика
    ax2.stem(range(len(coeffs)), coeffs, linefmt='C2-', markerfmt='C2o')
    ax2.set_xlabel('Коэффициент k')
    ax2.set_ylabel('h[k]')
    ax2.set_title(f'Коэффициенты фильтра (N={len(coeffs)})')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Усиление на {freq_hz} Гц: {gain:.4f}")
    print(f"Фазовый сдвиг: {np.degrees(phase):.1f} градусов")
    
    return x, y

# Быстрый тест
if __name__ == "__main__":
    # Тестовые коэффициенты (простой НЧ-фильтр)
    #coeffs = [0.1, 0.15, 0.2, 0.3, 0.2, 0.15, 0.1]
    coeffs = fir_coeffs = [-0.17929851,-0.15620984,0.20098885,0.3777962,0.20098885,-0.15620984,-0.17929851]

    # Частота в полосе пропускания
    #simple_fir_visualization(50, 1000, coeffs)
    simple_fir_visualization(100, 42000, coeffs)

    # Частота в полосе подавления
    #simple_fir_visualization(400, 1000, coeffs)
    simple_fir_visualization(4000, 42000, coeffs)