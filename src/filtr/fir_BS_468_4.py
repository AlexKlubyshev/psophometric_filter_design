import numpy as np
from scipy import interpolate
from scipy import signal

# Этот класс использую для построения фильтра через обратное ДПФ для сравнения с результатом построения фильтра статистическим методом
class BS468ReferenceResponse_fir:
    """
    Генератор эталонной характеристики BS.468-4 для проектирования фильтров
    Возвращает АЧХ и ФЧХ на произвольных частотах
    """
    
    def __init__(self, fs=48000):
        """
        Инициализация с эталонными данными BS.468-4
        
        Параметры:
        ----------
        fs : float
            Частота дискретизации для расчета групповой задержки
        """
        self.fs = fs
        
        # Табличные данные АЧХ BS.468-4 (частоты в Гц, ослабление в дБ)
        self.ref_freqs = np.array([
            10, 20, 31.5, 63, 100, 125, 200, 250, 300, 400,
            500, 600, 750, 800, 900, 1000, 1400, 2000, 2200,
            2500, 2800, 3150, 4000, 5000, 6300, 8000, 9000,
            10000, 14000, 20000
        ])
        
        self.ref_att_db = np.array([
            -50.5, -44.7, -29.9, -19.8, -13.8, -10.9, -6.6,
            -4.7, -3.6, -2.3, -1.7, -1.4, -1.1, -1.0, -1.6,
            -3.1, -5.7, 0.0, 0.5, 0.0, -0.3, -0.8, -2.5,
            -4.5, -6.6, -10.5, -12.5, -15.5, -25.0, -40.0
        ])
        
        # Нормируем на 2200 Гц (0 дБ)
        idx_2200 = np.where(self.ref_freqs == 2200)[0][0]
        self.ref_att_db_norm = self.ref_att_db - self.ref_att_db[idx_2200]
        
        # Расчет фазовой характеристики на основе аналоговой модели
        self._setup_phase_response()
        
        # Создаем интерполяторы
        self._create_interpolators()
    
    def _setup_phase_response(self):
        """Настройка фазовой характеристики на основе аналоговой модели"""
        # Параметры аналогового фильтра BS.468-4
        # Эти значения аппроксимируют табличную АЧХ и типичную ФЧХ
        self.zeros_analog = np.array([
            -2 * np.pi * 31.6,    # Нуль НЧ
            -2 * np.pi * 1000,    # Нуль СЧ
        ])
        
        self.poles_analog = np.array([
            -2 * np.pi * 20.0,    # Полюс НЧ
            -2 * np.pi * 400.0,   # Полюс
            -2 * np.pi * 1600.0,  # Полюс
            -2 * np.pi * 5000.0,  # Полюс ВЧ
            -2 * np.pi * 12000.0, # Полюс ВЧ
        ])
        
        # Коэффициент усиления
        self.gain = 1.0
        # Нормировка на 2200 Гц
        self._normalize_gain()
    
    def _normalize_gain(self):
        """Нормировка коэффициента усиления на 2200 Гц"""
        f_ref = 2200
        w_ref = 2 * np.pi * f_ref
        
        H_num = 1.0
        H_den = 1.0
        
        for z in self.zeros_analog:
            H_num *= (1j * w_ref - z)
        
        for p in self.poles_analog:
            H_den *= (1j * w_ref - p)
        
        H_ref = H_num / H_den
        self.gain = 1.0 / np.abs(H_ref)
    
    def _analog_response(self, freqs_hz):
        """
        Расчет аналоговой характеристики
        
        Параметры:
        ----------
        freqs_hz : array_like
            Частоты в Гц
            
        Возвращает:
        -----------
        mag_linear : ndarray
            АЧХ в линейных единицах
        phase_rad : ndarray
            ФЧХ в радианах
        """
        w = 2 * np.pi * freqs_hz
        
        # Инициализация
        H = np.ones_like(freqs_hz, dtype=complex) * self.gain
        
        # Умножаем на нули
        for z in self.zeros_analog:
            H *= (1j * w - z)
        
        # Делим на полюса
        for p in self.poles_analog:
            H /= (1j * w - p)
        
        mag_linear = np.abs(H)
        phase_rad = np.angle(H)
        
        return mag_linear, phase_rad
    
    def _create_interpolators(self):
        """Создание интерполяторов для быстрого доступа"""
        # Создаем частотную сетку для интерполяции
        # Используем логарифмическую шкалу для лучшей точности
        self.interp_freqs = np.logspace(
            np.log10(10), 
            np.log10(20000), 
            1000
        )
        
        # Рассчитываем характеристики на этой сетке
        mag_linear, phase_rad = self._analog_response(self.interp_freqs)
        
        # АЧХ в дБ
        mag_db = 20 * np.log10(mag_linear)
        
        # Нормируем АЧХ на 2200 Гц
        idx_2200 = np.argmin(np.abs(self.interp_freqs - 2200))
        mag_db_norm = mag_db - mag_db[idx_2200]
        
        # Разворачиваем фазу
        phase_unwrapped = np.unwrap(phase_rad)
        
        # Создаем интерполяторы
        # Для АЧХ используем кубические сплайны
        self.interp_mag_db = interpolate.CubicSpline(
            np.log10(self.interp_freqs),
            mag_db_norm,
            extrapolate=True
        )
        
        # Для ФЧХ используем линейную интерполяцию (меньше колебаний)
        self.interp_phase_rad = interpolate.interp1d(
            np.log10(self.interp_freqs),
            phase_unwrapped,
            kind='linear',
            fill_value='extrapolate'
        )
        
        # Для линейной АЧХ (коэффициент передачи)
        self.interp_mag_linear = interpolate.CubicSpline(
            np.log10(self.interp_freqs),
            mag_linear,
            extrapolate=True
        )
    
    def get_response(self, frequencies_hz, return_complex=False):
        """
        Получить полную характеристику BS.468-4 на заданных частотах
        
        Параметры:
        ----------
        frequencies_hz : array_like
            Частоты в Гц
        return_complex : bool
            Если True, возвращает комплексную характеристику
            
        Возвращает:
        -----------
        В зависимости от return_complex:
        - False: (magnitude_db, phase_deg)
        - True: complex_response
        """
        # Преобразуем частоты в логарифмическую шкалу
        log_freqs = np.log10(np.maximum(frequencies_hz, 1e-10))
        
        # Интерполируем значения
        magnitude_db = self.interp_mag_db(log_freqs)
        phase_rad = self.interp_phase_rad(log_freqs)
        
        # Конвертируем фазу в градусы
        phase_deg = np.degrees(phase_rad)
        
        if return_complex:
            # Возвращаем комплексную характеристику
            magnitude_linear = 10 ** (magnitude_db / 20)
            complex_response = magnitude_linear * np.exp(1j * phase_rad)
            return complex_response
        else:
            return magnitude_db, phase_deg
    
    def get_magnitude_db(self, frequencies_hz):
        """Получить только АЧХ в дБ"""
        log_freqs = np.log10(np.maximum(frequencies_hz, 1e-10))
        return self.interp_mag_db(log_freqs)
    
    def get_magnitude_linear(self, frequencies_hz):
        """Получить только АЧХ в линейных единицах"""
        log_freqs = np.log10(np.maximum(frequencies_hz, 1e-10))
        return self.interp_mag_linear(log_freqs)
    
    def get_phase_deg(self, frequencies_hz):
        """Получить только ФЧХ в градусах"""
        log_freqs = np.log10(np.maximum(frequencies_hz, 1e-10))
        phase_rad = self.interp_phase_rad(log_freqs)
        return np.degrees(phase_rad)
    
    def get_phase_rad(self, frequencies_hz):
        """Получить только ФЧХ в радианах"""
        log_freqs = np.log10(np.maximum(frequencies_hz, 1e-10))
        return self.interp_phase_rad(log_freqs)
    
    def get_group_delay(self, frequencies_hz):
        """
        Получить групповое запаздывание
        
        Параметры:
        ----------
        frequencies_hz : array_like
            Частоты в Гц
            
        Возвращает:
        -----------
        group_delay_seconds : ndarray
            Групповое запаздывание в секундах
        """
        # Вычисляем производную фазы по частоте
        if np.isscalar(frequencies_hz):
            frequencies_hz = np.array([frequencies_hz])
        
        # Используем численное дифференцирование
        eps = 1e-6
        phase_rad = self.get_phase_rad(frequencies_hz)
        
        if len(frequencies_hz) > 1:
            # Для массива частот
            group_delay = -np.gradient(phase_rad) / (2 * np.pi * np.gradient(frequencies_hz))
        else:
            # Для одной частоты
            f = frequencies_hz[0]
            f1 = f * (1 - eps)
            f2 = f * (1 + eps)
            phase1 = self.get_phase_rad(f1)
            phase2 = self.get_phase_rad(f2)
            group_delay = np.array([-(phase2 - phase1) / (2 * np.pi * (f2 - f1))])
        
        return group_delay
    
    def generate_fir_coefficients(self, num_taps=129, fs=48000):
        """
        Генерация коэффициентов КИХ-фильтра методом частотной выборки
        
        Параметры:
        ----------
        num_taps : int
            Количество отводов фильтра (должно быть нечетным)
        fs : float
            Частота дискретизации
            
        Возвращает:
        -----------
        coefficients : ndarray
            Коэффициенты КИХ-фильтра
        """
        if num_taps % 2 == 0:
            raise ValueError("Количество отводов должно быть нечетным для линейной фазы")
        
        # Частоты для частотной выборки
        N = num_taps
        freq_points = np.arange(N) * fs / N
        
        # Желаемая частотная характеристика на этих частотах
        desired_response = self.get_response(freq_points[:N//2 + 1], return_complex=True)
        
        # Для КИХ-фильтра с линейной фазой:
        # 1. Создаем симметричную характеристику
        full_response = np.zeros(N, dtype=complex)
        full_response[:N//2 + 1] = desired_response
        full_response[N//2 + 1:] = np.conj(desired_response[1:][::-1])
        
        # 2. Преобразуем обратным ДПФ
        coefficients = np.fft.ifft(full_response).real
        
        # 3. Применяем окно для уменьшения пульсаций
        #window = np.hamming(N)
        #coefficients *= window
        
        return coefficients

########################### Использование  #######
def design_bs468_fir_filter(num_taps=257, fs=48000, plot_response=True):
    """
    Проектирование КИХ-фильтра BS.468-4
    
    Параметры:
    ----------
    num_taps : int
        Количество коэффициентов фильтра
    fs : float
        Частота дискретизации
    plot_response : bool
        Визуализировать характеристику
        
    Возвращает:
    -----------
    b : ndarray
        Коэффициенты фильтра
    """
    # Создаем генератор эталонной характеристики
    bs468_ref = BS468ReferenceResponse_fir(fs=fs)
    
    # Генерируем коэффициенты КИХ-фильтра
    b = bs468_ref.generate_fir_coefficients(num_taps=num_taps, fs=fs)
    
    # Проверяем характеристику полученного фильтра
    if plot_response:
        import matplotlib.pyplot as plt
        
        # Частоты для анализа
        freqs = np.logspace(np.log10(20), np.log10(20000), 1000)
        
        # Эталонная характеристика
        ref_mag_db, ref_phase_deg = bs468_ref.get_response(freqs)
        
        # Характеристика спроектированного фильтра
        w, h = signal.freqz(b, worN=4096, fs=fs)
        fir_mag_db = 20 * np.log10(np.abs(h))
        fir_phase_rad = np.unwrap(np.angle(h))
        fir_phase_deg = np.degrees(fir_phase_rad)
        
        # Создаем графики
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # АЧХ в дБ
        ax = axes[0, 0]
        ax.semilogx(freqs, ref_mag_db, 'b-', linewidth=2, label='Эталон BS.468-4')
        ax.semilogx(w, fir_mag_db, 'r--', linewidth=1.5, label=f'КИХ ({num_taps} отводов)')
        ax.set_xlabel('Частота, Гц')
        ax.set_ylabel('Ослабление, дБ')
        ax.set_title('АЧХ фильтра BS.468-4')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()
        ax.set_xlim(20, 20000)
        ax.set_ylim(-50, 5)
        
        # Ошибка АЧХ
        ax = axes[0, 1]
        # Интерполируем FIR характеристику на частоты эталона
        fir_mag_interp = np.interp(freqs, w, fir_mag_db)
        error = fir_mag_interp - ref_mag_db
        ax.semilogx(freqs, error, 'g-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Частота, Гц')
        ax.set_ylabel('Ошибка, дБ')
        ax.set_title('Ошибка аппроксимации АЧХ')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.set_xlim(20, 20000)
        ax.set_ylim(-2, 2)
        
        # ФЧХ
        ax = axes[1, 0]
        # Интерполируем FIR фазу на частоты эталона
        fir_phase_interp = np.interp(freqs, w, fir_phase_deg)
        ax.semilogx(freqs, ref_phase_deg, 'b-', linewidth=2, label='Эталон')
        ax.semilogx(freqs, fir_phase_interp, 'r--', linewidth=1.5, label='КИХ')
        ax.set_xlabel('Частота, Гц')
        ax.set_ylabel('Фаза, градусы')
        ax.set_title('ФЧХ фильтра BS.468-4')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()
        ax.set_xlim(20, 20000)
        
        # Групповая задержка
        ax = axes[1, 1]
        group_delay_ref = bs468_ref.get_group_delay(freqs)
        group_delay_fir = -np.gradient(fir_phase_rad) / (2 * np.pi * np.gradient(w))
        group_delay_fir_interp = np.interp(freqs, w, group_delay_fir)
        
        ax.semilogx(freqs, group_delay_ref * 1000, 'b-', linewidth=2, label='Эталон')
        ax.semilogx(freqs, group_delay_fir_interp * 1000, 'r--', linewidth=1.5, label='КИХ')
        ax.set_xlabel('Частота, Гц')
        ax.set_ylabel('Групповая задержка, мс')
        ax.set_title('Групповое запаздывание')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()
        ax.set_xlim(20, 20000)
        
        plt.tight_layout()
        plt.show()
    
    return b