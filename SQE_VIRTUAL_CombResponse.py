import InstrumentDriver
import numpy as np


class Driver(InstrumentDriver.InstrumentWorker):
    """Virtual pump-referenced comb-response generator for Labber.

    - Lines at f_n = n * f_p, n = Min order ... Max order
    - Amplitude decreases with harmonic order (envelope in n)
    - 'Spectrum'        -> complex VECTOR_COMPLEX (PNA-X-like I/Q)
    - 'Spectrum (dBm)'  -> real VECTOR in dBm (SA-like power spectrum)
    - 'Time trace'      -> real VECTOR in V (QCS-like voltage vs time)
    """

    # ----------------------- Standard hooks -----------------------

    def performOpen(self, options={}):
        # No real hardware to open
        pass

    def performClose(self, bError=False, options={}):
        # No real hardware to close
        pass

    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        # Just store the value, Labber handles the bookkeeping
        return value

    # ----------------- Common harmonic configuration -----------------

    def _get_orders_and_amps(self):
        """Return harmonic orders, line frequencies, and envelope amplitudes (normalized).

        Returns
        -------
        orders : np.ndarray of int
            Harmonic indices n (n_min ... n_max)
        line_freqs : np.ndarray of float
            Harmonic frequencies f_n = n * f_p
        amps : np.ndarray of float
            Real positive amplitudes for each harmonic (max normalized to 1)
        """
        # Pump and harmonic range
        fp = self.getValue('Pump frequency')
        n_min = int(self.getValue('Min order'))
        n_max = int(self.getValue('Max order'))

        if n_min < 1:
            n_min = 1
        if n_max < n_min:
            n_max = n_min

        orders = np.arange(n_min, n_max + 1, dtype=int)
        line_freqs = orders * fp

        # Envelope
        env_type = self.getValue('Amplitude envelope')

        if env_type == 'Power law':
            alpha = self.getValue('Power-law exponent')
            alpha = max(alpha, 0.0)
            amps = 1.0 / (orders.astype(float) ** alpha)

        elif env_type == 'Gaussian in n':
            sigma_n = self.getValue('Gaussian width in n')
            if sigma_n <= 0:
                sigma_n = 1.0
            amps = np.exp(-0.5 * ((orders - 1.0) / sigma_n) ** 2)

        else:  # 'Custom'
            s = self.getValue('Custom amplitudes')
            try:
                vals = np.array(
                    [float(x) for x in s.split(',') if x.strip() != '']
                )
                N = len(orders)
                if len(vals) < N:
                    vals = np.pad(vals, (0, N - len(vals)), mode='edge')
                elif len(vals) > N:
                    vals = vals[:N]
                amps = vals
            except Exception:
                amps = np.ones_like(orders, dtype=float)

        # Normalize so maximum amplitude is 1
        max_amp = np.max(np.abs(amps))
        if max_amp > 0:
            amps = amps / max_amp
        else:
            amps = np.ones_like(orders, dtype=float)

        return orders, line_freqs, amps

    # ----------------- Frequency-domain calculation -----------------

    def _calculate_complex_spectrum(self):
        """Build frequency axis and complex comb response (continuous vs f)."""
        # 1) Frequency axis
        f0 = self.getValue('Start frequency')
        f1 = self.getValue('Stop frequency')
        n_points = int(self.getValue('Number of points'))
        f = np.linspace(f0, f1, n_points)

        # 2) Harmonics and envelope
        orders, line_freqs, amps = self._get_orders_and_amps()
        gamma = self.getValue('Linewidth')  # FWHM-like
        gamma_rad = gamma / 2.0

        global_phase = self.getValue('Global phase')

        # 3) Sum Lorentzian-like complex lines
        signal = np.zeros_like(f, dtype=complex)
        for a, fn in zip(amps, line_freqs):
            if gamma_rad > 0:
                signal += a / (1.0 + 1j * (f - fn) / gamma_rad)
            else:
                # If gamma=0, fall back to very narrow Gaussian spike
                signal += a * np.exp(-((f - fn) ** 2) / (2 * (1.0e3 ** 2)))

        # Global phase
        signal *= np.exp(1j * global_phase)

        # 4) Add noise (magnitude & phase per frequency point)
        if self.getValue('Add noise'):
            mag_noise = max(self.getValue('Magnitude noise'), 0.0)
            phase_noise = max(self.getValue('Phase noise'), 0.0)

            mag_factor = 1.0 + np.sqrt(
                np.random.uniform(0.0, mag_noise, len(signal))
            )
            phase_rand = np.random.uniform(
                0.0, phase_noise * np.pi, len(signal)
            )
            signal = signal * mag_factor * np.exp(1j * phase_rand)

        # 5) Gain & offset (complex)
        gain = self.getValue('Gain')
        offset = self.getValue('Offset')
        signal = gain * signal + offset

        return f, signal

    # ----------------- Time-domain calculation -----------------

    def _calculate_time_trace(self):
        """Build time vector and real voltage trace from discrete harmonics.

        This corresponds to a periodic train of pulses with period 1/f_p,
        like the upper panel of Fig. 1c (QCS-style time-domain signal).
        """
        # Time axis
        fs = self.getValue('Sample rate')          # Hz
        n_t = int(self.getValue('Number of time points'))
        if fs <= 0:
            fs = 1e9  # fallback
        t = np.arange(n_t) / fs

        # Harmonics and envelope
        orders, line_freqs, amps = self._get_orders_and_amps()
        global_phase = self.getValue('Global phase')

        # Complex harmonic coefficients
        coeffs = amps.astype(complex) * np.exp(1j * global_phase)

        # Optional static noise per harmonic (amplitude & phase)
        if self.getValue('Add noise'):
            mag_noise = max(self.getValue('Magnitude noise'), 0.0)
            phase_noise = max(self.getValue('Phase noise'), 0.0)
            mag_factor = 1.0 + np.sqrt(
                np.random.uniform(0.0, mag_noise, len(coeffs))
            )
            phase_rand = np.random.uniform(
                0.0, phase_noise * np.pi, len(coeffs)
            )
            coeffs = coeffs * mag_factor * np.exp(1j * phase_rand)

        # Build time signal: sum_n Re[ a_n e^{i 2π n f_p t} ]
        v_complex = np.zeros_like(t, dtype=complex)
        for a_n, f_n in zip(coeffs, line_freqs):
            v_complex += a_n * np.exp(1j * 2.0 * np.pi * f_n * t)

        v = np.real(v_complex)

        # Gain & offset in time-domain (voltage)
        gain = self.getValue('Gain')
        offset = self.getValue('Offset')
        v = gain * v + offset

        return t, v

    # ------------------------ GetValue logic ------------------------

    def performGetValue(self, quant, options={}):
        """Handle Labber getValue requests."""
        name = quant.name

        # -------- Frequency-domain outputs --------
        if name in ('Spectrum', 'Spectrum (dBm)'):
            f, signal = self._calculate_complex_spectrum()

            if len(f) > 1:
                dt = f[1] - f[0]  # used as x-step
            else:
                dt = 1.0

            if name == 'Spectrum':
                trace = quant.getTraceDict(signal, t0=0.0, dt=dt)
                return trace

            if name == 'Spectrum (dBm)':
                power_linear = np.abs(signal) ** 2
                power_linear[power_linear <= 1e-24] = 1e-24
                power_dBm = 10.0 * np.log10(power_linear)
                trace = quant.getTraceDict(power_dBm, t0=0.0, dt=dt)
                return trace

        # -------- Time-domain output --------
        if name == 'Time trace':
            t, v = self._calculate_time_trace()

            if len(t) > 1:
                dt = t[1] - t[0]
            else:
                dt = 1.0

            trace = quant.getTraceDict(v, t0=0.0, dt=dt)
            return trace

        # Other scalar quantities: just return stored value
        return quant.getValue()
