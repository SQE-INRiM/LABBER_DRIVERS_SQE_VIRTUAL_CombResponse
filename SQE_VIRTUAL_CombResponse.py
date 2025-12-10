import InstrumentDriver
import numpy as np


class Driver(InstrumentDriver.InstrumentWorker):
    """Virtual SQUID-like comb generator using time-domain pulses.

    Steps:
      1. Build a periodic voltage pulse train in time (alternating sign).
      2. FFT -> complex comb in frequency domain.
      3. Interpolate FFT onto the Labber frequency grid.
      4. Apply noise / gain / offset / global phase.
      5. Return 'Spectrum' as VECTOR_COMPLEX vs frequency.
    """

    # ----------------------- Standard hooks -----------------------

    def performOpen(self, options={}):
        # Purely virtual instrument
        pass

    def performClose(self, bError=False, options={}):
        pass

    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        # Just store the value
        return value

    # ----------------- Time-domain pulse construction -----------------

    def _build_time_pulse_train(self):
        """Construct a periodic voltage pulse train v(t).

        Returns
        -------
        t : np.ndarray
            Time axis [s].
        v : np.ndarray
            Voltage vs time [V].
        """
        # Pulse / pump parameters
        fp = self.getValue('Pump frequency')          # pump / repetition frequency [Hz]
        A = self.getValue('Pulse amplitude')          # [V]
        frac_width = self.getValue('Pulse width (fraction of period)')
        n_periods = int(self.getValue('Number of periods'))
        samples_per_period = int(self.getValue('Samples per period'))
        remove_dc = bool(self.getValue('DC removal'))
        alt_sign = bool(self.getValue('Alternating sign'))

        # Safety / sanity
        if fp <= 0.0:
            fp = 1e9
        if samples_per_period < 8:
            samples_per_period = 8
        if n_periods < 1:
            n_periods = 1
        # Limit width fraction to sensible range
        frac_width = max(1e-3, min(frac_width, 0.9))

        T = 1.0 / fp
        fs = fp * samples_per_period   # sampling rate [Hz]
        dt = 1.0 / fs
        n_t = n_periods * samples_per_period

        # Time axis
        t = np.arange(n_t) * dt

        # Gaussian pulse parameters
        # Interpret "Pulse width (fraction of period)" as FWHM = frac_width * T
        fwhm = frac_width * T
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # FWHM -> sigma

        # Build pulse train
        v = np.zeros_like(t, dtype=float)
        for k in range(n_periods):
            # Place one pulse per period, centered at middle of the period
            t0 = (k + 0.5) * T
            sign = (-1)**k if alt_sign else 1.0
            v += sign * A * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

        # Remove DC offset if requested
        if remove_dc:
            v = v - np.mean(v)

        return t, v, fs

    # ----------------- Frequency-domain spectrum from FFT -----------------

    def _calculate_complex_spectrum(self):
        """Compute the comb spectrum from the time-domain pulse train.

        Returns
        -------
        f_spec : np.ndarray
            Frequency axis for requested spectrum (Labber sweep) [Hz].
        signal_spec : np.ndarray
            Complex spectrum at those frequencies.
        """
        # 1) Build time-domain pulse train
        t, v, fs = self._build_time_pulse_train()
        n_t = len(t)

        # 2) FFT (one-sided real FFT)
        # Use rfft since v is real; frequency axis from 0 to fs/2
        V_f = np.fft.rfft(v)
        freqs_fft = np.fft.rfftfreq(n_t, d=1.0/fs)

        # Normalize (optional); here we normalize by number of points
        V_f = V_f / n_t

        # 3) Desired Labber spectrum axis (PNA-X-like sweep)
        f0 = self.getValue('Start frequency')
        f1 = self.getValue('Stop frequency')
        n_points = int(self.getValue('Number of points'))

        if n_points < 2:
            n_points = 2
        f_spec = np.linspace(f0, f1, n_points)

        # 4) Interpolate complex spectrum onto f_spec
        # Outside [0, fs/2] we set spectrum to zero.
        # Interpolate real and imag separately.
        real_interp = np.interp(
            f_spec,
            freqs_fft,
            V_f.real,
            left=0.0,
            right=0.0
        )
        imag_interp = np.interp(
            f_spec,
            freqs_fft,
            V_f.imag,
            left=0.0,
            right=0.0
        )
        signal_spec = real_interp + 1j * imag_interp

        # 5) Apply global phase, noise, gain, offset in frequency domain
        # Global phase
        global_phase = self.getValue('Global phase')
        signal_spec *= np.exp(1j * global_phase)

        # Noise
        if self.getValue('Add noise'):
            mag_noise = max(self.getValue('Magnitude noise'), 0.0)
            phase_noise = max(self.getValue('Phase noise'), 0.0)
            mag_factor = 1.0 + np.sqrt(
                np.random.uniform(0.0, mag_noise, len(signal_spec))
            )
            phase_rand = np.random.uniform(
                0.0, phase_noise * np.pi, len(signal_spec)
            )
            signal_spec = signal_spec * mag_factor * np.exp(1j * phase_rand)

        # Gain & offset
        gain = self.getValue('Gain')
        offset = self.getValue('Offset')
        signal_spec = gain * signal_spec + offset

        return f_spec, signal_spec

    # ----------------- GetValue -----------------

    def performGetValue(self, quant, options={}):
        name = quant.name

        if name == 'Spectrum':
            # Compute spectrum from time-domain pulse train
            f_spec, signal_spec = self._calculate_complex_spectrum()

            # Encode frequency axis as t0 + n*dt for Labber
            if len(f_spec) > 1:
                df = f_spec[1] - f_spec[0]
            else:
                df = 1.0
            f0 = f_spec[0] if len(f_spec) > 0 else 0.0

            trace = quant.getTraceDict(signal_spec, t0=f0, dt=df)
            return trace        

        # All other quantities: just return stored value
        return quant.getValue()
