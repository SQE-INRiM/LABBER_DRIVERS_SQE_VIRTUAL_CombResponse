import InstrumentDriver
import numpy as np


class Driver(InstrumentDriver.InstrumentWorker):
    """Virtual SQUID comb response in the frequency domain.

    Physics:
      - Flux drive: Phi(t)/Phi0 = Phi_DC + Phi_RF * cos(2π f_p t)
      - Phase:      phi(t) = pi * Phi(t)/Phi0
      - Voltage:    V(t) = -(r/2) * [(1 + tan^2 phi)/(1 + r^2 tan^2 phi)] * d(Phi)/dt
      - Here Phi is expressed in units of Phi0, so dPhi/dt = Phi0 * d(Phi_norm)/dt.
      - We build V(t), FFT it, and interpolate onto the requested
        [Start frequency, Stop frequency] axis.
      - Output: complex 'Spectrum' (VECTOR_COMPLEX), like a PNA-X trace.
    """

    # ----------------------- Standard hooks -----------------------

    def performOpen(self, options={}):
        # Purely virtual instrument
        pass

    def performClose(self, bError=False, options={}):
        # Purely virtual instrument
        pass

    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        # Just store the value
        return value

    # ----------------- Time-domain SQUID waveform -----------------

    def _build_squid_time_trace(self):
        """Construct time axis and SQUID voltage V(t) using the flux model.

        Returns
        -------
        t : np.ndarray
            Time axis [s].
        v : np.ndarray
            Voltage vs time [V].
        fs : float
            Sampling rate [Hz].
        """
        # Read parameters
        fp = self.getValue('Pump frequency')            # [Hz]
        phi_dc = self.getValue('DC flux bias')          # Phi_DC / Phi0
        phi_rf = self.getValue('RF flux amplitude')     # Phi_RF / Phi0
        r = self.getValue('Asymmetry r')                # asymmetry
        A_scale = self.getValue('Pulse amplitude scale')      # dimensionless gain

        n_periods = int(self.getValue('Number of periods'))
        samples_per_period = int(self.getValue('Samples per period'))
        remove_dc = bool(self.getValue('DC removal'))

        # Sanity checks
        if fp <= 0.0:
            fp = 1e9
        if n_periods < 1:
            n_periods = 1
        if samples_per_period < 8:
            samples_per_period = 8

        # Time grid
        T = 1.0 / fp
        fs = fp * samples_per_period
        dt = 1.0 / fs
        n_t = n_periods * samples_per_period
        t = np.arange(n_t, dtype=float) * dt

        # Normalized flux: Phi_norm(t) = Phi(t)/Phi0
        phi_norm = phi_dc + phi_rf * np.cos(2.0 * np.pi * fp * t)
        # Phase phi(t) = pi * Phi_norm(t)
        phi = np.pi * phi_norm

        # Avoid divergence near tan(pi/2)
        tan_phi = np.tan(phi)
        tan_phi = np.clip(tan_phi, -1e4, 1e4)

        # Nonlinear prefactor
        numerator = 1.0 + tan_phi**2
        denominator = 1.0 + (r**2) * tan_phi**2
        nonlinear_factor = numerator / denominator

        # d(Phi_norm)/dt  [1/s]
        dphi_norm_dt = -2.0 * np.pi * fp * phi_rf * np.sin(2.0 * np.pi * fp * t)

        # Flux quantum [Wb]
        PHI0 = 2.067833848e-15

        # Voltage from theory (in volts, up to overall scale A_scale)
        # V(t) = -(r/2) * factor * PHI0 * d(Phi_norm)/dt
        v_th = -0.5 * r * nonlinear_factor * dphi_norm_dt * PHI0

        # Apply scaling
        v = A_scale * v_th

        # Remove DC component if requested
        if remove_dc:
            v = v - np.mean(v)

        return t, v, fs

    # ----------------- Frequency-domain spectrum -----------------

    def _calculate_complex_spectrum(self):
        """Compute complex comb spectrum from SQUID time-domain response."""
        # 1) Time-domain SQUID signal
        t, v, fs = self._build_squid_time_trace()
        n_t = len(t)

        # 2) FFT (one-sided)
        V_f = np.fft.rfft(v)
        freqs_fft = np.fft.rfftfreq(n_t, d=1.0 / fs)

        # Normalize so amplitude doesn't scale with record length
        V_f = V_f / n_t

        # 3) Requested spectrum axis
        f0 = self.getValue('Start frequency')
        f1 = self.getValue('Stop frequency')
        n_points = int(self.getValue('Number of points'))
        if n_points < 2:
            n_points = 2
        f_spec = np.linspace(f0, f1, n_points)

        # 4) Interpolate complex FFT onto f_spec
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

        # 5) Apply global phase, noise, gain, offset
        global_phase = self.getValue('Global phase')
        signal_spec *= np.exp(1j * global_phase)

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

        gain = self.getValue('Gain')
        offset = self.getValue('Offset')
        signal_spec = gain * signal_spec + offset

        return f_spec, signal_spec

    # ----------------- GetValue -----------------

    def performGetValue(self, quant, options={}):
        name = quant.name

        if name == 'Spectrum':
            f_spec, signal_spec = self._calculate_complex_spectrum()

            # Encode frequency axis as t0 + n*dt (Labber will label it Frequency [Hz])
            if len(f_spec) > 1:
                dt = f_spec[1] - f_spec[0]
            else:
                dt = 1.0
            f0 = f_spec[0] if len(f_spec) > 0 else 0.0

            trace = quant.getTraceDict(signal_spec, t0=f0, dt=dt)
            return trace

        # All other scalar quantities
        return quant.getValue()
