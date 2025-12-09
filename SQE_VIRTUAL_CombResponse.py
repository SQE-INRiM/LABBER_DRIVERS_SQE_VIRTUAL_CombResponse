import InstrumentDriver
import numpy as np

class Driver(InstrumentDriver.InstrumentWorker):
    """Virtual comb-response generator for Labber"""

    def performOpen(self, options={}):
        # Nothing to open – purely virtual
        pass

    def performClose(self, bError=False, options={}):
        # Nothing to close
        pass

    def performSetValue(self, quant, value, sweepRate=0.0, options={}):
        # Just store and return
        return value

    def performGetValue(self, quant, options={}):
        if quant.name == 'Spectrum':
            # 1) Frequency axis
            f0 = self.getValue('Start frequency')
            f1 = self.getValue('Stop frequency')
            n_points = int(self.getValue('Number of points'))
            f = np.linspace(f0, f1, n_points)

            # 2) Comb parameters
            fc = self.getValue('Center frequency')
            df = self.getValue('Line spacing')
            N  = int(self.getValue('Number of lines'))
            gamma = self.getValue('Linewidth')  # FWHM (approx)

            # Ensure odd N so we can center the comb nicely
            if N % 2 == 0:
                N += 1

            # Line index: e.g. -K,...,0,...,+K
            K = (N - 1) // 2
            idx = np.arange(-K, K+1)
            line_freqs = fc + idx * df

            # 3) Amplitude envelope
            env_type = self.getValue('Amplitude envelope')
            if env_type == 'Flat':
                amps = np.ones_like(line_freqs, dtype=float)
            elif env_type == 'Gaussian':
                # Simple Gaussian envelope vs index
                sigma_idx = max(1.0, N/6.0)
                amps = np.exp(-0.5 * (idx / sigma_idx)**2)
            else:  # 'Custom'
                s = self.getValue('Custom amplitudes')
                try:
                    vals = np.array([float(x) for x in s.split(',') if x.strip()!=''])
                    # pad or crop to N
                    if len(vals) < N:
                        vals = np.pad(vals, (0, N-len(vals)), mode='edge')
                    elif len(vals) > N:
                        vals = vals[:N]
                    amps = vals
                except Exception:
                    amps = np.ones_like(line_freqs, dtype=float)
            # Normalize envelope to max 1
            if np.max(np.abs(amps)) > 0:
                amps = amps / np.max(np.abs(amps))

            # 4) Build comb spectrum (Lorentzian lines)
            signal = np.zeros_like(f, dtype=complex)
            gamma_rad = gamma / 2.0  # half-width

            global_phase = self.getValue('Global phase')
            for a, fl in zip(amps, line_freqs):
                # Complex Lorentzian: a / (1 + i * (f - fl)/gamma_rad)
                signal += a / (1.0 + 1j * (f - fl) / gamma_rad)

            # Apply global phase
            signal *= np.exp(1j * global_phase)

            # 5) Add noise if requested
            if self.getValue('Add noise'):
                mag_noise = self.getValue('Magnitude noise')
                phase_noise = self.getValue('Phase noise')
                mag_factor = 1 + np.sqrt(
                    np.random.uniform(0, mag_noise, len(signal))
                )
                phase_rand = np.random.uniform(0, phase_noise * np.pi, len(signal))
                signal = signal * mag_factor * np.exp(1j * phase_rand)

            # 6) Gain & offset
            gain = self.getValue('Gain')
            offset = self.getValue('Offset')
            signal = gain * signal + offset

            # 7) Return as trace
            try:
                dt = f[1] - f[0]
            except Exception:
                dt = 1.0
            trace = quant.getTraceDict(signal, t0=0.0, dt=dt)
            return trace

        else:
            # Any other quantity: just return its current value
            return quant.getValue()

