"""Exercise environmental options through real registered pyEFPEHM plugins."""
import numpy as np
import pytest

pyEFPEHM = pytest.importorskip('pyEFPEHM')
from pyEFPEHM.waveform.plugins import native_parameters
from pycbc.waveform import (get_fd_waveform, get_td_waveform,
                            get_fd_waveform_modes, get_td_waveform_modes)


@pytest.mark.parametrize('domain', ['FD', 'TD'])
@pytest.mark.parametrize('method', ['phase_only', 'stationary_phase'])
@pytest.mark.parametrize('model', [None, 'roemer', 'bhl', 'supersonic'])
def test_environmental_mode_plugins_reconstruct_full_waveform(domain, method, model):
    params = dict(approximant='pyEFPEHM', mass1=150, mass2=100,
                  eccentricity=0.1, spin1x=0.2, spin2y=0.15,
                  f_lower=0.09, f_ref=0.09, f_final=0.1,
                  coa_phase=0.4, mean_per_ano=0.3, inclination=0.9,
                  dquad_mon1=0, dquad_mon2=0, pn_amplitude_order=0,
                  SUA_kmax=3, environmental_model=model,
                  environmental_fd_method=method, environmental_phase_sign=-1,
                  environmental_redshift=0.2, gas_density=1e-8,
                  sound_speed=1e4, tertiary_mass=1e5, tertiary_distance=1e14)
    harmonics = [(2, 2, 2), (2, 2, 3)]
    native = native_parameters(params)
    native['harmonic_array'] = harmonics
    metadata = pyEFPEHM.pyEFPE(native)
    projector = metadata.Apc_projs[0]
    params['mode_array'] = harmonics
    if domain == 'FD':
        params['delta_f'] = 5e-4
        full = get_fd_waveform(**params)
        real, imag = get_fd_waveform_modes(**params)
        projector = projector.conj()
    else:
        params['delta_t'] = 4
        full = get_td_waveform(**params)
        real, imag = get_td_waveform_modes(**params)
    summed = np.zeros((2, len(full[0])), dtype=complex)
    for n in (2, 3):
        for column, mp in enumerate(range(-2, 3)):
            key = (2, mp, n, 2)
            mode = np.asarray(real[key])+1j*np.asarray(imag[key])
            summed += projector[column, :, None]*mode
    if domain == 'TD':
        summed = summed.real
    reference = np.array([np.asarray(full[0]), np.asarray(full[1])])
    assert np.linalg.norm(summed-reference)/np.linalg.norm(reference) < 1e-9
    assert metadata.q1 == 1 and metadata.q2 == 1
