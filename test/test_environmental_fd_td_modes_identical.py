#!/usr/bin/env python3
"""Compare PyCBC mode-plugin polarizations in the time domain.

Dependencies: numpy, scipy, PyCBC with FD and TD mode-plugin support,
and the pyEFPEHM branch with the environmental corrections and both mode
plugins installed. Both waveform arrays come from PyCBC. The pyEFPEHM
object supplies the orbital clock, stationary times and native projector.
The default 0.09–0.10 Hz band limits the FD allocation because the current
plugin allocates bins through ISCO. Use --target-phase 0 to keep the input
mass or density rather than rescaling it for the test.
Optional plots also need matplotlib.

Examples:
    python check_environmental_fd_td_pycbc.py --check
    python check_environmental_fd_td_pycbc.py --harmonic 2 2 3 --check
    python check_environmental_fd_td_pycbc.py --model roemer --plot comparison.png

The waveform tolerance defaults to 1% and the environmental-residual
tolerance to 5%. These are adjustable diagnostic choices, not universal
accuracy guarantees or noise-weighted detector mismatches.
The metrics use the central half of the trajectory and both polarizations.
No fitted time, phase, or amplitude alignment is performed. FD SPA/SUA and
the first-order TD phase mapping need not agree exactly.
"""  # describe the comparison, dependencies, and example commands
import argparse  # handle command-line arguments and help messages
import sys  # report errors to stderr and return process exit codes

import numpy as np  # use NumPy for arrays, waveform comparisons, and phase calculations
from scipy.fft import irfft, next_fast_len, rfftfreq  # import the real inverse FFT, its frequency grid, and efficient FFT sizes

import pycbc  # identify the PyCBC installation running the test
import pyEFPEHM  # access the orbital metadata and native polarization convention
from pycbc.pnutils import f_SchwarzISCO  # estimate the FD plugin's allocation
from pycbc.waveform import get_fd_waveform_modes, get_td_waveform_modes  # generate the mode arrays
from pycbc.waveform.waveform_modes import (  # inspect both installed mode registries
    fd_waveform_mode_approximants,  # list approximants registered for FD modes
    td_waveform_mode_approximants,  # list approximants registered for TD modes
)  # finish importing the registry lookups


def parse_args():  # define and validate the command-line options
    parser = argparse.ArgumentParser(description=__doc__,  # use the module docstring as the command-line description
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # keep the examples and line breaks in the help output
    parser.add_argument("--mass1", type=float, default=100)  # set the first component mass in solar masses
    parser.add_argument("--mass2", type=float, default=100)  # set the second component mass in solar masses
    parser.add_argument("--eccentricity", type=float, default=0.1)  # set the eccentricity at the reference frequency
    parser.add_argument("--f-start", type=float, default=0.09)  # set the starting quadrupole frequency in Hz
    parser.add_argument("--f-end", type=float, default=0.1)  # set the ending quadrupole frequency in Hz
    parser.add_argument("--dt", type=float, default=2)  # set the time sample spacing in seconds
    parser.add_argument("--harmonic", type=int, nargs=3, default=[2, 2, 2],  # select one harmonic using three integers
                        metavar=("L", "M", "N"))  # label the degree, azimuthal number, and eccentric harmonic index
    parser.add_argument("--model", choices=["all", "roemer", "bhl", "supersonic"],  # list the available environmental models
                        default="all")  # run all three models by default
    parser.add_argument("--redshift", type=float, default=0.0)  # set the redshift used by the environmental prescriptions
    parser.add_argument("--phase-sign", type=int, choices=[-1, 1], default=1)  # choose the sign of the phase correction
    parser.add_argument("--tertiary-mass", type=float, default=1.0)  # set the tertiary mass in solar masses
    parser.add_argument("--tertiary-distance", type=float, default=1e14)  # set the tertiary distance in metres
    parser.add_argument("--gas-density", type=float, default=1e-7)  # set the gas density in kilograms per cubic metre
    parser.add_argument("--sound-speed", type=float, default=1e4)  # set the sound speed in metres per second
    parser.add_argument("--target-phase", type=float, default=0.1,  # set the desired phase magnitude near the start, in radians
                        help="Near-initial harmonic FD phase magnitude; 0 disables rescaling.")  # document that zero keeps the supplied physical parameters
    parser.add_argument("--max-samples", type=int, default=2_000_000,  # cap the number of samples in the padded time grid
                        help="Refuse a larger FFT before allocating it.")  # explain that the limit is checked before FFT allocation
    parser.add_argument("--max-fd-bins", type=int, default=2_000_000,  # bound the plugin grid through ISCO
                        help="Limit the number of bins in each PyCBC FD mode.")  # explain the plugin allocation limit
    parser.add_argument("--check", action="store_true",  # enable threshold checks for automated runs
                        help="Exit 1 if the selected error thresholds are exceeded.")  # document the failure exit code
    parser.add_argument("--waveform-tolerance", type=float, default=0.01)  # set the relative error limit for the complete waveform
    parser.add_argument("--effect-tolerance", type=float, default=0.05)  # set the relative error limit for the change due to the environment
    parser.add_argument("--plot", metavar="PNG", help="Optionally save diagnostic plots.")  # accept an optional PNG filename for the plots
    args = parser.parse_args()  # parse the options supplied on the command line
    numbers = [args.mass1, args.mass2, args.f_start, args.f_end, args.dt,  # collect the quantities that must be positive and finite
               args.tertiary_mass, args.tertiary_distance, args.sound_speed,  # include the tertiary and gas parameters that must be positive
               args.waveform_tolerance, args.effect_tolerance]  # apply the same requirement to both error tolerances
    if not all(np.isfinite(x) and x > 0 for x in numbers):  # check every quantity in the list
        parser.error("Masses, frequencies, cadence, distances, sound speed and tolerances must be positive and finite.")  # report invalid positive-valued inputs through the argument parser
    if args.f_end <= args.f_start or not 0 <= args.eccentricity < 1:  # check the frequency ordering and the allowed eccentricity range
        parser.error("Require f-end > f-start and 0 <= eccentricity < 1.")  # report the required frequency ordering and eccentricity bounds
    if not np.isfinite(args.target_phase) or args.target_phase < 0:  # check that the phase target is finite and nonnegative
        parser.error("target-phase must be finite and nonnegative.")  # report an invalid phase target
    if not np.isfinite(args.gas_density) or args.gas_density <= 0:  # check that the gas density is finite and positive
        parser.error("gas-density must be positive and finite.")  # report an invalid gas density
    if not np.isfinite(args.redshift) or args.redshift < 0 or min(args.max_samples, args.max_fd_bins) < 16:  # validate the redshift and grid limits
        parser.error("Require finite nonnegative redshift and both grid limits >= 16.")  # report which bounds those two inputs must satisfy
    l, m, n = args.harmonic  # unpack the harmonic label into its three indices
    if [l, m] not in [[2, 0], [2, 2]] or n < 1:  # check which multipoles and harmonic indices this 0PN test supports
        parser.error("This 0PN diagnostic supports (l,m)=(2,0) or (2,2), with n >= 1.")  # report the allowed 0PN harmonic labels
    return args  # return the validated options


def taper(times, start, end):  # define the endpoint taper on the supplied time grid
    """Sin-squared ramps over each outer 15% of the vacuum trajectory."""  # describe the taper shape and width
    fraction = (np.asarray(times) - start) / (end - start)  # express each time as a fraction of the waveform duration
    rise = np.sin(0.5 * np.pi * np.clip(fraction / 0.15, 0, 1)) ** 2  # ramp up smoothly over the first 15 percent
    fall = np.sin(0.5 * np.pi * np.clip((1 - fraction) / 0.15, 0, 1)) ** 2  # ramp down smoothly over the last 15 percent
    return rise * fall  # combine both ramps into one window


def compare(candidate, reference, mask):  # compare two waveforms on the selected time samples
    a, b = candidate[:, mask].ravel(), reference[:, mask].ravel()  # flatten the selected samples across both polarizations
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)  # measure the amplitude of each waveform with its L2 norm
    if norm_a == 0 or norm_b == 0:  # check that both norms can serve as divisors
        raise ValueError("Zero waveform or environmental residual; comparison is undefined.")  # stop if a waveform has no signal in the comparison window
    error = np.linalg.norm(a - b) / norm_b  # normalize the difference by the reference waveform norm
    overlap = np.clip(np.dot(a, b) / (norm_a * norm_b), -1, 1)  # compute the normalized inner product and clip roundoff at its bounds
    return float(error), float(1 - overlap)  # return the relative L2 error and the unweighted mismatch



def project_modes(result, key, projector, spacing, domain, indices=slice(None)):  # project one harmonic's native plugin modes
    real_modes, imag_modes = result  # separate the two dictionaries returned by PyCBC
    degree, azimuth, harmonic = key  # unpack the input (l, m, n) selection
    labels = [(degree, mp, harmonic, azimuth) for mp in range(-degree, degree + 1)]  # use the plugin's (l, mprime, n, m) keys
    if set(real_modes) != set(labels) or set(imag_modes) != set(labels):  # check that every expected inertial contribution was returned
        raise ValueError(f"Unexpected {domain} mode keys for {key}: {list(real_modes)}.")  # report any unexpected selection
    first = real_modes[labels[0]]  # get a representative series for length and grid checks
    frequency_domain = domain == "FD"  # choose the native FD or TD projection convention
    spacing_name = "delta_f" if frequency_domain else "delta_t"  # find the sampling interval property for this domain
    epoch_name = "epoch" if frequency_domain else "start_time"  # find the series epoch property for this domain
    epoch = float(getattr(first, epoch_name))  # read the plugin's time reference
    for label in labels:  # validate both components of each inertial mode
        for series in (real_modes[label], imag_modes[label]):  # inspect the real and imaginary series
            if len(series) != len(first) or getattr(series, spacing_name) != spacing:  # require a common sampling grid
                raise ValueError(f"Inconsistent {domain} sampling grid for {label}.")  # identify the mismatched mode
            if float(getattr(series, epoch_name)) != epoch:  # require the same time reference across modes
                raise ValueError(f"Inconsistent {domain} epoch for {label}.")  # identify the mismatched mode
    if isinstance(indices, np.ndarray) and indices[-1] >= len(first):  # make sure the FD grid reaches the selected harmonic's support
        raise ValueError("The PyCBC FD grid stops inside the selected harmonic's band.")  # report an ISCO truncation
    coefficients = projector.conj() if frequency_domain else projector  # apply pyEFPEHM's own polarization convention
    projected = None  # allocate the polarization arrays on the first contribution
    for label, coefficient in zip(labels, coefficients):  # keep the inertial index and projector row in the same order
        real = np.asarray(real_modes[label])[indices]  # extract the real part on the requested bins
        imag = np.asarray(imag_modes[label])[indices]  # extract the matching imaginary part
        contribution = coefficient[:, None] * (real + 1j * imag)  # combine and project the native mode
        if projected is None:  # initialize the accumulation from the first contribution
            projected = contribution  # reuse this array for the polarization sum
        else:  # combine any remaining inertial contributions
            projected += contribution  # accumulate both polarizations in place
    return projected, epoch  # keep the complex projection for the phase comparison


def carrier_rates(waveform, times, key):  # measure the selected harmonic's instantaneous frequency and chirp rate
    _, azimuth, harmonic = key  # extract the coefficients of the two orbital phases
    rates = []  # hold the first and second time derivatives
    for derivative in (1, 2):  # inspect phase frequency and acceleration
        orbital, periastron = waveform.sol(times, derivative=derivative, idxs=[2, 3])  # sample the vacuum phase derivatives
        environmental = waveform._environmental_tdomain_orbital_phase(times, derivative=derivative)  # read the shared orbital correction
        rates.append(harmonic * (orbital + environmental) + (azimuth - harmonic) * periastron)  # apply each harmonic coefficient
    return rates  # return angular frequency and its time derivative


def environment(args, model):  # build model parameters and identify the parameter used to scale the correction
    if model == "roemer":  # handle the model driven by the tertiary mass and distance
        return dict(environmental_model=model, tertiary_mass=args.tertiary_mass,  # pass the requested tertiary mass to the Roemer model
                    tertiary_distance=args.tertiary_distance), "tertiary_mass"  # pass the distance and identify the mass as the adjustable strength
    options = dict(environmental_model=model, gas_density=args.gas_density)  # initialize the shared density input for the gas models
    if model == "bhl":  # check whether the model needs a sound speed
        options["sound_speed"] = args.sound_speed  # include the sound speed in the BHL parameters
    return options, "gas_density"  # return the gas parameters with density as the adjustable strength



def main():  # run the same comparison using both PyCBC mode entry points
    args = parse_args()  # read the binary, waveform, and comparison settings
    key = tuple(args.harmonic)  # store the requested harmonic as a dictionary key
    if ("pyEFPEHM" not in fd_waveform_mode_approximants()  # verify that the FD entry point is registered
            or "pyEFPEHM" not in td_waveform_mode_approximants()):  # verify that the TD entry point is registered
        raise ValueError("Install pyEFPEHM with both mode entry points in a PyCBC environment that loads FD and TD mode plugins.")  # point to the missing integration
    common = dict(mass1=args.mass1, mass2=args.mass2, eccentricity=args.eccentricity,  # share the binary inputs between both APIs
                  distance=100.0, inclination=0.9, pn_amplitude_order=0, SUA_kmax=0,  # retain the original waveform settings
                  environmental_redshift=args.redshift, environmental_phase_sign=args.phase_sign)  # share the environmental frame and sign
    orbit_parameters = dict(common, harmonic_array=[list(key)], f22_start=args.f_start,  # select one harmonic for the orbital metadata
                            f22_ref=args.f_start, f22_end=args.f_end, phase=0.0,  # match the plugin's frequency and phase inputs
                            mean_anomaly=0.0, q1=1.0, q2=1.0)  # align both models with explicit default orbital parameters
    plugin_parameters = dict(common, approximant="pyEFPEHM", mode_array=[key],  # select the same harmonic through PyCBC
                             f_lower=args.f_start, f_ref=args.f_start, f_final=args.f_end,  # map the frequency bounds to the plugin's input names
                             coa_phase=0.0, mean_per_ano=0.0, dquad_mon1=1.0, dquad_mon2=1.0)  # use a shared orbital phase and quadrupole convention
    print("PyCBC source", pycbc.__file__)  # identify the PyCBC checkout or installed package
    print("pyEFPEHM source", pyEFPEHM.__file__)  # identify the waveform implementation behind the plugin
    print("PyCBC inputs", plugin_parameters, flush=True)  # show the parameters passed through the mode API
    vacuum = pyEFPEHM.pyEFPE(orbit_parameters)  # get the vacuum orbital clock and projection metadata
    start, end = vacuum.return_start_time(), vacuum.return_end_time()  # read the physical bounds of the comparison
    duration = end - start  # measure the length of the vacuum trajectory
    requested_count = int(np.ceil(1.5 * duration / args.dt))  # reserve 50 percent additional time for FFT padding
    if requested_count > args.max_samples:  # enforce the time-domain allocation limit
        raise ValueError(f"Requested {requested_count:,} time samples. Narrow the band or adjust --dt / --max-samples.")  # report how to fit the FFT
    count = 2 * next_fast_len(max(8, (requested_count + 1) // 2), real=True)  # choose an efficient even FFT size
    if count > args.max_samples:  # check the rounded size as well
        raise ValueError(f"Rounded FFT size {count:,} exceeds --max-samples.")  # identify the allocation that would exceed the limit
    df = 1.0 / (count * args.dt)  # use the PyCBC bin spacing implied by the padded FFT
    isco = float(f_SchwarzISCO(args.mass1 + args.mass2))  # reproduce the FD plugin's grid ceiling
    fd_bins = int(np.floor(isco / df)) + 1  # estimate the FD series length including DC
    if fd_bins > args.max_fd_bins:  # stop before the plugin can allocate the extended spectrum
        raise ValueError(  # explain the separate FD allocation limit
            f"The FD plugin allocates {fd_bins:,} bins per series through ISCO, "  # report the estimated plugin allocation
            f"above --max-fd-bins {args.max_fd_bins:,}. Choose a shorter frequency "  # suggest a smaller trajectory
            "interval, or raise the limit if your machine has enough memory.")  # leave memory-intensive runs available explicitly
    probe_times = np.linspace(start, end, 1025)  # inspect the vacuum carrier across the model trajectory
    vacuum_omega, _ = carrier_rates(vacuum, probe_times, key)  # determine the harmonic's actual frequency range
    probed_hz = vacuum_omega / (2 * np.pi)  # convert angular frequency to Hz
    if probed_hz[0] < args.f_start - 1e-8 * args.f_start or max(probed_hz) > isco:  # require that PyCBC covers the complete harmonic band
        raise ValueError("The selected harmonic extends outside the FD plugin's f_lower-to-ISCO grid.")  # explain why the two waveforms could not share this support
    freqs = rfftfreq(count, args.dt)  # construct the one-sided spectrum coordinates for the inverse FFT
    bins, _, stationary_times, _, _ = vacuum.stationary_times(freqs[1:-1])  # get the vacuum stationary-time metadata for tapering
    bins = bins + 1  # restore the index offset from omitting DC
    if len(bins) < 32 or len(np.unique(bins)) != len(bins):  # require a sufficiently resolved single stationary branch
        raise ValueError("Expected at least 32 bins and one stationary time per bin.")  # identify incompatible stationary-time metadata
    if bins[0] < int(np.ceil(args.f_start / df)) or bins[-1] >= fd_bins:  # check the exact discrete support against PyCBC's grid limits
        raise ValueError("The selected harmonic's bins extend outside the PyCBC FD grid.")  # report a clipped frequency interval
    pad = int(np.floor(0.25 * duration / args.dt))  # reserve a whole number of leading samples
    origin = start - pad * args.dt  # align the padded grid with the plugin's TD epoch
    times = origin + np.arange(count) * args.dt  # construct the shared time grid
    left, right = start + 0.25 * duration, end - 0.25 * duration  # define the central half of the trajectory
    interior = (times >= left) & (times <= right)  # restrict metrics to the region away from endpoint tapers
    phase_mask = (stationary_times >= left) & (stationary_times <= right)  # choose the corresponding FD bins for the phase diagnostic
    if np.count_nonzero(interior) < 64 or not np.any(phase_mask):  # require enough TD samples and overlapping FD metadata
        raise ValueError("Too few interior samples or stationary times for comparison.")  # report the sampling limitation
    projector = vacuum.Apc_projs[0]  # reuse the selected multipole's native plus/cross projector
    taper_fd = taper(stationary_times, start, end)  # taper the FD amplitudes at their stationary times
    fft_shift = np.exp(2j * np.pi * freqs[bins] * origin)  # shift the absolute-time FD phase to the padded FFT origin
    reference_frequency = np.array([probed_hz[1]])  # choose an interior frequency for phase-strength normalization
    reference_phase = vacuum.stationary_times(reference_frequency)[3][0]  # read the vacuum reference phase without generating modes
    print(f"FFT samples {count:,}. Estimated plugin FD bins {fd_bins:,} per series.", flush=True)  # report both allocations
    print("Metrics use both polarizations and the central 50% of the waveform.")  # describe the shared comparison region

    def evaluate(waveform, options):  # generate and compare one model on the common grids
        omega, chirp_rate = carrier_rates(waveform, probe_times, key)  # check the model's phase derivatives before generating arrays
        if np.min(omega) <= 0 or np.min(chirp_rate) <= 0:  # require a positive increasing chirp for the SPA comparison
            raise ValueError("The selected harmonic is not a positive increasing chirp.")  # stop when the comparison assumptions fail
        if max(omega) / (2 * np.pi) >= 0.8 * freqs[-1]:  # keep the harmonic below the Nyquist margin
            raise ValueError("Harmonic frequency is too close to Nyquist. Decrease --dt.")  # request finer sampling when necessary
        td_result = get_td_waveform_modes(**plugin_parameters, delta_t=args.dt, **options)  # generate the actual TD modes via PyCBC
        td, td_epoch = project_modes(td_result, key, projector, args.dt, "TD")  # obtain complex projected TD polarizations
        del td_result  # release the per-mode TD arrays before allocating the FD series
        fd_result = get_fd_waveform_modes(**plugin_parameters, delta_f=df, **options)  # generate the actual FD modes via PyCBC
        fd, fd_epoch = project_modes(fd_result, key, projector, df, "FD", bins)  # project the selected FD bins
        del fd_result  # release the extended PyCBC spectra
        if not np.allclose([td_epoch, fd_epoch], start, rtol=0, atol=1e-8):  # check that both plugin epochs match the orbital metadata
            raise ValueError("Plugin epochs differ from the model start time.")  # report an inconsistent time origin
        stop = pad + td.shape[1]  # locate where the plugin TD samples end on the padded grid
        if stop > count or abs(td.shape[1] * args.dt - duration) > 1.000001 * args.dt:  # check that the uniform TD series covers the expected trajectory
            raise ValueError("The PyCBC TD waveform length differs from the model duration.")  # report an inconsistent TD grid
        direct = np.zeros((2, count))  # allocate real plus and cross time series with zero padding
        direct[:, pad:stop] = td.real * taper(times[pad:stop], start, end)  # taper the real TD polarizations on their sample times
        spectrum = np.zeros((2, len(freqs)), dtype=complex)  # allocate a Hermitian-compatible one-sided polarization spectrum
        spectrum[:, bins] = fd * taper_fd * fft_shift  # taper the projected modes and apply the FFT time-origin shift
        inverse = irfft(spectrum, n=count, axis=-1) / args.dt  # invert and convert continuous Fourier amplitudes to strain
        if not np.all(np.isfinite(direct)) or not np.all(np.isfinite(inverse)):  # validate the arrays before measuring agreement
            raise ValueError("The plugin returned nonfinite waveform samples.")  # report invalid waveform values
        return direct, inverse, fd  # retain the actual FD plugin polarizations for phase diagnostics

    direct_vacuum, inverse_vacuum, vacuum_fd = evaluate(vacuum, {})  # compute both vacuum waveforms through the plugin
    vacuum_error, vacuum_mismatch = compare(inverse_vacuum, direct_vacuum, interior)  # measure the vacuum baseline
    print(f"vacuum: relative L2 {vacuum_error:.6g}; unweighted mismatch {vacuum_mismatch:.6g}",  # show the FFT comparison metrics
          flush=True)  # report the vacuum result before starting the environmental calculations
    passed = vacuum_error <= args.waveform_tolerance  # initialize the threshold status from the baseline
    models = ["roemer", "bhl", "supersonic"] if args.model == "all" else [args.model]  # select which environmental corrections to run
    plot_data = []  # prepare storage for the optional plot traces
    for model in models:  # run one model at a time to limit memory
        options, strength_key = environment(args, model)  # get this model's PyCBC keyword arguments
        waveform = pyEFPEHM.pyEFPE({**orbit_parameters, **options})  # obtain the environmental phase metadata
        if args.target_phase:  # optionally choose a measurable correction strength
            near_initial_phase = waveform.stationary_times(reference_frequency)[3][0] - reference_phase  # measure the applied FD correction at the reference bin
            if not np.isfinite(near_initial_phase) or near_initial_phase == 0:  # check that the rescaling denominator is valid
                raise ValueError(f"Cannot normalize the {model} phase.")  # report an unusable reference correction
            options[strength_key] *= args.target_phase / abs(near_initial_phase)  # rescale mass or density to the desired phase magnitude
            waveform = pyEFPEHM.pyEFPE({**orbit_parameters, **options})  # keep metadata consistent with the plugin inputs
        direct, inverse, fd = evaluate(waveform, options)  # compute both environmental waveforms via PyCBC
        fd_rotation = np.sum(fd[:, phase_mask] * vacuum_fd[:, phase_mask].conj(), axis=0)  # measure the plugin's phase rotation using both polarizations
        fd_phase = np.angle(fd_rotation)  # extract the phase shift with the public FD Fourier sign
        td_phase = key[2] * waveform._environmental_tdomain_orbital_phase(stationary_times[phase_mask])  # evaluate the TD prescription at the same emission epochs
        phase_norm = np.linalg.norm(np.angle(np.exp(1j * td_phase)))  # use the TD correction magnitude on the principal phase interval
        if phase_norm == 0 or np.any(fd_rotation == 0):  # require a measurable phase and a nonzero FD reference
            raise ValueError(f"Cannot measure the {model} phase on the selected interior bins.")  # explain why the phase comparison is undefined
        prescription_error = np.linalg.norm(np.angle(np.exp(1j * (fd_phase - td_phase)))) / phase_norm  # compare phase rotations modulo one cycle
        if not np.isfinite(prescription_error):  # validate the phase diagnostic before reporting it
            raise ValueError("Nonfinite phase-prescription discrepancy.")  # stop if the phase comparison is invalid
        error, mismatch = compare(inverse, direct, interior)  # compare the complete environmental waveforms
        delta_direct = direct - direct_vacuum  # subtract vacuum from the direct TD waveform
        delta_inverse = inverse - inverse_vacuum  # subtract vacuum after both FD waveforms have been inverse-transformed
        effect_size = np.linalg.norm(delta_direct[:, interior]) / np.linalg.norm(  # measure the TD environmental change as a fraction of the vacuum norm
            direct_vacuum[:, interior])  # normalize using the same interior time samples
        if effect_size < 1e-8:  # check that the environmental change is large enough to measure
            raise ValueError(f"{model} waveform change {effect_size:.3g} is too small to test.")  # report the change that fell below the diagnostic floor
        effect_error, _ = compare(delta_inverse, delta_direct, interior)  # compare the environmental residuals in the two domains
        print(f"{model}: {options}")  # print the model parameters actually used
        print(f"  relative phase-prescription discrepancy {prescription_error:.6g}")  # report the relative disagreement between the phase prescriptions
        print(f"  relative L2 {error:.6g}; unweighted mismatch {mismatch:.6g}")  # report the complete waveform error and mismatch
        print(f"  TD change/vacuum {effect_size:.6g}; relative environmental residual error {effect_error:.6g}",  # print the size of the TD change and the error between residuals
              flush=True)  # flush the model metrics before moving to the next run
        passed = passed and error <= args.waveform_tolerance and effect_error <= args.effect_tolerance  # require both waveform and residual errors to meet their limits
        if args.plot:  # keep plot traces only when a plot was requested
            plot_data.append((model, direct[0, interior], inverse[0, interior],  # save the interior plus-polarization waveforms for this model
                              delta_direct[0, interior], delta_inverse[0, interior]))  # include the corresponding changes relative to vacuum

    if args.plot:  # generate a plot if an output filename was supplied
        import matplotlib  # load Matplotlib only for the optional plot
        matplotlib.use("Agg")  # select the backend that renders without a display
        import matplotlib.pyplot as plt  # import plotting functions after choosing the backend
        fig, axes = plt.subplots(len(plot_data), 2, figsize=(12, 3 * len(plot_data)),  # allocate two panels per model to show the waveform and residual
                                 squeeze=False)  # keep the axes in a two-dimensional array for a single model
        elapsed = times[interior] - start  # express the plotted times in seconds since the trajectory start
        visible = elapsed <= elapsed[0] + min(1000.0, 0.5 * duration)  # show at most the first 1000 seconds of the interior comparison window
        for row, (model, direct, inverse, delta_direct, delta_inverse) in enumerate(plot_data):  # assign each environmental model to a plot row
            for column, (a, b, title) in enumerate([  # pair each panel's direct and inverse-transformed traces with a title
                (direct, inverse, model),  # use the left panel for the complete waveforms
                (delta_direct, delta_inverse, model + " change relative to vacuum"),  # use the right panel for the changes relative to vacuum
            ]):  # finish defining the two panels for this model
                ax = axes[row, column]  # select the axes for the current panel
                ax.plot(elapsed[visible], a[visible], label="PyCBC TD")  # plot the direct time-domain result
                ax.plot(elapsed[visible], b[visible], "--", label="Inverse FFT of PyCBC FD")  # overlay the inverse-transformed frequency-domain result with a dashed line
                ax.set(xlabel="Seconds since trajectory start", ylabel="Plus strain", title=title)  # label the time axis, strain axis, and panel
                ax.legend()  # identify the two traces in a legend
        fig.tight_layout()  # adjust spacing so labels fit
        fig.savefig(args.plot, dpi=160)  # write the diagnostic figure to the requested path
        plt.close(fig)  # close the figure to release its resources
        print("Saved plot", args.plot)  # confirm where the image was saved
    if args.check:  # report pass or fail when threshold checks are enabled
        print("PASS" if passed else "FAIL", "using the supplied diagnostic tolerances.")  # show the result of the accumulated checks
        return 0 if passed else 1  # return exit code 1 if any threshold was exceeded
    print("Diagnostic complete. Use --check to enforce error thresholds.")  # explain how to enable threshold checks for a later run
    return 0  # return success after a completed diagnostic-only run


if __name__ == "__main__":  # run the entry point when launched as a script
    try:  # catch expected input and waveform selection errors
        sys.exit(main())  # exit using the comparison outcome
    except (ValueError, KeyError) as error:  # intercept errors with a direct explanation for the caller
        print(f"Comparison stopped: {error}", file=sys.stderr)  # print the reason for stopping to stderr
        sys.exit(2)  # return exit code 2 when the comparison cannot run
