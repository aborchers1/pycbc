#!/usr/bin/env python3
"""Compare separately generated pyEFPEHM harmonics with a single PyCBC waveform call.

Use an environment with pyEFPEHM's FD and TD mode plugins installed in PyCBC.
The script compares both polarizations on the native grid of each domain.
pyEFPEHM supplies the projection coefficients for the mode arrays returned
by PyCBC.

Examples:
    python check_pycbc_mode_sum.py
    python check_pycbc_mode_sum.py --model roemer --precessing
    python check_pycbc_mode_sum.py --domain td --harmonic 2 2 2 --harmonic 2 2 3
"""  # describe what the test compares and how to run it

import argparse  # parse the binary, model, and sampling options
import sys  # return an exit status suitable for an automated test

import numpy as np  # project and compare the sampled waveforms
import pycbc  # identify the PyCBC installation that loads the plugins
import pyEFPEHM  # obtain the native polarization projector
from pycbc.pnutils import f_SchwarzISCO  # estimate the FD plugin's grid size
from pycbc.waveform import (  # import the public PyCBC waveform APIs
    fd_approximants,  # list summed FD waveform generators
    get_fd_waveform,  # generate the summed FD polarizations
    get_fd_waveform_modes,  # generate individual FD harmonics
    get_td_waveform,  # generate the summed TD polarizations
    get_td_waveform_modes,  # generate individual TD harmonics
    td_approximants,  # list summed TD waveform generators
)  # finish importing the waveform APIs
from pycbc.waveform.waveform_modes import (  # import the mode-plugin registries
    fd_waveform_mode_approximants,  # list registered FD mode generators
    td_waveform_mode_approximants,  # list registered TD mode generators
)  # finish importing the registry lookups
from pyEFPEHM.waveform.plugins import rename_parameters  # reuse the plugin's parameter translation


def parse_args():  # define the settings for a mode-sum comparison
    parser = argparse.ArgumentParser(description=__doc__,  # use the module description and examples
                                     formatter_class=argparse.RawDescriptionHelpFormatter)  # preserve the example layout
    parser.add_argument("--mass1", type=float, default=100.0)  # set the first detector-frame mass in solar masses
    parser.add_argument("--mass2", type=float, default=100.0)  # set the second detector-frame mass in solar masses
    parser.add_argument("--eccentricity", type=float, default=0.1)  # set eccentricity at the starting frequency
    parser.add_argument("--f-start", type=float, default=0.09)  # set the starting quadrupole frequency in Hz
    parser.add_argument("--f-end", type=float, default=0.1)  # set the ending quadrupole frequency in Hz
    parser.add_argument("--dt", type=float, default=2.0)  # set the time-domain cadence in seconds
    parser.add_argument("--df", type=float, default=2e-5)  # set the independent frequency-domain spacing in Hz
    parser.add_argument("--harmonic", action="append", nargs=3, type=int, metavar=("L", "M", "N"),  # accept one triple per flag
                        help="Repeat this option for each harmonic in the sum.")  # explain how to select several harmonics
    parser.add_argument("--model", default="all", choices=("all", "vacuum", "roemer", "bhl", "supersonic"))  # choose which corrections to check
    parser.add_argument("--domain", default="both", choices=("both", "fd", "td"))  # choose which PyCBC APIs to exercise
    parser.add_argument("--precessing", action="store_true")  # enable an example with transverse spins
    parser.add_argument("--tertiary-mass", type=float, default=1e5)  # set tertiary mass in solar masses
    parser.add_argument("--tertiary-distance", type=float, default=1e14)  # set tertiary distance in metres
    parser.add_argument("--bhl-density", type=float, default=4e-11)  # set BHL gas density in kilograms per cubic metre
    parser.add_argument("--supersonic-density", type=float, default=1e-4)  # set supersonic gas density in kilograms per cubic metre
    parser.add_argument("--sound-speed", type=float, default=1e4)  # set the BHL sound speed in metres per second
    parser.add_argument("--tolerance", type=float, default=1e-9)  # set the regression tolerance for numerical differences
    parser.add_argument("--max-samples", type=int, default=2_000_000)  # limit the number of samples per series before generation
    args = parser.parse_args()  # read options from the command line
    if not np.all(np.isfinite([args.mass1, args.mass2, args.f_start, args.f_end, args.dt, args.df, args.tolerance])):  # validate numeric controls
        parser.error("Masses, frequencies, sampling spacings, and tolerance must be finite.")  # report invalid numeric controls
    if min(args.mass1, args.mass2, args.f_start, args.dt, args.df, args.tolerance) <= 0 or args.max_samples < 1:  # require positive controls
        parser.error("Masses, start frequency, sampling spacings, tolerance, and sample limit must be positive.")  # report invalid bounds
    if args.f_end <= args.f_start or not 0 < args.eccentricity < 1:  # require an ordered band and eccentric harmonics with potential support
        parser.error("Require f-end > f-start and 0 < eccentricity < 1.")  # report the domain of the eccentric test
    harmonics = [tuple(key) for key in (args.harmonic or [(2, 2, 2), (2, 2, 3), (2, 0, 2)])]  # exercise two multipoles and two harmonic indices by default
    if len(harmonics) < 2 or len(set(harmonics)) != len(harmonics):  # require a sum over distinct harmonics
        parser.error("Select at least two distinct --harmonic L M N entries.")  # reject duplicates that the plugin would deduplicate
    if any((l, m) not in {(2, 0), (2, 2)} or n < 1 for l, m, n in harmonics):  # restrict the test to the positive-index 0PN harmonics
        parser.error("Use 0PN multipoles (2, 0) or (2, 2), with N >= 1.")  # report the allowed harmonic labels
    return args, harmonics  # return the options and normalized input types


def series_grid(series, domain):  # read a PyCBC series' uniform grid metadata
    if domain == "FD":  # handle a spectrum whose bins begin at zero frequency
        return len(series), float(series.delta_f), float(series.epoch)  # retain the bin spacing and start-time metadata
    return len(series), float(series.delta_t), float(series.start_time)  # retain the cadence and start time for a time series


def sum_harmonics(get_modes, parameters, harmonics, projectors, reference_grid, domain):  # accumulate independently requested harmonics
    dtype = np.complex128 if domain == "FD" else np.float64  # preserve complex FD and real TD polarizations
    summed = np.zeros((2, reference_grid[0]), dtype=dtype)  # allocate the sum on the full waveform's grid
    for l, m, n in harmonics:  # call PyCBC separately for each source harmonic
        real_modes, imag_modes = get_modes(**parameters, mode_array=[(l, m, n)])  # retrieve real and imaginary native components
        expected = {(l, mp, n, m) for mp in range(-l, l + 1)}  # include each inertial index for the chosen co-precessing harmonic
        if set(real_modes) != expected or set(imag_modes) != expected:  # check that the selected harmonic was returned in full
            raise ValueError(f"{domain} harmonic {(l, m, n)} returned keys {sorted(real_modes)}. Check its support on the requested grid.")  # identify mismatched mode labels
        projector = projectors[l, abs(m)]  # use the native projection for this multipole
        coefficients = projector.conj() if domain == "FD" else projector  # apply the domain-specific polarization convention
        peak = 0.0  # track whether the selected harmonic contributes a nonzero native mode
        for column, mp in enumerate(range(-l, l + 1)):  # associate each native mode with its projector row
            key = (l, mp, n, m)  # preserve the source index when multiple harmonics share an inertial mode
            real, imag = real_modes[key], imag_modes[key]  # read both components of this inertial contribution
            if series_grid(real, domain) != reference_grid or series_grid(imag, domain) != reference_grid:  # verify alignment before adding samples
                raise ValueError(f"{domain} mode {key} and the summed waveform have different grids.")  # stop on a length, spacing, or epoch mismatch
            mode = np.asarray(real) + 1j * np.asarray(imag)  # reconstruct the complex native mode
            peak = max(peak, np.max(np.abs(mode)))  # record the strongest inertial contribution
            projected = coefficients[column, :, None] * mode  # project onto plus and cross
            summed += projected if domain == "FD" else projected.real  # add the polarizations in their native domain
        if peak == 0:  # catch a zero-amplitude harmonic in the requested case
            raise ValueError(f"{domain} harmonic {(l, m, n)} is identically zero for these parameters.")  # prevent a vacuous mode-sum check
        del real_modes, imag_modes  # release this harmonic's mode series before requesting the next one
    return summed  # return the polarization sum for comparison


def main():  # compare the mode sum against the complete waveform for each model
    args, harmonics = parse_args()  # read the comparison settings
    parameters = dict(approximant="pyEFPEHM", mass1=args.mass1, mass2=args.mass2,  # select the registered waveform model and binary masses
                      eccentricity=args.eccentricity, distance=100.0, inclination=0.9,  # use the previous test's distance and inclination
                      f_lower=args.f_start, f_ref=args.f_start, f_final=args.f_end,  # pass the same frequency parameters to every PyCBC call
                      pn_amplitude_order=0, SUA_kmax=1, coa_phase=0.2)  # use 0PN amplitudes and the FD precession correction
    if args.precessing:  # exercise contributions from multiple inertial indices
        parameters.update(spin1x=0.2, spin1z=0.15, spin2y=0.15)  # add example misaligned spins
    native = parameters.copy()  # prepare to get the projector from the underlying model
    native.pop("approximant")  # remove the PyCBC registry name before constructing the model
    native["harmonic_array"] = harmonics  # request projectors for the full harmonic selection
    rename_parameters(native)  # match the plugin's phase and frequency conventions
    for name in ("f_lower", "f_ref", "f_final", "coa_phase"):  # remove aliases after translation
        native.pop(name)  # retain only parameter names read by pyEFPEHM
    metadata = pyEFPEHM.pyEFPE(native)  # build projector and duration metadata without generating samples
    projectors = dict(zip(map(tuple, metadata.mode_array), metadata.Apc_projs))  # index the native projectors by multipole
    domains = {  # associate each domain with its public PyCBC calls and sample spacing
        "FD": (get_fd_waveform, get_fd_waveform_modes, fd_approximants, fd_waveform_mode_approximants, {"delta_f": args.df}),  # compare spectra directly
        "TD": (get_td_waveform, get_td_waveform_modes, td_approximants, td_waveform_mode_approximants, {"delta_t": args.dt}),  # compare time series directly
    }  # finish defining the generation routes
    selected_domains = list(domains) if args.domain == "both" else [args.domain.upper()]  # resolve the requested run domains
    settings = {  # set explicit strengths so the environmental changes are measurable on this band
        "vacuum": {},  # use the plugin's vacuum default
        "roemer": dict(environmental_model="roemer", tertiary_mass=args.tertiary_mass, tertiary_distance=args.tertiary_distance),  # configure Roemer delay
        "bhl": dict(environmental_model="bhl", gas_density=args.bhl_density, sound_speed=args.sound_speed),  # configure BHL drag
        "supersonic": dict(environmental_model="supersonic", gas_density=args.supersonic_density),  # configure supersonic drag
    }  # finish defining the environmental runs
    models = ["vacuum"]  # use vacuum as the reference for any environmental run
    if args.model == "all":  # include every supported correction
        models.extend(("roemer", "bhl", "supersonic"))  # run the corrections independently
    elif args.model != "vacuum":  # handle one requested correction
        models.append(args.model)  # add the requested environmental run
    duration = metadata.return_end_time() - metadata.return_start_time()  # measure the TD integration span in seconds
    fd_count = int(np.floor(f_SchwarzISCO(args.mass1 + args.mass2) / args.df)) + 1  # count the FD plugin's bins through ISCO
    estimates = {"FD": fd_count, "TD": int(np.ceil(duration / args.dt))}  # estimate memory use in samples per series
    print("PyCBC source", pycbc.__file__)  # show which installation resolved the waveform calls
    print("pyEFPEHM source", pyEFPEHM.__file__)  # show which pyEFPEHM checkout supplies the implementations
    print(f"Harmonics {harmonics}. Quadrupole band {args.f_start:g}–{args.f_end:g} Hz.", flush=True)  # report the requested mode set and frequency bounds
    passed = True  # collect the results across models and domains
    for domain in selected_domains:  # keep all comparisons on the native grid for each domain
        get_full, get_modes, full_registry, mode_registry, sampling = domains[domain]  # select the public calls and spacing
        if "pyEFPEHM" not in full_registry() or "pyEFPEHM" not in mode_registry():  # verify both plugin entry points were loaded
            raise ValueError(f"PyCBC did not register both pyEFPEHM {domain} plugins in this environment.")  # identify the missing registration
        if estimates[domain] > args.max_samples:  # avoid large allocations before calling the mode plugin
            raise ValueError(f"{domain} needs about {estimates[domain]:,} samples per series. Adjust the sampling or --max-samples.")  # report the limiting grid size
        for model in models:  # compare the independent and simultaneous calls for this environment
            options = dict(parameters, **sampling, **settings[model])  # share all physical and sampling parameters between the two paths
            plus, cross = get_full(**options, mode_array=list(harmonics))  # request the complete mode set in one PyCBC waveform call
            grid = series_grid(plus, domain)  # record the target length, spacing, and epoch
            if series_grid(cross, domain) != grid:  # verify that the reference polarizations share the same grid
                raise ValueError(f"{domain} reference plus and cross use different grids.")  # reject a malformed reference
            reference = np.stack((np.asarray(plus), np.asarray(cross)))  # combine the full polarizations into one comparison array
            summed = sum_harmonics(get_modes, options, harmonics, projectors, grid, domain)  # assemble the same polarizations from independent calls
            if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(summed)):  # check that all samples are valid
                raise ValueError(f"{domain} {model} returned nonfinite samples.")  # report the affected run
            norm = np.linalg.norm(reference)  # get the full waveform norm over both polarizations
            if norm == 0:  # prevent a pass for a zero reference waveform
                raise ValueError(f"{domain} {model} reference waveform is zero.")  # report the empty reference
            relative_error = np.linalg.norm(summed - reference) / norm  # compare the complete arrays without shifting or fitting
            max_error = np.max(np.abs(summed - reference)) / np.max(np.abs(reference))  # detect a local mismatch as well as aggregate error
            case_passed = max(relative_error, max_error) <= args.tolerance  # require both error metrics to meet the tolerance
            print(f"{domain} {model:>10}  relative L2 {relative_error:.3e}  peak-normalized error {max_error:.3e}", end="")  # report sum agreement
            if model == "vacuum":  # save the baseline for the environmental effect check
                vacuum = reference  # keep the simultaneous vacuum waveform
                vacuum_grid = grid  # retain the grid for verifying comparability
            else:  # confirm that the model changed the waveform through PyCBC
                if grid != vacuum_grid:  # check that vacuum and environment share the same sample coordinates
                    raise ValueError(f"{domain} vacuum and {model} waveforms have different grids.")  # avoid comparing mismatched samples
                effect = np.linalg.norm(reference - vacuum) / np.linalg.norm(vacuum)  # measure the environmental change relative to vacuum
                case_passed = case_passed and effect > 1e-10  # fail if the selected environmental effect is numerically absent
                print(f"  change from vacuum {effect:.3e}", end="")  # report the size of the forwarded correction
            print("  PASS" if case_passed else "  FAIL", flush=True)  # show the result of this comparison
            passed = passed and case_passed  # retain any failure through the end of the script
    return 0 if passed else 1  # return success only if every requested comparison passes


if __name__ == "__main__":  # run the comparison when called as a script
    try:  # catch expected input or grid errors
        sys.exit(main())  # pass the test result to the shell
    except (ValueError, KeyError) as error:  # report a setup error separately from a numerical failure
        print(f"Comparison stopped: {error}", file=sys.stderr)  # show the reason the comparison could not run
        sys.exit(2)  # use a distinct status for an invalid test setup
