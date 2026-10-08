"""
Random antisite-substitution structure generation (composition sweeps).

`generate_random_binary_structures` hits a list of target concentrations by
randomly substituting `main_element` <-> `mixing_element` atoms -- i.e.
random *antisite* defects, not a general random-alloy generator (that would
be its `'reshuffle'` mode, a full re-randomization of the whole structure at
fixed composition, which is not the default and not what most callers want).

Unlike the rest of `point_defects` (single, individually tracked defects via
`StructureContainer`), this generates one-shot structures at each target
concentration directly from a base structure, with no lineage tracking --
for building composition sweeps (e.g. Al-Mg antisite structures from 0 to
100% Mg) rather than single-defect formation-energy structures.

Carried over from an earlier notebook module as-is.
"""

import operator

import numpy as np
import pandas as pd

from .container import UID_KEY, StructureContainer
from .defects import create_substitution_batch


def get_element_fractions(atoms, element=None):
    """
    Calculate element-wise fractions from an ASE Atoms object.
    Returns a dict {element: fraction}
    """
    total = len(atoms)

    if element is not None:
        fraction = {element: atoms.symbols.count(element) / total}
        return fraction

    elements = set(atoms.get_chemical_symbols())
    fraction = {el: atoms.symbols.count(el) / total for el in elements}

    return fraction


def generate_random_binary_structures(
        base_structure,
        main_element='Al',
        mixing_element='Mg',
        phase_type='fcc',
        reference_phase='solid',
        concentrations=None,
        approximations=None,
        seed=None):
    """
    Adjusts the concentration of mixing_element to desired values by substitution.
    Initial structure may contain both elements already.
    """
    if concentrations is None:
        concentrations = [0, 0.5, 1]
    if approximations is None:
        approximations = ['antisites']
    else:
        approximations = list(approximations)  # avoid mutating the caller's list

    rng = np.random.default_rng(seed)
    n_sites = len(base_structure)
    structures = []
    rows = []

    if 'reshuffle' in approximations and 'antisites' not in approximations:
        approximations.append('antisites')

    for i, target_conc in enumerate(concentrations):
        atoms = base_structure.copy()
        symbols_orig = np.array(atoms.get_chemical_symbols(), dtype=object)

        n_curr_mix = np.count_nonzero(symbols_orig == mixing_element)
        curr_conc = n_curr_mix/len(symbols_orig)
        # print(f'Current concentration of the mixing element {mixing_element} is {curr_conc}')
        n_target_mix = int(round(target_conc * n_sites))
        delta_n = n_target_mix - n_curr_mix

        main_indices = np.where(symbols_orig == main_element)[0]
        mix_indices = np.where(symbols_orig == mixing_element)[0]

        approx = list(approximations)

        if 'antisites' in approximations:
            if delta_n > 0:
                # Need to substitute main_element → mixing_element
                if delta_n > len(main_indices):
                    raise ValueError(f"Cannot reach concentration {target_conc}: not enough {main_element} to replace.")
                replace_indices = rng.choice(main_indices, size=delta_n, replace=False)
                symbols_orig[replace_indices] = mixing_element
                approx = ['antisites']

            elif delta_n < 0:
                # Need to substitute mixing_element → main_element
                delta_n = abs(delta_n)
                if delta_n > len(mix_indices):
                    raise ValueError(f"Cannot reach concentration {target_conc}: not enough {mixing_element} to replace.")
                replace_indices = rng.choice(mix_indices, size=delta_n, replace=False)
                symbols_orig[replace_indices] = main_element
                approx = ['antisites']

            elif delta_n == 0:
                approx = ["stoichiometric"]

        final_seed = seed

        if 'reshuffle' in approximations:
            if i>0:
                # Full reshuffle: randomize assignment for target_conc
                seed_i = (seed if seed is not None else 0) + i
                rng = np.random.default_rng(seed_i)

                rng.shuffle(symbols_orig)

                approx = ['reshuffle', 'antisites']
                final_seed = seed_i

        atoms.set_chemical_symbols(symbols_orig.tolist())
        structures.append(atoms)
        # Optionally print summary per structure (remove/comment if not needed)
        # print(f"[{target_conc:.2f}] {mixing_element} count: {np.count_nonzero(symbols_orig==mixing_element)}")

        row = {
            'symbol' : atoms.get_chemical_formula(),
            'main_element' : main_element,
            'mixing_element' : mixing_element,
            'fractions' : get_element_fractions(atoms),
            'c' : get_element_fractions(atoms, element=mixing_element)[mixing_element],
            'c_in' : target_conc,
            'atoms' : atoms,
            'phase_type' : phase_type,
            'reference_phase' : reference_phase,
            'approximations': approx,
            'seed': final_seed
        }
        rows.append(row)

    structures_df = pd.DataFrame(rows)

    return structures_df


def generate_antisite_structures(
        base_structure,
        main_element='Al',
        mixing_element='Mg',
        phase_type='fcc',
        reference_phase='solid',
        concentrations=None,
        seed=None):
    """
    Random antisite structures at target concentrations of `mixing_element`, built with
    `create_substitution_batch`, each one on top of the previous one.

    The targets on either side of the base structure's concentration form a chain that starts at
    the base structure and moves outwards: the target closest to the base is made from the base
    structure, the next one from that structure, and so on. Every step randomly substitutes just the
    atoms that are missing: `main_element` -> `mixing_element` on the side above the base
    concentration, `mixing_element` -> `main_element` on the side below it. A structure therefore
    contains all antisites of the structures between it and the base, plus new ones, and
    neighbouring concentrations differ only by those. The base structure may already contain both
    elements (an intermetallic such as Mg17Al12 can be swept across its stoichiometric composition
    to the Al-rich and the Mg-rich side); a target that equals the base composition returns the
    base structure unchanged.

    The chain is defined by the set of targets, not by the order they are given in: the rows come
    back in the order of `concentrations`. A step to target concentration ``c`` is drawn with the
    seed ``seed + n``, where ``n = round(c * N)`` is its number of `mixing_element` atoms. A
    structure thus depends on the base structure, `seed` and the targets between the base and
    itself; adding a target changes the structures beyond it, not those before it. ``seed=None``
    draws a random seed; the seed that was used is recorded in the ``seed`` column.

    Parameters
    ----------
    base_structure : ase.Atoms
    main_element, mixing_element : str
        The two elements. Concentration is the fraction of `mixing_element` (``round(c * N)`` atoms).
    phase_type, reference_phase : str
        Stored in the result for later bookkeeping.
    concentrations : list of float, optional
        Target fractions of `mixing_element`; default ``[0, 0.5, 1]``.
    seed : int, optional

    Returns
    -------
    pandas.DataFrame
        One row per target concentration: ``symbol``, ``main_element``, ``mixing_element``,
        ``fractions``, ``c`` (achieved), ``c_in`` (requested), ``atoms``, ``phase_type``,
        ``reference_phase``, ``approximations`` (``['antisites']`` or ``['stoichiometric']``),
        ``seed`` (the seed passed in, or drawn), and ``substituted_indices`` (indices of the atoms
        whose element differs from the base structure).

    Raises
    ------
    ValueError
        If a target cannot be reached (outside [0, 1], or not enough atoms of the element that
        would have to be replaced), if the two elements are the same, or if `seed` is negative.
    TypeError
        If `seed` is not an integer.
    """
    if concentrations is None:
        concentrations = [0, 0.5, 1]
    if main_element == mixing_element:
        raise ValueError(f"main_element and mixing_element must differ (both are {main_element!r}).")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2 ** 32))
    else:
        seed = operator.index(seed)  # TypeError for anything that is not an integer (e.g. 6919.0)
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}.")

    n_sites = len(base_structure)
    base_symbols = np.array(base_structure.get_chemical_symbols(), dtype=object)
    n_mixing = int(np.count_nonzero(base_symbols == mixing_element))

    n_targets = [int(round(target_conc * n_sites)) for target_conc in concentrations]
    for target_conc, n_target in zip(concentrations, n_targets):
        if not 0 <= n_target <= n_sites:
            raise ValueError(f"Cannot reach concentration {target_conc}: outside 0 to 1.")

    container = StructureContainer()
    container.add_pristine(base_structure.copy())

    atoms_by_n_mixing = {n_mixing: base_structure.copy()}
    for on_upper_side in (True, False):
        side_targets = sorted(
            {n for n in n_targets if (n > n_mixing) == on_upper_side and n != n_mixing},
            key=lambda n: abs(n - n_mixing),
        )
        from_element, to_element = (main_element, mixing_element) if on_upper_side else (mixing_element, main_element)
        parent_index, parent_n = 0, n_mixing
        for n_target in side_targets:
            parent_symbols = np.array(container.get_structure(parent_index)["structure"].get_chemical_symbols(), dtype=object)
            if abs(n_target - parent_n) > np.count_nonzero(parent_symbols == from_element):
                target_conc = next(c for c, n in zip(concentrations, n_targets) if n == n_target)
                raise ValueError(f"Cannot reach concentration {target_conc}: not enough {from_element} to replace.")
            container = create_substitution_batch(
                container,
                target_indices=[parent_index],
                to_element=to_element,
                from_element=from_element,
                n=abs(n_target - parent_n),
                seed=seed + n_target,
            )
            parent_index, parent_n = len(container) - 1, n_target
            atoms = container.get_structure(parent_index)["structure"].copy()
            atoms.arrays.pop(UID_KEY, None)  # container bookkeeping, not part of the structure
            atoms_by_n_mixing[n_target] = atoms

    rows = []
    for target_conc, n_target in zip(concentrations, n_targets):
        atoms = atoms_by_n_mixing[n_target].copy()
        symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
        rows.append({
            'symbol': atoms.get_chemical_formula(),
            'main_element': main_element,
            'mixing_element': mixing_element,
            'fractions': get_element_fractions(atoms),
            'c': get_element_fractions(atoms, element=mixing_element)[mixing_element],
            'c_in': target_conc,
            'atoms': atoms,
            'phase_type': phase_type,
            'reference_phase': reference_phase,
            'approximations': ['stoichiometric'] if n_target == n_mixing else ['antisites'],
            'seed': seed,
            'substituted_indices': np.flatnonzero(symbols != base_symbols).tolist(),
        })

    return pd.DataFrame(rows)
