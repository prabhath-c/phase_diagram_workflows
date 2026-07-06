import numpy as np
import pandas as pd


def get_element_fractions(atoms, element=None):
    """
    Calculate element-wise fractions from an ASE Atoms object.
    Returns a dict {element: fraction}
    """
    total = len(atoms)

    if element is not None:
        fraction = {element: atoms.symbols.count(element)/total}
        return fraction

    elements = set(atoms.get_chemical_symbols())
    fraction = {el: atoms.symbols.count(el)/total for el in elements}

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
