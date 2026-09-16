"""
Interstitial site discovery and symmetry-unique sublattice discovery.

Two independent things live here:

- Interstitial site finders (`get_voronoi_interstitial_sites`,
  `get_delaunay_interstitial_sites`): given a host structure, return
  candidate void positions to insert an interstitial atom at. Both are
  scipy-based (Voronoi tessellation / Delaunay circumcenters) with no
  optional dependencies.
- Sublattice discovery (`discover_atomic_sublattices`,
  `discover_interstitial_sublattices`, plus the `tile_*`/`validate_*`
  helpers): groups atom sites (or discovered void sites) into
  symmetry-unique orbits via spglib, for picking one representative site per
  orbit rather than enumerating every symmetry-equivalent atom/void.

Both feed `phase_diagram_workflows.structures.point_defects.defects`:
site-finder output (`sublattice` arrays) is the input to
`create_interstitial`/`create_interstitial_batch`, and sublattice orbits are
typically discovered on a small unit cell and tiled (`tile_atomic_sublattices`,
`tile_interstitial_sublattices`) into the production supercell's index space.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay, Voronoi


# ============================================================================
# Interstitial site finding
# ============================================================================


def _voronoi_or_delaunay_sites(
    atoms: Atoms,
    primitive_atoms: Optional[Atoms],
    repeat: Optional[tuple],
    r_min: float,
    cluster_tol: float,
    n_images: int,
    tessellate,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared candidate-generation/filter/cluster pipeline for the scipy-based site finders."""
    if (primitive_atoms is None) != (repeat is None):
        raise ValueError("primitive_atoms and repeat must be provided together or not at all.")

    use_primitive = primitive_atoms is not None
    work_atoms = primitive_atoms if use_primitive else atoms

    cell = work_atoms.get_cell().array  # (3, 3)
    cell_inv = np.linalg.inv(cell)
    pos = work_atoms.get_positions()  # (N, 3)

    offsets = np.array(
        [
            [i, j, k]
            for i in range(-n_images, n_images + 1)
            for j in range(-n_images, n_images + 1)
            for k in range(-n_images, n_images + 1)
        ],
        dtype=float,
    )
    pos_ext = np.vstack([pos + off @ cell for off in offsets])  # (N*(2n+1)^3, 3)

    candidates_all = tessellate(pos_ext)

    # keep only candidates inside the primitive unit cell
    frac = candidates_all @ cell_inv
    inside = np.all((frac >= -1e-8) & (frac < 1 - 1e-8), axis=1)
    candidates = candidates_all[inside]

    if len(candidates) == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

    # filter: too close to any host atom (minimum image convention)
    diff = candidates[:, None, :] - pos[None, :, :]  # (C, N, 3)
    diff_frac = diff @ cell_inv
    diff_frac -= np.round(diff_frac)
    diff_cart = diff_frac @ cell
    min_dist = np.linalg.norm(diff_cart, axis=-1).min(axis=1)  # (C,)
    candidates = candidates[min_dist >= r_min]

    if len(candidates) == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, empty

    # scipy.spatial's Qhull-based tessellations don't guarantee a stable vertex
    # order across separate process runs for degenerate/highly symmetric point
    # sets -- impose a canonical order here so clustering (and anything
    # downstream, e.g. content-hash-based caching) is reproducible regardless
    # of Qhull's internal ordering. Collapse exact ties first (order-independent
    # mean) so np.unique's lexicographic order has no ties left to break
    # unpredictably.
    round_key = np.round(candidates, 6)
    _, inverse = np.unique(round_key, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_groups = inverse.max() + 1 if len(inverse) else 0
    candidates = np.array([candidates[inverse == g].mean(axis=0) for g in range(n_groups)], dtype=float)

    # cluster: merge candidates within cluster_tol
    used = np.zeros(len(candidates), dtype=bool)
    clusters = []
    for i in range(len(candidates)):
        if used[i]:
            continue
        d = np.linalg.norm(candidates - candidates[i], axis=1)
        mask = d < cluster_tol
        used[mask] = True
        clusters.append(candidates[mask].mean(axis=0))

    all_sites_prim = np.array(clusters, dtype=float)  # void centers in primitive cell
    prim_key = np.round(all_sites_prim, 6)
    _, prim_inverse = np.unique(prim_key, axis=0, return_inverse=True)
    prim_inverse = prim_inverse.ravel()
    n_prim_groups = prim_inverse.max() + 1 if len(prim_inverse) else 0
    all_sites_prim = np.array(
        [all_sites_prim[prim_inverse == g].mean(axis=0) for g in range(n_prim_groups)], dtype=float
    )
    unique_sites = all_sites_prim.copy()

    if use_primitive:
        n1, n2, n3 = int(repeat[0]), int(repeat[1]), int(repeat[2])
        tiles = []
        for f in all_sites_prim:
            for i in range(n1):
                for j in range(n2):
                    for k in range(n3):
                        tiles.append(f + np.array([i, j, k], dtype=float) @ cell)
        all_sites = np.array(tiles, dtype=float)
    else:
        all_sites = all_sites_prim

    return unique_sites, all_sites


def get_voronoi_interstitial_sites(
    atoms: Atoms,
    primitive_atoms: Optional[Atoms] = None,
    repeat: Optional[tuple] = None,
    r_min: float = 0.8,
    cluster_tol: float = 0.5,
    n_images: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find Voronoi interstitial sites via scipy -- fast, no pymatgen dependency.

    Voronoi vertices of the atom positions (plus periodic image shells) are
    exactly the void centers of the tessellation. No symmetry analysis is
    performed; all geometric void centers are returned directly. Drop-in
    replaceable with `get_delaunay_interstitial_sites`::

        unique_sites, all_sites = get_voronoi_interstitial_sites(
            atoms, primitive_atoms=prim, repeat=(3, 3, 3))

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object (host supercell). Used only for validation when
        `primitive_atoms`/`repeat` are provided; otherwise tessellation runs
        directly on this structure.
    primitive_atoms : Atoms or None
        Primitive unit cell. When given together with `repeat`, tessellation
        runs on the primitive cell and results are tiled.
    repeat : tuple of int or None
        `(n1, n2, n3)` such that `primitive_atoms.repeat(repeat)` reproduces `atoms`.
    r_min : float
        Minimum distance (angstrom) a void candidate must keep from every host atom.
    cluster_tol : float
        Candidates closer than this (angstrom) are merged into one (centroid);
        removes near-duplicate Voronoi vertices on or near cell boundaries.
    n_images : int
        Number of periodic image shells to include before tessellation.

    Returns
    -------
    unique_sites : (N, 3) ndarray
        One representative per cluster, Cartesian coordinates (angstrom).
    all_sites : (M, 3) ndarray
        All deduplicated void centers, Cartesian coordinates (angstrom), replicated
        across the supercell when `repeat` is given.

    See Also
    --------
    get_delaunay_interstitial_sites : Equivalent via Delaunay circumcenters.
    """

    def tessellate(pos_ext):
        return Voronoi(pos_ext).vertices

    return _voronoi_or_delaunay_sites(atoms, primitive_atoms, repeat, r_min, cluster_tol, n_images, tessellate)


def get_delaunay_interstitial_sites(
    atoms: Atoms,
    primitive_atoms: Optional[Atoms] = None,
    repeat: Optional[tuple] = None,
    r_min: float = 0.8,
    cluster_tol: float = 0.5,
    n_images: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find interstitial sites via Delaunay circumcenters -- faster than Voronoi
    tessellation for large or complex primitive cells.

    Circumcenters of Delaunay tetrahedra are natural void centers. No
    pymatgen dependency; uses only scipy and numpy. Drop-in replaceable with
    `get_voronoi_interstitial_sites` (same parameters and return shape).

    See Also
    --------
    get_voronoi_interstitial_sites : Scipy-based Voronoi alternative (same speed, no pymatgen).
    """

    def tessellate(pos_ext):
        tri = Delaunay(pos_ext)
        verts = pos_ext[tri.simplices]  # (T, 4, 3)
        A = verts[:, 1:] - verts[:, :1]  # (T, 3, 3), edge vectors from v0
        b = 0.5 * (A**2).sum(axis=-1)  # (T, 3)

        # Drop degenerate tetrahedra (coplanar vertices -> singular A) before the
        # batched solve; det near 0 identifies them cheaply.
        At = A.swapaxes(1, 2)  # (T, 3, 3)
        nondegenerate = np.abs(np.linalg.det(At)) > 1e-10
        At_nd = At[nondegenerate]
        b_nd = b[nondegenerate]
        v0_nd = verts[nondegenerate, 0]

        x = np.linalg.solve(At_nd, b_nd[..., None])[..., 0]  # (T', 3), b needs shape (..., m, 1)
        return x + v0_nd

    return _voronoi_or_delaunay_sites(atoms, primitive_atoms, repeat, r_min, cluster_tol, n_images, tessellate)


# ============================================================================
# Sublattice discovery (multiplicity-aware, aliasing-safe)
# ============================================================================


def discover_atomic_sublattices(atoms: Atoms, symprec: float = 1e-3, formula_units: int = 1) -> List[dict]:
    """
    Discover symmetry-unique atomic sublattices (vacancy/substitution
    candidate sites) directly on the given structure.

    Groups atoms by spglib's `equivalent_atoms` id -- the ground truth for
    which atoms are truly related by a symmetry operation of `atoms` --
    rather than by the bare `(species, wyckoff_letter)` string. spglib
    legitimately reuses the same generic Wyckoff letter for multiple
    physically distinct orbits that share a site-symmetry class and
    multiplicity; deduplicating on the letter alone silently merges those
    distinct orbits into one and drops every site but the last one seen.
    Label collisions are instead resolved with a numeric `_1, _2, ...` suffix,
    mirroring the convention used by `discover_interstitial_sublattices`.

    Run this on the actual structure defects will be placed on (typically the
    relaxed host), not an idealized/unrelaxed reference cell -- for
    lower-symmetry relaxed structures the idealized cell's Wyckoff positions
    do not represent the real orbits of the structure being simulated.

    Parameters
    ----------
    atoms : Atoms
        Host structure.
    symprec : float
        Symmetry tolerance for spglib.
    formula_units : int
        Number of primitive/conventional-cell repeats baked into `atoms`
        (e.g. `N_REPEAT**3` for a supercell built via
        `primitive.repeat(N_REPEAT)`). When > 1, the human-readable label
        uses the textbook per-formula-unit multiplicity
        (`multiplicity // formula_units`, e.g. `Al_4a`) while the returned
        `multiplicity` field still reports the true count in `atoms` (e.g.
        108) -- so labels stay stable across supercell-size choices without
        changing anything a sum-check consumes. Leave at the default (1) for
        structures that are already their own complete conventional cell.
        Raises `ValueError` if an orbit's multiplicity isn't an exact
        multiple of `formula_units` (a sign the supplied value is wrong).

    Returns
    -------
    list of dict, one per symmetry-unique orbit, with keys:
        `label` (e.g. "Al_4a"; suffixed with `_1, _2, ...` on collision),
        `species`, `atom_indices` (into `atoms`), `multiplicity` (true count),
        `label_multiplicity` (per-formula-unit count used in `label`),
        `wyckoff_letter`, `eq_id` (spglib's raw `equivalent_atoms` id).

    See Also
    --------
    validate_sublattice_coverage : Check that multiplicities sum to len(atoms).
    discover_interstitial_sublattices : Analogous discovery for voids.
    """
    import spglib

    cell_tuple = (atoms.get_cell()[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    ds = spglib.get_symmetry_dataset(cell_tuple, symprec=symprec)
    eq = np.asarray(ds.equivalent_atoms)
    syms = atoms.get_chemical_symbols()

    orbits = []
    for eq_id in np.unique(eq):
        atom_indices = np.flatnonzero(eq == eq_id).tolist()
        species = syms[atom_indices[0]]
        letter = ds.wyckoffs[atom_indices[0]]
        mult = len(atom_indices)
        if mult % formula_units != 0:
            raise ValueError(
                f"Orbit {species}_{mult}{letter} (eq_id={eq_id}) has multiplicity {mult}, not an "
                f"exact multiple of formula_units={formula_units}. formula_units is likely wrong "
                f"for this structure."
            )
        label_mult = mult // formula_units
        orbits.append(
            {
                "species": species,
                "atom_indices": atom_indices,
                "multiplicity": mult,
                "label_multiplicity": label_mult,
                "wyckoff_letter": letter,
                "eq_id": int(eq_id),
                "base_label": f"{species}_{label_mult}{letter}",
            }
        )

    _assign_collision_safe_labels(orbits, key=lambda o: (o["base_label"], o["eq_id"]))
    return orbits


def discover_interstitial_sublattices(
    atoms: Atoms,
    r_min: float = 0.8,
    cluster_tol: float = 0.5,
    n_images: int = 1,
    symprec: float = 1e-3,
    formula_units: int = 1,
) -> List[dict]:
    """
    Discover symmetry-unique interstitial void sublattices directly on the
    given structure.

    Raw geometric void centers come from `get_voronoi_interstitial_sites`
    (deterministic across runs). They are grouped into orbits under the full
    host space group -- not just geometric proximity -- and each orbit's
    Wyckoff label is read by inserting *all* orbit members simultaneously as
    dummy `Ne` atoms: inserting members one at a time breaks the host
    symmetry and yields a wrong, spuriously-low-multiplicity label. Label
    collisions across distinct orbits are resolved the same way as in
    `discover_atomic_sublattices`.

    Parameters
    ----------
    atoms : Atoms
        Host structure (should be the same structure passed to
        `discover_atomic_sublattices`).
    r_min, cluster_tol, n_images : float, float, int
        Forwarded to `get_voronoi_interstitial_sites`.
    symprec : float
        Symmetry tolerance for spglib.
    formula_units : int
        Same meaning and default as in `discover_atomic_sublattices`.

    Returns
    -------
    list of dict, one per symmetry-unique void orbit, with keys:
        `label` (e.g. "int_4b"; suffixed on collision), `cart_positions`
        ((M, 3) ndarray, every symmetry-equivalent void position, Cartesian
        angstrom), `multiplicity` (true count), `label_multiplicity`, `wyckoff_letter`.

    See Also
    --------
    discover_atomic_sublattices : Analogous discovery for host atom sites.
    """
    import spglib

    cell = np.array(atoms.get_cell())
    inv = np.linalg.inv(cell)

    def cell_tuple(a):
        return (a.get_cell()[:], a.get_scaled_positions(), a.get_atomic_numbers())

    ds = spglib.get_symmetry_dataset(cell_tuple(atoms), symprec=symprec)
    rots, trans = ds.rotations, ds.translations

    def orbit(frac):
        images = []
        for R, t in zip(rots, trans):
            img = (R @ frac + t) % 1.0
            if not any(np.linalg.norm(img - o - np.round(img - o)) < 1e-3 for o in images):
                images.append(img)
        return images

    void_cart, _ = get_voronoi_interstitial_sites(atoms, r_min=r_min, cluster_tol=cluster_tol, n_images=n_images)
    if len(void_cart) == 0:
        return []
    void_frac = void_cart @ inv

    assigned = np.zeros(len(void_cart), dtype=bool)
    raw_orbits = []
    for i in range(len(void_cart)):
        if assigned[i]:
            continue
        orb = orbit(void_frac[i])
        for j in range(i, len(void_cart)):
            if assigned[j]:
                continue
            for o in orb:
                diff = void_frac[j] - o
                if np.linalg.norm(diff - np.round(diff)) < 1e-3:
                    assigned[j] = True
                    break

        extended = atoms.copy()
        for f in orb:
            extended.append("Ne")
            extended.positions[-1] = f @ cell
        ds_ext = spglib.get_symmetry_dataset(cell_tuple(extended), symprec=symprec)
        ne_idx = len(atoms)
        mult = int(np.sum(ds_ext.equivalent_atoms == ds_ext.equivalent_atoms[ne_idx]))
        letter = ds_ext.wyckoffs[ne_idx]
        if mult % formula_units != 0:
            raise ValueError(
                f"Void orbit int_{mult}{letter} has multiplicity {mult}, not an exact multiple of "
                f"formula_units={formula_units}. formula_units is likely wrong for this structure."
            )
        label_mult = mult // formula_units
        cart_positions = np.array([f @ cell for f in orb], dtype=float)
        raw_orbits.append(
            {
                "cart_positions": cart_positions,
                "multiplicity": mult,
                "label_multiplicity": label_mult,
                "wyckoff_letter": letter,
                "base_label": f"int_{label_mult}{letter}",
            }
        )

    _assign_collision_safe_labels(raw_orbits, key=lambda o: (o["base_label"], tuple(np.round(o["cart_positions"][0], 6))))
    return raw_orbits


def _assign_collision_safe_labels(orbits: List[dict], key) -> None:
    """
    Sort `orbits` by `key` (for determinism independent of spglib/Qhull
    internal traversal order), then pop each `base_label` and set `label` to
    it, suffixed with `_1, _2, ...` when multiple orbits share one. Mutates
    `orbits` in place.
    """
    orbits.sort(key=key)
    label_count: dict = {}
    for o in orbits:
        label_count[o["base_label"]] = label_count.get(o["base_label"], 0) + 1
    label_seen: dict = {}
    for o in orbits:
        base = o.pop("base_label")
        if label_count[base] > 1:
            label_seen[base] = label_seen.get(base, 0) + 1
            o["label"] = f"{base}_{label_seen[base]}"
        else:
            o["label"] = base


def tile_atomic_sublattices(atomic_sublattices: List[dict], repeat: tuple, n_unitcell_atoms: int) -> List[dict]:
    """
    Expand atomic sublattices discovered on a small unit cell into the full
    set of atom indices in a supercell built via `unit_cell.repeat(repeat)`.

    Uses ASE's exact `Atoms.repeat()` index convention (verified
    empirically): the atom at unit-cell index `idx`, tile offset `(i, j, k)`,
    lands at supercell index
    `((i * ny + j) * nz + k) * n_unitcell_atoms + idx` for
    `repeat = (nx, ny, nz)`. This avoids re-deriving orbits on the
    (potentially much larger and slower to analyze) supercell directly, while
    still producing indices directly usable against the real supercell.

    Parameters
    ----------
    atomic_sublattices : list of dict
        Output of `discover_atomic_sublattices` called on the small unit cell
        (with `formula_units=1`, since there's nothing to rescale yet).
    repeat : tuple of int
        `(nx, ny, nz)` as passed to `Atoms.repeat()`.
    n_unitcell_atoms : int
        Number of atoms in the unit cell.

    Returns
    -------
    list of dict, same keys as `discover_atomic_sublattices`, except:
        `atom_indices` (into the full supercell), `multiplicity` (true
        supercell count), `unitcell_multiplicity` (original per-unit-cell count).

    See Also
    --------
    tile_interstitial_sublattices : Analogous expansion for void orbits.
    """
    nx, ny, nz = repeat
    tile_ids = [(i * ny + j) * nz + k for i in range(nx) for j in range(ny) for k in range(nz)]
    result = []
    for orbit in atomic_sublattices:
        tiled_indices = sorted(
            tile_id * n_unitcell_atoms + idx for idx in orbit["atom_indices"] for tile_id in tile_ids
        )
        new_orbit = dict(orbit)
        new_orbit["unitcell_multiplicity"] = orbit["multiplicity"]
        new_orbit["atom_indices"] = tiled_indices
        new_orbit["multiplicity"] = len(tiled_indices)
        result.append(new_orbit)
    return result


def tile_interstitial_sublattices(
    interstitial_sublattices: List[dict], repeat: tuple, unitcell_cell, supercell_cell
) -> List[dict]:
    """
    Expand interstitial void sublattices discovered on a small unit cell into
    the full set of Cartesian void positions in a supercell built via
    `unit_cell.repeat(repeat)`.

    Each orbit's Cartesian positions are converted to fractional coordinates
    using the small unit cell's own lattice (fractional void positions are
    scale-invariant), tiled by every integer lattice shift `(i, j, k)`, and
    mapped back to Cartesian using the *actual* supercell's lattice divided
    by `repeat` -- so a relaxation-driven change to the cell (e.g. volume
    relaxation) is correctly reflected in the tiled positions, not just the
    idealized unit cell's geometry.

    Parameters
    ----------
    interstitial_sublattices : list of dict
        Output of `discover_interstitial_sublattices` on the small unit cell
        (with `formula_units=1`).
    repeat : tuple of int
        `(nx, ny, nz)`.
    unitcell_cell : (3, 3) array-like
        Lattice vectors of the small unit cell used for discovery.
    supercell_cell : (3, 3) array-like
        Lattice vectors of the actual (relaxed) supercell.

    Returns
    -------
    list of dict, same keys as `discover_interstitial_sublattices`, except:
        `cart_positions` (within the full supercell), `multiplicity` (true
        supercell count), `unitcell_multiplicity` (original per-unit-cell count).

    See Also
    --------
    tile_atomic_sublattices : Analogous expansion for atomic orbits.
    """
    nx, ny, nz = repeat
    unitcell_cell = np.asarray(unitcell_cell, dtype=float)
    unitcell_inv = np.linalg.inv(unitcell_cell)
    real_conv_cell = np.asarray(supercell_cell, dtype=float) / np.array([[nx], [ny], [nz]], dtype=float)

    result = []
    for orbit in interstitial_sublattices:
        frac = orbit["cart_positions"] @ unitcell_inv
        tiled = [(frac + np.array([i, j, k], dtype=float)) @ real_conv_cell for i in range(nx) for j in range(ny) for k in range(nz)]
        tiled_positions = np.vstack(tiled)
        new_orbit = dict(orbit)
        new_orbit["unitcell_multiplicity"] = orbit["multiplicity"]
        new_orbit["cart_positions"] = tiled_positions
        new_orbit["multiplicity"] = len(tiled_positions)
        result.append(new_orbit)
    return result


def validate_sublattice_coverage(atomic_sublattices: List[dict], n_atoms: int) -> None:
    """
    Assert that discovered atomic sublattices exactly partition the host
    structure -- every atom belongs to exactly one orbit, none missing or
    double-counted.

    Raises
    ------
    ValueError
        If the multiplicities do not sum to `n_atoms`, naming the
        shortfall/excess and every sublattice found.
    """
    total = sum(s["multiplicity"] for s in atomic_sublattices)
    if total != n_atoms:
        diff = total - n_atoms
        detail = ", ".join(f"{s['label']}={s['multiplicity']}" for s in atomic_sublattices)
        raise ValueError(
            f"Sublattice site counts do not cover the host structure: sum(multiplicity)={total} "
            f"but the structure has {n_atoms} atoms ({'excess' if diff > 0 else 'shortfall'} of "
            f"{abs(diff)}). Sublattices found: {detail}"
        )
