"""
Multi-step workflows built on top of the building-block modules elsewhere in
this package (`structures`, `free_energies`, `phase_diagram`).

Where those modules provide single-purpose functions (create one defect,
run one TI calculation, build one convex hull), `workflows` chains several
of them into an end-to-end pipeline -- e.g. running TI temperature-bracket
convergence and feeding the converged bracket into a point-defect
formation-energy calculation, runnable step-by-step or all at once.
"""
