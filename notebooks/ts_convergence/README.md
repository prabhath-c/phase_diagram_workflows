# TS convergence notebooks

Temperature-scaling (TS) bracket convergence: narrow a `(t_low, t_high)` bracket until the forward and backward reversible-scaling sweeps agree (the *criterion* is below a *tolerance*). Every notebook runs the same small Al case (`_common.py`: 32 atoms, 100 + 100 steps, bracket `(700, 720)` K, Mishin potential), so each takes a minute or two. The numbers are for speed, not production-quality free energies.

## Which driver?

| You want | Driver | Notebook |
|---|---|---|
| look at every bracket yourself, call again by hand | `refine_temperature_bracket_manually` | `single/manual.ipynb` |
| submit the whole sequence up front; the executor (or SLURM) holds the dependencies | `refine_temperature_bracket_with_chain` | `single/chain.ipynb` |
| submit once and close everything; each bracket job submits the next from its own criterion | `refine_temperature_bracket_with_resubmission` | `single/resubmission.ipynb` |
| the same for many structures of a phase, one tolerance found from their first brackets | `refine_many_structures_with_discovered_tolerance` | `many/discovered_tolerance.ipynb` |

Chain vs resubmission: the chain fixes the brackets before any result exists and reserves jobs that may not be needed; resubmission chooses each bracket from the previous criterion and only submits what is needed. Only `SlurmClusterExecutor` and `FluxClusterExecutor` let a job end after handing over (`ExecutorSpec.detach`); with `SingleNodeExecutor`, as in these notebooks, the calls return when everything is done.

## Running

Use the `pyiron-dev-latest` kernel. Results go to `results/<notebook>/` (gitignored); delete that folder to start a notebook from scratch. Every notebook ends with assertions, so running one end to end also tests the driver:

```
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=pyiron-dev-latest single/chain.ipynb
```

On a cluster, replace the `SingleNodeExecutor` in the `ExecutorSpec` / executor with `SlurmClusterExecutor` (and a `resource_dict`); nothing else changes. That variant is not exercised here.
