# Load Flow Learning Lab — V1

A local desktop calculator for learning and checking balanced AC load-flow
homework. Configure a small network, enter the unknown voltage guesses, inspect
**Iteration 0**, and advance either Gauss–Seidel or polar Newton–Raphson one
iteration at a time. A saved history exposes calculated injections, applicable
mismatches, corrections, and solver details.

The application implements both numerical methods directly in Python and NumPy.
The Tkinter interface uses native file dialogs; no web server, account, Excel,
MATLAB, or external power-flow package is required. Excel can read its CSV exports.

## Install and launch

Use **Python 3.11 or newer**, with Tcl/Tk support, and run commands from this folder.
Runtime dependencies are in `requirements.txt`; test dependencies are in
`requirements-dev.txt`. Setup uses a project-local virtual environment.

### Windows quick setup

If Python is not installed, obtain it from [python.org](https://www.python.org/downloads/windows/),
include its **Tcl/Tk** component, and enable **Add Python to PATH** or install the
Python launcher. Double-click **`install_dependencies.cmd`**, or open PowerShell
in this folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The execution-policy option applies to this invocation; it does not change the
machine's permanent policy. The script checks Python/Tkinter, creates `.venv`, and
installs the runtime requirements. After setup, double-click **`main.py`** if
Windows associates `.py` files with Python, or run:

```powershell
python main.py
```

`main.py` automatically uses this folder's `.venv` when present, so no activation
or `.cmd` file is needed. It also works when invoked from another directory.
Without `.venv`, it uses the Python installation running the script, which must
have the runtime requirements installed. The existing `launch.cmd` shortcut
and direct module launch remain available:

```powershell
.\.venv\Scripts\python.exe -m loadflow_app
```

To include the test tools in setup:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Dev
.\.venv\Scripts\python.exe -m pytest
```

### Windows manual setup

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m tkinter
.\.venv\Scripts\python.exe -m loadflow_app
```

Use `python` in place of `py -3` if your installation does not provide the launcher.
No virtual-environment activation is required for these explicit commands.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m tkinter
python -m loadflow_app
```

`python -m tkinter` should open a small demonstration window; close it to continue.
Tkinter ships with many Python distributions, but some Linux installations need an
OS package such as `python3-tk` (and `python3-venv` for virtual environments).
Install it through your operating system or Python distribution. **Tkinter is not
a pip requirement.** A graphical desktop/display is needed for the application;
the numerical engine and tests can run without one.

With a suitable Python environment active, the standard launch command is:

```bash
python -m loadflow_app
```

Optional package installation is supported with `python -m pip install -e .`,
which also provides the `loadflow-lab` launcher.

## First run

1. The app opens on **1 Configure** with the example already loaded; **Load example**
   restores it later. The supplied case is also saved in
   `examples/three_bus.json`. Choose Newton–Raphson or Gauss–Seidel and leave the
   tolerance at `1e-6` for normal use.
2. Choose **Continue to initial guesses →**. The example has three unknowns, ordered as
   angle at Bus 2, angle at Bus 3, and magnitude at Bus 2. **Flat start** fills
   unknown magnitudes with 1 pu and unknown angles with 0°, preserving specified
   Slack/PV values.
3. Select **Calculate initial mismatches**. This stores Iteration 0 without an
   update. The example starts with residual `[-2, +1, -0.5]` pu and maximum
   applicable mismatch 2 pu.
4. On **Results**, use **Update values — 1 iteration**, **Run N iterations**, or
   **Run until converged**. Run N means N *additional* complete updates. A batch
   stops early on convergence, failure, Stop, or the total iteration cap.
5. Review earlier snapshots with the history control. Reviewing history leaves
   the current solver state intact. The calculation-details panel shows the
   source iteration for the Newton residual/Jacobian/correction, or temporary
   Gauss–Seidel PV reactive injections.
6. **Export results** writes original inputs, the actual Ybus, all snapshots,
   events, and solver details to CSV. Export is available at Iteration 0, during
   an unfinished run, and after failure. In Excel, filter `record_type` to
   `iteration_bus` for the ordinary history table.

**Save case** and **Open case** use versioned JSON. Case files preserve the
configuration and original guesses, not a resumable iteration history. A malformed
file is rejected before replacing the active case. Reset restores the original
guesses/types and starts a new run; applying configuration changes also starts a
new run. Exporting never advances the solver.

## Enter your own network

In **branch-table mode**, use **Add bus** / **Add branch**, or double-click an
existing table row to edit it. Add the buses and ordinary lines, then enter line
resistance `r`, reactance `x`, and total charging susceptance `b`, all in pu.
The app stamps `1/(r+jx)` into Ybus and half of `j*b` at each end. Parallel lines
accumulate. Optional bus `shunt_g + j*shunt_b` is added to that bus's diagonal;
positive susceptance is capacitive. Zero resistance is supported; zero total
series impedance is rejected. Removing a bus removes its incident branches.

In **direct Ybus mode**, enter one matrix row per line and separate entries by
commas or whitespace. Use numeric complex literals such as:

```text
0-12.5j, 0+10j, 0+2.5j
0+10j, 0-15j, 0+5j
0+2.5j, 0+5j, 0-7.5j
```

The `i` suffix is also accepted. No expressions are evaluated. Matrix row/column
order follows the displayed bus table. Choose **Apply matrix dimensions / refresh
preview** after editing to reconcile the number of buses with the square matrix.
In direct mode, bus Add/Remove controls are disabled; resize the matrix to change
the bus count and then edit the bus rows as needed. A direct matrix already contains the
aggregate network, including any supplied shunt or transformer model; separate
branch-mode shunt fields are not stamped into it again. Nonsymmetric matrices and
nonzero row sums are permitted. The application does not infer physical branch
counts or individual branch flows from Ybus.

Resizing a direct matrix requires a complete valid resulting matrix. Simply
deleting a bus row/column does not generally model removing all its physical lines:
remaining diagonals may need changes. This operation is not Kron reduction.

## Electrical and numerical conventions

All inputs are **per unit on consistent bases**. Optional positive base MVA is
metadata; changing it does not convert any existing input. Enter/display angles
in degrees; the numerical engine uses radians internally.

Scheduled powers are **net injections**: `P = PG − PL`, `Q = QG − QL`. Generation
is positive and consumption is negative. A load consuming 2 pu real and 0.5 pu
reactive power therefore has schedules `P=-2`, `Q=-0.5`. Calculated injections are
`S = V * conjugate(Ybus @ V)`, with an elementwise voltage/current product.

| Bus type | Specified inputs | Unknowns |
|---|---|---|
| Slack | Magnitude, angle | Net P and Q are calculated outputs |
| PV | Net P, specified magnitude | Angle; net Q is a calculated output |
| PQ | Net P, net Q | Angle and magnitude |

The mismatch is scheduled minus calculated injection. The equation order is P at
all non-slack buses, followed by Q at effective PQ buses, retaining bus-table order
within each group. Unknowns use angles at non-slack buses, followed by magnitudes
at effective PQ buses. Slack power mismatches and an unconstrained PV's Q mismatch
are **N/A**, not zero. Before any conversion, there are `nPV + 2*nPQ` unknowns.

Convergence requires the maximum absolute applicable mismatch to be at most the
tolerance, valid finite voltages, preserved fixed voltage constraints, and completed
Q-limit checks when enabled. Small voltage changes alone do not imply convergence.
The final permitted successful update is still allowed to converge.

One **Gauss–Seidel** iteration is one full sweep in bus-table order, using newly
updated values as they become available. An effective PV bus calculates temporary
net Q for its update and then restores its specified magnitude. The temporary
Q used during the sweep can differ from Q recalculated after the whole sweep;
the app records both separately. V1 uses no acceleration.

**Newton–Raphson uses absolute magnitude corrections:** it solves `J Δx = r`
with `numpy.linalg.solve`, adds angle corrections in radians, and adds magnitude
corrections in pu. A lecture formulation using fractional `η=Δv/v` would instead
multiply magnitude columns of J by their current magnitudes and update
`v ← v*(1+η)`. These conventions must not be mixed. Internal angles are not wrapped
after each step; exported corrections remain unambiguous. V1 applies no damping
and never substitutes an inverse, pseudoinverse, or least-squares solution for a
singular Newton solve.

A nonfinite state, singular Jacobian, or nonpositive attempted PQ magnitude stops
the run with a diagnostic and preserves its last complete valid snapshot. A failed
attempt does not count as a successful iteration. A finite increase in mismatch
does not by itself stop the solver.

## Optional generator reactive limits

Limits are off by default. For a PV generator, enter local reactive load `QL` and
generator bounds. The checked generator output is `QG = Qnet_calculated + QL`.
Both bounds must be present for a configured limit, and `Qmin ≤ Qmax`. PV buses
without configured bounds remain unbounded.

When enabled, the app first solves the current equation set to tolerance. It then
checks all remaining PV generators using the same pu tolerance, converts all
violators to effective PQ in bus order, and sets their net targets to
`Qnet_target = Qbound − QL`. It continues from the existing voltages without
resetting the global iteration count. A clamped bus is not restored to PV during
that run; Reset restores original classifications.

Conversion is an event, not a solver update. History records states before and
after conversion with the same iteration number and different snapshot IDs. This
outer-loop policy can differ from textbook schemes that switch during every GS
sweep. Intermediate Q values are provisional until the inner solve converges and
limits have been checked.

To try a binding limit, load the example, enable limits, and change Bus 3's maximum
to `0.2` pu. Its solved magnitude becomes about `0.98142277` pu while its original
1.0 pu target remains recorded. Using `QL=0.1` and generator maximum `0.3` produces
the same net target and solution.

## Example results and tests

With the supplied flat start and no binding limits, both methods reach the
high-voltage solution below. Use `1e-10` mismatch tolerance to compare these digits.

| Bus | Magnitude pu | Angle degrees | Net P pu | Net Q pu |
|---|---:|---:|---:|---:|
| 1 | 1.000000000 | 0.000000000 | 1.000000000 | 0.524506675 |
| 2 | 0.954905106 | -6.902024491 | -2.000000000 | -0.500000000 |
| 3 | 1.000000000 | 3.383052861 | 1.000000000 | 0.306550266 |

`examples/three_bus_run.csv` is an actual Newton run of the example with `1e-10`
tolerance. [CSV schema documentation](docs/csv_schema.md) explains every record
type, unit, convention, and the authoritative final-status metadata.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The automated suite checks admittance stamping, analytic derivatives against
finite differences, initial and first-step reference values, convergence of both
methods, voltage constraints, reactive limits and local loads, iteration/history
accounting, invalid inputs/numerical failures, and JSON/CSV preservation. Numerical
tests do not require a display. GUI tests use real Tk widgets and the event loop;
they skip if a usable Tk/display environment is unavailable. Native file-dialog
return values are stubbed in the automated tests.

The V1 verification run recorded **123 passed, 2 skipped**. The two skips came
from intermittent Tcl/Tk initialization in the verification environment; those
workflows passed separately. The 112 numerical and file-format tests all passed.
See [verification notes](docs/verification.md) for the tested environment and
remaining manual checks.

## Project layout and scope

`loadflow_app/models.py`, `network.py`, `equations.py`, `solvers.py`, and
`run_controller.py` implement the independently testable numerical layer.
`case_io.py` handles validated JSON; `export.py` handles rectangular CSV;
`gui.py` provides the desktop workflow. `tests/` contains automated checks.
The original specification and lecture materials remain in the project folder.

V1 targets approximately 2–50 buses using dense NumPy arrays, with no hard-coded
three-bus solver. It requires one connected network and exactly one Slack bus.
It implements balanced single-phase-equivalent steady-state AC power flow and
one equivalent regulating generator per PV bus. The branch editor supports
ordinary lines and shunts; a direct Ybus can contain externally modeled
transformers. It does not calculate individual branch flows, expose transformer
tap/phase-shifter controls, convert MW/Mvar inputs, implement Slack Q limits or
generator capability curves, or provide optimal power flow, state estimation,
unbalanced three-phase analysis, plots, or a packaged executable. Difficult cases
can fail or reach the iteration cap; alternative guesses may reach different
solutions. Use the visible assumptions and residuals when checking homework.
