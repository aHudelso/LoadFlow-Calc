# V1 verification

Verified on Windows with Python 3.12.14 and Tk 8.6. The initial numerical suite
used NumPy 2.3.5. The supplied `setup.ps1 -Dev` then created a project-local
virtual environment, installed NumPy 2.5.4 and pytest 9.1.1, and successfully
checked imports and a withdrawn Tk window. The full suite also ran in that
installed environment.

## Automated checks

The final complete run collected 125 cases: **123 passed, 2 skipped**. All 112
numerical, network-validation, case-file, and CSV cases passed. Two GUI cases
skipped because the bundled Tcl/Tk runtime intermittently could not initialize
its script files. Every one of the 13 GUI cases passed across the full and
targeted verification runs. No assertion failures remain.

Covered behavior includes:

- Both first-step and converged reference solutions, including binding generator
  limits with and without local reactive load.
- The analytic Jacobian against central finite differences, including a
  nonsymmetric matrix and nonzero conductance.
- Independent constructed solutions for networks up to 50 buses with mixed bus
  types and reordered buses. A separate 50-bus ring/chord case recovered the
  known solution to below `5e-11` complex voltage error: Newton took 3 updates,
  Gauss-Seidel 706 at `1e-10` mismatch tolerance. A higher iteration cap was used
  for that Gauss-Seidel check.
- Iteration-zero, stepping/batching, convergence and iteration-cap boundaries,
  Q-limit conversion snapshots, reset, invalid inputs, and preservation of valid
  history on numerical failure.
- Complete rectangular CSV exports, JSON round trips, and malformed-file
  rejection.
- Real Tk callbacks and event-loop behavior for both methods, Run N, Stop,
  history review, reset, changed inputs, matrix resizing, integer bus IDs,
  branch/bus edits, reactive-limit events, and export/save/open error handling.

GUI tests stub the native file dialogs' returned paths/cancellations. They verify
the application's handling of those results, not Windows' own dialog controls.

## Visible desktop checks and limitations

The application was launched on the Windows desktop. Configure, Initial guesses,
and Results at Iteration 0 were inspected. The three-bus input values, unknown
list, fixed voltage quantities, N/A mismatches, and initial residual were correct.

Computer Use was stopped with the physical Escape key during desktop checking.
No further desktop interaction was performed. Native Save As/Open interactions,
including cancellation, therefore remain unverified by manual desktop automation;
their callback paths are covered by the automated GUI tests.

The initial local `.venv` was created with the available bundled Python runtime.
Virtual environments depend on their base Python installation and are not
portable. If that runtime changes or this folder moves to another computer,
recreate the environment with a regular Python 3.11+ installation using the
README instructions. No system-wide Python settings were changed.

The original planning document, lecture PDF, and homework DOCX were preserved.
