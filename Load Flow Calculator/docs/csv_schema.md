# Results CSV schema, version 1

An export is one rectangular UTF-8 CSV with a single header and a fixed superset of
columns. Every row has `record_type` and `run_id`. Irrelevant values are empty cells,
including nonapplicable power mismatches; these are **not zero**. Numbers retain
Python/NumPy floating-point precision rather than the GUI's display rounding.

In Excel, open or import the CSV, enable filters on the header, and filter
`record_type` to **iteration_bus** for the ordinary bus-by-bus iteration history.
Filter `bus_id` to follow one bus. Use `snapshot_id` to distinguish states before
and after a reactive-limit conversion with the same `iteration` number. `row_index`,
`column_index`, and `bus_order` are one-based; snapshot IDs begin at zero. Bus IDs
are user identifiers (strings or integers in case files).

| Record type | Contents and columns |
|---|---|
| `metadata` | One `key` / `value` pair per row: schema/app versions, run ID, UTC export time, solver method, tolerances, iteration cap, Q-limit setting/policy, base MVA, input mode/counts, final status, completed iterations, snapshot count, and numerical conventions. |
| `bus_input` | `bus_order`, `bus_id`, `bus_name`, `original_type`, `p_schedule`, `q_schedule`, `v_set`, `angle_set_deg`, `v_guess`, `angle_guess_deg`, `q_min`, `q_max`, `q_load`, `shunt_g`, `shunt_b`. These are the original configuration, including guesses. |
| `branch_input` | `branch_id`, `from_bus`, `to_bus`, `r`, `x`, `line_b`. Present only in branch-table mode. `line_b` is total charging susceptance, divided equally between the line ends in Ybus. |
| `ybus_entry` | All n × n entries, including zero entries: `row_index`, `column_index`, `row_bus_id`, `column_bus_id`, `y_real`, `y_imag`. This is the actual aggregate matrix used by the solver. |
| `iteration_bus` | One row per bus per saved snapshot; includes Iteration 0 and constraint changes. See below. |
| `event` | `iteration`, optional `snapshot_id`/`status`, `event_type`, `message`. Failed updates include `attempted_iteration`. Conversions include `bus_id`, `binding_limit`, `generator_q_limit`, and `net_q_target`. Stops preserve completed states. |
| `solver_detail` | Pre-update residual/Jacobian/corrections or temporary GS PV Q. See below. |

## Iteration records

| Columns | Meaning |
|---|---|
| `snapshot_id`, `iteration`, `phase`, `status` | Unique snapshot identity, completed update count, snapshot phase, and status at that snapshot. |
| `bus_order`, `bus_id`, `bus_name`, `original_type`, `effective_type` | Original bus identity and the classification used in that snapshot. |
| `voltage_real`, `voltage_imag`, `voltage_magnitude`, `angle_deg` | Present voltage in pu and unwrapped angle in degrees. |
| `p_schedule`, `q_schedule` | Original net scheduled injections in pu, blank where not specified. |
| `p_target`, `q_target` | Active net injection targets, including the net Q target after a generator limit binds. |
| `p_calculated`, `q_calculated` | Net injections calculated from this complete snapshot's voltage vector. |
| `q_generator` | Original PV generator output: `q_calculated + q_load`, including after conversion to PQ. Blank for other original bus types. |
| `delta_p`, `delta_q` | Applicable target minus calculated injection; blank for equations excluded from the solve. |
| `delta_magnitude`, `delta_angle_deg` | Actual change from the previous valid voltage state; blank when no numerical update occurred (initial and constraint-change snapshots). Magnitude correction is absolute pu. |
| `v_set`, `fixed_v_set` | Original specified magnitude, and currently enforced specified magnitude. After PV-to-PQ conversion `v_set` is retained but `fixed_v_set` is blank. |
| `max_mismatch`, `max_mismatch_label`, `tolerance` | Maximum absolute applicable residual, its equation, and the configured pu tolerance. |
| `binding_limit`, `message` | Any binding Q-limit label and optional snapshot event text. |

The final run status is **metadata where `key=final_status`**. A failed attempt can
leave the last valid numerical snapshot unchanged, so that snapshot's `status`
need not equal the final run status. Exporting does not take a solver step.

## Solver details

Every detail row identifies the receiving `snapshot_id` and the pre-update
`source_iteration`. `detail_kind` selects its meaning:

| `detail_kind` | Fields |
|---|---|
| `residual` | `row_index`, `row_label`, `value`, `unit=pu`: residual used for the update. |
| `jacobian` | `row_index`, `column_index`, `row_label`, `column_label`, `value`, and derivative `unit` (`pu/rad` or `pu/pu`). Includes all entries. |
| `correction` | `column_index`, `column_label`, `value`, `unit`: Newton angle corrections are radians; magnitude corrections are absolute pu. |
| `gs_pv_q_used` | `bus_id`, `value`, `unit=pu`: temporary net Q used during the sequential GS sweep. It may differ from the final snapshot's `q_calculated`. |

Residual order is P at every non-slack bus, then Q at each effective PQ bus.
Unknown order is angles at every non-slack bus, then magnitudes at each effective
PQ bus. Both groups retain bus-table order. Snapshot angle changes are in degrees;
raw Newton detail angle corrections are in radians, explicitly labeled by `unit`.

## Case files

JSON cases use `schema_version: 1` and store `network_mode`, `base_mva`, ordered
`buses`, `branches`, `ybus`, and `solver`. Bus/branch keys match the Python data
models. Angles are degrees in case files. A direct Ybus entry is a two-element
`[real, imaginary]` array, never executable text. Branch mode uses `ybus: null`;
direct mode uses an empty `branches` array. File loading rejects unknown keys,
duplicate object keys, nonfinite values, unsupported versions, and invalid cases.
Loading a case creates fresh configuration; JSON does not restore run history.
