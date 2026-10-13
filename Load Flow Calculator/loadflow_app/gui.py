"""Tkinter interface; all numerical state lives in RunController."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from .models import BusData, BranchData, SolverConfig, NetworkCase
from .network import parse_ybus, validate_case, build_ybus, unknown_labels
from .run_controller import RunController
from .case_io import example_case, load_case, save_case
from .export import export_csv


def number(text: str, label: str, optional: bool = False):
    text = text.strip()
    if optional and not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{label}: enter a number.") from None
    if not math.isfinite(value):
        raise ValueError(f"{label}: enter a finite number.")
    return value


def positive_integer(text: str, label: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{label}: enter a positive integer.") from None
    if value < 1:
        raise ValueError(f"{label}: enter a positive integer.")
    return value


def fmt(value, precision=9):
    return "N/A" if value is None else f"{value:.{precision}g}"


def matrix_text(ybus):
    return "\n".join("  ".join(f"{z.real:.17g}{z.imag:+.17g}j" for z in row) for row in ybus)


def text_set(widget, value):
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", value)
    widget.configure(state="disabled")


def scroll_text(parent, **options):
    """A bounded text view that remains usable for wider/larger networks."""
    holder = ttk.Frame(parent)
    widget = tk.Text(holder, **options)
    vertical = ttk.Scrollbar(holder, orient="vertical", command=widget.yview)
    horizontal = ttk.Scrollbar(holder, orient="horizontal", command=widget.xview)
    widget.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
    widget.grid(row=0, column=0, sticky="nsew")
    vertical.grid(row=0, column=1, sticky="ns")
    horizontal.grid(row=1, column=0, sticky="ew")
    holder.columnconfigure(0, weight=1)
    holder.rowconfigure(0, weight=1)
    return holder, widget


def table(parent, columns, height=6):
    frame = ttk.Frame(parent)
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings", height=height, selectmode="browse")
    for key, title, width in columns:
        tree.heading(key, text=title)
        tree.column(key, width=width, minwidth=65, stretch=False, anchor="center")
    ybar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    xbar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
    tree.grid(row=0, column=0, sticky="nsew")
    ybar.grid(row=0, column=1, sticky="ns")
    xbar.grid(row=1, column=0, sticky="ew")
    return frame, tree


class FieldDialog(tk.Toplevel):
    """Small form with per-field validation and bus-type dependent entries."""

    BUS_FIELDS = [
        ("id", "Bus ID", None), ("name", "Display name (optional)", None),
        ("type", "Original bus type", ["Slack", "PV", "PQ"]),
        ("p", "Net scheduled P (pu)", None), ("q", "Net scheduled Q (pu)", None),
        ("v_set", "Specified magnitude (pu)", None), ("angle_set", "Specified angle (degrees)", None),
        ("q_min", "Generator Q minimum (pu, optional)", None),
        ("q_max", "Generator Q maximum (pu, optional)", None),
        ("q_load", "Local reactive load QL (pu)", None),
        ("shunt_g", "Bus shunt conductance g (pu)", None),
        ("shunt_b", "Bus shunt susceptance b (pu)", None),
    ]

    def __init__(self, app, kind, item=None):
        super().__init__(app)
        self.app, self.kind, self.item = app, kind, item
        self.title(("Edit " if item else "Add ") + kind)
        self.transient(app)
        self.resizable(False, False)
        self.variables, self.entries = {}, {}
        body = ttk.Frame(self, padding=18)
        body.pack(fill="both", expand=True)
        if kind == "bus":
            schema = self.BUS_FIELDS
            defaults = dict(id=app.next_bus_id(), type="PQ", p=0, q=0, q_load=0, shunt_g=0, shunt_b=0)
        else:
            ids = [str(b.id) for b in app.buses]
            schema = [("id", "Branch ID", None), ("from_bus", "From bus", ids), ("to_bus", "To bus", ids),
                      ("r", "Series resistance r (pu)", None), ("x", "Series reactance x (pu)", None),
                      ("b", "Total line charging b (pu)", None)]
            defaults = dict(id=app.next_branch_id(), from_bus=ids[0], to_bus=ids[-1], r=0, x=0.1, b=0)
        for row, (key, title, choices) in enumerate(schema):
            value = getattr(item, key) if item is not None else defaults.get(key)
            var = tk.StringVar(value="" if value is None else str(value))
            self.variables[key] = var
            ttk.Label(body, text=title).grid(row=row, column=0, sticky="w", padx=(0, 20), pady=4)
            widget = ttk.Combobox(body, textvariable=var, values=choices, state="readonly", width=24) if choices else ttk.Entry(body, textvariable=var, width=27)
            widget.grid(row=row, column=1, sticky="ew", pady=4)
            self.entries[key] = widget
        if kind == "bus":
            self.variables["type"].trace_add("write", lambda *_: self.update_applicability())
            self.update_applicability()
            if app.mode.get() == "direct":
                for key in ("shunt_g", "shunt_b"):
                    self.entries[key].configure(state="disabled")
        ttk.Label(body, text="Generation is positive; consumption is negative net injection.\nPositive shunt b is capacitive. All electrical values are per unit.", foreground="#465769").grid(row=len(schema), column=0, columnspan=2, pady=12)
        buttons = ttk.Frame(body)
        buttons.grid(row=len(schema)+1, column=0, columnspan=2, sticky="e")
        ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="left", padx=5)
        ttk.Button(buttons, text="Apply", command=self.apply).pack(side="left")
        self.bind("<Escape>", lambda _: self.destroy())
        self.grab_set()
        self.entries["id"].focus_set()

    def update_applicability(self):
        kind = self.variables["type"].get()
        active = {"p": kind != "Slack", "q": kind == "PQ", "v_set": kind in ("Slack", "PV"),
                  "angle_set": kind == "Slack", "q_min": kind == "PV", "q_max": kind == "PV", "q_load": kind == "PV"}
        for key, enabled in active.items():
            self.entries[key].configure(state="normal" if enabled else "disabled")
        for key, default in (("v_set", "1"), ("angle_set", "0"), ("p", "0"), ("q", "0")):
            if active[key] and not self.variables[key].get():
                self.variables[key].set(default)

    def apply(self):
        try:
            values = {k: v.get().strip() for k, v in self.variables.items()}
            if not values["id"]:
                raise ValueError("An ID is required.")
            others = self.app.buses if self.kind == "bus" else self.app.branches
            if any(x is not self.item and str(x.id) == values["id"] for x in others):
                raise ValueError(f"Duplicate {self.kind} ID: {values['id']}.")
            if self.kind == "bus":
                bus_type = values["type"]
                for key in ("p", "q", "v_set", "angle_set", "q_min", "q_max", "q_load", "shunt_g", "shunt_b"):
                    if key in ("shunt_g", "shunt_b") and self.app.mode.get() == "direct":
                        values[key] = getattr(self.item, key, 0)
                        continue
                    enabled = str(self.entries[key].cget("state")) != "disabled"
                    values[key] = number(values[key], f"Bus {values['id']} {key}", key in ("q_min", "q_max")) if enabled else (0 if key == "q_load" else None)
                if values["v_set"] is not None and values["v_set"] <= 0:
                    raise ValueError("Specified voltage magnitude must be positive.")
                if values["q_min"] is not None and values["q_max"] is not None and values["q_min"] > values["q_max"]:
                    raise ValueError("Generator Q minimum must not exceed its maximum.")
                updated = BusData(**values)
                if self.item:
                    updated.v_guess, updated.angle_guess = self.item.v_guess, self.item.angle_guess
            else:
                endpoint_ids = {str(bus.id): bus.id for bus in self.app.buses}
                values["from_bus"] = endpoint_ids[values["from_bus"]]
                values["to_bus"] = endpoint_ids[values["to_bus"]]
                for key in ("r", "x", "b"):
                    values[key] = number(values[key], f"Branch {values['id']} {key}")
                if values["from_bus"] == values["to_bus"]:
                    raise ValueError("A branch must connect two distinct buses.")
                if values["r"] == 0 and values["x"] == 0:
                    raise ValueError("Branch series impedance cannot be zero.")
                updated = BranchData(**values)
            if self.item is not None and str(self.item.id) == str(updated.id):
                updated.id = self.item.id
            if self.item is None:
                others.append(updated)
            else:
                others[others.index(self.item)] = updated
                if self.kind == "bus" and self.item.id != updated.id:
                    for branch in self.app.branches:
                        if branch.from_bus == self.item.id:
                            branch.from_bus = updated.id
                        if branch.to_bus == self.item.id:
                            branch.to_bus = updated.id
            self.app.mark_dirty()
            self.app.refresh_network()
            self.destroy()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Check inputs", str(exc), parent=self)


class LoadFlowApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Load Flow Learning Lab — V1")
        self.geometry("1220x860")
        self.minsize(1040, 720)
        self.run = None
        self.case = None
        self.buses, self.branches = [], []
        self.guess_vars = {}
        self.busy = False
        self.batch_remaining = 0
        self.batch_after = None
        self.loading = True
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", foreground="#425a6e")
        style.configure("Treeview", rowheight=26)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Load Flow Learning Lab", style="Title.TLabel").pack(side="left")
        ttk.Button(header, text="Help & conventions", command=self.show_help).pack(side="right")
        ttk.Label(outer, text="AC power flow • per-unit inputs • degrees on screen • inspect every iteration", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 10))
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.configure_tab = ttk.Frame(self.tabs, padding=12)
        self.guesses_tab = ttk.Frame(self.tabs, padding=12)
        self.results_tab = ttk.Frame(self.tabs, padding=12)
        for frame, label in ((self.configure_tab, "1  Configure"), (self.guesses_tab, "2  Initial guesses"), (self.results_tab, "3  Results")):
            self.tabs.add(frame, text=label)
        self.footer = tk.StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.footer, style="Subtitle.TLabel").pack(anchor="w", pady=(8, 0))
        self.build_configure()
        self.build_guesses()
        self.build_results()
        self.apply_case(example_case())
        self.protocol("WM_DELETE_WINDOW", self.close)

    def build_configure(self):
        frame = self.configure_tab
        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(0, 10))
        for text, command in (("Load example", lambda: self.apply_case(example_case())), ("Open case…", self.open_case), ("Save case…", self.save_case)):
            ttk.Button(bar, text=text, command=command).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Continue to initial guesses →", command=self.continue_to_guesses).pack(side="right")
        settings = ttk.LabelFrame(frame, text="Solver settings", padding=8)
        settings.pack(fill="x", pady=(0, 8))
        self.method = tk.StringVar(value="Newton-Raphson")
        self.tolerance = tk.StringVar(value="1e-6")
        self.max_iterations = tk.StringVar(value="100")
        self.base_mva = tk.StringVar()
        self.enforce_limits = tk.BooleanVar(value=False)
        for col, (label, var, width) in enumerate((("Method", self.method, 20), ("Tolerance (pu)", self.tolerance, 12), ("Max iterations", self.max_iterations, 12), ("Base MVA (optional)", self.base_mva, 12))):
            ttk.Label(settings, text=label).grid(row=0, column=col, sticky="w", padx=5)
            widget = ttk.Combobox(settings, textvariable=var, values=["Newton-Raphson", "Gauss-Seidel"], state="readonly", width=width) if col == 0 else ttk.Entry(settings, textvariable=var, width=width)
            widget.grid(row=1, column=col, sticky="w", padx=5)
        ttk.Checkbutton(settings, text="Enforce generator Q limits", variable=self.enforce_limits).grid(row=1, column=4, padx=15)
        ttk.Label(frame, text="Net injection: generation +, consumption −. Scheduled P and Q are case inputs; calculated P and Q come from V and Ybus.", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 4))
        busbar = ttk.Frame(frame)
        busbar.pack(fill="x")
        self.bus_count = tk.StringVar()
        ttk.Label(busbar, textvariable=self.bus_count).pack(side="left")
        self.add_bus_button = ttk.Button(busbar, text="Add bus", command=lambda: FieldDialog(self, "bus"))
        self.add_bus_button.pack(side="right", padx=4)
        self.remove_bus_button = ttk.Button(busbar, text="Remove bus", command=self.remove_bus)
        self.remove_bus_button.pack(side="right", padx=4)
        ttk.Button(busbar, text="Edit selected bus", command=self.edit_bus).pack(side="right", padx=4)
        busframe, self.bus_tree = table(frame, [("id", "Bus", 70), ("name", "Name", 120), ("type", "Type", 70), ("p", "P schedule (pu)", 118), ("q", "Q schedule (pu)", 118), ("v_set", "V specified (pu)", 115), ("angle_set", "Angle specified (°)", 130), ("q_min", "QG min (pu)", 100), ("q_max", "QG max (pu)", 100), ("q_load", "QL (pu)", 90), ("shunt_g", "Shunt g (pu)", 100), ("shunt_b", "Shunt b (pu)", 100)], height=4)
        busframe.pack(fill="both", expand=True, pady=4)
        self.bus_tree.bind("<Double-1>", lambda _: self.edit_bus())
        modebar = ttk.Frame(frame)
        modebar.pack(fill="x", pady=(6, 4))
        self.mode = tk.StringVar(value="branches")
        ttk.Label(modebar, text="Network input:").pack(side="left")
        ttk.Radiobutton(modebar, text="Branch table", variable=self.mode, value="branches", command=self.change_mode).pack(side="left", padx=12)
        ttk.Radiobutton(modebar, text="Direct Ybus", variable=self.mode, value="direct", command=self.change_mode).pack(side="left")
        self.branch_count = tk.StringVar()
        ttk.Label(modebar, textvariable=self.branch_count).pack(side="right")
        self.network_holder = ttk.Frame(frame)
        self.network_holder.pack(fill="both", expand=True)
        self.branch_frame = ttk.Frame(self.network_holder)
        branchbar = ttk.Frame(self.branch_frame)
        branchbar.pack(fill="x")
        ttk.Label(branchbar, text="Ordinary lines and bus shunts; transformer taps/phase shifters require direct Ybus.", style="Subtitle.TLabel").pack(side="left")
        ttk.Button(branchbar, text="Add branch", command=self.add_branch).pack(side="right", padx=4)
        ttk.Button(branchbar, text="Remove", command=self.remove_branch).pack(side="right", padx=4)
        ttk.Button(branchbar, text="Edit", command=self.edit_branch).pack(side="right", padx=4)
        bf, self.branch_tree = table(self.branch_frame, [("id", "Branch", 95), ("from_bus", "From bus", 110), ("to_bus", "To bus", 110), ("r", "r (pu)", 130), ("x", "x (pu)", 130), ("b", "Total charging b (pu)", 180)], height=3)
        bf.pack(fill="both", expand=True, pady=4)
        self.branch_tree.bind("<Double-1>", lambda _: self.edit_branch())
        self.direct_frame = ttk.Frame(self.network_holder)
        ttk.Label(self.direct_frame, text="One row per line; comma or whitespace separated. Use 0-12.5j, 0+10j, 1.2-4.6j (or i). No expressions.").pack(anchor="w")
        direct_holder, self.direct_text = scroll_text(self.direct_frame, height=5, wrap="none", font=("Consolas", 10), undo=True)
        direct_holder.pack(fill="both", expand=True, pady=4)
        self.direct_text.bind("<<Modified>>", self.matrix_modified)
        ttk.Button(self.direct_frame, text="Apply matrix dimensions / refresh preview", command=self.apply_matrix).pack(anchor="w")
        ttk.Label(self.direct_frame, text="Resizing requires the complete new matrix. Deleting rows/columns alone does not remove incident line stamps.", style="Subtitle.TLabel").pack(anchor="w", pady=3)
        self.preview_label = tk.StringVar()
        ttk.Label(frame, textvariable=self.preview_label, wraplength=1000).pack(anchor="w", pady=(6, 0))
        preview_holder, self.preview = scroll_text(frame, height=3, wrap="none", font=("Consolas", 9), state="disabled", background="#f3f6f8")
        preview_holder.pack(fill="x", pady=(3, 0))
        for var in (self.method, self.tolerance, self.max_iterations, self.base_mva, self.enforce_limits):
            var.trace_add("write", lambda *_: self.mark_dirty())

    def build_guesses(self):
        frame = self.guesses_tab
        self.unknowns_text = tk.StringVar()
        ttk.Label(frame, text="Initial voltage estimates", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 10))
        ttk.Label(frame, text="Only unknown voltages are editable. Fixed Slack values and PV magnitudes are preserved.", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 10))
        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(0, 12))
        ttk.Button(bar, text="← Edit configuration", command=lambda: self.tabs.select(self.configure_tab)).pack(side="left")
        ttk.Button(bar, text="Flat start", command=self.flat_start).pack(side="left", padx=8)
        ttk.Button(bar, text="Calculate initial mismatches →", command=self.calculate_initial).pack(side="right")
        holder = ttk.Frame(frame)
        holder.pack(fill="both", expand=True)
        self.guess_canvas = tk.Canvas(holder, highlightthickness=0)
        scroll = ttk.Scrollbar(holder, orient="vertical", command=self.guess_canvas.yview)
        self.guess_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.guess_canvas.pack(side="left", fill="both", expand=True)
        self.guess_form = ttk.Frame(self.guess_canvas, padding=8)
        self.guess_window = self.guess_canvas.create_window((0, 0), window=self.guess_form, anchor="nw")
        self.guess_form.bind("<Configure>", lambda _: self.guess_canvas.configure(scrollregion=self.guess_canvas.bbox("all")))
        self.guess_canvas.bind("<Configure>", lambda event: self.guess_canvas.itemconfigure(self.guess_window, width=event.width))

    def build_results(self):
        frame = self.results_tab
        self.summary = tk.StringVar()
        ttk.Label(frame, textvariable=self.summary, font=("Segoe UI", 11), wraplength=1100).pack(fill="x", pady=(0, 10))
        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(0, 7))
        self.step_button = ttk.Button(bar, text="Update values — 1 iteration", command=self.step_once)
        self.step_button.pack(side="left", padx=(0, 8))
        self.batch_n = tk.StringVar(value="5")
        self.batch_entry = ttk.Entry(bar, textvariable=self.batch_n, width=6)
        self.batch_entry.pack(side="left")
        self.batch_button = ttk.Button(bar, text="Run N iterations", command=lambda: self.start_batch(False))
        self.batch_button.pack(side="left", padx=5)
        self.until_button = ttk.Button(bar, text="Run until converged", command=lambda: self.start_batch(True))
        self.until_button.pack(side="left", padx=5)
        self.stop_button = ttk.Button(bar, text="Stop", command=self.stop_batch, state="disabled")
        self.stop_button.pack(side="left", padx=5)
        self.export_button = ttk.Button(bar, text="Export results…", command=self.export_results)
        self.export_button.pack(side="right")
        bar2 = ttk.Frame(frame)
        bar2.pack(fill="x", pady=(0, 8))
        self.edit_button = ttk.Button(bar2, text="Edit configuration", command=self.edit_configuration)
        self.edit_button.pack(side="right", padx=4)
        self.reset_button = ttk.Button(bar2, text="Reset to initial guesses", command=self.reset_run)
        self.reset_button.pack(side="right", padx=4)
        ttk.Label(bar2, text="History:").pack(side="left", padx=(0, 6))
        self.history_choice = tk.StringVar()
        self.history_combo = ttk.Combobox(bar2, textvariable=self.history_choice, state="readonly", width=54)
        self.history_combo.pack(side="left")
        self.history_combo.bind("<<ComboboxSelected>>", lambda _: self.show_snapshot())
        self.viewing = tk.StringVar()
        ttk.Label(frame, textvariable=self.viewing, style="Subtitle.TLabel").pack(anchor="w", pady=(0, 4))
        columns = [("bus", "Bus", 65), ("type", "Original / effective", 135), ("v", "|V| (pu)", 106), ("angle", "Angle (°)", 105), ("p", "P calc (pu)", 112), ("q", "Q net calc (pu)", 112), ("pt", "P target (pu)", 106), ("qt", "Q target (pu)", 106), ("dp", "ΔP (pu)", 108), ("dq", "ΔQ (pu)", 108), ("dv", "Δ|V| (pu)", 105), ("da", "Δangle (°)", 105), ("qg", "Q generator (pu)", 125), ("limit", "Binding Q limit", 112)]
        tf, self.result_tree = table(frame, columns, height=7)
        tf.pack(fill="both", expand=True)
        self.details_visible = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Show calculation details (source residual, Jacobian, corrections / GS temporary Q)", variable=self.details_visible, command=self.toggle_details).pack(anchor="w", pady=7)
        self.details_frame = ttk.Frame(frame)
        self.details_text = tk.Text(self.details_frame, height=9, wrap="none", font=("Consolas", 10), state="disabled")
        dsy = ttk.Scrollbar(self.details_frame, orient="vertical", command=self.details_text.yview)
        dsx = ttk.Scrollbar(self.details_frame, orient="horizontal", command=self.details_text.xview)
        self.details_text.configure(yscrollcommand=dsy.set, xscrollcommand=dsx.set)
        self.details_text.grid(row=0, column=0, sticky="nsew")
        dsy.grid(row=0, column=1, sticky="ns")
        dsx.grid(row=1, column=0, sticky="ew")
        self.details_frame.columnconfigure(0, weight=1)
        self.log_label = ttk.Label(frame, text="Events and diagnostics")
        self.log_label.pack(anchor="w", pady=(4, 3))
        self.event_log = tk.Text(frame, height=4, wrap="word", state="disabled", font=("Segoe UI", 9), background="#f3f6f8")
        self.event_log.pack(fill="x")

    def next_bus_id(self):
        ids = {str(b.id) for b in self.buses}
        i = 1
        while str(i) in ids:
            i += 1
        return str(i)

    def next_branch_id(self):
        ids = {str(b.id) for b in self.branches}
        i = 1
        while f"L{i}" in ids:
            i += 1
        return f"L{i}"

    def selected(self, tree, objects):
        selection = tree.selection()
        if not selection:
            raise ValueError("Select a row first.")
        return objects[int(selection[0])]

    def edit_bus(self):
        try:
            FieldDialog(self, "bus", self.selected(self.bus_tree, self.buses))
        except ValueError as exc:
            messagebox.showinfo("Select a bus", str(exc), parent=self)

    def remove_bus(self):
        try:
            bus = self.selected(self.bus_tree, self.buses)
            incident = [b for b in self.branches if bus.id in (b.from_bus, b.to_bus)]
            description = ", ".join(str(b.id) for b in incident) or "none"
            if messagebox.askyesno("Remove bus", f"Remove Bus {bus.id}?\nIncident branches to remove: {description}.", parent=self):
                self.buses.remove(bus)
                self.branches = [b for b in self.branches if b not in incident]
                self.mark_dirty()
                self.refresh_network()
        except ValueError as exc:
            messagebox.showinfo("Select a bus", str(exc), parent=self)

    def add_branch(self):
        if len(self.buses) < 2:
            messagebox.showerror("Add buses first", "A branch requires at least two buses.", parent=self)
            return
        FieldDialog(self, "branch")

    def edit_branch(self):
        try:
            FieldDialog(self, "branch", self.selected(self.branch_tree, self.branches))
        except ValueError as exc:
            messagebox.showinfo("Select a branch", str(exc), parent=self)

    def remove_branch(self):
        try:
            self.branches.remove(self.selected(self.branch_tree, self.branches))
            self.mark_dirty()
            self.refresh_network()
        except ValueError as exc:
            messagebox.showinfo("Select a branch", str(exc), parent=self)

    def mark_dirty(self, guesses_only=False):
        if self.loading:
            return
        self.run = None
        self.tabs.tab(self.results_tab, state="disabled")
        if not guesses_only:
            self.case = None
            self.guess_vars = {}
            self.tabs.tab(self.guesses_tab, state="disabled")
        self.footer.set("Inputs changed. Calculate initial mismatches to begin a new run." if guesses_only else "Configuration changed. Continue to validate and prepare a new run.")

    def matrix_modified(self, _=None):
        if self.direct_text.edit_modified():
            if self.mode.get() == "direct":
                self.mark_dirty()
            self.direct_text.edit_modified(False)

    def change_mode(self):
        self.mark_dirty()
        if self.mode.get() == "direct":
            try:
                ybus = build_ybus(self.buses, self.branches)
                self.set_matrix(ybus)
            except ValueError:
                pass
        self.refresh_network()

    def set_matrix(self, ybus):
        self.direct_text.delete("1.0", "end")
        self.direct_text.insert("1.0", matrix_text(ybus))
        self.direct_text.edit_modified(False)

    def apply_matrix(self):
        try:
            ybus = parse_ybus(self.direct_text.get("1.0", "end"))
            self.reconcile_buses(len(ybus))
            self.mark_dirty()
            self.refresh_network()
        except ValueError as exc:
            messagebox.showerror("Ybus input", str(exc), parent=self)

    def reconcile_buses(self, size):
        self.buses = self.buses[:size]
        while len(self.buses) < size:
            if not self.buses:
                self.buses.append(BusData(id=self.next_bus_id(), type="Slack", v_set=1, angle_set=0))
            else:
                self.buses.append(BusData(id=self.next_bus_id(), type="PQ", p=0, q=0))
        # Branches are a separate editable draft and never inferred from a matrix.
        ids = {b.id for b in self.buses}
        self.branches = [b for b in self.branches if b.from_bus in ids and b.to_bus in ids]

    def refresh_network(self):
        self.bus_tree.delete(*self.bus_tree.get_children())
        for i, bus in enumerate(self.buses):
            values = [bus.id, bus.name, bus.type] + [fmt(getattr(bus, key)) for key in ("p", "q", "v_set", "angle_set", "q_min", "q_max", "q_load", "shunt_g", "shunt_b")]
            self.bus_tree.insert("", "end", iid=str(i), values=values)
        self.bus_count.set(f"Buses: {len(self.buses)}  •  table order = Ybus row/column order  •  double-click to edit")
        self.branch_tree.delete(*self.branch_tree.get_children())
        for i, branch in enumerate(self.branches):
            self.branch_tree.insert("", "end", iid=str(i), values=(branch.id, branch.from_bus, branch.to_bus, fmt(branch.r), fmt(branch.x), fmt(branch.b)))
        direct = self.mode.get() == "direct"
        self.branch_count.set("Branch data not supplied" if direct else f"Branches: {len(self.branches)}")
        self.add_bus_button.configure(state="disabled" if direct else "normal")
        self.remove_bus_button.configure(state="disabled" if direct else "normal")
        self.branch_frame.pack_forget()
        self.direct_frame.pack_forget()
        (self.direct_frame if direct else self.branch_frame).pack(fill="both", expand=True)
        ids = ", ".join(str(b.id) for b in self.buses)
        self.preview_label.set(f"Ybus preview (pu) — row and column order: {ids}")
        try:
            ybus = parse_ybus(self.direct_text.get("1.0", "end")) if direct else build_ybus(self.buses, self.branches)
            text_set(self.preview, matrix_text(ybus))
        except ValueError as exc:
            text_set(self.preview, f"Preview unavailable: {exc}")

    def apply_case(self, case):
        validate_case(case)
        self.loading = True
        self.run = None
        self.case = None
        self.guess_vars = {}
        self.buses, self.branches = deepcopy(case.buses), deepcopy(case.branches or [])
        self.method.set(case.config.method)
        self.tolerance.set(str(case.config.tolerance))
        self.max_iterations.set(str(case.config.max_iterations))
        self.enforce_limits.set(case.config.enforce_q_limits)
        self.base_mva.set("" if case.base_mva is None else str(case.base_mva))
        self.mode.set(case.mode)
        self.set_matrix(validate_case(case))
        self.refresh_network()
        self.tabs.tab(self.guesses_tab, state="disabled")
        self.tabs.tab(self.results_tab, state="disabled")
        self.tabs.select(self.configure_tab)
        self.loading = False
        self.footer.set("Case loaded. Review the network and solver settings, then Continue.")

    def case_from_ui(self):
        direct = self.mode.get() == "direct"
        ybus = parse_ybus(self.direct_text.get("1.0", "end")) if direct else None
        if direct and len(ybus) != len(self.buses):
            self.reconcile_buses(len(ybus))
            self.refresh_network()
        config = SolverConfig(method=self.method.get(), tolerance=number(self.tolerance.get(), "Tolerance"),
                              max_iterations=positive_integer(self.max_iterations.get(), "Maximum iterations"), enforce_q_limits=self.enforce_limits.get())
        case = NetworkCase(buses=deepcopy(self.buses), mode=self.mode.get(), branches=[] if direct else deepcopy(self.branches), ybus=ybus,
                           base_mva=number(self.base_mva.get(), "System base MVA", optional=True), config=config)
        self.read_guesses(case)
        validate_case(case)
        return case

    def continue_to_guesses(self):
        try:
            self.case = self.case_from_ui()
            self.run = None
            self.render_guesses()
            self.tabs.tab(self.guesses_tab, state="normal")
            self.tabs.tab(self.results_tab, state="disabled")
            self.tabs.select(self.guesses_tab)
            self.footer.set("Enter unknown guesses. Initial mismatches creates Iteration 0 without a solver update.")
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Check configuration", str(exc), parent=self)

    def render_guesses(self):
        for child in self.guess_form.winfo_children():
            child.destroy()
        self.guess_vars = {}
        labels = unknown_labels(self.case.buses)
        self.unknowns_text.set(f"{len(labels)} unknowns, in solver order:\n" + ", ".join(labels))
        ttk.Label(self.guess_form, textvariable=self.unknowns_text, wraplength=960, font=("Segoe UI", 11)).grid(row=0, column=0, columnspan=4, sticky="w", padx=16, pady=(0, 16))
        for col, label in enumerate(("Bus", "Original type", "Magnitude (pu)", "Angle (degrees)")):
            ttk.Label(self.guess_form, text=label, font=("Segoe UI", 10, "bold")).grid(row=1, column=col, sticky="w", padx=16, pady=8)
        for row, bus in enumerate(self.case.buses, start=2):
            ttk.Label(self.guess_form, text=f"{bus.id}  {bus.name}").grid(row=row, column=0, sticky="w", padx=16, pady=8)
            ttk.Label(self.guess_form, text=bus.type).grid(row=row, column=1, sticky="w", padx=16)
            self.guess_vars[bus.id] = {}
            for col, key, editable, value in ((2, "v_guess", bus.type == "PQ", bus.v_guess if bus.type == "PQ" else bus.v_set),
                                              (3, "angle_guess", bus.type != "Slack", bus.angle_set if bus.type == "Slack" else bus.angle_guess)):
                if editable:
                    var = tk.StringVar(value=str(value))
                    self.guess_vars[bus.id][key] = var
                    ttk.Entry(self.guess_form, textvariable=var, width=23).grid(row=row, column=col, sticky="w", padx=16)
                    var.trace_add("write", lambda *_: self.mark_dirty(guesses_only=True))
                else:
                    ttk.Label(self.guess_form, text=f"{fmt(value)}  (specified)").grid(row=row, column=col, sticky="w", padx=16)

    def read_guesses(self, case):
        for bus in case.buses:
            for key, var in self.guess_vars.get(bus.id, {}).items():
                if (key == "v_guess" and bus.type == "PQ") or (key == "angle_guess" and bus.type != "Slack"):
                    setattr(bus, key, number(var.get(), f"Bus {bus.id} {key.replace('_', ' ')}"))

    def flat_start(self):
        for variables in self.guess_vars.values():
            for key, var in variables.items():
                var.set("1.0" if key == "v_guess" else "0.0")

    def calculate_initial(self):
        try:
            case = deepcopy(self.case)
            self.read_guesses(case)
            run = RunController(case)
            self.case, self.run = case, run
            self.buses = deepcopy(case.buses)
            self.tabs.tab(self.results_tab, state="normal")
            self.tabs.select(self.results_tab)
            self.refresh_results()
            self.footer.set("Iteration 0 calculated. No solver update has been taken.")
        except (ValueError, TypeError, ArithmeticError) as exc:
            messagebox.showerror("Check initial guesses", str(exc), parent=self)

    def refresh_results(self):
        if self.run is None:
            return
        run, snapshot = self.run, self.run.current
        remaining = max(0, run.case.config.max_iterations - snapshot.iteration)
        self.summary.set(f"{run.case.config.method}  |  {run.status}  |  Completed iterations: {snapshot.iteration}  |  Remaining: {remaining}\n"
                         f"Maximum applicable mismatch: {snapshot.max_mismatch:.8g} pu ({snapshot.max_mismatch_label})  |  Tolerance: {run.case.config.tolerance:g} pu")
        options = [f"Snapshot {s.snapshot_id}  •  Iteration {s.iteration}  •  {s.phase}" for s in run.history.snapshots]
        self.history_combo.configure(values=options)
        self.history_combo.current(len(options)-1)
        self.show_snapshot()
        logs = [f"{'Attempted iteration' if 'attempted_iteration' in e else 'Iteration'} "
                f"{e.get('attempted_iteration', e.get('iteration', '?'))} — "
                f"{e.get('type', 'event')}: {e.get('message', '')}" for e in run.history.events]
        if run.case.config.enforce_q_limits:
            logs.insert(0, "Q limits are checked after the current equation set converges. Intermediate generator Q is provisional.")
        text_set(self.event_log, "\n".join(logs) if logs else "No constraint switches or failures.")
        self.update_buttons()

    def show_snapshot(self):
        if self.run is None:
            return
        index = self.history_combo.current()
        if index < 0:
            return
        snapshot = self.run.history.snapshots[index]
        self.viewing.set(f"Viewing iteration {snapshot.iteration}, snapshot {snapshot.snapshot_id} ({snapshot.phase}; {snapshot.status}). Live state is iteration {self.run.state.iteration}. History selection never changes the live state.")
        self.result_tree.delete(*self.result_tree.get_children())
        for i, bus in enumerate(self.run.case.buses):
            dv = None if snapshot.delta_magnitude is None else snapshot.delta_magnitude[i]
            da = None if snapshot.delta_angle is None else np.degrees(snapshot.delta_angle[i])
            qg = snapshot.q_calculated[i] + bus.q_load if bus.type == "PV" else None
            values = (bus.id, f"{bus.type} / {snapshot.effective_types[i]}", fmt(abs(snapshot.voltages[i])), fmt(np.degrees(snapshot.angles[i])), fmt(snapshot.p_calculated[i]), fmt(snapshot.q_calculated[i]), fmt(snapshot.p_targets[i]), fmt(snapshot.q_targets[i]), fmt(snapshot.delta_p[i]), fmt(snapshot.delta_q[i]), fmt(dv), fmt(da), fmt(qg), snapshot.binding_limits[i] or "N/A")
            self.result_tree.insert("", "end", values=values)
        text_set(self.details_text, self.details_description(snapshot))

    def details_description(self, snapshot):
        details = snapshot.details
        if not details:
            if snapshot.phase == "constraint_change":
                return "This snapshot activates reactive constraints without a voltage correction. Select the preceding snapshot to inspect the last solver update.\n\n" + (snapshot.event or "")
            return "No solver correction at this snapshot. Iteration 0 is the uncorrected initial state."
        lines = [f"Pre-update quantities from iteration {details.get('source_iteration', '?')}; displayed bus powers are recalculated AFTER the update."]
        if "jacobian" in details:
            rows = details["equation_labels"]
            cols = details["unknown_labels"]
            lines += ["Unknown order: " + ", ".join(cols), "Residual (scheduled − calculated):"]
            lines += [f"  {label}: {value:.12g}" for label, value in zip(rows, details["residual"])]
            lines += ["Jacobian of calculated injections (angles in radians; magnitudes in pu):", "Row / column".ljust(23) + " ".join(str(c).rjust(19) for c in cols)]
            lines += [str(label).ljust(23) + " ".join(f"{v:19.11g}" for v in row) for label, row in zip(rows, details["jacobian"])]
            lines += ["Correction vector (angle in radians, absolute magnitude in pu):"]
            lines += [f"  {label}: {value:.12g}" for label, value in zip(cols, details["correction"])]
        if "pv_q_used" in details:
            lines += ["Temporary PV net Q used during the Gauss–Seidel sweep (pu):"]
            lines += [f"  Bus {bus}: {value:.12g}" for bus, value in details["pv_q_used"].items()]
            lines += ["These mixed-state Q values may differ from the final calculated Q in the results table."]
        return "\n".join(lines)

    def toggle_details(self):
        if self.details_visible.get():
            self.details_frame.pack(fill="both", before=self.log_label)
        else:
            self.details_frame.pack_forget()

    def update_buttons(self):
        can_step = self.run is not None and self.run.can_step and not self.busy
        for button in (self.step_button, self.batch_button, self.until_button):
            button.configure(state="normal" if can_step else "disabled")
        self.stop_button.configure(state="normal" if self.busy else "disabled")
        for button in (self.reset_button, self.edit_button, self.export_button, self.batch_entry):
            button.configure(state="disabled" if self.busy else "normal")
        self.tabs.tab(self.configure_tab, state="disabled" if self.busy else "normal")
        self.tabs.tab(self.guesses_tab, state="disabled" if self.busy else "normal")

    def step_once(self):
        if self.run is not None and self.run.can_step and not self.busy:
            self.run.step()
            self.refresh_results()

    def start_batch(self, until):
        if self.busy or self.run is None or not self.run.can_step:
            return
        try:
            self.batch_remaining = self.run.case.config.max_iterations - self.run.state.iteration if until else positive_integer(self.batch_n.get(), "Run N iterations")
        except ValueError as exc:
            messagebox.showerror("Batch count", str(exc), parent=self)
            return
        self.busy = True
        self.update_buttons()
        self.batch_after = self.after(1, self.batch_tick)

    def batch_tick(self):
        self.batch_after = None
        if not self.busy or self.run is None:
            return
        self.run.step()
        self.batch_remaining -= 1
        self.refresh_results()
        if self.batch_remaining > 0 and self.run.can_step:
            self.batch_after = self.after(10, self.batch_tick)
        else:
            self.busy = False
            if self.run.can_step:
                self.run.status = "Paused"
            self.refresh_results()

    def stop_batch(self):
        if not self.busy:
            return
        if self.batch_after is not None:
            self.after_cancel(self.batch_after)
            self.batch_after = None
        self.busy = False
        self.run.stop()
        self.refresh_results()
        self.footer.set("Stopped between complete updates. Continue or export the preserved history.")

    def reset_run(self):
        if self.run is not None and not self.busy:
            self.run.reset()
            self.refresh_results()
            self.footer.set("Reset: original guesses and bus types restored in a fresh run.")

    def edit_configuration(self):
        if not self.busy:
            self.tabs.select(self.configure_tab)

    def open_case(self):
        path = filedialog.askopenfilename(parent=self, title="Open load-flow case", filetypes=[("Load-flow case", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            case = load_case(path)
            self.apply_case(case)
            self.footer.set(f"Opened {Path(path).name}. Continue to begin a new run.")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            messagebox.showerror("Could not open case", str(exc), parent=self)

    def save_case(self):
        try:
            case = self.case_from_ui()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("Check configuration", str(exc), parent=self)
            return
        path = filedialog.asksaveasfilename(parent=self, title="Save load-flow case", defaultextension=".json", initialfile="load_flow_case.json", filetypes=[("Load-flow case", "*.json")])
        if not path:
            return
        try:
            save_case(case, path)
            self.footer.set(f"Saved case and original guesses: {path}")
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror("Could not save case", str(exc), parent=self)

    def export_results(self):
        if self.run is None:
            return
        path = filedialog.asksaveasfilename(parent=self, title="Export full load-flow run", defaultextension=".csv", initialfile=f"load_flow_{self.run.case.config.method.lower().replace('-', '_')}_iteration_{self.run.state.iteration}.csv", filetypes=[("CSV spreadsheet", "*.csv")])
        if not path:
            return
        try:
            export_csv(self.run, path)
            self.footer.set(f"Exported all {len(self.run.history.snapshots)} snapshots and inputs: {path}")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not export results", str(exc), parent=self)

    def show_help(self):
        help_window = tk.Toplevel(self)
        help_window.title("Load flow — conventions and workflow")
        help_window.geometry("780x620")
        text = tk.Text(help_window, wrap="word", padx=20, pady=20, font=("Segoe UI", 11))
        text.pack(fill="both", expand=True)
        text.insert("1.0", "All electrical inputs are per unit on consistent bases. Angles are displayed in degrees. Base MVA is metadata; changing it does not convert inputs.\n\n"
            "NET INJECTIONS\nP scheduled = generation − consumption; Q scheduled = generation − consumption. A load of 2 + j0.5 pu is P = −2, Q = −0.5. Calculated injections use S = V × conjugate(Ybus V), element by element. Mismatch = scheduled − calculated for active equations.\n\n"
            "BUS TYPES AND UNKNOWNS\nSlack: specify magnitude and angle; calculate P and Q. PV: specify net P and magnitude; guess angle and calculate net Q. PQ: specify net P and Q; guess magnitude and angle. Unknown order is all non-slack angles in table order, then all PQ magnitudes in table order. N/A means there is no equation for that quantity.\n\n"
            "ITERATIONS\nIteration 0 only calculates injections and mismatches. A Gauss–Seidel iteration is a full sequential sweep; temporary PV Q used in the sweep is shown separately from final Q. Newton–Raphson solves J Δx = residual with absolute magnitude corrections in pu and angle corrections in radians. Some lectures use fractional ΔV/V: that convention scales magnitude columns by current V and updates V(1 + ΔV/V). This app uses V + ΔV. No hidden damping or acceleration is applied.\n\n"
            "CONVERGENCE AND Q LIMITS\nConvergence uses the maximum absolute applicable power mismatch. With limits enabled, solve the current equation set first, then check QG = Qnet + QL. Violating PV buses become PQ at Qnet target = generator bound − QL. The current voltage is retained; all violations switch in table order. Switches have separate snapshot IDs at the same iteration count. No switch back occurs until reset. Intermediate Q values are provisional.\n\n"
            "NETWORK AND EXPORT\nBranch input stamps ordinary series lines, half charging at each end, and bus shunts. Direct Ybus can include transformer models; it already includes all shunts. Bus-shunt entries apply only in branch mode. Direct matrix row/column order follows the bus table. Removing a row/column alone does not remove incident-line diagonal contributions.\n\n"
            "Run N adds up to N updates and stops early at convergence or failure. Stop works between complete updates. History review never rewinds the solver. Export includes inputs and every snapshot; filter record_type = iteration_bus in Excel. JSON saves configuration and original guesses; opening creates a new run.\n\n"
            "V1 targets small balanced AC teaching networks (2–50 buses). It does not calculate individual branch flows, build incidence matrices, perform optimal power flow, or control transformer taps.")
        text.configure(state="disabled")

    def close(self):
        if self.batch_after is not None:
            self.after_cancel(self.batch_after)
        self.destroy()


def main():
    LoadFlowApp().mainloop()
