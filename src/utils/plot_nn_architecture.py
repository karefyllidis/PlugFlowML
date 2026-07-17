#!/usr/bin/env python3
"""
Neural-network architecture diagrams (matplotlib + TikZ).

Two complementary renderers for a feed-forward MLP described by a list of
layer sizes (``layer_sizes``), human-readable names (``layer_names``), and
the operations on each edge (``layer_ops``):

* :func:`draw_mlp_architecture` — miloharper-style schematic (empty circles
  connected by thin lines) in matplotlib. Horizontal layout, with vertical
  truncation + ellipsis for wide hidden layers. Renders consistently with the
  PlugFlowML matplotlib aesthetics set by ``src.utils.plot_style.setup_matplotlib``.

* :func:`write_tikz_mlp` — emits a standalone TikZ ``.tex`` document that
  draws rectangular layer blocks with arrows. Compile externally with
  ``pdflatex`` to obtain a vector publication figure.

The functions accept pure-Python inputs (lists of ints / strings + a path)
and do not import ``torch``, so they can be reused by any PlugFlowML notebook
or script that knows its layer sizes.

Author: Nikolas Karefyllidis, PhD
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt


__all__ = ["draw_mlp_architecture", "write_tikz_mlp"]


def _fmt_dropout_p(dropout_p: float) -> str:
    """Format dropout probability for figure labels (exactly 4 fractional digits, half-up)."""
    x = float(dropout_p)
    d = Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return format(d, "f")


# ─────────────────────────────────────────────────────────────────────────────
# (a) Matplotlib diagram
# ─────────────────────────────────────────────────────────────────────────────
def draw_mlp_architecture(
    layer_sizes: Sequence[int],
    layer_names: Sequence[str],
    layer_ops: Sequence[str],
    dropout_p: float = 0.0,
    save_path: Optional[Union[str, Path]] = None,
    max_neurons: int = 12,
    neuron_radius: float = 0.25,
    neuron_dy: float = 1.15,
    layer_dx: float = 6.0,
    gap_rows: int = 2,
    connection_color: str = "b",
    connection_alpha: float = 1.0,
    title: Optional[str] = None,
) -> plt.Figure:
    """Schematic feed-forward MLP diagram (horizontal layout, empty circles).

    Layers are drawn as columns of circles, with input on the left and output
    on the right. Connection lines start and end exactly on the circle edges
    using a unit-vector offset, so circles are not over-drawn. Layers wider
    than ``max_neurons`` are truncated to roughly half on each end plus an
    explicit empty gap of ``gap_rows`` rows in the middle where a vertical
    ellipsis ("⋮") is rendered, so a 128-unit hidden layer still fits cleanly.

    Parameters
    ----------
    layer_sizes
        Number of units in each layer, from input to output.
    layer_names
        Display name for each layer (same length as ``layer_sizes``).
    layer_ops
        Operation label between adjacent layers (length ``len(layer_sizes) - 1``),
        e.g. ``"Linear + ReLU + Dropout"``. The substring ``"Dropout"`` is
        expanded to ``"Dropout (p = <dropout_p>)"`` automatically.
    dropout_p
        Dropout probability used in the model; interpolated into op labels
        and the figure title.
    save_path
        If given, save the rendered figure as PNG/PDF/SVG. Format is inferred
        from the extension.
    max_neurons
        Maximum circles drawn per layer (wider layers are truncated).
    neuron_radius, neuron_dy, layer_dx, gap_rows
        Geometry knobs for circle size, intra-column spacing, inter-column
        spacing, and the empty gap in truncated columns.
    connection_color, connection_alpha
        Style of the inter-neuron lines.
    title
        Custom title; default shows dropout with **four** decimal places
        (``f"SimpleNN architecture  (dropout p = {_fmt_dropout_p(dropout_p)})"``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure (also saved if ``save_path`` was provided).
    """
    n_layers    = len(layer_sizes)
    drawn_sizes = [min(s, max_neurons) for s in layer_sizes]
    tallest     = max(drawn_sizes)

    # Per-layer (x, y array, was_truncated)
    layer_xy = []
    for i, (n_orig, n_draw) in enumerate(zip(layer_sizes, drawn_sizes)):
        x         = i * layer_dx
        truncated = (n_orig != n_draw)
        if not truncated:
            y_top = (n_draw - 1) * neuron_dy / 2.0
            ys    = y_top - np.arange(n_draw) * neuron_dy
        else:
            n_top  = n_draw // 2
            n_bot  = n_draw - n_top
            span   = (n_top + n_bot + gap_rows - 1) * neuron_dy
            y_max  = span / 2.0
            ys_top = y_max - np.arange(n_top) * neuron_dy
            ys_bot = -y_max + np.arange(n_bot)[::-1] * neuron_dy
            ys     = np.concatenate([ys_top, ys_bot])
        layer_xy.append((x, ys, truncated))

    # Figure sizing scales with depth and tallest column
    fig_w = max(8.5, n_layers * 1.9 + 3.0)
    fig_h = max(5.5, tallest * neuron_dy * 0.45 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Connection lines (terminate exactly on the circle edges)
    for i in range(n_layers - 1):
        x1, ys1, _ = layer_xy[i]
        x2, ys2, _ = layer_xy[i + 1]
        for y1 in ys1:
            for y2 in ys2:
                dx, dy = x2 - x1, y2 - y1
                d      = np.hypot(dx, dy)
                if d == 0:
                    continue
                ux, uy = dx / d, dy / d
                ax.plot(
                    [x1 + ux * neuron_radius, x2 - ux * neuron_radius],
                    [y1 + uy * neuron_radius, y2 - uy * neuron_radius],
                    color=connection_color, lw=0.55,
                    alpha=connection_alpha, zorder=1,
                )

    # Neuron circles (empty)
    for x, ys, truncated in layer_xy:
        for y in ys:
            ax.add_patch(plt.Circle(
                (x, y), neuron_radius,
                fill=False, edgecolor="black", lw=1.1, zorder=3,
            ))
        if truncated:
            ax.text(
                x, 0, "\u22ee",
                ha="center", va="center",
                fontsize=max(10, int(11 * neuron_dy)),
                color="0.25", zorder=4,
            )

    # Layer labels above each column (use the *actual* max y span). Two-line
    # block: "units = N" sits just above the topmost circle, the layer name
    # sits above that. Spacing is in neuron_dy units so it scales with the
    # column geometry.
    y_span_max = max(abs(ys).max() for _, ys, _ in layer_xy)
    y_units    = y_span_max + neuron_dy * 1.4                # subtitle row
    y_name     = y_units    + neuron_dy * 1.2                # layer-name row
    for i, (x, _ys, _trunc) in enumerate(layer_xy):
        ax.text(x, y_name,  layer_names[i],
                ha="center", va="bottom", fontsize=11)
        ax.text(x, y_units, f"units = {layer_sizes[i]}",
                ha="center", va="bottom", fontsize=9, color="0.4")

    # Operation labels below the gap between columns (italic). Long ops are
    # wrapped at " + Dropout" so each label fits within one inter-column gap.
    y_op = -(y_span_max + neuron_dy * 1.6)
    dp_lbl = _fmt_dropout_p(dropout_p)
    for i, op in enumerate(layer_ops):
        x_mid = (layer_xy[i][0] + layer_xy[i + 1][0]) / 2.0
        text  = op if "Dropout" not in op else f"{op} (p = {dp_lbl})"
        text  = text.replace(" + Dropout", "\n+ Dropout")
        ax.text(x_mid, y_op, text, ha="center", va="top",
                fontsize=9, color="0.25", style="italic",
                linespacing=1.25)

    # Frame
    ax.set_xlim(-layer_dx * 0.6, (n_layers - 1) * layer_dx + layer_dx * 0.6)
    ax.set_ylim(y_op - neuron_dy * 2.5, y_name + neuron_dy * 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        title or f"SimpleNN architecture  (dropout p = {dp_lbl})",
        fontsize=12, pad=10,
    )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# (b) Standalone TikZ source
# ─────────────────────────────────────────────────────────────────────────────
def write_tikz_mlp(
    layer_sizes: Sequence[int],
    layer_names: Sequence[str],
    layer_ops: Sequence[str],
    dropout_p: float,
    tex_path: Union[str, Path],
) -> Path:
    """Emit a self-contained TikZ document for the MLP, compilable with pdflatex.

    The produced ``.tex`` file uses the ``standalone`` class and the TikZ
    ``positioning`` / ``arrows.meta`` / ``fit`` libraries. Compile with::

        pdflatex architecture_diagram.tex

    Parameters
    ----------
    layer_sizes, layer_names, layer_ops, dropout_p
        Same semantics as :func:`draw_mlp_architecture`.
    tex_path
        Destination path; parent directory must already exist.

    Returns
    -------
    pathlib.Path
        Path the TikZ source was written to.
    """
    head = (
        r"% Auto-generated by PlugFlowML (src/utils/plot_nn_architecture.py). "
        r"Compile with: pdflatex" "\n"
        r"\documentclass[tikz, border=8pt]{standalone}" "\n"
        r"\usepackage{tikz}" "\n"
        r"\usetikzlibrary{positioning, arrows.meta, fit}" "\n"
        r"\begin{document}" "\n"
        r"\begin{tikzpicture}[" "\n"
        r"  io/.style={draw, thick, rounded corners=2pt, fill=gray!10," "\n"
        r"            minimum width=22mm, minimum height=14mm, align=center, font=\small}," "\n"
        r"  hl/.style={draw, thick, rounded corners=2pt, fill=blue!18," "\n"
        r"            minimum width=22mm, minimum height=14mm, align=center, font=\small}," "\n"
        r"  op/.style={font=\footnotesize\itshape, align=center, text=black!70}," "\n"
        r"  >={Latex[length=2.4mm,width=2.0mm]}, node distance=14mm," "\n"
        r"]" "\n"
    )

    lines = []
    last_id = None
    dp_lbl = _fmt_dropout_p(dropout_p)
    for i, (size, name) in enumerate(zip(layer_sizes, layer_names)):
        nid  = f"L{i}"
        kind = "io" if (i == 0 or i == len(layer_sizes) - 1) else "hl"
        if i == 0:
            label = fr"{name}\\$n={size}$"
        elif i == len(layer_sizes) - 1:
            label = fr"{name}\\$m={size}$"
        else:
            label = fr"{name}\\$h_{i}={size}$"
        position = "" if last_id is None else f", right=of {last_id}"
        lines.append(fr"\node[{kind}{position}] ({nid}) {{{label}}};")
        if last_id is not None:
            op_label = (
                layer_ops[i - 1]
                .replace("+", r"$+$")
                .replace("Dropout", f"Dropout (p={dp_lbl})")
            )
            lines.append(
                fr"\node[op, above=2mm of ${last_id}.north east!0.5!{nid}.north west$] "
                fr"{{{op_label}}};"
            )
            lines.append(fr"\draw[->] ({last_id}) -- ({nid});")
        last_id = nid

    foot = "\n" + r"\end{tikzpicture}" + "\n" + r"\end{document}" + "\n"
    tex_path = Path(tex_path)
    tex_path.write_text(head + "\n".join(lines) + foot, encoding="utf-8")
    return tex_path
