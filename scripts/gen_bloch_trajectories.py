"""Bloch-sphere trajectory comparison, emitted as TikZ (algebraic chapter).

Five panels, target U3(1.9, 1.4, 0.95) applied to |+>, eps = 1e-2 (the
U3-decomposition example of the bloch animations in ~/Documents/research/bloch):

  (a) the target unitary as ONE continuous rotation: the single arc about the
      unitary's own axis, start to end (the ground truth);
  (b) the Euler route U = Rz(alpha) Ry(beta) Rz(gamma): three single-axis
      rotations, one color each;
  (c) the same route discretized per rotation (the gridsynth route): Rz(gamma)
      and Rz(alpha) synthesized to Clifford+T directly, Ry(beta) synthesized as
      S H Rz(beta) H S^dag (exact Clifford conjugation, drawn gray); discrete
      arcs inherit the parent rotation's color;
  (d) the direct Clifford+T circuit for the whole unitary;
  (e) the direct Clifford+sqrt(T) circuit.
In (d)/(e) arcs are colored by gate family: H flips vs z-rotations vs sqrt(T).

All discrete paths are honest: gate strings come from cyclosynth in-process
(requires the `cyclosynth` conda env), each gate drawn as its Bloch rotation
arc. Orthographic camera; far-hemisphere segments drawn first at low opacity.

PAPER COPY (docs/paper FIG. 1). Diverged intentionally from the dissertation's
figure-sources/algebraic/scripts/gen_bloch_trajectories.py (2026-07-02); edits
here affect the paper only.

Outputs (in scripts/data/):
  bloch_trajectories_tikz.tex        -- bare tikzpicture fragment
  bloch_trajectories_standalone.tex  -- self-contained wrapper
  bloch_trajectories.pdf             -- compiled figure (pdflatex)
Copy the PDF to docs/paper/figures/bloch_trajectories.pdf to update the paper.
NOTE: cyclosynth's search is not run-to-run deterministic; gate counts in the
panel titles change between runs, so regenerate deliberately.
"""
import numpy as np
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "data" / "bloch_trajectories_tikz.tex"
U3 = (1.51, 1.43, -2.55)  # front-facing arcs, early arcs clear of the ring, ring near frame center
EPS = 1e-2              # the animations' epsilon
R = 1.0
STEP = np.deg2rad(4)
SCALE = 2.2
PITCH = 2.5             # panel spacing

# Bloch rotation (axis, angle) per gate token (lowercase = adjoint).
AXZ, AXX, AXY = np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
AXH = np.array([1.0, 0, 1.0]) / np.sqrt(2)
GATES = {
    "H": (AXH, np.pi), "X": (AXX, np.pi), "Y": (AXY, np.pi), "Z": (AXZ, np.pi),
    "S": (AXZ, np.pi / 2), "s": (AXZ, -np.pi / 2),
    "T": (AXZ, np.pi / 4), "t": (AXZ, -np.pi / 4),
    "Q": (AXZ, np.pi / 8), "q": (AXZ, -np.pi / 8),
}
HCOL, ZCOL, QCOL, XCOL = "trajH", "trajZ", "trajQ", "trajX"
GCOL = "gtruth"                                       # ground-truth single rotation
ECOLS = ["eulerA", "eulerB", "eulerC"]                # Rz(gamma), Ry(beta), Rz(alpha)

# ---- camera (orthographic) --------------------------------------------------
elev, azim = np.deg2rad(22), np.deg2rad(-60)
d = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
r_ = np.cross(np.array([0, 0, 1.0]), d); r_ /= np.linalg.norm(r_)
u_ = np.cross(d, r_)


def proj(p):
    return float(p @ r_), float(p @ u_), float(p @ d)   # (x2d, y2d, depth)


def rot(v, axis, ang):
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def arc(v, axis, ang):
    """Sample the rotation arc of Bloch vector v about axis by ang."""
    n = max(2, int(abs(ang) / STEP) + 1)
    return [rot(v, axis, ang * i / n) for i in range(1, n + 1)]


def runs(points):
    """Split a 3-D polyline into (is_front, [2d pts]) runs by camera depth."""
    out, cur, front = [], [], None
    for p in points:
        x, y, dep = proj(p)
        f = dep >= 0
        if front is None or f == front:
            cur.append((x, y))
        else:
            cur.append((x, y))            # share the boundary point
            out.append((front, cur))
            cur = [(x, y)]
        front = f
    if cur:
        out.append((front, cur))
    return out


def path_tex(points, color, lw="0.9pt", arrow=False):
    """TikZ draws for one colored polyline, far-hemisphere runs dimmed."""
    lines = []
    for front, pts in runs(points):
        if len(pts) < 2:
            continue
        op = "0.92" if front else "0.28"
        coords = " -- ".join(f"({x:.4f},{y:.4f})" for x, y in pts)
        lines.append(f"\\draw[{color}, line width={lw}, opacity={op}, "
                     f"line cap=round, line join=round] {coords};")
    lines.sort(key=lambda s: "opacity=0.92" in s)   # back runs first
    if arrow and len(points) >= 2:
        xe, ye, _ = proj(points[-2]); xf, yf, _ = proj(points[-1])
        lines.append(f"\\draw[{color}, line width={lw}, opacity=0.92, "
                     f"-{{Stealth[length=1.7mm]}}] ({xe:.4f},{ye:.4f}) -- ({xf:.4f},{yf:.4f});")
    return lines


def sphere_tex(x0, title, extra_lines):
    """One panel: sphere, equator, poles, |+>, then the trajectory lines."""
    L = [f"\\begin{{scope}}[shift={{({x0},0)}}]",
         f"\\fill[ballbg] (0,0) circle ({R});",
         f"\\draw[ballrim] (0,0) circle ({R});"]
    eq = [np.array([np.cos(t), np.sin(t), 0]) for t in np.linspace(0, 2 * np.pi, 120)]
    for front, pts in runs(eq):
        style = "eqfront" if front else "eqback"
        coords = " -- ".join(f"({x:.4f},{y:.4f})" for x, y in pts)
        L.append(f"\\draw[{style}] {coords};")
    for p, lab, anch in [(np.array([0, 0, 1.0]), r"\ket{0}", "south"),
                         (np.array([0, 0, -1.0]), r"\ket{1}", "north"),
                         (np.array([1.0, 0, 0]), r"\ket{+}", "west")]:
        x, y, _ = proj(p)
        off = 0.18 if anch == "south" else -0.18 if anch == "north" else 0
        dx = 0.28 if anch == "west" else 0
        L.append(f"\\fill[black!60] ({x:.4f},{y:.4f}) circle (0.55pt);")
        L.append(f"\\node[font=\\scriptsize, black!70] at ({x + dx:.4f},{y + off:.4f}) {{${lab}$}};")
    L += extra_lines
    L.append(f"\\node[font=\\small, align=center] at (0,{-R - 0.5}) {{{title}}};")
    L.append("\\end{scope}")
    return L


def walk(v0, steps):
    """Walk v0 through (color, axis, angle, lw, arrow) steps; return tikz lines."""
    lines, v = [], v0
    for col, axis, ang, lw, arrow in steps:
        pts = [v] + arc(v, axis, ang)
        lines += path_tex(pts, col, lw=lw, arrow=arrow)
        v = pts[-1]
    return lines, v


def u3_matrix(a, b, c):
    return np.array([[np.cos(b / 2), -np.exp(1j * c) * np.sin(b / 2)],
                     [np.exp(1j * a) * np.sin(b / 2),
                      np.exp(1j * (a + c)) * np.cos(b / 2)]])


def axis_angle(U):
    """Rotation axis and angle of a 2x2 unitary (up to global phase)."""
    Up = U / np.sqrt(np.linalg.det(U))                    # SU(2)
    ca = np.real(Up[0, 0] + Up[1, 1]) / 2                 # cos(theta/2)
    sx = -np.imag(Up[0, 1] + Up[1, 0]) / 2
    sy = np.real(Up[1, 0] - Up[0, 1]) / 2
    sz = -np.imag(Up[0, 0] - Up[1, 1]) / 2
    sv = np.array([sx, sy, sz])
    sn = np.linalg.norm(sv)
    theta = 2 * np.arctan2(sn, ca)
    return sv / sn, theta


def main():
    import cyclosynth
    al, be, ga = U3
    vplus = np.array([1.0, 0, 0])

    # exact target of |+> under U = Rz(al) Ry(be) Rz(ga)
    vt = rot(rot(rot(vplus, AXZ, ga), AXY, be), AXZ, al)

    # ground-truth single rotation: axis-angle of the whole unitary
    gax, gth = axis_angle(u3_matrix(al, be, ga))
    vg = rot(vplus, gax, gth)
    if np.linalg.norm(vg - vt) > 1e-9:                    # fix rotation sense
        gax, gth = -gax, gth
        vg = rot(vplus, gax, gth)
    assert np.linalg.norm(vg - vt) < 1e-9, "axis-angle does not reach the ZYZ target"

    def marks():
        xs, ys, _ = proj(vplus); xt, yt, _ = proj(vt)
        return [f"\\fill[black] ({xs:.4f},{ys:.4f}) circle (1.4pt);",
                f"\\draw[targetmark] ({xt:.4f},{yt:.4f}) circle (1.9pt);"]

    panels = []

    # ---- (a) ground truth: ONE continuous rotation ---------------------------
    lines_g, _ = walk(vplus, [(GCOL, gax, gth, "1.5pt", True)])
    panels.append(("Target $U$\\\\ one rotation", lines_g + marks()))

    # ---- (b) continuous Euler ZYZ ---------------------------------------------
    steps_e = [(ECOLS[0], AXZ, ga, "1.5pt", True),
               (ECOLS[1], AXY, be, "1.5pt", True),
               (ECOLS[2], AXZ, al, "1.5pt", True)]
    lines_e, _ = walk(vplus, steps_e)
    panels.append(("$R_z(\\alpha)\\,R_y(\\beta)\\,R_z(\\gamma)$\\\\ 3 rotations",
                   lines_e + marks()))

    # ---- (c) per-rotation Clifford+T (the gridsynth route) --------------------
    # Rz(ga), Rz(al) synthesized directly; Ry(be) = S H Rz(be) H S^dag with the
    # conjugating Cliffords exact (gray).
    def rz_steps(ang, col):
        res = cyclosynth.Synthesizer(epsilon=EPS, sqrt_t=False).synthesize_u1(ang)
        toks = list(reversed(list(res.gates)))
        return [(col, *GATES[g], "0.7pt", False) for g in toks], len(toks), int(res.cost)

    steps_c, total, cost = [], 0, 0
    sc, nc, cc = rz_steps(ga, ECOLS[0]); steps_c += sc; total += nc; cost += cc
    steps_c += [(XCOL, *GATES["s"], "0.9pt", False), (XCOL, *GATES["H"], "0.9pt", False)]
    sb, nb, cb = rz_steps(be, ECOLS[1]); steps_c += sb; total += nb; cost += cb
    steps_c += [(XCOL, *GATES["H"], "0.9pt", False), (XCOL, *GATES["S"], "0.9pt", False)]
    sa, na, ca = rz_steps(al, ECOLS[2]); steps_c += sa; total += na; cost += ca
    total += 4
    lines_c, _ = walk(vplus, steps_c)
    panels.append((f"$R_z(\\alpha)\\,R_y(\\beta)\\,R_z(\\gamma) \\rightarrow{{}}$\\cliffordt\\\\ {total} gates, cost {cost}",
                   lines_c + marks()))

    # ---- (d), (e) direct synthesis --------------------------------------------
    for st, title in [(False, "$U \\rightarrow{}$\\cliffordt"),
                      (True, "$U \\rightarrow{}$\\cliffordq")]:
        # NB: cyclosynth's synthesize_u3 takes (theta, phi, lam) Qiskit-style,
        # i.e. the POLAR angle first: Rz(phi) Ry(theta) Rz(lam). Our target is
        # Rz(al) Ry(be) Rz(ga), so pass (be, al, ga).
        res = cyclosynth.Synthesizer(epsilon=EPS, sqrt_t=st).synthesize_u3(be, al, ga)
        toks = list(reversed(list(res.gates)))
        col = "dirQ" if st else "dirT"        # cyclosynth paper scheme, one color per panel
        steps = [(col, *GATES[g], "0.7pt", False) for g in toks]
        lines, vend = walk(vplus, steps)
        assert np.linalg.norm(vend - vt) < 0.05, f"direct path misses target: {np.linalg.norm(vend - vt):.3f}"
        panels.append((f"{title}\\\\ {len(toks)} gates, cost {int(res.cost)}",
                       lines + marks()))

    body = [
        "% Auto-generated by gen_bloch_trajectories.py -- do not hand-edit;",
        f"% target U3{U3}, eps={EPS:g}, start |+>, ZYZ decomposition.",
        f"\\begin{{tikzpicture}}[scale={SCALE}]",
        "\\definecolor{gtruth}{HTML}{2a7f62}",   # ground truth: green
        "\\definecolor{eulerA}{HTML}{c66a12}",   # Rz(gamma) orange
        "\\definecolor{eulerB}{HTML}{2e8b8b}",   # Ry(beta) teal
        "\\definecolor{eulerC}{HTML}{7a4b9e}",   # Rz(alpha) purple
        "\\definecolor{dirT}{HTML}{0072B2}",    # cyclosynth Clifford+T (paper blue)
        "\\definecolor{dirQ}{HTML}{CC79A7}",    # cyclosynth Clifford+sqrt(T) (paper reddish-purple)
        "\\colorlet{trajX}{black!45}",           # exact Cliffords (conjugation)
        "\\colorlet{ballbg}{cyan!7}",
        "\\colorlet{ballrim}{black!55}",
        "\\tikzset{eqfront/.style={black!40, line width=0.5pt},",
        "         eqback/.style={black!22, line width=0.4pt, dash pattern=on 1.6pt off 1.4pt},",
        "         targetmark/.style={green!45!black, line width=0.9pt}}",
    ]
    for i, (title, lines) in enumerate(panels):
        body += sphere_tex(i * PITCH, f"({chr(97 + i)}) {title}", lines)
    # tight bounding box: hug the spheres and their labels, no side slack
    body.append(f"\\pgfresetboundingbox")
    body.append(f"\\useasboundingbox ({-R - 0.01},{-R - 0.69}) rectangle "
                f"({4 * PITCH + R + 0.13},{R + 0.18});")
    body.append("\\end{tikzpicture}")
    OUT.write_text("\n".join(body) + "\n")
    print(f"wrote {OUT} ({len(body)} lines)")
    write_wrapper_and_compile()


WRAPPER = r"""% Self-contained wrapper for the Bloch trajectory figure (paper FIG. 1).
\documentclass[border=2pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\usepackage{braket}
\usepackage{xspace}
\newcommand{\cliffordt}{Clifford+\ensuremath{T}\xspace}
\newcommand{\cliffordq}{Clifford+\ensuremath{\sqrt{T}}\xspace}
\begin{document}
\input{bloch_trajectories_tikz.tex}
\end{document}
"""


def write_wrapper_and_compile():
    import shutil
    import subprocess
    data = OUT.parent
    (data / "bloch_trajectories_standalone.tex").write_text(WRAPPER)
    if shutil.which("pdflatex") is None:
        print("pdflatex not found; compile bloch_trajectories_standalone.tex manually")
        return
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "bloch_trajectories_standalone.tex"],
                       cwd=data, capture_output=True, text=True)
    if r.returncode != 0:
        print("pdflatex FAILED; see", data / "bloch_trajectories_standalone.log")
        return
    (data / "bloch_trajectories_standalone.pdf").replace(data / "bloch_trajectories.pdf")
    print(f"compiled {data / 'bloch_trajectories.pdf'}")
    print("copy it to docs/paper/figures/bloch_trajectories.pdf to update the paper")


if __name__ == "__main__":
    main()
