"""
Fourier Series Visualizer — drawing shapes with rotating epicycles
====================================================================

The core idea (from Fourier's theorem): any closed curve traced by a
periodic function f(t) = x(t) + i*y(t)   (t in [0, 2*pi))
can be written as a sum of rotating vectors ("epicycles"):

    f(t)  =  sum_{n=-N}^{N}  c_n * exp(i * n * t)

The coefficients c_n are found by projecting f(t) onto each complex
exponential — literally the Fourier integral:

    c_n = (1 / 2*pi) * INTEGRAL_0^2pi  f(t) * exp(-i * n * t) dt

Since we only have the shape as a finite list of sampled (x, y) points
(not a closed-form function), that integral can't be solved
analytically. Instead we approximate it with a Riemann sum over the
sampled points — this is the "integral approximation" part:

    c_n  ≈  (1 / M) * sum_{k=0}^{M-1}  f(t_k) * exp(-i * n * t_k)

where t_k = 2*pi*k/M are M equally spaced samples. As M -> infinity,
this Riemann sum converges to the true integral — the same limiting
argument used to define the Riemann integral in calculus.

Each c_n becomes one epicycle:
    - radius            = |c_n|
    - starting angle    = arg(c_n)
    - rotation speed     = n  (revolutions per period)

Chained tip-to-tail, in order of decreasing |c_n| (most "important"
frequency first), their sum traces the original shape and the pen tip
draws it out over one full rotation.

Usage
-----
    python fourier_epicycles.py                      # draws a heart
    python fourier_epicycles.py --shape star
    python fourier_epicycles.py --shape custom --points path.csv
    python fourier_epicycles.py --terms 41 --out epicycles.gif
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _resample_by_arclength(x, y, num_points):
    """Re-sample a closed polyline (x, y) at `num_points` equally spaced
    positions along its arc length, using linear interpolation."""
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    seg = np.hypot(np.diff(x), np.diff(y))
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    total = arc[-1]
    targets = np.linspace(0, total, num_points, endpoint=False)
    xs = np.interp(targets, arc, x)
    ys = np.interp(targets, arc, y)
    return xs, ys


def heart_shape(num_points=400):
    t = np.linspace(0, 2 * np.pi, 2000)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return _resample_by_arclength(x, y, num_points)


def star_shape(num_points=400, spikes=5, outer=1.0, inner=0.42):
    angles = np.linspace(0, 2 * np.pi, spikes * 2, endpoint=False)
    radii = np.array([outer if i % 2 == 0 else inner for i in range(spikes * 2)])
    x = radii * np.cos(angles + np.pi / 2)
    y = radii * np.sin(angles + np.pi / 2)
    xs_full, ys_full = [], []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        x1, y1 = x[(i + 1) % len(x)], y[(i + 1) % len(y)]
        seg_t = np.linspace(0, 1, 60, endpoint=False)
        xs_full.append(x0 + (x1 - x0) * seg_t)
        ys_full.append(y0 + (y1 - y0) * seg_t)
    x = np.concatenate(xs_full) * 12
    y = np.concatenate(ys_full) * 12
    return _resample_by_arclength(x, y, num_points)


def infinity_shape(num_points=400):
    t = np.linspace(0, 2 * np.pi, 2000)
    a = 12
    x = a * np.cos(t) / (1 + np.sin(t) ** 2)
    y = a * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
    return _resample_by_arclength(x, y, num_points)


def load_custom_points(path, num_points=400):
    """Load a CSV file of `x,y` rows (one point per line) and resample it."""
    data = np.loadtxt(path, delimiter=",")
    x, y = data[:, 0], data[:, 1]
    return _resample_by_arclength(x, y, num_points)


SHAPES = {
    "heart": heart_shape,
    "star": star_shape,
    "infinity": infinity_shape,
}


def compute_fourier_coefficients(x, y, num_terms):
    """
    Approximate c_n = (1/M) * sum_k f(t_k) * exp(-i n t_k)  for
    n = -num_terms .. +num_terms, where f(t_k) = x_k + i*y_k.

    Returns a list of (n, c_n) sorted by |c_n| descending — the order
    epicycles should be chained in so the biggest, most important
    "strokes" of the shape are drawn by the innermost circles.
    """
    M = len(x)
    f = x + 1j * y                      
    t = np.linspace(0, 2 * np.pi, M, endpoint=False)

    coeffs = []
    for n in range(-num_terms, num_terms + 1):
       
        c_n = np.sum(f * np.exp(-1j * n * t)) / M
        coeffs.append((n, c_n))

    coeffs.sort(key=lambda pair: -np.abs(pair[1]))
    return coeffs


def animate_epicycles(coeffs, frames=240, out_path="epicycles.gif", fps=30,
                       title="Fourier Series Epicycles"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor("#0b0f19")
    fig.patch.set_facecolor("#0b0f19")
    ax.set_aspect("equal")
    ax.axis("off")

    max_radius = sum(abs(c) for _, c in coeffs)
    lim = max_radius * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(title, color="#e8e8e8", fontsize=13, pad=12)

    circle_lines, radius_lines = [], []
    for _ in coeffs:
        (c_line,) = ax.plot([], [], color="#3a5a80", lw=0.8, alpha=0.6)
        circle_lines.append(c_line)
    (vector_line,) = ax.plot([], [], color="#7fa8d9", lw=1.3)
    (trace_line,) = ax.plot([], [], color="#f2c744", lw=2.0)
    (pen_dot,) = ax.plot([], [], "o", color="#f2c744", markersize=5)

    trail_x, trail_y = [], []
    theta = np.linspace(0, 2 * np.pi, 80)

    def init():
        trail_x.clear()
        trail_y.clear()
        return circle_lines + [vector_line, trace_line, pen_dot]

    def update(frame):
        t = 2 * np.pi * frame / frames
        x, y = 0.0, 0.0
        vx, vy = [0.0], [0.0]
        for (n, c), circ_line in zip(coeffs, circle_lines):
            r = abs(c)
            circ_line.set_data(x + r * np.cos(theta), y + r * np.sin(theta))
            x += (c * np.exp(1j * n * t)).real
            y += (c * np.exp(1j * n * t)).imag
            vx.append(x)
            vy.append(y)

        vector_line.set_data(vx, vy)
        trail_x.append(x)
        trail_y.append(y)
        trace_line.set_data(trail_x, trail_y)
        pen_dot.set_data([x], [y])
        return circle_lines + [vector_line, trace_line, pen_dot]

    anim = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=1000 / fps
    )
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out_path




def main():
    parser = argparse.ArgumentParser(description="Fourier series epicycle visualizer")
    parser.add_argument("--shape", default="heart", choices=list(SHAPES) + ["custom"])
    parser.add_argument("--points", default=None, help="CSV file of x,y points (for --shape custom)")
    parser.add_argument("--samples", type=int, default=300, help="points sampled along the shape")
    parser.add_argument("--terms", type=int, default=41, help="number of epicycles = 2*terms+1")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", default="epicycles.gif")
    args = parser.parse_args()

    if args.shape == "custom":
        if not args.points:
            raise SystemExit("--shape custom requires --points path/to/file.csv")
        x, y = load_custom_points(args.points, args.samples)
    else:
        x, y = SHAPES[args.shape](args.samples)

    coeffs = compute_fourier_coefficients(x, y, args.terms)
    out = animate_epicycles(coeffs, frames=args.frames, out_path=args.out, fps=args.fps,
                             title=f"{args.shape.title()} — {2*args.terms+1} epicycles")
    print(f"Saved animation to {out}")


if __name__ == "__main__":
    main()
