"""Sections and optional plots of probability-space spectrahedra.

A section is ``p = center + B @ y`` with positivity constraint
``sum_i p_i A_i >= 0``. Both real symmetric and complex Hermitian pencils
are supported. Matplotlib and Plotly are imported only when plotting.
Legacy capitalized ``2D``/``3D`` names remain available for notebooks.
"""

import numpy as np
from scipy.linalg import eigh

__all__ = [
    "make_sum_zero_section",
    "section_coordinates",
    "make_icosphere",
    "probability_spectrahedron_section_2D",
    "probability_spectrahedron_section_3D",
    "plot_section_2D_matplotlib",
    "plot_section_3D_matplotlib",
    "mesh_edges_trace",
    "plot_section_3D_plotly",
]


def make_sum_zero_section(directions, tol=1e-12):
    """Project row vectors into the sum-zero hyperplane and orthonormalize.

    Input has shape ``(section_dimension, outcomes)``; the returned basis
    has shape ``(outcomes, section_dimension)``. All projected directions
    must be independent. One-dimensional sections are supported too.
    """
    directions = np.asarray(directions, dtype=float)
    if (
        directions.ndim != 2
        or not directions.size
        or not np.all(np.isfinite(directions))
    ):
        raise ValueError("directions must be a finite array of row vectors.")
    count, n = directions.shape
    if count >= n:
        raise ValueError("At most n-1 independent sum-zero directions are possible.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite.")
    projected = directions.T - directions.mean(axis=1)
    singular_values = np.linalg.svd(projected, compute_uv=False)
    if singular_values[-1] <= tol * max(1.0, singular_values[0]):
        raise ValueError("The projected directions are not independent.")
    return np.linalg.qr(projected, mode="reduced")[0]


def section_coordinates(
    probability_vectors, x_center, B, tol=1e-8, require_in_section=True
):
    """Return coordinates and Euclidean residuals from an affine section.

    Solve ``p = x_center + B @ y`` by least squares. A single input vector
    returns ``(y, scalar_residual)``; shape ``(k,n)`` returns ``(Y,residuals)``.
    The basis need not be orthonormal, but must have independent columns.
    Unless ``require_in_section=False``, residuals above ``tol`` raise an
    error. This is a coordinate conversion, not a positivity check.
    """
    points = np.asarray(probability_vectors, dtype=float)
    center = np.asarray(x_center, dtype=float)
    basis = np.asarray(B, dtype=float)
    single = points.ndim == 1
    if points.ndim not in (1, 2) or center.ndim != 1 or not center.size:
        raise ValueError(
            "Expected a vector or rows of points and a nonempty center vector."
        )
    points = np.atleast_2d(points)
    if basis.ndim != 2 or basis.shape[0] != len(center) or not basis.shape[1]:
        raise ValueError("B must have one row per probability component.")
    if points.shape[1] != len(center):
        raise ValueError("Each probability vector must have the same size as x_center.")
    if not all(np.all(np.isfinite(a)) for a in (points, center, basis)):
        raise ValueError("Section inputs must be finite.")
    if np.linalg.matrix_rank(basis) != basis.shape[1]:
        raise ValueError("B must have independent columns.")
    coordinates = np.linalg.lstsq(basis, (points - center).T, rcond=None)[0].T
    residuals = np.linalg.norm(points - center - coordinates @ basis.T, axis=1)
    if require_in_section and np.any(residuals > tol):
        bad = np.flatnonzero(residuals > tol)
        raise ValueError(
            f"Points {bad.tolist()} do not lie in the section; residuals={residuals[bad]}."
        )
    return (coordinates[0], residuals[0]) if single else (coordinates, residuals)


def _ray_section(A_matrices, x_center, B, directions, tol):
    """Shoot rays until the first simplex facet or PSD singularity."""
    matrices = np.asarray(A_matrices, dtype=complex)
    center = np.asarray(x_center, dtype=float)
    basis = np.asarray(B, dtype=float)
    if center.ndim != 1 or not center.size:
        raise ValueError("x_center must be a nonempty probability vector.")
    n, section_dimension = len(center), directions.shape[1]
    if (
        matrices.ndim != 3
        or matrices.shape[0] != n
        or matrices.shape[1] != matrices.shape[2]
        or not matrices.size
    ):
        raise ValueError("A_matrices must have shape (outcomes, size, size).")
    if basis.shape != (n, section_dimension):
        raise ValueError(f"B must have shape ({n}, {section_dimension}).")
    if not all(np.all(np.isfinite(a)) for a in (matrices, center, basis)):
        raise ValueError("Section inputs must be finite.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite.")
    if not np.isclose(center.sum(), 1, atol=tol, rtol=0) or np.min(center) <= tol:
        raise ValueError("x_center must sum to one and have strictly positive entries.")
    if not np.allclose(basis.sum(axis=0), 0, atol=tol, rtol=0):
        raise ValueError("Each column of B must sum to zero.")
    if np.linalg.matrix_rank(basis, tol=tol) != section_dimension:
        raise ValueError("B must have independent columns.")
    matrices = (matrices + matrices.conj().swapaxes(-1, -2)) / 2
    origin = np.einsum("i,ijk->jk", center, matrices)
    if np.linalg.eigvalsh(origin).min() <= tol:
        raise ValueError("A(x_center) must be positive definite for ray shooting.")

    boundary = np.empty_like(directions)
    probabilities = np.empty((len(directions), n))
    active = []
    for index, direction in enumerate(directions):
        velocity = basis @ direction
        decreasing = velocity < -tol
        simplex_radius = (
            np.min(-center[decreasing] / velocity[decreasing])
            if decreasing.any()
            else np.inf
        )
        pencil_direction = np.einsum("i,ijk->jk", velocity, matrices)
        eigenvalue = eigh(
            pencil_direction, origin, eigvals_only=True, check_finite=False
        )[0]
        psd_radius = -1 / eigenvalue if eigenvalue < -tol else np.inf
        radius = min(simplex_radius, psd_radius)
        if not np.isfinite(radius):
            raise ValueError(
                "An unbounded ray was found; directions may be too small relative to tol."
            )
        boundary[index] = radius * direction
        probabilities[index] = center + radius * velocity
        if (
            np.isfinite(simplex_radius)
            and np.isfinite(psd_radius)
            and np.isclose(simplex_radius, psd_radius, atol=tol, rtol=tol)
        ):
            active.append("both")
        else:
            active.append("simplex" if simplex_radius < psd_radius else "PSD")
    return {
        "boundary_y": boundary,
        "boundary_x": probabilities,
        "active": np.asarray(active),
        "A_matrices": matrices,
        "x_center": center,
        "B": basis,
    }


def probability_spectrahedron_section_2D(
    A_matrices, x_center, B, num_angles=1000, tol=1e-10
):
    """Sample a 2D section of ``p>=0, sum(p)=1, sum_i p_i A_i>=0``.

    ``A_matrices`` has shape ``(n,m,m)``, ``x_center`` shape ``(n,)``, and
    ``B`` shape ``(n,2)`` with independent sum-zero columns. The center
    must have strictly positive coordinates and a positive definite
    matrix pencil. Input matrices are Hermitianized before calculation.

    Return a dictionary with intrinsic ``boundary_y``, corresponding
    ``boundary_x`` probabilities, ``active_constraint`` labels (simplex,
    PSD, or both), and the input pencil, center, and basis. Rays sample
    equally spaced angles; the result is a boundary approximation.
    """
    if (
        isinstance(num_angles, bool)
        or not isinstance(num_angles, (int, np.integer))
        or num_angles < 3
    ):
        raise ValueError("num_angles must be an integer of at least three.")
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    result = _ray_section(
        A_matrices, x_center, B, np.column_stack([np.cos(angles), np.sin(angles)]), tol
    )
    result["active_constraint"] = result.pop("active")
    return result


def make_icosphere(subdivisions=3):
    """
    Return approximately uniform vertices on S^2 and triangular faces.
    """

    phi = (1.0 + np.sqrt(5.0)) / 2.0

    vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=float,
    )

    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=int,
    )

    if (
        isinstance(subdivisions, bool)
        or not isinstance(subdivisions, (int, np.integer))
        or subdivisions < 0
    ):
        raise ValueError("subdivisions must be a nonnegative integer.")

    for _ in range(subdivisions):
        midpoint_cache = {}
        new_vertices = vertices.tolist()
        new_faces = []

        def midpoint_index(i, j):
            key = tuple(sorted((i, j)))

            if key in midpoint_cache:
                return midpoint_cache[key]

            midpoint = (vertices[i] + vertices[j]) / 2.0
            midpoint /= np.linalg.norm(midpoint)

            index = len(new_vertices)
            new_vertices.append(midpoint.tolist())
            midpoint_cache[key] = index
            return index

        for i, j, k in faces:
            a = midpoint_index(i, j)
            b = midpoint_index(j, k)
            c = midpoint_index(k, i)

            new_faces.extend(
                [
                    [i, a, c],
                    [j, b, a],
                    [k, c, b],
                    [a, b, c],
                ]
            )

        vertices = np.asarray(new_vertices, dtype=float)
        faces = np.asarray(new_faces, dtype=int)

    return vertices, faces


def probability_spectrahedron_section_3D(
    A_matrices, x_center, B, subdivisions=3, tol=1e-10
):
    """Triangulate a 3D section by shooting rays through an icosphere.

    Inputs follow :func:`probability_spectrahedron_section_2D`, with ``B``
    of shape ``(n,3)``. Subdivisions 0, 1, 2, 3 give 12, 42, 162, 642
    boundary vertices. Return ``boundary_y``, ``boundary_x``, triangular
    ``faces``, ``active`` labels (simplex, PSD, or both), and the input
    pencil, center, and basis. Faces approximate the curved boundary.
    """
    directions, faces = make_icosphere(subdivisions)
    result = _ray_section(A_matrices, x_center, B, directions, tol)
    result["faces"] = faces
    return result


def _plot_sections(sections, section_labels):
    sections, labels = list(sections), list(section_labels)
    if not sections:
        raise ValueError("At least one section is required.")
    if labels and len(labels) != len(sections):
        raise ValueError("Provide one label per section, or no labels.")
    return sections, labels


def _marked(section):
    points = section.get("marked_probabilities", [])
    if not len(points):
        return np.empty((0, section["B"].shape[1])), []
    points = np.atleast_2d(points)
    coordinates, _ = section_coordinates(points, section["x_center"], section["B"])
    labels = section.get("labels", [""] * len(points))
    if len(labels) != len(points):
        raise ValueError("Provide one label per marked probability vector.")
    return coordinates, labels


def plot_section_2D_matplotlib(sections, section_labels=(), *, show=True):
    """Plot 2D boundaries and marked points; return a Matplotlib figure.

    Each section is the dictionary returned by the 2D ray-shooting helper.
    Optional ``marked_probabilities`` and ``labels`` add annotated points.
    ``show=False`` supports notebooks, exports, and noninteractive tests.
    """
    import matplotlib.pyplot as plt

    sections, labels = _plot_sections(sections, section_labels)
    figure, axes = plt.subplots(figsize=(7, 7))
    for index, section in enumerate(sections):
        boundary = section["boundary_y"]
        closed = np.vstack([boundary, boundary[:1]])
        (line,) = axes.plot(
            closed[:, 0], closed[:, 1], label=labels[index] if labels else None
        )
        axes.fill(boundary[:, 0], boundary[:, 1], color=line.get_color(), alpha=0.15)
        marked, annotations = _marked(section)
        if len(marked):
            axes.scatter(marked[:, 0], marked[:, 1], color=line.get_color(), zorder=5)
            for point, label in zip(marked, annotations):
                axes.annotate(label, point, xytext=(5, 5), textcoords="offset points")
    axes.set_aspect("equal")
    if labels:
        axes.legend()
    if show:
        plt.show()
    return figure


def plot_section_3D_matplotlib(sections, section_labels=(), *, show=True):
    """Plot 3D section meshes with optional marked probabilities and labels.

    Return a Matplotlib figure. Section colors are deterministic and the
    axes enclose all supplied sections. ``show=False`` suppresses display.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sections, labels = _plot_sections(sections, section_labels)
    figure = plt.figure(figsize=(9, 8))
    axes = figure.add_subplot(111, projection="3d")
    for index, section in enumerate(sections):
        boundary = section["boundary_y"]
        surface = Poly3DCollection(
            boundary[section["faces"]],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.15,
            label=labels[index] if labels else None,
            facecolors=plt.get_cmap("tab10")(index % 10),
        )
        axes.add_collection3d(surface)
        marked, annotations = _marked(section)
        if len(marked):
            axes.scatter(*marked.T, color="red", depthshade=False)
            for point, label in zip(marked, annotations):
                axes.text(*point, label)
    points = np.concatenate([section["boundary_y"] for section in sections])
    for axis, setter in enumerate((axes.set_xlim, axes.set_ylim, axes.set_zlim)):
        setter(points[:, axis].min(), points[:, axis].max())
    axes.set_box_aspect(np.ptp(points, axis=0))
    if labels:
        axes.legend()
    if show:
        plt.show()
    return figure


def mesh_edges_trace(vertices, faces, width=1):
    """Return a Plotly line trace with every undirected mesh edge once."""
    import plotly.graph_objects as go

    vertices, faces = np.asarray(vertices), np.asarray(faces)
    edges = {
        tuple(sorted(pair)) for i, j, k in faces for pair in ((i, j), (j, k), (k, i))
    }
    coordinates = [[], [], []]
    for first, second in sorted(edges):
        for axis in range(3):
            coordinates[axis].extend(
                (vertices[first, axis], vertices[second, axis], None)
            )
    return go.Scatter3d(
        x=coordinates[0],
        y=coordinates[1],
        z=coordinates[2],
        mode="lines",
        line={"color": "black", "width": width},
        hoverinfo="skip",
        showlegend=False,
    )


def plot_section_3D_plotly(sections, section_labels=(), *, show=True):
    """Return an interactive Plotly section figure, optionally displaying it.

    Section dictionaries may contain ``marked_probabilities`` and matching
    ``labels``. Plotly is an optional dependency, imported only on this call.
    """
    import plotly.graph_objects as go

    sections, labels = _plot_sections(sections, section_labels)
    figure = go.Figure()
    for index, section in enumerate(sections):
        boundary, faces = section["boundary_y"], section["faces"]
        figure.add_trace(
            go.Mesh3d(
                x=boundary[:, 0],
                y=boundary[:, 1],
                z=boundary[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.6,
                flatshading=True,
                showlegend=bool(labels),
                name=labels[index] if labels else "",
                hovertemplate="y1=%{x:.4f}<br>y2=%{y:.4f}<br>y3=%{z:.4f}<extra></extra>",
            )
        )
        figure.add_trace(mesh_edges_trace(boundary, faces))
        marked, annotations = _marked(section)
        if len(marked):
            figure.add_trace(
                go.Scatter3d(
                    x=marked[:, 0],
                    y=marked[:, 1],
                    z=marked[:, 2],
                    mode="markers+text",
                    text=annotations,
                    textposition="top center",
                    marker={"size": 6},
                    showlegend=False,
                )
            )
    figure.update_layout(
        width=1000,
        height=850,
        margin={"l": 0, "r": 0, "b": 0, "t": 30},
        scene={"aspectmode": "data", "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 1.2}}},
    )
    if show:
        figure.show(config={"responsive": True})
    return figure
