"""Analysis and optional Matplotlib figures for historical sky/ground tasks."""

from pathlib import Path
import numpy as np

from .sky_ground import (
    DEFAULT_BASE_DIR,
    BasisMeasurementAfterWHPOVMOnBasisStatesTask,
    BasisMeasurementOnBasisStatesTask,
    CharacterizeWHReferenceDeviceTask,
    exactify,
    load_sky_ground_results,
)

####################################################################################


def remove_task_segment(path):
    """Helper for turning task paths into something more readable."""
    parts = path.split("/")
    filtered = [p for p in parts if not p.endswith("Task")]
    return "/".join(filtered)


def add_caption(fig, text, pad=0.02, **kwargs):
    """Adds a caption to the bottom of the figure."""
    caption = fig.text(0.5, 0, text, ha="center", va="bottom", **kwargs)

    fig.canvas.draw()  # needed so the text has a real size
    renderer = fig.canvas.get_renderer()
    bbox = caption.get_window_extent(renderer=renderer)
    fig_height = fig.bbox.height
    text_height = bbox.height / fig_height
    margin = text_height + pad

    fig.subplots_adjust(bottom=margin)
    caption.set_position((0.5, -text_height - 0.01))


def plot_matrix_comparison(
    plot_title, matrices, labels, filename, inset="", show=False
):
    """Save a shared-color-scale comparison as ``filename + '.pdf'``.

    Accepts one or more finite real matrices, including rectangular matrices
    with different shapes. Returns the output path. Figures are closed when
    ``show=False`` so repeated batch rendering does not accumulate GUI state.
    """
    import matplotlib.pyplot as plt

    if not matrices or len(matrices) != len(labels):
        raise ValueError("Supply one label per matrix and at least one matrix.")
    matrices = [np.asarray(matrix) for matrix in matrices]
    if any(matrix.ndim != 2 or not matrix.size for matrix in matrices):
        raise ValueError("Each matrix must be nonempty and two dimensional.")
    if any(
        not np.isrealobj(matrix) or not np.all(np.isfinite(matrix))
        for matrix in matrices
    ):
        raise ValueError("Each matrix must be finite and real.")
    fig, axes = plt.subplots(
        1, len(matrices), figsize=(5 * len(matrices), 4), squeeze=False
    )
    axes = axes.ravel()
    least = min(np.min(matrix) for matrix in matrices)
    most = max(np.max(matrix) for matrix in matrices)
    for ax, mat, title in zip(axes, matrices, labels):
        im = ax.imshow(mat, cmap="plasma", vmin=least, vmax=most)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.tick_params(labelsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="")
    fig.suptitle(plot_title, fontsize=16)
    add_caption(fig, inset, fontsize=20)
    plt.tight_layout()
    output = Path(f"{filename}.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output


####################################################################################


def P_img(specs, img_dir, base_dir=None, show=False):
    """Generates matrix comparison plot for the P matrix."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=base_dir)
    task = tasks[CharacterizeWHReferenceDeviceTask]
    plot_title = remove_task_segment(task.fn)
    exact_P = exactify(task)["P"]
    empirical_P = sg_results["P"]
    matrices = [exact_P, empirical_P, exact_P - empirical_P]
    labels = [
        f"$P_{{\\text{{exact}}}}$",
        f"$P_{{\\text{{empirical}}}}$",
        f"$P_{{\\text{{exact}}}} - P_{{\\text{{empirical}}}}$",
    ]
    filename = f"{img_dir}/{plot_title.replace('/', '-')}-P"
    P_err = np.round(np.linalg.norm(exact_P - empirical_P), 6)
    inset = f"$||P_{{\\text{{exact}}}} - P_{{\\text{{empirical}}}}|| = {P_err}$"
    return plot_matrix_comparison(
        plot_title, matrices, labels, filename, inset=inset, show=show
    )


def Phi_img(specs, img_dir, base_dir=None, show=False):
    """Generates matrix comparison plot for the Phi matrix."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=base_dir)
    task = tasks[CharacterizeWHReferenceDeviceTask]
    plot_title = remove_task_segment(task.fn)
    exact_Phi = np.linalg.pinv(exactify(task)["P"])
    empirical_Phi = np.linalg.pinv(sg_results["P"])
    matrices = [exact_Phi, empirical_Phi, exact_Phi - empirical_Phi]
    labels = [
        f"$\\Phi_{{\\text{{exact}}}}$",
        f"$\\Phi_{{\\text{{empirical}}}}$",
        f"$\\Phi_{{\\text{{exact}}}} - \\Phi_{{\\text{{empirical}}}}$",
    ]
    filename = f"{img_dir}/{plot_title.replace('/', '-')}-Phi"
    Phi_err = np.round(np.linalg.norm(exact_Phi - empirical_Phi), 6)
    empirical_quantumness = np.linalg.norm(
        np.eye(empirical_Phi.shape[0]) - empirical_Phi
    )
    exact_quantumness = np.linalg.norm(np.eye(exact_Phi.shape[0]) - exact_Phi)
    rel_quantumness = np.round(empirical_quantumness / exact_quantumness, 6)
    inset = (
        f"$||\\Phi_{{\\text{{exact}}}} - \\Phi_{{\\text{{empirical}}}}|| = {Phi_err}$"
        f"\n$||I - \\Phi_{{\\text{{empirical}}}}||/||I - \\Phi_{{\\text{{exact}}}}|| = {rel_quantumness}$"
    )
    return plot_matrix_comparison(
        plot_title, matrices, labels, filename, inset=inset, show=show
    )


def q_img(specs, img_dir, base_dir=None, show=False):
    """Generates matrix comparison plot for the q matrix."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=base_dir)
    plot_title = remove_task_segment(tasks[CharacterizeWHReferenceDeviceTask].fn)
    exact_q = exactify(tasks[BasisMeasurementOnBasisStatesTask])["q"]
    empirical_q = sg_results["q"]
    empirical_born_rule = (
        sg_results["C"] @ np.linalg.pinv(sg_results["P"]) @ sg_results["r"]
    )

    matrices = [
        exact_q,
        empirical_q,
        empirical_born_rule,
        empirical_q - empirical_born_rule,
    ]
    labels = [
        f"$q_{{\\text{{exact}}}}$",
        f"$q_{{\\text{{empirical}}}}$",
        f"$(C\\Phi r)_{{\\text{{empirical}}}}$",
        f"$q_{{\\text{{empirical}}}} - (C\\Phi r)_{{\\text{{empirical}}}}$",
    ]
    filename = f"{img_dir}/{plot_title.replace('/', '-')}-born"
    q_err = np.round(np.linalg.norm(exact_q - empirical_q), 6)
    born_rule_err = np.round(np.linalg.norm(empirical_q - empirical_born_rule), 6)
    inset = (
        f"$||q_{{\\text{{exact}}}} - q_{{\\text{{empirical}}}}|| = {q_err}$"
        f"\n$||q_{{\\text{{empirical}}}} - (C \\Phi r)_{{\\text{{empirical}}}}|| = {born_rule_err}$"
    )
    return plot_matrix_comparison(
        plot_title, matrices, labels, filename, inset=inset, show=show
    )


def p_img(specs, img_dir, base_dir=None, show=False):
    """Generates matrix comparison plot for the p matrix."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=base_dir)
    plot_title = remove_task_segment(tasks[CharacterizeWHReferenceDeviceTask].fn)
    exact_p = exactify(tasks[BasisMeasurementAfterWHPOVMOnBasisStatesTask])["p"]
    empirical_p = sg_results["p"]
    empirical_LTP = sg_results["C"] @ sg_results["r"]

    matrices = [exact_p, empirical_p, empirical_LTP, empirical_p - empirical_LTP]
    labels = [
        f"$p_{{\\text{{exact}}}}$",
        f"$p_{{\\text{{empirical}}}}$",
        f"$(C r)_{{\\text{{empirical}}}}$",
        f"$p_{{\\text{{empirical}}}} - (C r)_{{\\text{{empirical}}}}$",
    ]
    filename = f"{img_dir}/{plot_title.replace('/', '-')}-ltp"
    p_err = np.round(np.linalg.norm(exact_p - empirical_p), 6)
    LTP_err = np.round(np.linalg.norm(empirical_p - empirical_LTP), 6)
    inset = (
        f"$||p_{{\\text{{exact}}}} - p_{{\\text{{empirical}}}}|| = {p_err}$"
        f"\n$||p_{{\\text{{empirical}}}} - (C r)_{{\\text{{empirical}}}}|| = {LTP_err}$"
    )
    return plot_matrix_comparison(
        plot_title, matrices, labels, filename, inset=inset, show=show
    )


def pq_img(specs, img_dir, base_dir=None, show=False):
    """Generates matrix comparison plot for p/q matrix."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=base_dir)
    plot_title = remove_task_segment(tasks[CharacterizeWHReferenceDeviceTask].fn)
    ltp = sg_results["C"] @ sg_results["r"]
    born = sg_results["C"] @ np.linalg.pinv(sg_results["P"]) @ sg_results["r"]
    matrices = [ltp, born, ltp - born]
    labels = [
        f"$(C r)_{{\\text{{empirical}}}}$",
        f"$(C \\Phi r)_{{\\text{{empirical}}}}$",
        f"$(C r)_{{\\text{{empirical}}}} - (C \\Phi r)_{{\\text{{empirical}}}}$",
    ]
    filename = f"{img_dir}/{plot_title.replace('/', '-')}-ltp_born"
    err = np.round(np.linalg.norm(ltp - born), 6)
    inset = f"$||(C r)_{{\\text{{empirical}}}} - (C \\Phi r)_{{\\text{{empirical}}}}|| = {err}$"
    return plot_matrix_comparison(
        plot_title, matrices, labels, filename, inset=inset, show=show
    )


sky_ground_img_funcs = [P_img, Phi_img, q_img, p_img, pq_img]


def sky_ground_images(specs, img_dir="img", base_dir=None, show=False):
    """Generates all the sky ground images from the results picked out by the specification dictionary `specs`."""
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for img_func in sky_ground_img_funcs:
        if specs["wh_implementation"] != "ak" and img_func == p_img:
            continue
        paths.append(img_func(specs, img_dir, base_dir=base_dir, show=show))
    return paths


####################################################################################


def sky_ground_metrics(specs, base_dir=None):
    """Compare stored probabilities with ideal circuits and the QBist constraint.

    Errors are Frobenius norms. ``born_rule_err`` is ``||q - C pinv(P) r||``;
    AK tasks also return the post-interaction residual ``||p - C r||``.
    Pseudoinversion is empirical and may amplify shot noise in poorly conditioned
    reference devices. Missing task records raise ``KeyError``.
    """
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=base_dir)
    exact_P = exactify(tasks[CharacterizeWHReferenceDeviceTask])["P"]
    empirical_P = sg_results["P"]
    P_err = np.linalg.norm(exact_P - empirical_P)

    exact_Phi = np.linalg.pinv(exact_P)
    empirical_Phi = np.linalg.pinv(empirical_P)
    Phi_err = np.linalg.norm(exact_Phi - empirical_Phi)
    empirical_quantumness = np.linalg.norm(
        np.eye(empirical_Phi.shape[0]) - empirical_Phi
    )
    exact_quantumness = np.linalg.norm(np.eye(exact_Phi.shape[0]) - exact_Phi)
    rel_quantumness = empirical_quantumness / exact_quantumness

    exact_q = exactify(tasks[BasisMeasurementOnBasisStatesTask])["q"]
    empirical_q = sg_results["q"]
    empirical_born_rule = (
        sg_results["C"] @ np.linalg.pinv(sg_results["P"]) @ sg_results["r"]
    )
    q_err = np.linalg.norm(exact_q - empirical_q)
    born_rule_err = np.linalg.norm(empirical_q - empirical_born_rule)

    metrics = {
        "P_err": P_err,
        "Phi_err": Phi_err,
        "rel_quantumness": rel_quantumness,
        "quantumness": empirical_quantumness,
        "q_err": q_err,
        "born_rule_err": born_rule_err,
    }

    if specs["wh_implementation"] == "ak":
        exact_p = exactify(tasks[BasisMeasurementAfterWHPOVMOnBasisStatesTask])["p"]
        empirical_p = sg_results["p"]
        empirical_LTP = sg_results["C"] @ sg_results["r"]

        p_err = np.linalg.norm(exact_p - empirical_p)
        LTP_err = np.linalg.norm(empirical_p - empirical_LTP)

        metrics["p_err"] = p_err
        metrics["LTP_err"] = LTP_err

    return metrics


__all__ = [
    "remove_task_segment",
    "add_caption",
    "plot_matrix_comparison",
    "P_img",
    "Phi_img",
    "q_img",
    "p_img",
    "pq_img",
    "sky_ground_img_funcs",
    "sky_ground_images",
    "sky_ground_metrics",
]
