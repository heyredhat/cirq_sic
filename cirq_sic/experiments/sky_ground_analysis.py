from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np

from .sky_ground import *

####################################################################################

def remove_task_segment(path):
    parts = path.split('/')
    filtered = [p for p in parts if not p.endswith('Task')]
    return '/'.join(filtered)

def add_caption(fig, text, pad=0.02, **kwargs):
    caption = fig.text(0.5, 0, text, ha="center", va="bottom", **kwargs)

    fig.canvas.draw()  # needed so the text has a real size
    renderer = fig.canvas.get_renderer()
    bbox = caption.get_window_extent(renderer=renderer)
    fig_height = fig.bbox.height
    text_height = bbox.height / fig_height
    margin = text_height + pad

    fig.subplots_adjust(bottom=margin)
    caption.set_position((0.5, -text_height-0.01))

def plot_matrix_comparison(plot_title, matrices, labels, filename, inset=""):
    fig, axes = plt.subplots(1, len(matrices), figsize=(15, 4))
    ticks = np.arange(matrices[0].shape[0])
    tick_labels = [str(tick) for tick in ticks]
    least = np.min(matrices)
    most = np.max(matrices)
    for ax, mat, title in zip(axes, matrices, labels):
        im = ax.imshow(mat, cmap='plasma', vmin=least, vmax=most)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='')
    fig.suptitle(plot_title, fontsize=16)
    add_caption(fig, inset, fontsize=20)
    plt.tight_layout()
    fig.savefig(f"{filename}.pdf", bbox_inches='tight')
    plt.show()

####################################################################################

def P_img(specs, img_dir, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=None)
    task = tasks[CharacterizeWHReferenceDeviceTask]
    plot_title = remove_task_segment(task.fn)
    exact_P = exactify(task)["P"]
    empirical_P = sg_results["P"]
    matrices = [exact_P, empirical_P, exact_P - empirical_P]
    labels = [f"$P_{{\\text{{exact}}}}$",
            f"$P_{{\\text{{empirical}}}}$",
            f"$P_{{\\text{{exact}}}} - P_{{\\text{{empirical}}}}$"]
    filename = f"{img_dir}/{plot_title.replace("/", "-")}-P.json"
    P_err = np.round(np.linalg.norm(exact_P - empirical_P), 6)
    inset = f"$||P_{{\\text{{exact}}}} - P_{{\\text{{empirical}}}}|| = {P_err}$"
    plot_matrix_comparison(plot_title, matrices, labels, filename, inset=inset)

def Phi_img(specs, img_dir, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=None)
    task = tasks[CharacterizeWHReferenceDeviceTask]
    plot_title = remove_task_segment(task.fn)
    exact_Phi = np.linalg.pinv(exactify(task)["P"])
    empirical_Phi = np.linalg.pinv(sg_results["P"])
    matrices = [exact_Phi, empirical_Phi, exact_Phi - empirical_Phi]
    labels = [f"$\\Phi_{{\\text{{exact}}}}$",
            f"$\\Phi_{{\\text{{empirical}}}}$",
            f"$\\Phi_{{\\text{{exact}}}} - \\Phi_{{\\text{{empirical}}}}$"]
    filename = f"{img_dir}/{plot_title.replace("/", "-")}-Phi.json"
    Phi_err = np.round(np.linalg.norm(exact_Phi - empirical_Phi), 6)
    empirical_quantumness = np.linalg.norm(np.eye(empirical_Phi.shape[0]) - empirical_Phi)
    exact_quantumness = np.linalg.norm(np.eye(exact_Phi.shape[0]) - exact_Phi)
    rel_quantumness = np.round(empirical_quantumness/exact_quantumness, 6)
    inset = (f"$||\\Phi_{{\\text{{exact}}}} - \\Phi_{{\\text{{empirical}}}}|| = {Phi_err}$"
            f"\n$||I - \\Phi_{{\\text{{empirical}}}}||/||I - \\Phi_{{\\text{{exact}}}}|| = {rel_quantumness}$")
    plot_matrix_comparison(plot_title, matrices, labels, filename, inset=inset)

def q_img(specs, img_dir, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=None)
    plot_title = remove_task_segment(tasks[CharacterizeWHReferenceDeviceTask].fn)
    exact_q = exactify(tasks[BasisMeasurementOnBasisStatesTask])["q"]
    empirical_q = sg_results["q"]
    empirical_born_rule = sg_results["C"] @ np.linalg.pinv(sg_results["P"]) @ sg_results["r"]

    matrices = [exact_q, empirical_q, empirical_born_rule, empirical_q - empirical_born_rule]
    labels = [f"$q_{{\\text{{exact}}}}$",
            f"$q_{{\\text{{empirical}}}}$",
            f"$(C\\Phi r)_{{\\text{{empirical}}}}$",
            f"$q_{{\\text{{empirical}}}} - (C\\Phi r)_{{\\text{{empirical}}}}$"]
    filename = f"{img_dir}/{plot_title.replace("/", "-")}-q.json"
    q_err = np.round(np.linalg.norm(exact_q - empirical_q), 6)
    born_rule_err = np.round(np.linalg.norm(empirical_q - empirical_born_rule), 6)
    inset = (f"$||q_{{\\text{{exact}}}} - q_{{\\text{{empirical}}}}|| = {q_err}$"
            f"\n$||q_{{\\text{{empirical}}}} - (C \\Phi r)_{{\\text{{empirical}}}}|| = {born_rule_err}$")
    plot_matrix_comparison(plot_title, matrices, labels, filename, inset=inset)

def p_img(specs, img_dir, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True)
    plot_title = remove_task_segment(tasks[CharacterizeWHReferenceDeviceTask].fn)
    exact_p = exactify(tasks[BasisMeasurementAfterWHPOVMOnBasisStatesTask])["p"]
    empirical_p = sg_results["p"]
    empirical_LTP = sg_results["C"] @ sg_results["r"]

    matrices = [exact_p, empirical_p, empirical_LTP, empirical_p - empirical_LTP]
    labels = [f"$p_{{\\text{{exact}}}}$",
            f"$p_{{\\text{{empirical}}}}$",
            f"$(C r)_{{\\text{{empirical}}}}$",
            f"$p_{{\\text{{empirical}}}} - (C r)_{{\\text{{empirical}}}}$"]
    filename = f"{img_dir}/{plot_title.replace("/", "-")}-q.json"
    p_err = np.round(np.linalg.norm(exact_p - empirical_p), 6)
    LTP_err = np.round(np.linalg.norm(empirical_p - empirical_LTP), 6)
    inset = (f"$||p_{{\\text{{exact}}}} - p_{{\\text{{empirical}}}}|| = {p_err}$"
            f"\n$||p_{{\\text{{empirical}}}} - (C r)_{{\\text{{empirical}}}}|| = {LTP_err}$")
    plot_matrix_comparison(plot_title, matrices, labels, filename, inset=inset)

sky_ground_img_funcs = [P_img, Phi_img, q_img, p_img]

def sky_ground_images(specs, img_path="img", base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    img_dir = Path(img_path)
    img_dir.mkdir(parents=True, exist_ok=True)
    for img_func in sky_ground_img_funcs:
        if specs["wh_implementation"] == "simple" and img_func == p_img:
            continue
        img_func(specs, img_dir, base_dir=None)

####################################################################################

def sky_ground_metrics(specs, base_dir=None):
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
    tasks, sg_results = load_sky_ground_results(specs, separate=True, base_dir=None)

    exact_P = exactify(tasks[CharacterizeWHReferenceDeviceTask])["P"]
    empirical_P = sg_results["P"]
    P_err = np.linalg.norm(exact_P - empirical_P)

    exact_Phi = np.linalg.pinv(exact_P)
    empirical_Phi = np.linalg.pinv(empirical_P)
    Phi_err = np.linalg.norm(exact_Phi - empirical_Phi)
    empirical_quantumness = np.linalg.norm(np.eye(empirical_Phi.shape[0]) - empirical_Phi)
    exact_quantumness = np.linalg.norm(np.eye(exact_Phi.shape[0]) - exact_Phi)
    rel_quantumness = empirical_quantumness/exact_quantumness

    exact_q = exactify(tasks[BasisMeasurementOnBasisStatesTask])["q"]
    empirical_q = sg_results["q"]
    empirical_born_rule = sg_results["C"] @ np.linalg.pinv(sg_results["P"]) @ sg_results["r"]
    q_err = np.linalg.norm(exact_q - empirical_q)
    born_rule_err = np.linalg.norm(empirical_q - empirical_born_rule)

    metrics = {"P_err": P_err, "Phi_err": Phi_err, "rel_quantumness": rel_quantumness,\
               "q_err": q_err, "born_rule_err": born_rule_err}
    
    if specs["wh_implementation"] == "ak":
        exact_p = exactify(tasks[BasisMeasurementAfterWHPOVMOnBasisStatesTask])["p"]
        empirical_p = sg_results["p"]
        empirical_LTP = sg_results["C"] @ sg_results["r"]

        p_err = np.linalg.norm(exact_p - empirical_p)
        LTP_err = np.linalg.norm(empirical_p - empirical_LTP)

        metrics["p_err"] = p_err
        metrics["LTP_err"] = LTP_err

    return metrics

####################################################################################

def FP_img(d, interpolation, img_dir="img"):
    fig, ax = plt.subplots(1, 1)
    T = np.linspace(0, 1, 15)
    ax.plot(T, [wh_2_frame_potential(interpolation(t)) for t in T])
    ax.axhline(y=0, c="black")
    ax.axvline(x=1, c="black")
    ax.set_title(f"d = {d}: Stabilizer to SIC state")
    ax.set_xlabel("t")
    ax.set_ylabel("Frame potential")
    plt.tight_layout()
    fig.savefig(f"{img_dir}/d{d}_frame_potential.pdf", bbox_inches='tight')
    plt.show()