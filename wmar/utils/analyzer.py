# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.​

import os

import numpy as np
from matplotlib.lines import Line2D

try:
    pass
except ImportError:
    pass
import json
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
from matplotlib import rcParams
from tabulate import tabulate
from wmar.augmentations.augmentation_manager import AugmentationManager


class Analyzer:

    def __init__(self, methods_dict, cache_path):
        self.methods_dict = methods_dict
        self.aug_manager = AugmentationManager(True, True, load_augs=False)

        self.all_augs = []
        self.all_augs.append(("roundtrips", None, [0, 1]))
        self.all_augs.extend(self.aug_manager.augs)

        self.all_methods = [k for k in methods_dict.keys()]
        self.all_metrics = {}
        self.all_orig_image_paths = {}
        self.N = {}
        self.cache_path = cache_path
        # does it exist
        if not os.path.exists(self.cache_path):
            json.dump({"all_metrics": {}, "all_orig_image_paths": {}, "N": {}}, open(self.cache_path, "w"))
        cache = json.load(open(self.cache_path))

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for full_label, (outdir, resultdir_prefix, method_id) in methods_dict.items():

                # If already there don't run
                if full_label in cache["all_metrics"]:
                    print(f"Found {full_label} in cache (N = {cache['N'][full_label]})")
                    self.all_metrics[full_label] = cache["all_metrics"][full_label]
                    self.all_orig_image_paths[full_label] = cache["all_orig_image_paths"][full_label]
                    self.N[full_label] = cache["N"][full_label]
                    continue

                futures.append(
                    executor.submit(
                        Analyzer.get_metrics_imagepaths_N,
                        self.all_augs,
                        full_label,
                        outdir,
                        resultdir_prefix,
                        method_id,
                    )
                )

        # Read and write
        new_entries = False
        for future in futures:
            new_entries = True
            full_label, metrics, image_paths, N = future.result()
            self.all_metrics[full_label] = metrics
            self.all_orig_image_paths[full_label] = image_paths
            self.N[full_label] = N

            # save to cache
            cache["all_metrics"][full_label] = self.all_metrics[full_label]
            cache["all_orig_image_paths"][full_label] = self.all_orig_image_paths[full_label]
            cache["N"][full_label] = self.N[full_label]

        # save cache
        if new_entries:
            print(f"There were new entries. Saving cache to {self.cache_path}")
            json.dump(cache, open(self.cache_path, "w"))

        ###################################################

        self.inverted_augs = [
            "jpeg",
            "upperleft-crop",
            "neural-compress",
        ]

        self.summary_metrics = {
            "gaussian-blur": ("Valuemetric", 9, "Gaussian Blur", "Gaussian Blur [kernel size]"),
            "gaussian-noise": ("Valuemetric", 0.1, "Gaussian Noise", "Gaussian Noise [stddev]"),
            "jpeg": ("Valuemetric", 25, "JPEG", "JPEG Compression [quality]"),
            "brightness": ("Valuemetric", 2, "Brighten", "Brighten [factor]"),
            "rotation": ("Geometric", 10, "Rotation", "Rotation [angle]"),
            "flip-h": ("Geometric", 1, "HFlip", "Horizontal Flip [is flipped]"),
            "upperleft-crop": ("Geometric", 0.75, "Crop", "Crop [percent kept]"),
            "diffpure": ("Adversarial Purification", 0.1, "DiffPure", "DiffPure [timestep]"),
            "neural-compress": ("Neural Compression", "q=3", "Neural Compression", "Neural Compression [bpp]"),
        }
        self.cats_sizes = {
            "None": 1,
            "Valuemetric": 4,
            "Geometric": 3,
            "Adversarial Purification": 1,
            "Neural Compression": 6,
        }
        # Colors
        self.palette_colors = [
            "#E63946",  # Bright Red,
            "#457B9D",  # Muted Blue
            "#2A9D8F",  # Teal
            "#F4A261",  # Warm Orange
            "#E9C46A",  # Golden Yellow
            "#264653",  # Deep Blue-Gray
            "#1D3557",  # Dark Navy
            "#A8DADC",  # Soft Cyan
            "#F94144",  # Strong Red-Orange
            "#000000",  # Black
            "#808080",  # Gray
        ]
        self.method_colors = {
            "original": 0,  # red
            "finetuned_noaugs": 1,  # blue
            "finetuned_augs": 2,  # green
            "finetuned_augs+sync": 4,  # yellow
        }
        self.method_nice_names = {
            "original": "Base",  # red
            "finetuned_noaugs": "FT",  # blue
            "finetuned_augs": "FT+Augs",  # green
            "finetuned_augs+sync": "FT+Augs+Sync",  # yellow
        }
        self.sort_priority = {
            self.method_nice_names["original"]: 1,
            self.method_nice_names["finetuned_noaugs"]: 2,
            self.method_nice_names["finetuned_augs"]: 3,
            self.method_nice_names["finetuned_augs+sync"]: 4,
        }
        self.nc_nice_names = {
            "bmshj2018-factorized": ("BMSHJ18 (Factorized)", "s"),
            "bmshj2018-hyperprior": ("BMSHJ18 (Hyperprior)", "p"),
            "mbt2018-mean": ("MBT18 (Scale)", "o"),
            "mbt2018": ("MBT18", "8"),
            "cheng2020-anchor": ("CSTK20 (Anchor)", "D"),
            "cheng2020-attn": ("CSTK20 (Attention)", "d"),
            "diffusers-sd-vae-ft-ema": ("SD VAE (ft-EMA)", "1"),
            "diffusers-sd-vae-fp16": ("SDXL VAE (fp16)", "2"),
            "diffusers-deep-compression": ("DC-AE", "3"),
            "diffusers-flux": ("FLUX VAE", "4"),
        }
        self.nc_sort_priority = {
            self.nc_nice_names["bmshj2018-factorized"][0]: 9,
            self.nc_nice_names["bmshj2018-hyperprior"][0]: 10,
            self.nc_nice_names["mbt2018-mean"][0]: 6,
            self.nc_nice_names["mbt2018"][0]: 5,
            self.nc_nice_names["cheng2020-anchor"][0]: 7,
            self.nc_nice_names["cheng2020-attn"][0]: 8,
            self.nc_nice_names["diffusers-sd-vae-ft-ema"][0]: 2,
            self.nc_nice_names["diffusers-sd-vae-fp16"][0]: 3,
            self.nc_nice_names["diffusers-deep-compression"][0]: 4,
            self.nc_nice_names["diffusers-flux"][0]: 1,
        }
    
    def set_up_latex(self):
        plt.rcdefaults()
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        rcParams['text.usetex'] = True
        rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

    @staticmethod
    def get_metrics_imagepaths_N(
        all_augs, full_label, root, resultdir_prefix, method_id, load_metrics=True, load_imagepaths=True
    ):
        all_metrics = {}
        all_orig_image_paths = {}
        N = -1

        # Walk outdir and find those that start with resultdir prefix
        dirs_all = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        dirs = [d for d in dirs_all if d.startswith(resultdir_prefix)]
        print(f"found {len(dirs_all)} dirs in {root}; filtered to {len(dirs)} dirs")

        result_dirs = []
        for dir in dirs:
            subdirs = os.listdir(os.path.join(root, dir))
            subdirs = [d for d in subdirs if os.path.isdir(os.path.join(root, dir, d))]
            result_dirs.extend([os.path.join(root, dir, d) for d in subdirs])
        print(f"[{full_label}] Found {len(result_dirs)} result dirs under {root} w/ {resultdir_prefix}")
        N = len(result_dirs)

        # Build all_metrics
        all_metrics = {}
        all_orig_image_paths = {}
        for aug, _, params in all_augs:
            for p in params:
                all_metrics[f"{aug}_{p}"] = []

        for i, result_dir in enumerate(result_dirs):
            found_roundtrips_0 = False
            toks = [t.split("=")[-1] for t in result_dir.split("/")[-1].split(",")]
            if len(toks) != 2:
                print(f"WARNING: {result_dir} has {len(toks)} tokens, expected 2 -- this will crash")
            imagenet_cls, idx = toks
            # need to walk through to get all jsons
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    if file.endswith(".json") and load_metrics:
                        idx, method, aug, param = file[:-5].split("_")
                        if method == method_id:
                            metrics_path = os.path.join(root, file)
                            metrics = json.load(open(metrics_path))
                            if f"{aug}_{param}" not in all_metrics:
                                all_metrics[f"{aug}_{param}"] = []
                            all_metrics[f"{aug}_{param}"].append(metrics)
                            if aug == "roundtrips" and param == "0":
                                found_roundtrips_0 = True
                    elif file.endswith(".png") and "roundtrips_0" in file and load_imagepaths:
                        idx, method, aug, param = file[:-4].split("_")
                        if method == method_id:
                            if imagenet_cls not in all_orig_image_paths:
                                all_orig_image_paths[imagenet_cls] = []
                            all_orig_image_paths[imagenet_cls].append(os.path.join(root, file))
                            if aug == "roundtrips" and param == "0":
                                found_roundtrips_0 = True
            if not found_roundtrips_0:
                print(f"WARNING: {full_label} in {result_dir} does not have roundtrips_0")
        print(f"Exited get_metrics_imagepaths_N for {full_label}")
        return full_label, all_metrics, all_orig_image_paths, N

    def process_bpp(self, metricsarr, method):
        bpps = [m["bpp"] for m in metricsarr]
        return np.mean(bpps)

    def plot_auc(self, curr_methods=None, xlim=None, save_to=None):
        fig, axes = plt.subplots(1, 1, figsize=(10, 6.3), dpi=120)
        ax = axes

        if curr_methods is None:
            curr_methods = self.all_methods
        augparam = "roundtrips_1"
        minxs = 1e-1
        z_orders = {
            "original": 4,
            "finetuned_noaugs": 3,
            "finetuned_augs": 2,
            "finetuned_augs+sync": 1,
        }
        for j, method in enumerate(curr_methods):
            clr = self.palette_colors[self.method_colors[method]]
            metricsarr = self.all_metrics[method][augparam]
            pvals = list(sorted([m["pvalue"] for m in metricsarr]))
            xs = []
            ys = []
            for i, pval in enumerate(pvals):
                xs.append(pval)
                ys.append((i + 1) / len(pvals))
            xs.append(1e-0)
            ys.append(ys[-1])
            nice_name = self.method_nice_names[method]
            ax.plot(xs, ys, color=clr, label=f"\\textsc{{{nice_name}}}", linewidth=7, zorder=z_orders[method])
            minxs = min(minxs, min(xs))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylabel("True Positive Rate", fontsize=40)
        ax.set_xlabel("False Positive Rate", fontsize=40)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=36)
        ax.tick_params(axis="x", labelsize=36)
        ax.grid(axis="y", linestyle="--", alpha=1)  # Light dashed horizontal lines
        ax.set_axisbelow(True)
        ax.set_facecolor((0.97, 0.97, 0.97))
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        ax.set_xscale("log", base=10)
        if xlim is not None:
            ax.set_xlim(xlim)
            if xlim[0] == 1e-82:
                ax.set_xticks([1e-82, 1e-66, 1e-50, 1e-34, 1e-18, 1e-2])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        else:
            ax.set_xlim([1e-50, 1e-0])
            ax.set_xticks([1e-50, 1e-34, 1e-18, 1e-2])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.axvline(x=1e-2, color="black", linestyle="--", linewidth=2)  # 98%
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        if save_to is None:
            fig.show()
        else:
            fig.savefig(save_to, bbox_inches="tight")

    def plot_l0_hist(self, curr_methods=None, save_to=None, is_rar=False):
        fig, axes = plt.subplots(1, 1, figsize=(10, 6.3), dpi=120)
        ax = axes

        if curr_methods is None:
            curr_methods = self.all_methods
        augparam = "roundtrips_1"
        z_orders = {
            "original": 4,
            "finetuned_noaugs": 3,
            "finetuned_augs": 2,
            "finetuned_augs+sync": 1,
        }
        for j, method in enumerate(curr_methods):
            clr = self.palette_colors[self.method_colors[method]]
            metricsarr = self.all_metrics[method][augparam]
            l0s = list(sorted([m["l0"] for m in metricsarr]))
            token_match = [1.0 - l0 for l0 in l0s]
            pcbig = sum([1 for t in token_match if t > 0.8]) / len(token_match)
            mean_tm = np.mean(token_match)
            median_tm = np.median(token_match)
            print(f"Mean,Median,%Above0.8 for {method}: {mean_tm:.3f} {median_tm:.3f} {pcbig:.3f}")
            nice_name = self.method_nice_names[method]
            n_bins = 50
            if "augs" in method and "sync" not in method and is_rar:
                n_bins = 20
            ax.hist(
                token_match,
                bins=n_bins,
                color=clr,
                label=f"\\textsc{{{nice_name}}}",
                alpha=0.8,
                weights=np.zeros_like(token_match) + 1.0 / len(token_match),
                zorder=z_orders[method],
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylabel("Relative Frequency", fontsize=40)
        ax.set_xlabel("Token Match", fontsize=40)
        if is_rar:
            ax.set_yticks([0, 0.1, 0.2, 0.3])
            ax.set_ylim(0, 0.35)
        else:
            ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=36)
        ax.tick_params(axis="x", labelsize=36)
        ax.grid(axis="y", linestyle="--", alpha=1)  # Light dashed horizontal lines
        ax.set_axisbelow(True)
        ax.set_facecolor((0.97, 0.97, 0.97))
        ax.legend(fontsize=30)
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        if save_to is None:
            fig.show()
        else:
            fig.savefig(save_to, bbox_inches="tight")

    def plot_robustness(self, curr_methods=None, save_to=None, legend_fontsize=32):
        if curr_methods is None:
            curr_methods = self.all_methods

        summary_score = {}
        summary_cnt = {}
        for method in curr_methods:
            summary_score[method] = {}
            summary_cnt[method] = {}
            for cat in self.cats_sizes.keys():
                summary_score[method][cat] = []
                summary_cnt[method][cat] = 0
            # Get None
            pvals = [m["pvalue"] for m in self.all_metrics[method]["flip-h_0"]]
            if pvals[0] is None:
                tpr_at_one = 0
            else:
                tpr_at_one = np.sum(np.array(pvals) < 0.01) / len(pvals)
            summary_score[method]["None"].append(tpr_at_one)
            summary_cnt[method]["None"] += 1
        LW = 4
        # Start the plot
        fig, axes = plt.subplots(2, 5, figsize=(35, 12), dpi=120, sharey=True)

        ijs_skip = [(0, 4)]
        for i, j in ijs_skip:
            axes[i][j].set_visible(False)

        ijs = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
        for idx, (i, j) in enumerate(ijs):
            ax = axes[i][j]
            aug, (cat, summary_param, aug_nice_name, param_nice_name) = list(self.summary_metrics.items())[idx]
            params = [a[2] for a in self.all_augs if aug == a[0]][0]

            # Set title
            ax.set_xlabel(param_nice_name, fontsize=40)
            if j == 0:
                ax.set_ylabel("TPR@1\\%", fontsize=40)

            # Plot
            for methodidx, method in enumerate(curr_methods):
                clr = self.palette_colors[self.method_colors[method]]
                xs, ys, aux = [], [], []
                for param in params:
                    augparam = f"{aug}_{param}"
                    if augparam not in self.all_metrics[method] or len(self.all_metrics[method][augparam]) == 0:
                        print(f"WARNING: {augparam} not found in runs for {method}")
                        continue
                    metricsarr = self.all_metrics[method][augparam]
                    if aug == "neural-compress":
                        # For neural compress our x axis will be the bpp, save name in aux
                        xs.append(self.process_bpp(metricsarr, method))
                        aux.append(str(param))
                    else:
                        if aug == "upperleft-crop":
                            xs.append(round(param * 100))
                        else:
                            xs.append(param)
                    pvals = [m["pvalue"] for m in metricsarr]
                    if pvals[0] is None:
                        tpr_at_one = 0
                    else:
                        tpr_at_one = np.sum(np.array(pvals) < 0.01) / len(pvals)
                    ys.append(tpr_at_one)

                    # FILL TABLE
                    if (isinstance(summary_param, str) and summary_param in str(param)) or summary_param == param:
                        summary_score[method][cat].append(tpr_at_one)
                        summary_cnt[method][cat] += 1

                # Always sort (x,y) pairs by xs
                xs, ys, aux = np.array(xs), np.array(ys), np.array(aux)
                sorted_idxs = np.argsort(xs)
                if aug in self.inverted_augs:
                    sorted_idxs = sorted_idxs[::-1]
                xs = xs[sorted_idxs]
                ys = ys[sorted_idxs]
                if len(aux) > 0:
                    aux = aux[sorted_idxs]

                if aug == "neural-compress":
                    for x, y, a in zip(xs, ys, aux):
                        a_stripped = a.replace("-q=1", "").replace("-q=3", "").replace("-q=6", "")
                        lab = self.nc_nice_names[a_stripped][0]
                        if method != "original" or "q=3" in a or "q=6" in a:
                            lab = None
                        if "q=" in a:
                            ax.scatter(
                                x,
                                y,
                                color=clr,
                                marker=self.nc_nice_names[a_stripped][1],
                                s=200,
                                zorder=20,
                                facecolors="None",
                                label=lab,
                            )
                        else:
                            ax.scatter(
                                x, y, color=clr, marker=self.nc_nice_names[a_stripped][1], s=400, zorder=20, label=lab
                            )

                    # fit a polynomial of degree 5 to these points
                    ax.plot(xs, np.poly1d(np.polyfit(xs, ys, 5))(xs), color=clr, linewidth=LW, zorder=0)
                else:
                    nice_name = self.method_nice_names[method] if method in self.method_nice_names else method
                    ax.plot(xs, ys, label=f"\\textsc{{{nice_name}}}", color=clr, linewidth=LW)
                    ax.scatter(xs, ys, color=clr, s=150, zorder=10, edgecolors="white")
            # If aug ==
            if aug == "neural-compress":
                # show legend but only once because the diffuser/compress method is same
                ax.legend(loc="center left", bbox_to_anchor=(-2.2, 0.6), fontsize=20, title="")
                leg = ax.get_legend()
                # deduplicate the legend
                leg.get_frame().set_alpha(1)
                [lgd.set_color("black") for lgd in leg.legend_handles]
                [lgd.set_facecolor("None") for lgd in leg.legend_handles]

                handles, labels = ax.get_legend().legend_handles, [t.get_text() for t in ax.get_legend().get_texts()]
                sorted_pairs = sorted(zip(handles, labels), key=lambda x: self.nc_sort_priority[x[1]])
                sorted_handles = [pair[0] for pair in sorted_pairs]
                sorted_labels = [pair[1] for pair in sorted_pairs]
                # insert empty at index 4
                ax.legend(
                    sorted_handles, sorted_labels, loc="upper center", bbox_to_anchor=(0.5, 2.2), fontsize=21, ncol=1
                )

            # Plot level params
            # show at -1 on y
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

            ax.tick_params(axis="y", which="both", left=False, right=False, labelsize=36)
            ax.tick_params(axis="x", labelsize=36)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

            # Some augmentations are upside down
            if aug == "gaussian-noise":
                rang = max(params) - min(params)
                ax.set_xlim(min(params) - rang * 0.05, max(params) + rang * 0.05)
                ax.set_xticks([0.0, 0.05, 0.1, 0.15, 0.2])
            if aug == "brightness":
                rang = max(params) - min(params)
                ax.set_xlim(min(params) - rang * 0.05, max(params) + rang * 0.05)
                ax.set_xticks([1, 1.5, 2, 2.5, 3])
            elif aug == "rotation":
                rang = max(params) - min(params)
                ax.set_xlim(min(params) - rang * 0.05, max(params) + rang * 0.05)
                ax.set_xticks([20, 10, 0, -10, -20])
            elif aug == "neural-compress":
                ax.set_xlim(2.1, -0.01)
                ax.set_xticks([2, 1.5, 1, 0.5, 0])
            elif aug == "jpeg":
                ax.set_xlim(max(params) + rang * 0.1, min(params) - rang * 0.05)
                ax.set_xticks([90, 70, 50, 30, 10])
            elif aug == "gaussian-blur":
                rang = max(params) - min(params)
                ax.set_xlim(min(params) - rang * 0.05, max(params) + rang * 0.05)
                ax.set_xticks([1, 4, 7, 10, 13, 16, 19])
            elif aug == "flip-h":
                ax.set_xticks([0, 1])
                ax.set_xlim(-0.05, 1.05)
            elif aug in self.inverted_augs:
                if aug == "upperleft-crop":
                    params = [round(p * 100) for p in params]
                    ax.set_xticks([100, 90, 80, 70, 60, 50])
                rang = max(params) - min(params)
                ax.set_xlim(max(params) + rang * 0.05, min(params) - rang * 0.05)
            else:
                rang = max(params) - min(params)
                ax.set_xlim(min(params) - rang * 0.05, max(params) + rang * 0.05)

            ax.grid(axis="y", linestyle="--", alpha=1)  # Light dashed horizontal lines
            ax.set_axisbelow(True)
            ax.set_facecolor((0.97, 0.97, 0.97))

        # add legend above
        fake_lines = []
        fake_labels = []
        for methodidx, method in enumerate(curr_methods):
            clr = self.palette_colors[self.method_colors[method]]
            nice_name = self.method_nice_names[method] if method in self.method_nice_names else method
            fake_lines.append(Line2D([0], [0], color=clr, linewidth=LW))
            fake_labels.append(f"\\textsc{{{nice_name}}}")
        fig.legend(
            handles=fake_lines, labels=fake_labels, loc="upper left", bbox_to_anchor=(0.20, 1.1), fontsize=40, ncols=4
        )

        # add vertical margin between rows
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.32, wspace=0.07)

        if save_to is not None:
            fig.savefig(save_to, bbox_inches="tight")
        else:
            fig.show()

        # TABLE
        LOW_SCORE_THRESHOLD = 0.6

        headers = ["Method"] + [k for k in self.cats_sizes.keys()]
        rows = []
        for method in curr_methods:
            row = []
            nice_name = self.method_nice_names[method] if method in self.method_nice_names else method
            row.append(nice_name)
            latex = f"& \\textsc{{{nice_name}}} "
            for cat, size in self.cats_sizes.items():
                assert (
                    len(summary_score[method][cat]) == size
                ), f"{method} {cat} has {len(summary_score[method][cat])} scores, expected {size}"
                score = np.sum(summary_score[method][cat]) / summary_cnt[method][cat]
                if round(score, 2) < LOW_SCORE_THRESHOLD:
                    latex += f"& \\textcolor{{red}}{{{score:.2f}}} "
                    score = f"!{score:.2f}!"
                else:
                    latex += f"& {score:.2f} "
                    score = f"{score:.2f}"
                row.append(score)
            rows.append(row)
            latex += "\\\\"
            print(latex)
        print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".2f"))
