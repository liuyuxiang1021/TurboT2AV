"""Diagnose whether SCM gradient spikes come from data or t-sampling.

For N fixed samples, run M forward passes with different random t values.
If within-sample std is high → t-sampling drives variance.
If between-sample mean differs a lot → data drives variance.

Usage:
    torchrun --nproc_per_node=8 -m ltx_distillation.tests.test_scm_variance \
        --config_path /path/to/config.yaml --num_samples 4 --num_repeats 20
"""
import argparse
import sys
import os
import numpy as np
import torch
import torch.distributed as dist

# Ensure the ltx_distillation package is importable
_src = os.path.join(os.path.dirname(__file__), "..", "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="/home/jovyan/codes/turbodiff/new_Turbo/OmniForcing/LTX-2/packages/ltx-distillation/configs/stage1_bidirectional_scm_rcm.yaml")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of fixed data samples to test")
    parser.add_argument("--num_repeats", type=int, default=20,
                        help="Number of different t-sampling repeats per sample")
    return parser.parse_args()


def run_diagnostic():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Import after dist init so FSDP works
    from ltx_distillation.dmd import DMD
    from ltx_distillation.train_distillation import parse_args as parse_train_args
    from ltx_distillation.data import ODERegressionLMDBDataset

    # Parse the training config
    train_args = parse_train_args(["--config_path", args.config_path])

    if rank == 0:
        print(f"=== SCM Variance Diagnostic ===")
        print(f"Config: {args.config_path}")
        print(f"Samples: {args.num_samples}, Repeats: {args.num_repeats}")
        print(f"scm_p_G_mean={getattr(train_args, 'scm_p_G_mean', -0.8)}, "
              f"scm_p_G_std={getattr(train_args, 'scm_p_G_std', 1.6)}")
        print(f"scm_fd_type={getattr(train_args, 'scm_fd_type', 1)}, "
              f"scm_loss_scale={getattr(train_args, 'scm_loss_scale', 100)}")

    # Build DMD (loads models, wraps with FSDP)
    if rank == 0:
        print("Initializing DMD (loading models)...")
    dmd = DMD(train_args)

    # Load SCM dataset (rank 0 only to get fixed samples)
    scm_data_path = getattr(train_args, "scm_data_path", None)
    if scm_data_path is None:
        raise ValueError("scm_data_path not set in config")

    if rank == 0:
        print(f"Loading SCM dataset from {scm_data_path}...")
    dataset = ODERegressionLMDBDataset(scm_data_path, max_pair=args.num_samples)

    # Collect results
    all_results = []

    for sample_idx in range(min(args.num_samples, len(dataset))):
        sample = dataset[sample_idx]
        prompt = sample["prompts"][:80] if isinstance(sample["prompts"], str) else str(sample["prompts"])[:80]

        if rank == 0:
            print(f"\n--- Sample {sample_idx} ---")
            print(f"  Prompt: {prompt}...")
            print(f"  ode_latent shape: {sample['ode_latent'].shape}")
            if sample.get("ode_audio_latent") is not None:
                print(f"  ode_audio_latent shape: {sample['ode_audio_latent'].shape}")
            print(f"  Running {args.num_repeats} repeats with different t...")

        sample_results = []
        for repeat in range(args.num_repeats):
            # Move data to GPU
            clean_video = sample["ode_latent"].unsqueeze(0).cuda()  # [1, T, F, C, H, W]
            clean_audio = None
            if sample.get("ode_audio_latent") is not None:
                clean_audio = sample["ode_audio_latent"].unsqueeze(0).cuda()

            # Build conditional dict (simplified — reuses training logic)
            # We need the prompt encoded. For the diagnostic, we use the
            # generator's internal encode/decode path via the SCM loss.
            # Actually, let's just call compute_scm_loss which handles everything.

            # The DMD.compute_scm_loss handles:
            # - prompt encoding
            # - t sampling
            # - teacher flow computation
            # - student JVP
            # - loss computation
            scm_loss, log_dict = dmd.compute_scm_loss(
                clean_video=clean_video,
                clean_audio=clean_audio,
                conditional_dict=None,  # Will be built from the sample prompt
                unconditional_dict=None,
            )

            grad_norm = 0.0  # We don't do backward in diagnostic mode
            sample_results.append({
                "scm_video_loss": log_dict.get("alignment/scm_video_loss", log_dict.get("train/scm_video_loss", 0)),
                "scm_audio_loss": log_dict.get("alignment/scm_audio_loss", log_dict.get("train/scm_audio_loss", 0)),
                "video_gap": log_dict.get("alignment/scm_video_direction_gap", 0),
                "video_tangent": log_dict.get("alignment/scm_video_tangent_norm", 0),
                "video_tangent_max": log_dict.get("alignment/scm_video_tangent_max", 0),
                "rf_time": log_dict.get("alignment/scm_rf_time_mean", 0),
            })

            if rank == 0 and repeat % 5 == 0:
                r = sample_results[-1]
                print(f"    repeat {repeat:3d}: loss_v={r['scm_video_loss']:.4f} "
                      f"loss_a={r['scm_audio_loss']:.4f} "
                      f"gap={r['video_gap']:.4f} "
                      f"tmean={r['video_tangent']:.4f} tmax={r['video_tangent_max']:.1f}")

        # Aggregate stats for this sample
        gaps = [r["video_gap"] for r in sample_results]
        tmaxs = [r["video_tangent_max"] for r in sample_results]
        tmeans = [r["video_tangent"] for r in sample_results]

        all_results.append({
            "sample_idx": sample_idx,
            "prompt": prompt,
            "gap_mean": np.mean(gaps),
            "gap_std": np.std(gaps),
            "gap_min": np.min(gaps),
            "gap_max": np.max(gaps),
            "tmean_mean": np.mean(tmeans),
            "tmean_std": np.std(tmeans),
            "tmax_mean": np.mean(tmaxs),
            "tmax_std": np.std(tmaxs),
            "tmax_min": np.min(tmaxs),
            "tmax_max": np.max(tmaxs),
        })

    # Print summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("SUMMARY: Per-sample statistics across different t-samplings")
        print("=" * 80)
        print(f"{'Sample':<8} {'gap_mean':<10} {'gap_std':<10} {'gap_range':<18} {'tmax_mean':<12} {'tmax_std':<10} {'tmax_range':<18}")
        print("-" * 80)
        for r in all_results:
            gap_range = f"[{r['gap_min']:.4f}, {r['gap_max']:.4f}]"
            tmax_range = f"[{r['tmax_min']:.0f}, {r['tmax_max']:.0f}]"
            print(f"{r['sample_idx']:<8} {r['gap_mean']:<10.4f} {r['gap_std']:<10.4f} "
                  f"{gap_range:<18} {r['tmax_mean']:<12.1f} {r['tmax_std']:<10.1f} "
                  f"{tmax_range:<18}")

        # Cross-sample comparison
        gap_means = [r["gap_mean"] for r in all_results]
        gap_stds = [r["gap_std"] for r in all_results]
        print(f"\nCross-sample gap: mean={np.mean(gap_means):.4f} "
              f"between-sample std={np.std(gap_means):.4f} "
              f"avg within-sample std={np.mean(gap_stds):.4f}")
        print(f"Ratio (between/within): {np.std(gap_means) / np.mean(gap_stds):.2f}x")
        print(f"\nInterpretation:")
        ratio = np.std(gap_means) / (np.mean(gap_stds) + 1e-8)
        if ratio > 2:
            print(f"  Ratio={ratio:.1f}x > 2: DATA dominates variance (different samples have very different gaps)")
            print(f"  → Pre-computing teacher flows or filtering hard samples may help")
        elif ratio < 0.5:
            print(f"  Ratio={ratio:.1f}x < 0.5: T-SAMPLING dominates variance (same sample varies a lot with t)")
            print(f"  → SCM loss itself is high-variance, need DMD or semi-continuous")
        else:
            print(f"  Ratio={ratio:.1f}x ~1: Both data and t-sampling contribute")
            print(f"  → Need both better data filtering AND loss stabilization")

    dist.destroy_process_group()


if __name__ == "__main__":
    run_diagnostic()
