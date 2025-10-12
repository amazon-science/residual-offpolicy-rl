# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

import wandb
from resfit.dexmg.environments.dexmg import VectorizedEnvWrapper
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent


def run_dexmg_evaluation(
    *,
    env: VectorizedEnvWrapper,
    agent: QAgent,
    num_episodes: int = 20,
    device: torch.device | str = "cpu",
    global_step: int | None = None,
    save_video: bool = False,
    save_q_plots: bool = False,
    run_name: str | None = None,
    output_dir: str | Path | None = "outputs",
) -> tuple[dict[str, float], float]:
    """Extended evaluation to match the richer functionality available in
    the *residual_td3_dexmg* evaluator.  In particular, this version:

    1. Annotates every rendered frame with useful metadata (env index,
       episode counter, step counter, predicted Q-value and SUCCESS/FAIL).
    2. Caches frames per-episode and flushes them into a single video file
       at the end of the evaluation.
    3. Keeps the original simple success-rate / return metrics so existing
       training code continues to work unchanged.
    """

    # ------------------------------------------------------------------
    # Helper functions (local to avoid polluting module namespace)
    # ------------------------------------------------------------------
    def _annotate_frame(
        frame: np.ndarray,
        *,
        env_idx: int,
        episode_num: int,
        total_episodes: int,
        step_idx: int,
        is_success: bool,
        q_value: float,
        font=None,
    ) -> np.ndarray:
        """Overlay evaluation metadata onto *frame* (H, W, C)."""

        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        # Status label ---------------------------------------------------
        status_text = "SUCCESS" if is_success else "FAIL"
        status_color = (0, 255, 0) if is_success else (255, 0, 0)

        y = 10
        dy = 15
        draw.text((10, y), f"Env {env_idx + 1}", fill=(255, 255, 255), font=font)
        y += dy
        draw.text((10, y), f"Episode {episode_num}/{total_episodes}", fill=(255, 255, 255), font=font)
        y += dy
        draw.text((10, y), f"Step {step_idx}", fill=(255, 255, 255), font=font)
        y += dy
        draw.text((10, y), status_text, fill=status_color, font=font)
        y += dy
        draw.text((10, y), f"Q = {q_value:.2f}", fill=(255, 255, 255), font=font)

        return np.asarray(pil_img)

    def _create_q_trajectory_plots(
        trajectories: list[list[float]],
        episode_lengths: list[int],
        successes: list[bool],
        output_path: Path,
        global_step: int | None = None,
    ) -> None:
        """Create Q-value trajectory plots for all episodes."""
        if not trajectories:
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: All Q-trajectories over time
        # Separate successful and failed episodes
        successful_trajs = [traj for i, traj in enumerate(trajectories) if successes[i]]
        failed_trajs = [traj for i, traj in enumerate(trajectories) if not successes[i]]

        # Plot all trajectories with different colors for success/failure
        for i, traj in enumerate(successful_trajs):
            steps = list(range(len(traj)))
            ax1.plot(steps, traj, "g-", alpha=0.6, linewidth=1, label="Success" if i == 0 else "")

        for i, traj in enumerate(failed_trajs):
            steps = list(range(len(traj)))
            ax1.plot(steps, traj, "r-", alpha=0.6, linewidth=1, label="Failure" if i == 0 else "")

        ax1.set_xlabel("Episode Step")
        ax1.set_ylabel("Q-Value")
        ax1.set_title(f"Q-Value Trajectories Over Time (Step {global_step or 'N/A'})")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Q-value distribution at different episode progress points
        progress_points = [0.25, 0.5, 0.75, 1.0]  # 25%, 50%, 75%, 100% of episode
        q_values_at_progress = {f"{int(p * 100)}%": [] for p in progress_points}

        for traj in trajectories:
            traj_len = len(traj)
            for p in progress_points:
                step_idx = min(int(p * traj_len), traj_len - 1)
                if step_idx < len(traj):
                    q_values_at_progress[f"{int(p * 100)}%"].append(traj[step_idx])

        # Create box plot
        box_data = [q_values_at_progress[f"{int(p * 100)}%"] for p in progress_points]
        box_labels = [f"{int(p * 100)}%" for p in progress_points]

        ax2.boxplot(box_data, labels=box_labels)
        ax2.set_xlabel("Episode Progress")
        ax2.set_ylabel("Q-Value")
        ax2.set_title("Q-Value Distribution at Different Episode Progress Points")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved Q-trajectory plots to: {output_path}")

    # ------------------------------------------------------------------
    # Initial setup -----------------------------------------------------
    # ------------------------------------------------------------------
    device = torch.device(device)
    agent.eval()

    num_envs: int = env.num_envs if hasattr(env, "num_envs") else 1

    # Per-environment episode buffers ----------------------------------
    ep_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    ep_q_preds: list[list[float]] = [[] for _ in range(num_envs)]

    successes: list[bool] = []  # episode-level success flags
    returns: list[float] = []  # episode-level undiscounted returns

    # Q-trajectory data for plotting ------------------------------------
    all_q_trajectories: list[list[float]] = []  # Store Q-trajectories for all episodes
    all_episode_lengths: list[int] = []  # Store episode lengths for plotting

    # Video buffers -----------------------------------------------------
    frame_buffer: list[list[np.ndarray]] | None = [[] for _ in range(num_envs)] if save_video else None
    all_frames: list[np.ndarray] | None = [] if save_video else None

    done_episodes = 0
    obs, _ = env.reset()

    # Initialize progress display with dots
    progress_dots = ["."] * num_episodes
    print(f"Evaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

    while done_episodes < num_episodes:
        # --------------------------------------------------------------
        # 1. Policy inference + Q-value prediction ---------------------
        # --------------------------------------------------------------
        with torch.no_grad():
            actions = q_actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            # Build features on-the-fly to obtain Q-predictions --------
            obs_q = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            obs_q["feat"] = agent._encode(obs_q, augment=False)

            # For Q-value computation, use combined action and clamp to [-1, 1] (consistent with training)
            if agent.residual_actor:
                q_actions = torch.clamp(obs["observation.base_action"] + actions, -1.0, 1.0)

            q_pred = (
                agent.critic.q_value(obs_q["feat"], obs_q["observation.state"], q_actions).detach().cpu().squeeze(-1)
            )

        # --------------------------------------------------------------
        # 2. Environment step ------------------------------------------
        # --------------------------------------------------------------
        next_obs, reward, terminated, truncated, _ = env.step(actions)
        done_flags = terminated | truncated

        # Capture frames ------------------------------------------------
        if save_video and frame_buffer is not None:
            frame = env.render()
            for env_idx in range(num_envs):
                frame_buffer[env_idx].append(frame[env_idx])

        # --------------------------------------------------------------
        # 3. Per-environment bookkeeping -------------------------------
        # --------------------------------------------------------------
        for env_idx in range(num_envs):
            ep_rewards[env_idx].append(reward[env_idx].item())
            ep_q_preds[env_idx].append(q_pred[env_idx].item())

            if done_flags[env_idx]:
                # Episode finished -- aggregate results ----------------
                ep_return = float(sum(ep_rewards[env_idx]))
                is_success = bool(reward[env_idx].item() == 1.0)

                # Update progress display
                progress_dots[done_episodes] = "✓" if is_success else "✗"
                print(f"\rEvaluating {num_episodes} episodes: {''.join(progress_dots)}", end="", flush=True)

                successes.append(is_success)
                returns.append(ep_return)

                # Store Q-trajectory data for plotting ------------------
                if save_q_plots:
                    all_q_trajectories.append(ep_q_preds[env_idx].copy())

                # Always track episode length for successful episodes logging
                all_episode_lengths.append(len(ep_q_preds[env_idx]))

                # Annotate and flush frames ---------------------------
                if save_video and frame_buffer is not None and all_frames is not None:
                    episode_frames = frame_buffer[env_idx]
                    episode_qs = ep_q_preds[env_idx]

                    episode_global_idx = done_episodes + 1  # 1-based

                    for step_idx, fr in enumerate(episode_frames):
                        annotated_fr = _annotate_frame(
                            fr,
                            env_idx=env_idx,
                            episode_num=episode_global_idx,
                            total_episodes=num_episodes,
                            step_idx=step_idx + 1,
                            is_success=is_success,
                            q_value=episode_qs[step_idx],
                        )
                        all_frames.append(annotated_fr)

                    # Clear per-episode frame buffer
                    frame_buffer[env_idx].clear()

                # Reset per-env caches --------------------------------
                ep_rewards[env_idx].clear()
                ep_q_preds[env_idx].clear()

                done_episodes += 1

                if done_episodes == num_episodes:
                    break

        # Prepare for next loop ----------------------------------------
        obs = next_obs

    print("Done")

    # ------------------------------------------------------------------
    # 4. Aggregate metrics ---------------------------------------------
    # ------------------------------------------------------------------
    # Sanity check: episode lengths must align 1:1 with successes
    if len(all_episode_lengths) != len(successes):
        raise RuntimeError(
            f"Episode length/success misalignment: lengths={len(all_episode_lengths)} successes={len(successes)}"
        )

    success_rate: float = float(np.mean(successes)) if successes else 0.0
    mean_return: float = float(np.mean(returns)) if returns else 0.0

    # Calculate mean episode length among successful episodes
    successful_episode_lengths = [length for length, is_success in zip(all_episode_lengths, successes) if is_success]
    mean_successful_episode_length: float = (
        float(np.mean(successful_episode_lengths)) if successful_episode_lengths else 0.0
    )

    metrics: dict[str, float] = {
        "eval/success_rate": success_rate,
        "eval/mean_return": mean_return,
        "eval/mean_successful_episode_length": mean_successful_episode_length,
    }

    if wandb.run is not None:
        wandb.log(metrics, step=global_step)

    # ------------------------------------------------------------------
    # 5. Q-trajectory plots --------------------------------------------
    # ------------------------------------------------------------------
    if save_q_plots and all_q_trajectories and run_name is not None:
        parent = Path(str(output_dir or "outputs")) / run_name.split("__")[0]
        parent.mkdir(parents=True, exist_ok=True)

        plot_name = f"eval_q_trajectories_{run_name}_step_{global_step if global_step is not None else 'NA'}.png"
        plot_path = parent / plot_name

        _create_q_trajectory_plots(
            trajectories=all_q_trajectories,
            episode_lengths=all_episode_lengths,
            successes=successes,
            output_path=plot_path,
            global_step=global_step,
        )

        # Log to W&B if available
        if wandb.run is not None:
            wandb.log({"value/q_trajectories": wandb.Image(str(plot_path))}, step=global_step)

    # ------------------------------------------------------------------
    # 6. Video dump + W&B logging --------------------------------------
    # ------------------------------------------------------------------
    if save_video and all_frames is not None and run_name is not None:
        parent = Path(str(output_dir or "outputs")) / run_name.split("__")[0]
        parent.mkdir(parents=True, exist_ok=True)

        vid_name = f"eval_{run_name}_step_{global_step if global_step is not None else 'NA'}.mp4"
        video_path = parent / vid_name

        fps_val = getattr(env, "fps", 20)

        writer = imageio.get_writer(video_path, fps=fps_val)
        for fr in all_frames:
            writer.append_data(fr)
        writer.close()

        if wandb.run is not None:
            wandb.log({"eval/video": wandb.Video(str(video_path), format="mp4")}, step=global_step)

    # Restore training mode --------------------------------------------
    agent.train(True)

    return metrics
