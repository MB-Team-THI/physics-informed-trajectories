import torch

def _squeeze_traj_dims(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B,N,1,T]; got {x.shape}")
    return x.squeeze(2)


def compute_sampling_metrics(all_predictions: list[dict[str, torch.Tensor]]) -> torch.Tensor | None:
    """
    Computes spread, diversity, and multimodality metrics averaged over the entire
    trajectory duration.

    Args:
        all_predictions: A list of dictionaries from each rollout, containing
                         'X_reshaped' and 'Y_reshaped' tensors.

    Returns:
        A tensor of shape [num_vehicles, 3] containing the time-averaged metrics
        (Spread, Diversity, Multimodality) for each vehicle.
    """
    # --- 1. Handle Edge Cases & Consolidate Data ---
    if not all_predictions or len(all_predictions) < 2:
        print("Not enough predictions to compute metrics (need at least 2).")
        return None

    try:
        all_trajs_list = [
            torch.stack([
                _squeeze_traj_dims(run['X_reshaped']),
                _squeeze_traj_dims(run['Y_reshaped'])
            ], dim=-1) for run in all_predictions
        ]
        # Shape: [num_rollouts, num_vehicles, target_len, 2]
        all_trajs = torch.cat(all_trajs_list, dim=0)
        device = all_trajs.device
    except (KeyError, ValueError) as e:
        print(f"Error processing prediction data: {e}")
        return None
    all_trajs = all_trajs.squeeze(3)
    num_rollouts, num_vehicles, target_len, _ = all_trajs.shape

    # --- 2. Initialize Tensors to Store Per-Timestep Metrics ---
    # We will store the metrics for each timestep before averaging.
    # Shape: [num_vehicles, target_len]
    temporal_spread = torch.zeros(num_vehicles, target_len, device=device)
    temporal_diversity = torch.zeros(num_vehicles, target_len, device=device)
    temporal_multimodality = torch.zeros(num_vehicles, target_len, device=device)

    # --- 3. Loop Through Each Timestep to Calculate Metrics ---
    for t in range(target_len):
        # Get all predicted (x,y) points at the current timestep 't'
        # Shape: [num_rollouts, num_vehicles, 2]
        points_at_t = all_trajs[:, :, t, :]

        for v_idx in range(num_vehicles):
            # Shape: [num_rollouts, 2]
            vehicle_points_at_t = points_at_t[:, v_idx, :]

            # --- Calculate metrics for this specific timestep 't' ---

            # Spread
            temporal_spread[v_idx, t] = torch.linalg.norm(torch.std(vehicle_points_at_t, dim=0))

            # Diversity
            pairwise_dists = torch.pdist(vehicle_points_at_t)
            temporal_diversity[v_idx, t] = torch.mean(pairwise_dists)

            # Multimodality
            if num_rollouts > 2:
                dist_matrix = torch.cdist(vehicle_points_at_t, vehicle_points_at_t)
                max_flat_idx = torch.argmax(dist_matrix)
                anchor_idx1, anchor_idx2 = max_flat_idx // num_rollouts, max_flat_idx % num_rollouts

                dists_to_1 = torch.linalg.norm(vehicle_points_at_t - vehicle_points_at_t[anchor_idx1], dim=-1)
                dists_to_2 = torch.linalg.norm(vehicle_points_at_t - vehicle_points_at_t[anchor_idx2], dim=-1)

                cluster1_mask = dists_to_1 < dists_to_2
                if torch.any(cluster1_mask) and torch.any(~cluster1_mask):
                    centroid1 = torch.mean(vehicle_points_at_t[cluster1_mask], dim=0)
                    centroid2 = torch.mean(vehicle_points_at_t[~cluster1_mask], dim=0)
                    temporal_multimodality[v_idx, t] = torch.linalg.norm(centroid1 - centroid2)

    # --- 4. Aggregate Metrics by Averaging Over Time ---
    # We take the mean across the 'target_len' dimension.
    avg_spread = torch.mean(temporal_spread, dim=1)
    avg_diversity = torch.mean(temporal_diversity, dim=1)
    avg_multimodality = torch.mean(temporal_multimodality, dim=1)

    # --- 5. Combine and Return Final Metrics ---
    # Shape: [num_vehicles, 3]
    all_metrics = torch.stack([avg_spread, avg_diversity, avg_multimodality], dim=1)

    # Optional: Print metrics for real-time feedback
    # print("\n--- Avg. Time-Based Sampling Metrics (per vehicle) ---")
    # print(f"{'Veh #':<6} | {'Avg Spread':<12} | {'Avg Diversity':<15} | {'Avg Multimodality':<18}")
    # print("-" * 60)
    # for v_idx in range(num_vehicles):
    #     s, d, m = all_metrics[v_idx].tolist()
    #     print(f"{v_idx + 1:<6} | {s:<12.2f} | {d:<15.2f} | {m:<18.2f}")
    # print("-" * 60)

    return all_metrics
