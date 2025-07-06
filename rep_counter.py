# rep_counter.py
import numpy as np
import torch


def calculate_angle(a, b, c):
    """Calculate angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Ensure points are 1D vectors
    if a.ndim > 1:
        a = a.flatten()
    if b.ndim > 1:
        b = b.flatten()
    if c.ndim > 1:
        c = c.flatten()

    ba = a - b
    bc = c - b

    # Add small epsilon to prevent division by zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 180.0  # Return straight angle if points are too close

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)


def count_reps_from_3d_pose(pose3d_seq, angle_threshold_down=110, angle_threshold_up=160,
                            joint_indices=None, min_frames_between_transitions=3):
    """
    Count repetitions from 3D pose sequence.

    Args:
        pose3d_seq: Can be torch tensor or numpy array with shape (batch, time, joints, 3) or (time, joints, 3)
        angle_threshold_down: Lower angle threshold for rep detection (bent position)
        angle_threshold_up: Upper angle threshold for rep detection (extended position)
        joint_indices: Tuple of indices (hip, knee, ankle) for angle calculation
        min_frames_between_transitions: Minimum frames required between state transitions to avoid noise
    """
    # Default joint indices for Human3.6M (right leg)
    if joint_indices is None:
        joint_indices = (4, 5, 6)  # Right hip, knee, ankle

    hip_idx, knee_idx, ankle_idx = joint_indices

    # Convert torch tensor to numpy if needed
    if isinstance(pose3d_seq, torch.Tensor):
        pose3d_seq = pose3d_seq.detach().cpu().numpy()

    # Handle different input shapes
    if pose3d_seq.ndim == 4:  # (batch, time, joints, 3)
        # Take the first batch
        pose3d_seq = pose3d_seq[0]
    elif pose3d_seq.ndim == 3:  # (time, joints, 3)
        pass  # Already correct shape
    else:
        raise ValueError(f"Unexpected pose sequence shape: {pose3d_seq.shape}")

    print(f"Processing pose sequence with shape: {pose3d_seq.shape}")
    print(
        f"Using joint indices - Hip: {hip_idx}, Knee: {knee_idx}, Ankle: {ankle_idx}")
    print(
        f"Thresholds - Down: {angle_threshold_down}°, Up: {angle_threshold_up}°")

    # Validate joint indices
    max_joint_idx = max(hip_idx, knee_idx, ankle_idx)
    if max_joint_idx >= pose3d_seq.shape[1]:
        raise ValueError(
            f"Joint index {max_joint_idx} exceeds available joints {pose3d_seq.shape[1]}")

    count = 0
    state = "neutral"  # "neutral", "down", "up"
    last_transition_frame = -min_frames_between_transitions
    angles = []

    for frame_idx, frame in enumerate(pose3d_seq):
        try:
            hip = frame[hip_idx]
            knee = frame[knee_idx]
            ankle = frame[ankle_idx]
        except IndexError:
            print(
                f"Frame {frame_idx} - Joint index out of bounds. Available joints: {frame.shape[0]}")
            continue

        angle = calculate_angle(hip, knee, ankle)
        angles.append(angle)

        # Debug print for first few frames
        if frame_idx < 5:
            print(
                f"Frame {frame_idx}: Hip={hip}, Knee={knee}, Ankle={ankle}, Angle={angle:.1f}°")

        # Ensure minimum frames between transitions to avoid noise
        if frame_idx - last_transition_frame < min_frames_between_transitions:
            continue

        # State machine for rep counting
        if state == "neutral":
            if angle < angle_threshold_down:
                state = "down"
                last_transition_frame = frame_idx
                print(
                    f"Frame {frame_idx}: Entering DOWN state (angle: {angle:.1f}°)")
            elif angle > angle_threshold_up:
                state = "up"
                last_transition_frame = frame_idx
                print(
                    f"Frame {frame_idx}: Entering UP state (angle: {angle:.1f}°)")

        elif state == "down":
            if angle > angle_threshold_up:
                state = "up"
                last_transition_frame = frame_idx
                print(
                    f"Frame {frame_idx}: DOWN->UP transition (angle: {angle:.1f}°)")

        elif state == "up":
            if angle < angle_threshold_down:
                state = "down"
                last_transition_frame = frame_idx
                count += 1
                print(
                    f"Frame {frame_idx}: UP->DOWN transition - Rep #{count} completed! (angle: {angle:.1f}°)")

    # Print summary statistics
    if angles:
        print(f"\nSummary:")
        print(f"Total reps counted: {count}")
        print(f"Angle range: {min(angles):.1f}° - {max(angles):.1f}°")
        print(f"Average angle: {np.mean(angles):.1f}°")
        print(f"Final state: {state}")

    return count


def smooth_angles(angles, window_size=5):
    """Apply moving average smoothing to angle sequence."""
    if len(angles) < window_size:
        return angles

    smoothed = []
    for i in range(len(angles)):
        start = max(0, i - window_size // 2)
        end = min(len(angles), i + window_size // 2 + 1)
        smoothed.append(np.mean(angles[start:end]))

    return smoothed


def count_reps_with_smoothing(pose3d_seq, angle_threshold_down=110, angle_threshold_up=160,
                              joint_indices=None, smoothing_window=5):
    """
    Count repetitions with angle smoothing for more robust detection.
    """
    # Convert and validate input (same as before)
    if joint_indices is None:
        joint_indices = (4, 5, 6)

    hip_idx, knee_idx, ankle_idx = joint_indices

    if isinstance(pose3d_seq, torch.Tensor):
        pose3d_seq = pose3d_seq.detach().cpu().numpy()

    if pose3d_seq.ndim == 4:
        pose3d_seq = pose3d_seq[0]
    elif pose3d_seq.ndim != 3:
        raise ValueError(f"Unexpected pose sequence shape: {pose3d_seq.shape}")

    # Calculate all angles first
    angles = []
    for frame in pose3d_seq:
        hip = frame[hip_idx]
        knee = frame[knee_idx]
        ankle = frame[ankle_idx]
        angle = calculate_angle(hip, knee, ankle)
        angles.append(angle)

    # Apply smoothing
    smoothed_angles = smooth_angles(angles, smoothing_window)

    # Count reps using smoothed angles
    count = 0
    state = "neutral"
    min_frames_between = 3
    last_transition = -min_frames_between

    for i, angle in enumerate(smoothed_angles):
        if i - last_transition < min_frames_between:
            continue

        if state == "neutral":
            if angle < angle_threshold_down:
                state = "down"
                last_transition = i
            elif angle > angle_threshold_up:
                state = "up"
                last_transition = i
        elif state == "down":
            if angle > angle_threshold_up:
                state = "up"
                last_transition = i
        elif state == "up":
            if angle < angle_threshold_down:
                state = "down"
                last_transition = i
                count += 1
                print(
                    f"Rep #{count} completed at frame {i} (smoothed angle: {angle:.1f}°)")

    print(f"Total reps with smoothing: {count}")
    return count, smoothed_angles
