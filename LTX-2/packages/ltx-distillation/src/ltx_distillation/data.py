"""
Dataset classes for DMD distillation.
"""

import bisect
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Simple text prompt dataset.

    Reads prompts from a text file where each line is one prompt.
    The prompts are assumed to be already processed (no enhancement needed).
    """

    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_path: Path to text file with one prompt per line
            max_samples: Maximum number of samples to load (None for all)
        """
        self.data_path = data_path
        self.prompts = self._load_prompts(data_path, max_samples)

    def _load_prompts(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """Load prompts from file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        if max_samples is not None:
            prompts = prompts[:max_samples]

        print(f"Loaded {len(prompts)} prompts from {data_path}")
        return prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


class ODERegressionLMDBDataset(Dataset):
    """
    LMDB dataset for ODE regression training.

    Stores pre-computed ODE trajectories for faster training.
    This is used when backward_simulation=False.

    LMDB stores:
        - video_latents_{idx}_data: video trajectory bytes [T, F, C, H, W]
        - audio_latents_{idx}_data: audio trajectory bytes [T, F_a, C]
        - prompts_{idx}_data: prompt string bytes
        - video_latents_shape: "[total, T, F, C, H, W]"
        - audio_latents_shape: "[total, T, F_a, C]"

    The input path can be either:
        - a single LMDB directory
        - a root directory containing multiple shard LMDBs named `shard_*`
    """

    def __init__(
        self,
        lmdb_path: str,
        max_pair: int = int(1e8),
    ):
        """
        Args:
            lmdb_path: Path to LMDB database
            max_pair: Maximum number of pairs to load
        """
        self.lmdb_path = lmdb_path
        self.max_pair = max_pair

        try:
            import lmdb
            self.envs = []
            self.shard_lengths = []
            self.cumulative_lengths = []
            self.shard_paths = self._discover_lmdb_paths(lmdb_path)
            self.is_sharded = len(self.shard_paths) > 1

            remaining = max_pair
            self.video_entry_shape = None
            self.audio_entry_shape = None
            self.has_audio = False

            for shard_path in self.shard_paths:
                if remaining <= 0:
                    break

                env = lmdb.open(
                    str(shard_path),
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                shard_length, video_entry_shape, audio_entry_shape, has_audio = self._read_shape_metadata(
                    env, str(shard_path)
                )
                shard_length = min(shard_length, remaining)
                remaining -= shard_length

                if self.video_entry_shape is None:
                    self.video_entry_shape = video_entry_shape
                    self.audio_entry_shape = audio_entry_shape
                    self.has_audio = has_audio
                else:
                    if self.video_entry_shape != video_entry_shape:
                        raise ValueError(
                            f"Inconsistent video shape in shard {shard_path}: "
                            f"{video_entry_shape} vs {self.video_entry_shape}"
                        )
                    if self.audio_entry_shape != audio_entry_shape:
                        raise ValueError(
                            f"Inconsistent audio shape in shard {shard_path}: "
                            f"{audio_entry_shape} vs {self.audio_entry_shape}"
                        )
                    if self.has_audio != has_audio:
                        raise ValueError(
                            f"Inconsistent audio availability in shard {shard_path}: "
                            f"{has_audio} vs {self.has_audio}"
                        )

                self.envs.append(env)
                self.shard_lengths.append(shard_length)
                cumulative = shard_length if not self.cumulative_lengths else self.cumulative_lengths[-1] + shard_length
                self.cumulative_lengths.append(cumulative)

            self.length = self.cumulative_lengths[-1] if self.cumulative_lengths else 0

        except ImportError:
            raise ImportError("lmdb package required for ODERegressionLMDBDataset")
        except Exception as e:
            raise RuntimeError(f"Failed to open LMDB at {lmdb_path}: {e}")

        if self.is_sharded:
            print(
                f"Loaded sharded LMDB dataset with {self.length} samples "
                f"from {lmdb_path} across {len(self.envs)} shard(s)"
            )
        else:
            print(f"Loaded LMDB dataset with {self.length} samples from {lmdb_path}")
        print(f"  Video shape per entry: {self.video_entry_shape}")
        if self.has_audio:
            print(f"  Audio shape per entry: {self.audio_entry_shape}")

    @staticmethod
    def _discover_lmdb_paths(lmdb_path: str) -> list[Path]:
        root = Path(lmdb_path)
        if not root.exists():
            raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

        if (root / "data.mdb").exists():
            return [root]

        shard_paths = sorted(
            (
                path
                for path in root.iterdir()
                if path.is_dir() and path.name.startswith("shard_") and (path / "data.mdb").exists()
            ),
            key=lambda path: path.name,
        )
        if shard_paths:
            return shard_paths

        raise FileNotFoundError(
            f"No LMDB found at {lmdb_path}. Expected either data.mdb or shard_* subdirectories."
        )

    @staticmethod
    def _read_shape_metadata(env, lmdb_path: str) -> tuple[int, list[int], Optional[list[int]], bool]:
        with env.begin(write=False) as txn:
            video_shape_bytes = txn.get("video_latents_shape".encode())
            if video_shape_bytes is None:
                raise ValueError(f"Missing video_latents_shape in LMDB: {lmdb_path}")
            video_shape = list(map(int, video_shape_bytes.decode().split()))
            count = video_shape[0]
            video_entry_shape = video_shape[1:]

            audio_shape_bytes = txn.get("audio_latents_shape".encode())
            if audio_shape_bytes is not None:
                audio_shape = list(map(int, audio_shape_bytes.decode().split()))
                audio_entry_shape = audio_shape[1:]
                has_audio = True
            else:
                audio_entry_shape = None
                has_audio = False

        return count, video_entry_shape, audio_entry_shape, has_audio

    def _get_env_for_index(self, idx: int):
        shard_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        previous_total = 0 if shard_idx == 0 else self.cumulative_lengths[shard_idx - 1]
        local_idx = idx - previous_total
        return self.envs[shard_idx], local_idx

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from LMDB.

        Returns:
            Dictionary containing:
                - prompts: Text prompt
                - ode_latent: ODE video trajectory [T, F, C, H, W]
                - ode_audio_latent: ODE audio trajectory [T, F_a, C] (if available)
        """
        env, local_idx = self._get_env_for_index(idx)

        with env.begin(write=False) as txn:
            # Load prompt
            prompt_key = f"prompts_{local_idx}_data".encode()
            prompt_bytes = txn.get(prompt_key)
            if prompt_bytes is None:
                raise KeyError(f"Prompt key {idx} (local {local_idx}) not found in LMDB")
            prompt = prompt_bytes.decode('utf-8')

            # Load video latents
            video_key = f"video_latents_{local_idx}_data".encode()
            video_bytes = txn.get(video_key)
            if video_bytes is None:
                raise KeyError(f"Video latents key {idx} (local {local_idx}) not found in LMDB")
            video_array = np.frombuffer(video_bytes, dtype=np.float16)
            video_array = video_array.reshape(self.video_entry_shape)
            video_tensor = torch.from_numpy(video_array.copy()).float()

            # Load audio latents (if available)
            audio_tensor = None
            if self.has_audio:
                audio_key = f"audio_latents_{local_idx}_data".encode()
                audio_bytes = txn.get(audio_key)
                if audio_bytes is not None:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float16)
                    audio_array = audio_array.reshape(self.audio_entry_shape)
                    audio_tensor = torch.from_numpy(audio_array.copy()).float()

        result = {
            "prompts": prompt,
            "ode_latent": video_tensor,  # [T, F, C, H, W]
        }

        if audio_tensor is not None:
            result["ode_audio_latent"] = audio_tensor  # [T, F_a, C]

        return result

    def __del__(self):
        self.close()

    def close(self):
        for env in getattr(self, "envs", []):
            try:
                env.close()
            except Exception:
                pass
        self.envs = []


def collate_text_prompts(batch: List[str]) -> List[str]:
    """Simple collate function for text prompts."""
    return batch


def collate_ode_data(batch: List[dict]) -> dict:
    """
    Collate function for ODE regression data.

    Handles both video and audio latents.
    Audio may be None if not available in the dataset.
    """
    prompts = [item["prompts"] for item in batch]
    ode_latents = torch.stack([item["ode_latent"] for item in batch])

    result = {
        "prompts": prompts,
        "ode_latent": ode_latents,  # [B, T, F, C, H, W]
    }

    # Check if audio is available (first item determines availability)
    if "ode_audio_latent" in batch[0] and batch[0]["ode_audio_latent"] is not None:
        ode_audio_latents = torch.stack([item["ode_audio_latent"] for item in batch])
        result["ode_audio_latent"] = ode_audio_latents  # [B, T, F_a, C]

    return result
