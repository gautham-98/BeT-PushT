from typing import Literal
import pickle

class Normalise:
    """
    A class to normalise and denormalise observation states and actions based on dataset statistics.
    """

    _stats = None

    @classmethod
    def set_stats(cls, stats: dict):
        cls._stats = stats
        with open("normaliser.stats", "wb") as f:
            pickle.dump(stats, f)

    @classmethod
    def set_stats_from_file(cls, stats_file: str = "normaliser.stats"):
        with open(stats_file, "rb") as f:
            stats = pickle.load(f)
        cls.set_stats(stats)
    
    @classmethod
    def forward(cls, sample, name: Literal["observation.state", "action"]):
        """Normalize the states in the sample dictionary."""
        if cls._stats is None:
            raise ValueError(
                "Dataset statistics not set. Please set stats using 'set_stats' method before normalizing."
            )

        mean = cls._stats[name]["mean"]
        std = cls._stats[name]["std"]

        # Normalize
        normalized_states = (sample - mean) / std
        return normalized_states

    @classmethod
    def inverse(cls, sample, name: Literal["observation.state", "action"]):
        """Denormalize the states in the sample dictionary."""
        if cls._stats is None:
            raise ValueError(
                "Dataset statistics not set. Please set stats using 'set_stats' method before denormalizing."
            )

        mean = cls._stats[name]["mean"]
        std = cls._stats[name]["std"]

        # Denormalize
        denormalized_states = sample * std + mean
        return denormalized_states
