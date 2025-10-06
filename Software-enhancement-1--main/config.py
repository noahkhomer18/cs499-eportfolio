"""
Configuration settings for the Treasure Hunt Game.
This file centralizes all the parameters so you can easily tweak the game behavior
without digging through the code.
"""

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class GameConfig:
    """Main configuration class that holds all game parameters."""
    
    # Maze settings - you can change these to create different mazes
    maze_size: Tuple[int, int] = (8, 8)
    maze_layout: str = "default"  # "default", "custom", or "random"
    
    # Training parameters - these control how the AI learns
    epochs: int = 200
    max_memory: int = 1000
    batch_size: int = 16
    data_size: int = 32
    
    # Learning parameters - these affect the AI's learning speed and behavior
    learning_rate: float = 0.001
    initial_epsilon: float = 0.1  # How much the AI explores vs exploits initially
    final_epsilon: float = 0.05   # Minimum exploration rate
    epsilon_decay_steps: int = 100  # How quickly exploration decreases
    
    # Neural network architecture
    hidden_layer_size: int = 64
    num_hidden_layers: int = 2
    
    # Reward system - these control what the AI considers "good" behavior
    treasure_reward: float = 1.0
    step_penalty: float = -0.04
    wall_penalty: float = -0.1
    win_bonus: float = 10.0
    
    # Visualization settings
    show_training: bool = True
    animation_speed: float = 0.5  # seconds between frames
    save_plots: bool = True
    
    # File paths - where to save results
    model_save_path: str = "models/"
    plots_save_path: str = "plots/"
    logs_save_path: str = "logs/"
    
    # Training settings
    early_stopping_patience: int = 50  # Stop if no improvement for this many epochs
    target_win_rate: float = 0.95  # Stop training when we reach this win rate
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.model_save_path, self.plots_save_path, self.logs_save_path]:
            os.makedirs(path, exist_ok=True)


# Default maze layout - the one from your original notebook
DEFAULT_MAZE = [
    [1., 0., 1., 1., 1., 1., 1., 1.],
    [1., 0., 1., 1., 1., 0., 1., 1.],
    [1., 1., 1., 1., 0., 1., 0., 1.],
    [1., 1., 1., 0., 1., 1., 1., 1.],
    [1., 1., 0., 1., 1., 1., 1., 1.],
    [1., 1., 1., 0., 1., 0., 0., 0.],
    [1., 1., 1., 0., 1., 1., 1., 1.],
    [1., 1., 1., 1., 0., 1., 1., 1.]
]


def get_config() -> GameConfig:
    """Get the default configuration. You can modify this function to load from files."""
    return GameConfig()


def save_config(config: GameConfig, filename: str = "config.json"):
    """Save configuration to a JSON file for reproducibility."""
    import json
    config_dict = {
        'maze_size': config.maze_size,
        'epochs': config.epochs,
        'learning_rate': config.learning_rate,
        'initial_epsilon': config.initial_epsilon,
        'final_epsilon': config.final_epsilon,
        'epsilon_decay_steps': config.epsilon_decay_steps,
        'treasure_reward': config.treasure_reward,
        'step_penalty': config.step_penalty,
        'wall_penalty': config.wall_penalty,
        'win_bonus': config.win_bonus,
    }
    
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to {filename}")


def load_config(filename: str = "config.json") -> GameConfig:
    """Load configuration from a JSON file."""
    import json
    with open(filename, 'r') as f:
        config_dict = json.load(f)
    
    # Create config with loaded values, using defaults for missing keys
    default_config = GameConfig()
    for key, value in config_dict.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)
    
    return default_config
