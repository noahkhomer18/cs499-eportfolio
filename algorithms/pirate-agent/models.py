"""
Neural network models for the Treasure Hunt Game.
This module contains the AI models that learn to play the game.
"""

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, PReLU, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Optional, Dict, Any
import os


class QNetwork:
    def __init__(self, input_size: int, num_actions: int, config: Optional[Dict[str, Any]] = None):
        self.input_size = input_size
        self.num_actions = num_actions
        self.config = config or {}
        
        self.hidden_size = self.config.get('hidden_layer_size', 64)
        self.num_hidden_layers = self.config.get('num_hidden_layers', 2)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.dropout_rate = self.config.get('dropout_rate', 0.1)
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        
        self.q_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.training_history = {
            'loss': [],
            'episode_rewards': [],
            'win_rates': []
        }
    
    def _build_model(self) -> Sequential:
        """
        Build the neural network architecture.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_size, input_shape=(self.input_size,)))
        model.add(PReLU())
        if self.dropout_rate > 0:
            model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for _ in range(self.num_hidden_layers - 1):
            model.add(Dense(self.hidden_size))
            model.add(PReLU())
            if self.dropout_rate > 0:
                model.add(Dropout(self.dropout_rate))
        
        # Output layer (Q-values for each action)
        model.add(Dense(self.num_actions, activation='linear'))
        
        # Compile with Adam optimizer
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        state_key = tuple(state.flatten())
        if state_key in self.q_cache:
            self.cache_hits += 1
            return self.q_cache[state_key]
        
        self.cache_misses += 1
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        q_values = self.model.predict(state, verbose=0)
        self.q_cache[state_key] = q_values
        return q_values
    
    def predict_target(self, state: np.ndarray) -> np.ndarray:
        """
        Predict Q-values using the target network (for stability).
        
        Args:
            state: Current game state
            
        Returns:
            Q-values for each action from target network
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return self.target_model.predict(state, verbose=0)
    
    def train(self, states: np.ndarray, targets: np.ndarray, 
              batch_size: int = 32, epochs: int = 1, verbose: int = 0) -> Dict[str, float]:
        """
        Train the model on a batch of experiences.
        
        Args:
            states: Batch of states
            targets: Batch of target Q-values
            batch_size: Training batch size
            epochs: Number of training epochs
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        history = self.model.fit(
            states, targets,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=0.1 if len(states) > 10 else 0
        )
        
        # Store training metrics
        self.training_history['loss'].extend(history.history['loss'])
        
        return history.history
    
    def update_target_network(self):
        """Update the target network with current network weights."""
        self.target_model.set_weights(self.model.get_weights())
    
    def save(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = load_model(filepath)
        self.target_model = load_model(filepath)  # Load same weights to target network
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """Get a string summary of the model architecture."""
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()


class PrioritizedExperienceReplay:
    def __init__(self, max_size: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.beta_increment = (1.0 - beta) / 10000
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, priority: float = None):
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
        
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int) -> tuple:
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def size(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()
        self.priorities.clear()
        self.position = 0

class ExperienceReplay:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode is finished
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size: int) -> tuple:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear the experience buffer."""
        self.buffer.clear()
        self.position = 0


class EpsilonScheduler:
    """
    Epsilon-greedy exploration scheduler.
    
    This controls how much the AI explores vs exploits. Initially, it explores
    a lot (high epsilon), but as it learns, it gradually exploits more (low epsilon).
    """
    
    def __init__(self, initial_epsilon: float = 1.0, final_epsilon: float = 0.1, 
                 decay_steps: int = 1000, decay_type: str = 'linear'):
        """
        Initialize the epsilon scheduler.
        
        Args:
            initial_epsilon: Starting exploration rate
            final_epsilon: Final exploration rate
            decay_steps: Number of steps to decay over
            decay_type: Type of decay ('linear' or 'exponential')
        """
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps
        self.decay_type = decay_type
        self.current_step = 0
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        if self.current_step >= self.decay_steps:
            return self.final_epsilon
        
        if self.decay_type == 'linear':
            # Linear decay
            progress = self.current_step / self.decay_steps
            return self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * progress
        elif self.decay_type == 'exponential':
            # Exponential decay
            decay_rate = np.log(self.final_epsilon / self.initial_epsilon) / self.decay_steps
            return self.initial_epsilon * np.exp(decay_rate * self.current_step)
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
    
    def step(self):
        """Advance the scheduler by one step."""
        self.current_step += 1
    
    def reset(self):
        """Reset the scheduler to initial state."""
        self.current_step = 0


def create_model(input_size: int, num_actions: int, config: Optional[Dict[str, Any]] = None) -> QNetwork:
    """
    Factory function to create a Q-Network with the given configuration.
    
    Args:
        input_size: Size of input state
        num_actions: Number of possible actions
        config: Optional configuration dictionary
        
    Returns:
        Configured QNetwork instance
    """
    return QNetwork(input_size, num_actions, config)
