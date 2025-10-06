"""
Enhanced training system for the Treasure Hunt Game.
This is the "Controller" in our MVC architecture - it orchestrates the training process.
"""

import numpy as np
import time
import datetime
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import logging

from config import GameConfig, DEFAULT_MAZE, get_config
from maze import TreasureMaze, Action
from models import QNetwork, ExperienceReplay, EpsilonScheduler
from visualizer import MazeVisualizer


class TrainingController:
    """
    Main controller for training the treasure hunt AI.
    
    This class orchestrates the entire training process, managing the maze environment,
    neural network, experience replay, and visualization. It's designed to be flexible
    and easy to use while providing detailed feedback on training progress.
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize the training controller.
        
        Args:
            config: Game configuration. If None, uses default config.
        """
        self.config = config or get_config()
        
        # Set up logging
        self._setup_logging()
        
        # Initialize components
        self.maze = None
        self.model = None
        self.experience_replay = None
        self.epsilon_scheduler = None
        self.visualizer = None
        
        # Training state
        self.training_history = {
            'loss': [],
            'episode_rewards': [],
            'win_rates': [],
            'epsilon': [],
            'episode_lengths': [],
            'exploration_rates': []
        }
        
        # Performance tracking
        self.best_win_rate = 0.0
        self.best_model_path = None
        self.training_start_time = None
        
        self.logger.info("Training controller initialized")
    
    def _setup_logging(self):
        """Set up logging for training progress."""
        log_dir = self.config.logs_save_path
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{log_dir}training_{int(time.time())}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_training(self, maze_layout: Optional[np.ndarray] = None) -> None:
        """
        Set up all components for training.
        
        Args:
            maze_layout: Optional custom maze layout. If None, uses default maze.
        """
        self.logger.info("Setting up training components...")
        
        # Create maze environment
        if maze_layout is None:
            maze_layout = np.array(DEFAULT_MAZE)
        
        maze_config = {
            'treasure_reward': self.config.treasure_reward,
            'step_penalty': self.config.step_penalty,
            'wall_penalty': self.config.wall_penalty,
            'win_bonus': self.config.win_bonus
        }
        
        self.maze = TreasureMaze(maze_layout, maze_config)
        
        # Create neural network
        self.model = QNetwork(
            input_size=self.maze.size,
            num_actions=4,  # up, down, left, right
            config={
                'hidden_layer_size': self.config.hidden_layer_size,
                'num_hidden_layers': self.config.num_hidden_layers,
                'learning_rate': self.config.learning_rate
            }
        )
        
        # Create experience replay buffer
        self.experience_replay = ExperienceReplay(max_size=self.config.max_memory)
        
        # Create epsilon scheduler
        self.epsilon_scheduler = EpsilonScheduler(
            initial_epsilon=self.config.initial_epsilon,
            final_epsilon=self.config.final_epsilon,
            decay_steps=self.config.epsilon_decay_steps,
            decay_type='linear'
        )
        
        # Create visualizer
        self.visualizer = MazeVisualizer({
            'save_plots': self.config.save_plots,
            'plots_save_path': self.config.plots_save_path
        })
        
        self.logger.info(f"Maze size: {self.maze.nrows}x{self.maze.ncols}")
        self.logger.info(f"Free cells: {len(self.maze.free_cells)}")
        self.logger.info(f"Model architecture: {self.config.hidden_layer_size} hidden units, "
                        f"{self.config.num_hidden_layers} layers")
    
    def train(self, epochs: Optional[int] = None, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the AI agent to play the treasure hunt game.
        
        Args:
            epochs: Number of training epochs. If None, uses config default.
            verbose: Whether to print training progress.
            
        Returns:
            Dictionary containing training results and statistics.
        """
        if self.maze is None or self.model is None:
            raise RuntimeError("Training not set up. Call setup_training() first.")
        
        epochs = epochs or self.config.epochs
        self.training_start_time = time.time()
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        self.logger.info(f"Initial epsilon: {self.epsilon_scheduler.initial_epsilon}")
        self.logger.info(f"Target win rate: {self.config.target_win_rate}")
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Run one training episode
            episode_reward, episode_length, win = self._run_episode()
            
            # Update training history
            self._update_history(episode_reward, episode_length, win)
            
            # Train the model on experience replay
            if self.experience_replay.size() >= self.config.batch_size:
                loss = self._train_model()
                self.training_history['loss'].append(loss)
            
            # Update epsilon
            self.epsilon_scheduler.step()
            current_epsilon = self.epsilon_scheduler.get_epsilon()
            self.training_history['epsilon'].append(current_epsilon)
            
            # Calculate win rate
            win_rate = self._calculate_win_rate()
            self.training_history['win_rates'].append(win_rate)
            
            # Update best model
            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self._save_best_model(epoch)
            
            # Log progress
            epoch_time = time.time() - epoch_start_time
            if verbose and epoch % 10 == 0:
                self._log_progress(epoch, epochs, episode_reward, episode_length, 
                                 win_rate, current_epsilon, epoch_time)
            
            # Early stopping
            if self._should_stop_early(epoch):
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Training completed
        training_time = time.time() - self.training_start_time
        results = self._generate_training_results(training_time)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Final win rate: {results['final_win_rate']:.2%}")
        self.logger.info(f"Training time: {training_time:.1f} seconds")
        
        return results
    
    def _run_episode(self) -> Tuple[float, int, bool]:
        """
        Run one training episode.
        
        Returns:
            Tuple of (total_reward, episode_length, win)
        """
        # Reset maze to random starting position
        self.maze.reset()
        state = self.maze.observe()
        
        total_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < 200:  # Prevent infinite episodes
            # Choose action using epsilon-greedy policy
            if np.random.random() < self.epsilon_scheduler.get_epsilon():
                # Explore: choose random valid action
                valid_actions = self.maze.valid_actions()
                if not valid_actions:
                    break
                action = np.random.choice(valid_actions)
            else:
                # Exploit: choose best action according to model
                q_values = self.model.predict(state)
                action = np.argmax(q_values[0])
            
            # Take action
            next_state, reward, game_status = self.maze.act(action)
            done = (game_status != 'not_over')
            
            # Store experience
            self.experience_replay.add(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            episode_length += 1
        
        win = (game_status == 'win')
        return total_reward, episode_length, win
    
    def _train_model(self) -> float:
        """
        Train the model on a batch of experiences.
        
        Returns:
            Average training loss
        """
        # Sample batch from experience replay
        states, actions, rewards, next_states, dones = self.experience_replay.sample(
            self.config.batch_size
        )
        
        # Calculate target Q-values
        target_q_values = self.model.predict(states)
        next_q_values = self.model.predict_target(next_states)
        
        # Update target Q-values using Bellman equation
        for i in range(len(states)):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
        
        # Train the model
        history = self.model.train(states, target_q_values, 
                                 batch_size=self.config.batch_size, epochs=1)
        
        return np.mean(history['loss'])
    
    def _update_history(self, episode_reward: float, episode_length: int, win: bool):
        """Update training history with episode results."""
        self.training_history['episode_rewards'].append(episode_reward)
        self.training_history['episode_lengths'].append(episode_length)
        
        # Calculate exploration rate
        if self.maze.state is not None:
            stats = self.maze.get_stats()
            self.training_history['exploration_rates'].append(stats['exploration_rate'])
    
    def _calculate_win_rate(self, window_size: int = 100) -> float:
        """Calculate win rate over recent episodes."""
        if len(self.training_history['episode_rewards']) < window_size:
            window_size = len(self.training_history['episode_rewards'])
        
        if window_size == 0:
            return 0.0
        
        # Count wins in recent episodes (episodes with positive reward are likely wins)
        recent_rewards = self.training_history['episode_rewards'][-window_size:]
        wins = sum(1 for reward in recent_rewards if reward > 0)
        
        return wins / window_size
    
    def _should_stop_early(self, epoch: int) -> bool:
        """Check if training should stop early."""
        if epoch < self.config.early_stopping_patience:
            return False
        
        # Check if win rate has been above target for enough epochs
        recent_win_rates = self.training_history['win_rates'][-self.config.early_stopping_patience:]
        return all(rate >= self.config.target_win_rate for rate in recent_win_rates)
    
    def _save_best_model(self, epoch: int):
        """Save the best model so far."""
        if self.best_model_path is None:
            self.best_model_path = f"{self.config.model_save_path}best_model_epoch_{epoch}.h5"
        
        self.model.save(self.best_model_path)
        self.logger.info(f"New best model saved (win rate: {self.best_win_rate:.2%})")
    
    def _log_progress(self, epoch: int, total_epochs: int, episode_reward: float, 
                     episode_length: int, win_rate: float, epsilon: float, epoch_time: float):
        """Log training progress."""
        template = (f"Epoch {epoch:3d}/{total_epochs} | "
                   f"Reward: {episode_reward:6.2f} | "
                   f"Length: {episode_length:3d} | "
                   f"Win Rate: {win_rate:.3f} | "
                   f"Epsilon: {epsilon:.3f} | "
                   f"Time: {epoch_time:.1f}s")
        self.logger.info(template)
    
    def _generate_training_results(self, training_time: float) -> Dict[str, Any]:
        """Generate comprehensive training results."""
        final_win_rate = self.training_history['win_rates'][-1] if self.training_history['win_rates'] else 0.0
        avg_reward = np.mean(self.training_history['episode_rewards']) if self.training_history['episode_rewards'] else 0.0
        final_epsilon = self.training_history['epsilon'][-1] if self.training_history['epsilon'] else 0.0
        
        return {
            'total_episodes': len(self.training_history['episode_rewards']),
            'final_win_rate': final_win_rate,
            'best_win_rate': self.best_win_rate,
            'avg_reward': avg_reward,
            'final_epsilon': final_epsilon,
            'training_time': training_time,
            'model_architecture': f"{self.config.hidden_layer_size} hidden, {self.config.num_hidden_layers} layers",
            'exploration_rate': np.mean(self.training_history['exploration_rates']) if self.training_history['exploration_rates'] else 0.0,
            'efficiency': final_win_rate / (training_time / 60) if training_time > 0 else 0.0,
            'training_history': self.training_history
        }
    
    def evaluate(self, num_episodes: int = 100, visualize: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            num_episodes: Number of episodes to run for evaluation
            visualize: Whether to create visualization plots
            
        Returns:
            Evaluation results
        """
        if self.model is None:
            raise RuntimeError("No model to evaluate. Train first.")
        
        self.logger.info(f"Evaluating model over {num_episodes} episodes...")
        
        # Set epsilon to 0 for evaluation (no exploration)
        original_epsilon = self.epsilon_scheduler.get_epsilon()
        self.epsilon_scheduler.current_step = self.epsilon_scheduler.decay_steps
        
        wins = 0
        total_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            self.maze.reset()
            state = self.maze.observe()
            
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done and episode_length < 200:
                # Always choose best action (no exploration)
                q_values = self.model.predict(state)
                action = np.argmax(q_values[0])
                
                state, reward, game_status = self.maze.act(action)
                done = (game_status != 'not_over')
                
                episode_reward += reward
                episode_length += 1
            
            if game_status == 'win':
                wins += 1
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Restore original epsilon
        self.epsilon_scheduler.current_step = int(original_epsilon * self.epsilon_scheduler.decay_steps)
        
        # Calculate evaluation metrics
        win_rate = wins / num_episodes
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        
        results = {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_episode_length': avg_length,
            'total_wins': wins,
            'total_episodes': num_episodes
        }
        
        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"  Win Rate: {win_rate:.2%}")
        self.logger.info(f"  Avg Reward: {avg_reward:.2f}")
        self.logger.info(f"  Avg Episode Length: {avg_length:.1f}")
        
        # Create visualizations
        if visualize:
            self._create_evaluation_plots(results)
        
        return results
    
    def _create_evaluation_plots(self, results: Dict[str, Any]):
        """Create plots for evaluation results."""
        # Plot training progress
        self.visualizer.plot_training_progress(self.training_history)
        
        # Plot final maze state
        self.visualizer.plot_maze(self.maze, "Final Training State", show_path=True)
        
        # Create summary report
        training_results = self._generate_training_results(0)  # Time not needed for report
        training_results.update(results)
        self.visualizer.create_summary_report(training_results)
    
    def save_training_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save training results to a JSON file."""
        if filename is None:
            filename = f"{self.config.logs_save_path}training_results_{int(time.time())}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict) and 'training_history' in key:
                # Handle training history specially
                serializable_results[key] = {}
                for hist_key, hist_value in value.items():
                    if isinstance(hist_value, list) and hist_value and isinstance(hist_value[0], np.ndarray):
                        serializable_results[key][hist_key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in hist_value]
                    else:
                        serializable_results[key][hist_key] = hist_value
            else:
                serializable_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Training results saved to {filename}")


def main():
    """Main function to run training with default settings."""
    print("🏴‍☠️  Welcome to the Enhanced Treasure Hunt Game! 🏴‍☠️")
    print("=" * 60)
    
    # Load configuration
    config = get_config()
    print(f"Configuration loaded: {config.epochs} epochs, {config.learning_rate} learning rate")
    
    # Create training controller
    controller = TrainingController(config)
    
    # Set up training
    controller.setup_training()
    
    # Train the model
    print("\n🚀 Starting training...")
    results = controller.train()
    
    # Evaluate the model
    print("\n📊 Evaluating trained model...")
    eval_results = controller.evaluate(num_episodes=50)
    
    # Save results
    controller.save_training_results(results)
    
    print("\n✅ Training completed successfully!")
    print(f"Final win rate: {results['final_win_rate']:.2%}")
    print(f"Best win rate: {results['best_win_rate']:.2%}")
    print(f"Training time: {results['training_time']:.1f} seconds")
    
    return controller, results


if __name__ == "__main__":
    controller, results = main()
