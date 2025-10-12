"""
Visualization module for the Treasure Hunt Game.
This is the "View" in our MVC architecture - it handles all the visual output.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
import os


class MazeVisualizer:
    """
    Handles visualization of the maze and training progress.
    
    This class creates beautiful, informative plots that help you understand
    what's happening during training and how well your AI is performing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration dictionary for visualization settings
        """
        self.config = config or {}
        self.save_plots = self.config.get('save_plots', True)
        self.plots_path = self.config.get('plots_save_path', 'plots/')
        
        # Create plots directory
        if self.save_plots:
            os.makedirs(self.plots_path, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        self.colors = {
            'wall': '#2C3E50',      # Dark blue-gray for walls
            'free': '#ECF0F1',      # Light gray for free space
            'visited': '#3498DB',   # Blue for visited cells
            'current': '#E74C3C',   # Red for current position
            'treasure': '#F39C12',  # Orange for treasure
            'path': '#27AE60'       # Green for optimal path
        }
    
    def plot_maze(self, maze_env, title: str = "Treasure Hunt Maze", 
                  show_path: bool = False, save: bool = True) -> plt.Figure:
        """
        Create a static plot of the maze.
        
        Args:
            maze_env: TreasureMaze instance
            title: Plot title
            show_path: Whether to highlight the optimal path
            save: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the maze visualization
        canvas = self._create_maze_canvas(maze_env, show_path)
        
        # Display the maze
        im = ax.imshow(canvas, cmap='viridis', vmin=0, vmax=1)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, maze_env.ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze_env.nrows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        
        # Customize the plot
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        # Add colorbar with custom labels
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9])
        cbar.set_ticklabels(['Wall', 'Current', 'Visited', 'Free', 'Treasure'])
        
        # Add statistics text
        if maze_env.state is not None:
            stats = maze_env.get_stats()
            stats_text = f"Position: {stats['position']}\n"
            stats_text += f"Reward: {stats['total_reward']:.2f}\n"
            stats_text += f"Visited: {stats['visited_cells']}/{stats['total_cells']}\n"
            stats_text += f"Distance to treasure: {stats['distance_to_treasure']}"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', 
                   facecolor='white', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        if save and self.save_plots:
            filename = f"{self.plots_path}maze_{int(time.time())}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Maze plot saved to {filename}")
        
        return fig
    
    def _create_maze_canvas(self, maze_env, show_path: bool = False) -> np.ndarray:
        """Create the visual representation of the maze."""
        canvas = np.copy(maze_env.maze)
        
        # Mark visited cells
        for row, col in maze_env.visited:
            canvas[row, col] = 0.5
        
        # Mark current position
        if maze_env.state is not None:
            row, col, _ = maze_env.state
            canvas[row, col] = 0.3
        
        # Mark treasure position
        canvas[maze_env.treasure_pos] = 0.9
        
        # Highlight optimal path if requested
        if show_path and maze_env.state is not None:
            path = self._find_optimal_path(maze_env)
            for row, col in path:
                if canvas[row, col] != 0.9:  # Don't overwrite treasure
                    canvas[row, col] = 0.7
        
        return canvas
    
    def _find_optimal_path(self, maze_env) -> List[Tuple[int, int]]:
        """Find the shortest path from current position to treasure using A*."""
        if maze_env.state is None:
            return []
        
        start = (maze_env.state[0], maze_env.state[1])
        goal = maze_env.treasure_pos
        
        # Simple A* pathfinding
        open_set = [start]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            open_set.remove(current)
            
            # Check neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if (0 <= neighbor[0] < maze_env.nrows and 
                    0 <= neighbor[1] < maze_env.ncols and 
                    maze_env.maze[neighbor] == 0):
                    
                    tentative_g = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                        
                        if neighbor not in open_set:
                            open_set.append(neighbor)
        
        return []
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for A*."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def plot_training_progress(self, history: Dict[str, List[float]], 
                              save: bool = True) -> plt.Figure:
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('Enhanced Training Progress Analytics', fontsize=16, fontweight='bold')
        
        # Plot loss
        if 'loss' in history and history['loss']:
            axes[0, 0].plot(history['loss'], color='#E74C3C', linewidth=2)
            axes[0, 0].set_title('Training Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot episode rewards
        if 'episode_rewards' in history and history['episode_rewards']:
            # Smooth the rewards for better visualization
            rewards = history['episode_rewards']
            window_size = min(50, len(rewards) // 10)
            if window_size > 1:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                axes[0, 1].plot(smoothed_rewards, color='#27AE60', linewidth=2, label='Smoothed')
                axes[0, 1].plot(rewards, color='#27AE60', alpha=0.3, label='Raw')
                axes[0, 1].legend()
            else:
                axes[0, 1].plot(rewards, color='#27AE60', linewidth=2)
            axes[0, 1].set_title('Episode Rewards', fontweight='bold')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot win rate
        if 'win_rates' in history and history['win_rates']:
            axes[1, 0].plot(history['win_rates'], color='#3498DB', linewidth=2)
            axes[1, 0].set_title('Win Rate', fontweight='bold')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
            axes[1, 0].legend()
        
        # Plot epsilon decay
        if 'epsilon' in history and history['epsilon']:
            axes[1, 1].plot(history['epsilon'], color='#F39C12', linewidth=2)
            axes[1, 1].set_title('Exploration Rate (Epsilon)', fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot episode lengths
        if 'episode_lengths' in history and history['episode_lengths']:
            lengths = history['episode_lengths']
            window_size = min(20, len(lengths) // 10)
            if window_size > 1:
                smoothed_lengths = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
                axes[2, 0].plot(smoothed_lengths, color='#8E44AD', linewidth=2, label='Smoothed')
                axes[2, 0].plot(lengths, color='#8E44AD', alpha=0.3, label='Raw')
                axes[2, 0].legend()
            else:
                axes[2, 0].plot(lengths, color='#8E44AD', linewidth=2)
            axes[2, 0].set_title('Episode Lengths', fontweight='bold')
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('Steps')
            axes[2, 0].grid(True, alpha=0.3)
        
        # Plot exploration rates
        if 'exploration_rates' in history and history['exploration_rates']:
            axes[2, 1].plot(history['exploration_rates'], color='#16A085', linewidth=2)
            axes[2, 1].set_title('Maze Exploration Rate', fontweight='bold')
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('Exploration %')
            axes[2, 1].set_ylim(0, 1)
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save and self.save_plots:
            filename = f"{self.plots_path}training_progress_{int(time.time())}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Training progress plot saved to {filename}")
        
        return fig
    
    def animate_training(self, maze_env, action_history: List[int], 
                        interval: int = 500, save: bool = True) -> animation.FuncAnimation:
        """
        Create an animation of the training process.
        
        Args:
            maze_env: TreasureMaze instance
            action_history: List of actions taken during training
            interval: Animation interval in milliseconds
            save: Whether to save the animation
            
        Returns:
            Matplotlib animation
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Initialize the maze
        canvas = self._create_maze_canvas(maze_env)
        im = ax.imshow(canvas, cmap='viridis', vmin=0, vmax=1)
        
        # Add grid
        ax.set_xticks(np.arange(-0.5, maze_env.ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze_env.nrows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        
        ax.set_title('Treasure Hunt Training Animation', fontsize=16, fontweight='bold')
        
        def animate(frame):
            if frame < len(action_history):
                # Take the action and update the maze
                maze_env.act(action_history[frame])
                canvas = self._create_maze_canvas(maze_env)
                im.set_array(canvas)
                
                # Update title with current stats
                stats = maze_env.get_stats()
                ax.set_title(f'Step {frame}: Reward={stats["total_reward"]:.2f}, '
                           f'Position={stats["position"]}', fontsize=14)
            
            return [im]
        
        anim = animation.FuncAnimation(fig, animate, frames=len(action_history),
                                     interval=interval, blit=True, repeat=True)
        
        if save and self.save_plots:
            filename = f"{self.plots_path}training_animation_{int(time.time())}.gif"
            anim.save(filename, writer='pillow', fps=2)
            print(f"Training animation saved to {filename}")
        
        return anim
    
    def create_summary_report(self, training_results: Dict[str, Any], 
                            save: bool = True) -> plt.Figure:
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('Enhanced Treasure Hunt Training Analytics Report', fontsize=20, fontweight='bold')
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Enhanced training statistics
        ax1 = fig.add_subplot(gs[0, :])
        stats_text = f"""
        Enhanced Training Analytics:
        • Total Episodes: {training_results.get('total_episodes', 'N/A')}
        • Final Win Rate: {training_results.get('final_win_rate', 'N/A'):.2%}
        • Best Win Rate: {training_results.get('best_win_rate', 'N/A'):.2%}
        • Training Time: {training_results.get('training_time', 'N/A'):.1f}s
        • Final Epsilon: {training_results.get('final_epsilon', 'N/A'):.3f}
        • Model Architecture: {training_results.get('model_architecture', 'N/A')}
        • Cache Hit Rate: {training_results.get('cache_hit_rate', 'N/A'):.1%}
        • Exploration Efficiency: {training_results.get('exploration_rate', 'N/A'):.1%}
        • Learning Stability: {training_results.get('learning_stability', 'N/A'):.2f}
        """
        ax1.text(0.1, 0.5, stats_text, transform=ax1.transAxes, fontsize=14,
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightblue', alpha=0.8))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Performance metrics
        ax2 = fig.add_subplot(gs[1, 0])
        metrics = ['Win Rate', 'Avg Reward', 'Exploration', 'Efficiency']
        values = [
            training_results.get('final_win_rate', 0),
            training_results.get('avg_reward', 0),
            training_results.get('exploration_rate', 0),
            training_results.get('efficiency', 0)
        ]
        bars = ax2.bar(metrics, values, color=['#3498DB', '#27AE60', '#F39C12', '#E74C3C'])
        ax2.set_title('Performance Metrics', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Learning curve
        ax3 = fig.add_subplot(gs[1, 1:])
        if 'win_rates' in training_results:
            ax3.plot(training_results['win_rates'], color='#3498DB', linewidth=2)
            ax3.set_title('Learning Curve (Win Rate)', fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Win Rate')
            ax3.set_ylim(0, 1)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.7)
        
        # Recommendations
        ax4 = fig.add_subplot(gs[2, :])
        recommendations = self._generate_recommendations(training_results)
        ax4.text(0.1, 0.5, recommendations, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='lightgreen', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        if save and self.save_plots:
            filename = f"{self.plots_path}summary_report_{int(time.time())}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Summary report saved to {filename}")
        
        return fig
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate recommendations based on training results."""
        recommendations = "Recommendations:\n"
        
        final_win_rate = results.get('final_win_rate', 0)
        if final_win_rate < 0.5:
            recommendations += "• Consider increasing training episodes or adjusting learning rate\n"
        elif final_win_rate < 0.8:
            recommendations += "• Good progress! Try fine-tuning hyperparameters for better performance\n"
        else:
            recommendations += "• Excellent performance! Consider testing on more complex mazes\n"
        
        if results.get('exploration_rate', 0) > 0.3:
            recommendations += "• High exploration rate - consider adjusting epsilon decay\n"
        
        if results.get('training_time', 0) > 300:  # 5 minutes
            recommendations += "• Long training time - consider reducing model complexity\n"
        
        return recommendations
