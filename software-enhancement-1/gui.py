"""
Graphical User Interface for the Treasure Hunt Game.
This provides a visual interface for training, monitoring, and playing the game.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import numpy as np
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import queue

from config import GameConfig, get_config
from train import TrainingController
from maze import TreasureMaze
from visualizer import MazeVisualizer


class TreasureHuntGUI:
    """
    Main GUI application for the Treasure Hunt Game.
    
    This provides a user-friendly interface to:
    - Configure training parameters
    - Monitor training progress in real-time
    - Visualize the maze and AI behavior
    - Save and load models
    - Run evaluations
    """
    
    def __init__(self):
        """Initialize the GUI application."""
        self.root = tk.Tk()
        self.root.title("🏴‍☠️ Enhanced Treasure Hunt Game")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2C3E50')
        
        # Application state
        self.config = get_config()
        self.controller: Optional[TrainingController] = None
        self.training_thread: Optional[threading.Thread] = None
        self.is_training = False
        self.training_queue = queue.Queue()
        
        # Create GUI components
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
        
        # Start GUI update loop
        self._update_gui()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        
        # Left panel - Configuration and Controls
        self.left_panel = ttk.LabelFrame(self.main_frame, text="Configuration & Controls", padding=10)
        
        # Configuration section
        self._create_config_widgets()
        
        # Control buttons
        self._create_control_widgets()
        
        # Progress section
        self._create_progress_widgets()
        
        # Right panel - Visualization
        self.right_panel = ttk.LabelFrame(self.main_frame, text="Visualization", padding=10)
        
        # Maze visualization
        self._create_maze_widgets()
        
        # Training plots
        self._create_plot_widgets()
    
    def _create_config_widgets(self):
        """Create configuration widgets."""
        # Epochs
        ttk.Label(self.left_panel, text="Epochs:").grid(row=0, column=0, sticky='w', pady=2)
        self.epochs_var = tk.StringVar(value=str(self.config.epochs))
        self.epochs_entry = ttk.Entry(self.left_panel, textvariable=self.epochs_var, width=10)
        self.epochs_entry.grid(row=0, column=1, sticky='w', padx=(5, 0), pady=2)
        
        # Learning rate
        ttk.Label(self.left_panel, text="Learning Rate:").grid(row=1, column=0, sticky='w', pady=2)
        self.lr_var = tk.StringVar(value=str(self.config.learning_rate))
        self.lr_entry = ttk.Entry(self.left_panel, textvariable=self.lr_var, width=10)
        self.lr_entry.grid(row=1, column=1, sticky='w', padx=(5, 0), pady=2)
        
        # Initial epsilon
        ttk.Label(self.left_panel, text="Initial Epsilon:").grid(row=2, column=0, sticky='w', pady=2)
        self.epsilon_var = tk.StringVar(value=str(self.config.initial_epsilon))
        self.epsilon_entry = ttk.Entry(self.left_panel, textvariable=self.epsilon_var, width=10)
        self.epsilon_entry.grid(row=2, column=1, sticky='w', padx=(5, 0), pady=2)
        
        # Hidden layer size
        ttk.Label(self.left_panel, text="Hidden Size:").grid(row=3, column=0, sticky='w', pady=2)
        self.hidden_var = tk.StringVar(value=str(self.config.hidden_layer_size))
        self.hidden_entry = ttk.Entry(self.left_panel, textvariable=self.hidden_var, width=10)
        self.hidden_entry.grid(row=3, column=1, sticky='w', padx=(5, 0), pady=2)
        
        # Target win rate
        ttk.Label(self.left_panel, text="Target Win Rate:").grid(row=4, column=0, sticky='w', pady=2)
        self.target_var = tk.StringVar(value=str(self.config.target_win_rate))
        self.target_entry = ttk.Entry(self.left_panel, textvariable=self.target_var, width=10)
        self.target_entry.grid(row=4, column=1, sticky='w', padx=(5, 0), pady=2)
    
    def _create_control_widgets(self):
        """Create control buttons."""
        self.control_frame = ttk.Frame(self.left_panel)
        self.control_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky='ew')
        
        # Train button
        self.train_btn = ttk.Button(self.control_frame, text="🚀 Start Training", 
                                   command=self._start_training, style='Accent.TButton')
        self.train_btn.pack(side='left', padx=(0, 5))
        
        # Stop button
        self.stop_btn = ttk.Button(self.control_frame, text="⏹️ Stop", 
                                  command=self._stop_training, state='disabled')
        self.stop_btn.pack(side='left', padx=(0, 5))
        
        # Evaluate button
        self.eval_btn = ttk.Button(self.control_frame, text="📊 Evaluate", 
                                  command=self._evaluate_model, state='disabled')
        self.eval_btn.pack(side='left', padx=(0, 5))
        
        # Save button
        self.save_btn = ttk.Button(self.control_frame, text="💾 Save Model", 
                                  command=self._save_model, state='disabled')
        self.save_btn.pack(side='left', padx=(0, 5))
        
        # Load button
        self.load_btn = ttk.Button(self.control_frame, text="📁 Load Model", 
                                  command=self._load_model)
        self.load_btn.pack(side='left')
    
    def _create_progress_widgets(self):
        """Create progress monitoring widgets."""
        # Progress bar
        ttk.Label(self.left_panel, text="Training Progress:").grid(row=6, column=0, sticky='w', pady=(10, 2))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.left_panel, variable=self.progress_var, 
                                           maximum=100, length=200)
        self.progress_bar.grid(row=7, column=0, columnspan=2, sticky='ew', pady=2)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to train")
        self.status_label = ttk.Label(self.left_panel, textvariable=self.status_var, 
                                     foreground='#27AE60')
        self.status_label.grid(row=8, column=0, columnspan=2, sticky='w', pady=2)
        
        # Stats display
        self.stats_frame = ttk.LabelFrame(self.left_panel, text="Training Statistics", padding=5)
        self.stats_frame.grid(row=9, column=0, columnspan=2, sticky='ew', pady=10)
        
        self.epoch_label = ttk.Label(self.stats_frame, text="Epoch: 0/0")
        self.epoch_label.pack(anchor='w')
        
        self.win_rate_label = ttk.Label(self.stats_frame, text="Win Rate: 0.0%")
        self.win_rate_label.pack(anchor='w')
        
        self.epsilon_label = ttk.Label(self.stats_frame, text="Epsilon: 0.0")
        self.epsilon_label.pack(anchor='w')
        
        self.reward_label = ttk.Label(self.stats_frame, text="Avg Reward: 0.0")
        self.reward_label.pack(anchor='w')
    
    def _create_maze_widgets(self):
        """Create maze visualization widgets."""
        # Maze display
        self.maze_frame = ttk.Frame(self.right_panel)
        self.maze_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Create matplotlib figure for maze
        self.maze_fig = Figure(figsize=(6, 6), dpi=100)
        self.maze_ax = self.maze_fig.add_subplot(111)
        self.maze_canvas = FigureCanvasTkAgg(self.maze_fig, self.maze_frame)
        self.maze_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Maze controls
        self.maze_control_frame = ttk.Frame(self.right_panel)
        self.maze_control_frame.pack(fill='x', pady=5)
        
        ttk.Button(self.maze_control_frame, text="🔄 Reset Maze", 
                  command=self._reset_maze).pack(side='left', padx=(0, 5))
        
        ttk.Button(self.maze_control_frame, text="🎯 Show Path", 
                  command=self._show_optimal_path).pack(side='left', padx=(0, 5))
        
        ttk.Button(self.maze_control_frame, text="🎮 Play Game", 
                  command=self._play_game).pack(side='left')
    
    def _create_plot_widgets(self):
        """Create training progress plot widgets."""
        # Create matplotlib figure for plots
        self.plot_fig = Figure(figsize=(8, 4), dpi=100)
        self.plot_ax = self.plot_fig.add_subplot(111)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, self.right_panel)
        self.plot_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initialize empty plot
        self.plot_ax.set_title("Training Progress")
        self.plot_ax.set_xlabel("Epoch")
        self.plot_ax.set_ylabel("Win Rate")
        self.plot_ax.grid(True, alpha=0.3)
        self.plot_canvas.draw()
    
    def _setup_layout(self):
        """Set up the main layout."""
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=2)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Place panels
        self.left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        self.right_panel.grid(row=0, column=1, sticky='nsew')
        
        # Configure left panel grid
        self.left_panel.columnconfigure(1, weight=1)
    
    def _bind_events(self):
        """Bind GUI events."""
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Configuration validation
        self.epochs_var.trace('w', self._validate_config)
        self.lr_var.trace('w', self._validate_config)
        self.epsilon_var.trace('w', self._validate_config)
        self.hidden_var.trace('w', self._validate_config)
        self.target_var.trace('w', self._validate_config)
    
    def _validate_config(self, *args):
        """Validate configuration inputs."""
        try:
            epochs = int(self.epochs_var.get())
            lr = float(self.lr_var.get())
            epsilon = float(self.epsilon_var.get())
            hidden = int(self.hidden_var.get())
            target = float(self.target_var.get())
            
            if epochs > 0 and 0 < lr <= 1 and 0 <= epsilon <= 1 and hidden > 0 and 0 <= target <= 1:
                self.train_btn.config(state='normal')
            else:
                self.train_btn.config(state='disabled')
        except ValueError:
            self.train_btn.config(state='disabled')
    
    def _start_training(self):
        """Start training in a separate thread."""
        if self.is_training:
            return
        
        # Update configuration
        self._update_config_from_gui()
        
        # Disable controls
        self.is_training = True
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_var.set("Setting up training...")
        
        # Start training thread
        self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.training_thread.start()
    
    def _training_worker(self):
        """Worker function for training thread."""
        try:
            # Create controller
            self.controller = TrainingController(self.config)
            self.controller.setup_training()
            
            # Update status
            self.training_queue.put(('status', 'Training started...'))
            
            # Train model
            results = self.controller.train(verbose=False)
            
            # Training completed
            self.training_queue.put(('complete', results))
            
        except Exception as e:
            self.training_queue.put(('error', str(e)))
    
    def _stop_training(self):
        """Stop training."""
        self.is_training = False
        self.status_var.set("Stopping training...")
        # Note: In a real implementation, you'd need to implement proper stopping
    
    def _evaluate_model(self):
        """Evaluate the trained model."""
        if self.controller is None:
            messagebox.showerror("Error", "No trained model available!")
            return
        
        self.status_var.set("Evaluating model...")
        
        def eval_worker():
            try:
                results = self.controller.evaluate(num_episodes=50, visualize=False)
                self.training_queue.put(('eval_complete', results))
            except Exception as e:
                self.training_queue.put(('error', str(e)))
        
        threading.Thread(target=eval_worker, daemon=True).start()
    
    def _save_model(self):
        """Save the trained model."""
        if self.controller is None:
            messagebox.showerror("Error", "No trained model available!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.controller.model.save(filename)
                messagebox.showinfo("Success", f"Model saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {e}")
    
    def _load_model(self):
        """Load a trained model."""
        filename = filedialog.askopenfilename(
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Create controller and load model
                self.controller = TrainingController(self.config)
                self.controller.setup_training()
                self.controller.model.load(filename)
                
                # Enable evaluation button
                self.eval_btn.config(state='normal')
                self.save_btn.config(state='normal')
                
                messagebox.showinfo("Success", f"Model loaded from {filename}")
                self.status_var.set("Model loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def _reset_maze(self):
        """Reset the maze visualization."""
        if self.controller and self.controller.maze:
            self.controller.maze.reset()
            self._update_maze_display()
    
    def _show_optimal_path(self):
        """Show the optimal path in the maze."""
        if self.controller and self.controller.maze:
            # This would show the optimal path using A* algorithm
            self._update_maze_display(show_path=True)
    
    def _play_game(self):
        """Play a game with the current model."""
        if self.controller is None:
            messagebox.showerror("Error", "No trained model available!")
            return
        
        # This would start an interactive game session
        messagebox.showinfo("Info", "Interactive game mode coming soon!")
    
    def _update_config_from_gui(self):
        """Update configuration from GUI values."""
        self.config.epochs = int(self.epochs_var.get())
        self.config.learning_rate = float(self.lr_var.get())
        self.config.initial_epsilon = float(self.epsilon_var.get())
        self.config.hidden_layer_size = int(self.hidden_var.get())
        self.config.target_win_rate = float(self.target_var.get())
    
    def _update_maze_display(self, show_path: bool = False):
        """Update the maze visualization."""
        if not self.controller or not self.controller.maze:
            return
        
        self.maze_ax.clear()
        
        # Create maze visualization
        canvas = np.copy(self.controller.maze.maze)
        
        # Mark visited cells
        for row, col in self.controller.maze.visited:
            canvas[row, col] = 0.5
        
        # Mark current position
        if self.controller.maze.state is not None:
            row, col, _ = self.controller.maze.state
            canvas[row, col] = 0.3
        
        # Mark treasure position
        canvas[self.controller.maze.treasure_pos] = 0.9
        
        # Display maze
        im = self.maze_ax.imshow(canvas, cmap='viridis', vmin=0, vmax=1)
        self.maze_ax.set_title("Treasure Hunt Maze")
        
        # Add grid
        self.maze_ax.set_xticks(np.arange(-0.5, self.controller.maze.ncols, 1), minor=True)
        self.maze_ax.set_yticks(np.arange(-0.5, self.controller.maze.nrows, 1), minor=True)
        self.maze_ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        
        self.maze_canvas.draw()
    
    def _update_plot_display(self, win_rates: list):
        """Update the training progress plot."""
        self.plot_ax.clear()
        
        if win_rates:
            self.plot_ax.plot(win_rates, color='#3498DB', linewidth=2)
            self.plot_ax.set_title("Training Progress - Win Rate")
            self.plot_ax.set_xlabel("Epoch")
            self.plot_ax.set_ylabel("Win Rate")
            self.plot_ax.set_ylim(0, 1)
            self.plot_ax.grid(True, alpha=0.3)
            self.plot_ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7)
        
        self.plot_canvas.draw()
    
    def _update_gui(self):
        """Update GUI elements from training queue."""
        try:
            while True:
                message_type, data = self.training_queue.get_nowait()
                
                if message_type == 'status':
                    self.status_var.set(data)
                    
                elif message_type == 'progress':
                    epoch, total_epochs, win_rate, epsilon, reward = data
                    self.progress_var.set((epoch / total_epochs) * 100)
                    self.epoch_label.config(text=f"Epoch: {epoch}/{total_epochs}")
                    self.win_rate_label.config(text=f"Win Rate: {win_rate:.1%}")
                    self.epsilon_label.config(text=f"Epsilon: {epsilon:.3f}")
                    self.reward_label.config(text=f"Avg Reward: {reward:.2f}")
                    
                    # Update plot
                    if hasattr(self.controller, 'training_history'):
                        self._update_plot_display(self.controller.training_history['win_rates'])
                    
                    # Update maze display
                    self._update_maze_display()
                    
                elif message_type == 'complete':
                    # Training completed
                    self.is_training = False
                    self.train_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    self.eval_btn.config(state='normal')
                    self.save_btn.config(state='normal')
                    self.status_var.set("Training completed!")
                    
                    # Show results
                    results = data
                    messagebox.showinfo("Training Complete", 
                                      f"Final Win Rate: {results['final_win_rate']:.2%}\n"
                                      f"Best Win Rate: {results['best_win_rate']:.2%}\n"
                                      f"Training Time: {results['training_time']:.1f}s")
                    
                elif message_type == 'eval_complete':
                    results = data
                    messagebox.showinfo("Evaluation Complete", 
                                      f"Win Rate: {results['win_rate']:.2%}\n"
                                      f"Avg Reward: {results['avg_reward']:.2f}\n"
                                      f"Avg Episode Length: {results['avg_episode_length']:.1f}")
                    
                elif message_type == 'error':
                    self.is_training = False
                    self.train_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    self.status_var.set("Error occurred")
                    messagebox.showerror("Error", data)
                    
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self._update_gui)
    
    def _on_closing(self):
        """Handle window closing."""
        if self.is_training:
            if messagebox.askokcancel("Quit", "Training is in progress. Do you want to quit?"):
                self.is_training = False
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main function to run the GUI."""
    app = TreasureHuntGUI()
    app.run()


if __name__ == "__main__":
    main()
