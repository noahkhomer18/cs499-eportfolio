"""
Enhanced maze environment for the Treasure Hunt Game.
This is the "Model" in our MVC architecture - it handles all the game logic.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import random


class GameStatus(Enum):
    """Enum for different game states - makes the code more readable."""
    NOT_OVER = "not_over"
    WIN = "win"
    LOSE = "lose"


class Action(Enum):
    """Enum for movement actions - prevents magic numbers."""
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


class TreasureMaze:
    """
    Enhanced maze environment with better error handling and features.
    
    This class represents the game world where the pirate agent moves around
    trying to find the treasure. It handles all the game logic including
    movement validation, reward calculation, and state management.
    """
    
    def __init__(self, maze: np.ndarray, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the maze environment.
        
        Args:
            maze: 2D numpy array where 1 = wall, 0 = free space
            config: Optional configuration dictionary for rewards and settings
        """
        self.maze = np.array(maze, dtype=float)
        self.nrows, self.ncols = self.maze.shape
        
        # Validate maze - make sure it's properly formatted
        self._validate_maze()
        
        # Game state
        self.state = None  # (row, col, total_reward)
        self.visited = set()
        self.total_reward = 0.0
        
        # Configuration with sensible defaults
        self.config = config or {}
        self.treasure_reward = self.config.get('treasure_reward', 1.0)
        self.step_penalty = self.config.get('step_penalty', -0.04)
        self.wall_penalty = self.config.get('wall_penalty', -0.1)
        self.win_bonus = self.config.get('win_bonus', 10.0)
        
        # Find all free cells (where the pirate can start)
        self.free_cells = self._find_free_cells()
        
        # Treasure location (bottom-right corner)
        self.treasure_pos = (self.nrows - 1, self.ncols - 1)
        
        # Validate that treasure is reachable
        if not self._is_treasure_reachable():
            raise ValueError("Treasure is not reachable from any starting position!")
    
    def _validate_maze(self):
        """Validate that the maze is properly formatted."""
        if len(self.maze.shape) != 2:
            raise ValueError("Maze must be a 2D array")
        
        if self.maze.size == 0:
            raise ValueError("Maze cannot be empty")
        
        # Check that maze only contains 0s and 1s
        unique_values = np.unique(self.maze)
        if not all(val in [0, 1] for val in unique_values):
            raise ValueError("Maze can only contain 0 (free) and 1 (wall) values")
        
        # Check that treasure position is free
        if self.maze[-1, -1] != 0:
            raise ValueError("Treasure position (bottom-right) must be free (0)")
    
    def _find_free_cells(self) -> List[Tuple[int, int]]:
        """Find all cells where the pirate can start (free cells)."""
        free_cells = []
        for row in range(self.nrows):
            for col in range(self.ncols):
                if self.maze[row, col] == 0:  # Free cell
                    free_cells.append((row, col))
        return free_cells
    
    def _is_treasure_reachable(self) -> bool:
        """
        Check if the treasure is reachable using a simple flood-fill algorithm.
        This prevents training on impossible mazes.
        """
        if not self.free_cells:
            return False
        
        # Start from any free cell and see if we can reach the treasure
        start_cell = self.free_cells[0]
        visited = set()
        stack = [start_cell]
        
        while stack:
            row, col = stack.pop()
            if (row, col) in visited:
                continue
            
            visited.add((row, col))
            
            # Check if we reached the treasure
            if (row, col) == self.treasure_pos:
                return True
            
            # Add neighboring free cells to the stack
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < self.nrows and 
                    0 <= new_col < self.ncols and 
                    self.maze[new_row, new_col] == 0 and
                    (new_row, new_col) not in visited):
                    stack.append((new_row, new_col))
        
        return False
    
    def reset(self, start_cell: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Reset the maze to initial state.
        
        Args:
            start_cell: Optional starting position. If None, chooses randomly.
            
        Returns:
            Initial state observation
        """
        if start_cell is None:
            start_cell = random.choice(self.free_cells)
        
        # Validate starting position
        if start_cell not in self.free_cells:
            raise ValueError(f"Invalid start cell {start_cell}. Must be a free cell.")
        
        # Reset game state
        self.state = (start_cell[0], start_cell[1], 0.0)
        self.visited = {start_cell}
        self.total_reward = 0.0
        
        return self.observe()
    
    def observe(self) -> np.ndarray:
        """
        Get current state observation as a flattened array.
        
        Returns:
            Flattened maze state that the neural network can understand
        """
        if self.state is None:
            raise RuntimeError("Maze not initialized. Call reset() first.")
        
        # Create a copy of the maze for observation
        observation = np.copy(self.maze)
        
        # Mark visited cells
        for row, col in self.visited:
            observation[row, col] = 0.5
        
        # Mark current position
        row, col, _ = self.state
        observation[row, col] = 0.3
        
        # Mark treasure position
        observation[self.treasure_pos] = 0.9
        
        return observation.flatten()
    
    def valid_actions(self, position: Optional[Tuple[int, int]] = None) -> List[int]:
        """
        Get list of valid actions from current or specified position.
        
        Args:
            position: Optional position to check. If None, uses current position.
            
        Returns:
            List of valid action indices
        """
        if position is None:
            if self.state is None:
                return []
            row, col, _ = self.state
        else:
            row, col = position
        
        valid_actions = []
        
        # Check each direction
        for action in Action:
            new_row, new_col = self._get_new_position(row, col, action)
            if self._is_valid_position(new_row, new_col):
                valid_actions.append(action.value)
        
        return valid_actions
    
    def _get_new_position(self, row: int, col: int, action: Action) -> Tuple[int, int]:
        """Get new position after taking an action."""
        if action == Action.LEFT:
            return row, col - 1
        elif action == Action.UP:
            return row - 1, col
        elif action == Action.RIGHT:
            return row, col + 1
        elif action == Action.DOWN:
            return row + 1, col
        else:
            raise ValueError(f"Invalid action: {action}")
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is valid (within bounds and not a wall)."""
        return (0 <= row < self.nrows and 
                0 <= col < self.ncols and 
                self.maze[row, col] == 0)
    
    def act(self, action: int) -> Tuple[np.ndarray, float, str]:
        """
        Take an action and return new state, reward, and game status.
        
        Args:
            action: Action to take (0=left, 1=up, 2=right, 3=down)
            
        Returns:
            Tuple of (new_observation, reward, game_status)
        """
        if self.state is None:
            raise RuntimeError("Maze not initialized. Call reset() first.")
        
        row, col, _ = self.state
        
        # Validate action
        if action not in [a.value for a in Action]:
            raise ValueError(f"Invalid action: {action}")
        
        action_enum = Action(action)
        new_row, new_col = self._get_new_position(row, col, action_enum)
        
        # Calculate reward and update state
        reward = self._calculate_reward(row, col, new_row, new_col, action_enum)
        game_status = self._get_game_status(new_row, new_col)
        
        # Update state if move is valid
        if self._is_valid_position(new_row, new_col):
            self.state = (new_row, new_col, self.total_reward + reward)
            self.visited.add((new_row, new_col))
        else:
            # Hit a wall - stay in same position but get penalty
            self.state = (row, col, self.total_reward + reward)
        
        self.total_reward += reward
        
        return self.observe(), reward, game_status.value
    
    def _calculate_reward(self, old_row: int, old_col: int, 
                         new_row: int, new_col: int, action: Action) -> float:
        """
        Calculate reward for taking an action.
        This is where you can experiment with different reward strategies.
        """
        # Check if we hit a wall
        if not self._is_valid_position(new_row, new_col):
            return self.wall_penalty
        
        # Check if we reached the treasure
        if (new_row, new_col) == self.treasure_pos:
            return self.treasure_reward + self.win_bonus
        
        # Small penalty for each step (encourages finding shortest path)
        return self.step_penalty
    
    def _get_game_status(self, row: int, col: int) -> GameStatus:
        """Determine game status based on current position."""
        if (row, col) == self.treasure_pos:
            return GameStatus.WIN
        else:
            return GameStatus.NOT_OVER
    
    @property
    def size(self) -> int:
        """Get total number of cells in the maze."""
        return self.maze.size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current game statistics."""
        if self.state is None:
            return {}
        
        row, col, total_reward = self.state
        return {
            'position': (row, col),
            'total_reward': total_reward,
            'visited_cells': len(self.visited),
            'total_cells': len(self.free_cells),
            'exploration_rate': len(self.visited) / len(self.free_cells),
            'distance_to_treasure': abs(row - self.treasure_pos[0]) + abs(col - self.treasure_pos[1])
        }
    
    def copy(self) -> 'TreasureMaze':
        """Create a copy of the maze for parallel training."""
        new_maze = TreasureMaze(self.maze, self.config)
        if self.state is not None:
            new_maze.state = self.state
            new_maze.visited = self.visited.copy()
            new_maze.total_reward = self.total_reward
        return new_maze
