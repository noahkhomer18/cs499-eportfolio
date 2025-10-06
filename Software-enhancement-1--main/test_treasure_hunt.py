"""
Comprehensive test suite for the Treasure Hunt Game.
This ensures all components work correctly and handles edge cases properly.
"""

import unittest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

from config import GameConfig, DEFAULT_MAZE, get_config
from maze import TreasureMaze, Action, GameStatus
from models import QNetwork, ExperienceReplay, EpsilonScheduler
from train import TrainingController


class TestTreasureMaze(unittest.TestCase):
    """Test cases for the TreasureMaze class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simple_maze = np.array([
            [1., 0., 1.],
            [0., 0., 0.],
            [1., 0., 1.]
        ])
        self.maze = TreasureMaze(self.simple_maze)
    
    def test_maze_initialization(self):
        """Test that maze initializes correctly."""
        self.assertEqual(self.maze.nrows, 3)
        self.assertEqual(self.maze.ncols, 3)
        self.assertEqual(self.maze.treasure_pos, (2, 2))
        self.assertIn((0, 1), self.maze.free_cells)
        self.assertIn((1, 0), self.maze.free_cells)
        self.assertIn((1, 1), self.maze.free_cells)
        self.assertIn((1, 2), self.maze.free_cells)
        self.assertIn((2, 1), self.maze.free_cells)
    
    def test_invalid_maze_raises_error(self):
        """Test that invalid mazes raise appropriate errors."""
        # Maze with treasure position blocked
        invalid_maze = np.array([
            [1., 0., 1.],
            [0., 0., 0.],
            [1., 0., 1.]  # Treasure position is blocked
        ])
        invalid_maze[2, 2] = 1  # Block treasure
        
        with self.assertRaises(ValueError):
            TreasureMaze(invalid_maze)
    
    def test_reset_functionality(self):
        """Test maze reset functionality."""
        # Reset to specific position
        state = self.maze.reset((1, 1))
        self.assertEqual(self.maze.state, (1, 1, 0.0))
        self.assertIn((1, 1), self.maze.visited)
        
        # Reset to random position
        state = self.maze.reset()
        self.assertIsNotNone(self.maze.state)
        self.assertIn(self.maze.state[:2], self.maze.free_cells)
    
    def test_observe_functionality(self):
        """Test state observation."""
        self.maze.reset((1, 1))
        observation = self.maze.observe()
        
        self.assertEqual(observation.shape, (9,))  # 3x3 maze flattened
        self.assertEqual(observation[4], 0.3)  # Current position
        self.assertEqual(observation[8], 0.9)  # Treasure position
    
    def test_valid_actions(self):
        """Test valid action detection."""
        self.maze.reset((1, 1))  # Center position
        valid_actions = self.maze.valid_actions()
        
        # Should have 4 valid actions from center
        self.assertEqual(len(valid_actions), 4)
        self.assertIn(Action.LEFT.value, valid_actions)
        self.assertIn(Action.UP.value, valid_actions)
        self.assertIn(Action.RIGHT.value, valid_actions)
        self.assertIn(Action.DOWN.value, valid_actions)
    
    def test_action_execution(self):
        """Test action execution and reward calculation."""
        self.maze.reset((1, 1))
        
        # Move right
        next_state, reward, game_status = self.maze.act(Action.RIGHT.value)
        self.assertEqual(self.maze.state[0], 1)  # Row stays same
        self.assertEqual(self.maze.state[1], 2)  # Column increases
        self.assertEqual(game_status, GameStatus.NOT_OVER.value)
        self.assertLess(reward, 0)  # Step penalty
    
    def test_win_condition(self):
        """Test win condition detection."""
        self.maze.reset((2, 1))  # Next to treasure
        next_state, reward, game_status = self.maze.act(Action.RIGHT.value)
        
        self.assertEqual(game_status, GameStatus.WIN.value)
        self.assertGreater(reward, 0)  # Positive reward for winning
    
    def test_wall_collision(self):
        """Test wall collision handling."""
        self.maze.reset((0, 1))  # Next to wall
        initial_state = self.maze.state
        
        # Try to move into wall
        next_state, reward, game_status = self.maze.act(Action.UP.value)
        
        # Should stay in same position
        self.assertEqual(self.maze.state, initial_state)
        self.assertLess(reward, 0)  # Wall penalty
    
    def test_get_stats(self):
        """Test statistics generation."""
        self.maze.reset((1, 1))
        stats = self.maze.get_stats()
        
        self.assertIn('position', stats)
        self.assertIn('total_reward', stats)
        self.assertIn('visited_cells', stats)
        self.assertIn('exploration_rate', stats)
        self.assertEqual(stats['position'], (1, 1))


class TestQNetwork(unittest.TestCase):
    """Test cases for the QNetwork class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 9  # 3x3 maze
        self.num_actions = 4
        self.model = QNetwork(self.input_size, self.num_actions)
    
    def test_model_creation(self):
        """Test that model is created correctly."""
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.target_model)
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.num_actions, self.num_actions)
    
    def test_prediction(self):
        """Test model prediction."""
        state = np.random.random(self.input_size)
        q_values = self.model.predict(state)
        
        self.assertEqual(q_values.shape, (1, self.num_actions))
        self.assertTrue(np.all(np.isfinite(q_values)))
    
    def test_training(self):
        """Test model training."""
        states = np.random.random((10, self.input_size))
        targets = np.random.random((10, self.num_actions))
        
        history = self.model.train(states, targets, batch_size=5, epochs=1)
        
        self.assertIn('loss', history)
        self.assertIsInstance(history['loss'], list)
        self.assertGreater(len(history['loss']), 0)
    
    def test_target_network_update(self):
        """Test target network weight update."""
        # Get initial weights
        initial_weights = self.model.model.get_weights()
        target_weights = self.model.target_model.get_weights()
        
        # Train the main model
        states = np.random.random((5, self.input_size))
        targets = np.random.random((5, self.num_actions))
        self.model.train(states, targets, batch_size=5, epochs=1)
        
        # Update target network
        self.model.update_target_network()
        
        # Check that target network has same weights as main network
        new_target_weights = self.model.target_model.get_weights()
        for w1, w2 in zip(self.model.model.get_weights(), new_target_weights):
            np.testing.assert_array_equal(w1, w2)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.h5')
            
            # Save model
            self.model.save(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Create new model and load
            new_model = QNetwork(self.input_size, self.num_actions)
            new_model.load(model_path)
            
            # Test that loaded model works
            state = np.random.random(self.input_size)
            original_pred = self.model.predict(state)
            loaded_pred = new_model.predict(state)
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)


class TestExperienceReplay(unittest.TestCase):
    """Test cases for the ExperienceReplay class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ExperienceReplay(max_size=100)
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(self.buffer.max_size, 100)
        self.assertEqual(self.buffer.size(), 0)
        self.assertEqual(len(self.buffer.buffer), 0)
    
    def test_add_experience(self):
        """Test adding experiences to buffer."""
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([4, 5, 6])
        done = False
        
        self.buffer.add(state, action, reward, next_state, done)
        
        self.assertEqual(self.buffer.size(), 1)
        self.assertEqual(len(self.buffer.buffer), 1)
    
    def test_buffer_overflow(self):
        """Test buffer behavior when it overflows."""
        # Fill buffer beyond max_size
        for i in range(150):
            state = np.array([i])
            self.buffer.add(state, 0, 0.0, state, False)
        
        self.assertEqual(self.buffer.size(), 100)  # Should be capped at max_size
    
    def test_sample_experience(self):
        """Test sampling experiences from buffer."""
        # Add some experiences
        for i in range(10):
            state = np.array([i])
            self.buffer.add(state, i % 4, float(i), state, i % 2 == 0)
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(5)
        
        self.assertEqual(len(states), 5)
        self.assertEqual(len(actions), 5)
        self.assertEqual(len(rewards), 5)
        self.assertEqual(len(next_states), 5)
        self.assertEqual(len(dones), 5)
    
    def test_sample_empty_buffer(self):
        """Test sampling from empty buffer."""
        states, actions, rewards, next_states, dones = self.buffer.sample(5)
        
        self.assertEqual(len(states), 0)
        self.assertEqual(len(actions), 0)
        self.assertEqual(len(rewards), 0)
        self.assertEqual(len(next_states), 0)
        self.assertEqual(len(dones), 0)


class TestEpsilonScheduler(unittest.TestCase):
    """Test cases for the EpsilonScheduler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = EpsilonScheduler(
            initial_epsilon=1.0,
            final_epsilon=0.1,
            decay_steps=100,
            decay_type='linear'
        )
    
    def test_initial_epsilon(self):
        """Test initial epsilon value."""
        self.assertEqual(self.scheduler.get_epsilon(), 1.0)
    
    def test_epsilon_decay(self):
        """Test epsilon decay over time."""
        initial_epsilon = self.scheduler.get_epsilon()
        
        # Step through decay
        for _ in range(50):
            self.scheduler.step()
        
        current_epsilon = self.scheduler.get_epsilon()
        self.assertLess(current_epsilon, initial_epsilon)
        self.assertGreater(current_epsilon, 0.1)
    
    def test_final_epsilon(self):
        """Test that epsilon reaches final value."""
        # Step through all decay steps
        for _ in range(self.scheduler.decay_steps + 10):
            self.scheduler.step()
        
        self.assertEqual(self.scheduler.get_epsilon(), 0.1)
    
    def test_reset_functionality(self):
        """Test scheduler reset."""
        # Step through some decay
        for _ in range(50):
            self.scheduler.step()
        
        # Reset
        self.scheduler.reset()
        
        # Should be back to initial epsilon
        self.assertEqual(self.scheduler.get_epsilon(), 1.0)


class TestTrainingController(unittest.TestCase):
    """Test cases for the TrainingController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GameConfig()
        self.config.epochs = 5  # Small number for testing
        self.config.max_memory = 100
        self.controller = TrainingController(self.config)
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        self.assertIsNotNone(self.controller.config)
        self.assertIsNotNone(self.controller.training_history)
        self.assertEqual(self.controller.best_win_rate, 0.0)
    
    def test_setup_training(self):
        """Test training setup."""
        self.controller.setup_training()
        
        self.assertIsNotNone(self.controller.maze)
        self.assertIsNotNone(self.controller.model)
        self.assertIsNotNone(self.controller.experience_replay)
        self.assertIsNotNone(self.controller.epsilon_scheduler)
        self.assertIsNotNone(self.controller.visualizer)
    
    @patch('train.TrainingController._log_progress')
    def test_training_loop(self, mock_log):
        """Test training loop execution."""
        self.controller.setup_training()
        
        # Run short training
        results = self.controller.train(epochs=3, verbose=False)
        
        # Check that training completed
        self.assertIn('total_episodes', results)
        self.assertIn('final_win_rate', results)
        self.assertIn('training_time', results)
        
        # Check that history was updated
        self.assertGreater(len(self.controller.training_history['episode_rewards']), 0)
        self.assertGreater(len(self.controller.training_history['win_rates']), 0)
    
    def test_evaluation(self):
        """Test model evaluation."""
        self.controller.setup_training()
        
        # Train briefly
        self.controller.train(epochs=2, verbose=False)
        
        # Evaluate
        results = self.controller.evaluate(num_episodes=5, visualize=False)
        
        self.assertIn('win_rate', results)
        self.assertIn('avg_reward', results)
        self.assertIn('total_episodes', results)
        self.assertEqual(results['total_episodes'], 5)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_full_training_pipeline(self):
        """Test the complete training pipeline."""
        config = GameConfig()
        config.epochs = 3
        config.max_memory = 50
        
        controller = TrainingController(config)
        controller.setup_training()
        
        # Train
        results = controller.train(verbose=False)
        
        # Evaluate
        eval_results = controller.evaluate(num_episodes=3, visualize=False)
        
        # Check that everything worked
        self.assertIsInstance(results['final_win_rate'], float)
        self.assertIsInstance(eval_results['win_rate'], float)
        self.assertGreaterEqual(results['final_win_rate'], 0.0)
        self.assertLessEqual(results['final_win_rate'], 1.0)
    
    def test_config_save_load(self):
        """Test configuration saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'test_config.json')
            
            # Create and save config
            config = GameConfig()
            config.epochs = 50
            config.learning_rate = 0.01
            
            # Save config
            import json
            config_dict = {
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
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f)
            
            # Load config
            with open(config_path, 'r') as f:
                loaded_dict = json.load(f)
            
            self.assertEqual(loaded_dict['epochs'], 50)
            self.assertEqual(loaded_dict['learning_rate'], 0.01)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTreasureMaze,
        TestQNetwork,
        TestExperienceReplay,
        TestEpsilonScheduler,
        TestTrainingController,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == '__main__':
    print("🧪 Running Treasure Hunt Game Test Suite...")
    print("=" * 60)
    
    result = run_tests()
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    print(f"\nTotal tests run: {result.testsRun}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
