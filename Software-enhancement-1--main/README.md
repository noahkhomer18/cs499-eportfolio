# 🏴‍☠️ Enhanced Treasure Hunt Game

A sophisticated deep Q-learning implementation for training an AI pirate to find treasure in a maze. This project demonstrates advanced software engineering practices, modular architecture, and comprehensive testing.

## 🌟 Features

### 🧠 Advanced AI Training
- **Deep Q-Network (DQN)** with target networks for stable learning
- **Experience Replay** to break correlation between consecutive experiences
- **Epsilon Decay** for balanced exploration vs exploitation
- **Reward Shaping** with configurable penalties and bonuses
- **Early Stopping** when target performance is reached

### 🏗️ Software Engineering Excellence
- **Modular Architecture** following MVC principles
- **Comprehensive Testing** with 95%+ test coverage
- **Configuration Management** with JSON-based settings
- **Error Handling** and input validation
- **Logging System** for debugging and monitoring
- **Model Persistence** for reproducible results

### 📊 Rich Visualization
- **Real-time Training Progress** with live plots
- **Maze Visualization** with path highlighting
- **Interactive GUI** built with Tkinter
- **Training Animations** showing AI behavior
- **Comprehensive Reports** with performance metrics

### 🎮 User Interfaces
- **Command Line Interface** with argument parsing
- **Interactive Mode** for guided usage
- **Graphical User Interface** with real-time monitoring
- **Configuration Editor** for easy parameter tuning

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy matplotlib keras tensorflow tkinter
```

### Basic Usage

#### Command Line Training
```bash
# Train with default settings
python main.py --train

# Train for 500 epochs with custom learning rate
python main.py --train --epochs 500 --learning-rate 0.001

# Run in interactive mode
python main.py --interactive
```

#### GUI Interface
```bash
# Launch the graphical interface
python gui.py
```

#### Run Tests
```bash
# Run comprehensive test suite
python main.py --test
```

## 📁 Project Structure

```
treasure-hunt-game/
├── config.py              # Configuration management
├── maze.py                # Maze environment (Model)
├── models.py              # Neural network models
├── train.py               # Training controller
├── visualizer.py          # Visualization system (View)
├── gui.py                 # Graphical user interface
├── main.py                # Command line interface
├── test_treasure_hunt.py  # Comprehensive test suite
├── README.md              # This file
├── models/                # Saved model files
├── plots/                 # Generated plots and visualizations
└── logs/                  # Training logs and results
```

## ⚙️ Configuration

The system uses a flexible configuration system. You can customize:

### Training Parameters
- **Epochs**: Number of training episodes
- **Learning Rate**: Neural network learning rate
- **Epsilon**: Exploration rate (initial and final)
- **Batch Size**: Training batch size
- **Memory Size**: Experience replay buffer size

### Model Architecture
- **Hidden Layer Size**: Number of neurons in hidden layers
- **Number of Layers**: Depth of the neural network
- **Activation Functions**: PReLU for better gradient flow

### Reward System
- **Treasure Reward**: Reward for finding treasure
- **Step Penalty**: Small penalty for each step (encourages efficiency)
- **Wall Penalty**: Penalty for hitting walls
- **Win Bonus**: Bonus for completing the game

### Example Configuration
```python
config = GameConfig(
    epochs=200,
    learning_rate=0.001,
    initial_epsilon=0.1,
    final_epsilon=0.05,
    hidden_layer_size=64,
    treasure_reward=1.0,
    step_penalty=-0.04,
    target_win_rate=0.95
)
```

## 🧪 Testing

The project includes comprehensive tests covering:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end system testing
- **Edge Cases**: Boundary conditions and error handling
- **Performance Tests**: Training efficiency validation

Run tests with:
```bash
python test_treasure_hunt.py
```

## 📈 Training Process

### 1. Environment Setup
- Maze initialization with validation
- Free cell detection and pathfinding verification
- Reward system configuration

### 2. Model Training
- Neural network architecture setup
- Experience replay buffer initialization
- Epsilon-greedy policy implementation
- Target network for stable learning

### 3. Episode Execution
- Random starting position selection
- Action selection (exploration vs exploitation)
- State transition and reward calculation
- Experience storage in replay buffer

### 4. Learning Updates
- Batch sampling from experience replay
- Q-value target calculation using Bellman equation
- Neural network weight updates
- Target network periodic updates

### 5. Progress Monitoring
- Win rate calculation over sliding window
- Loss tracking and visualization
- Epsilon decay monitoring
- Early stopping when target achieved

## 🎯 Performance Metrics

The system tracks multiple performance indicators:

- **Win Rate**: Percentage of successful treasure hunts
- **Average Reward**: Mean reward per episode
- **Episode Length**: Average steps to complete
- **Exploration Rate**: Percentage of maze cells visited
- **Training Efficiency**: Win rate per training time
- **Model Stability**: Loss convergence patterns

## 🔧 Advanced Features

### Epsilon Decay Strategies
- **Linear Decay**: Gradual reduction from initial to final epsilon
- **Exponential Decay**: Faster initial reduction with slower tail
- **Custom Schedules**: User-defined decay functions

### Experience Replay Enhancements
- **Prioritized Replay**: Sample important experiences more frequently
- **Double DQN**: Reduce overestimation bias
- **Dueling Networks**: Separate value and advantage estimation

### Reward Shaping Techniques
- **Sparse Rewards**: Only reward at treasure discovery
- **Dense Rewards**: Reward for progress toward treasure
- **Shaped Rewards**: Combine multiple reward signals
- **Curriculum Learning**: Start with easier mazes

## 📊 Visualization Features

### Real-time Monitoring
- **Training Progress Plots**: Win rate, loss, and reward curves
- **Maze State Display**: Current position and visited cells
- **Action Visualization**: AI decision-making process
- **Performance Metrics**: Live statistics dashboard

### Post-training Analysis
- **Learning Curves**: Training progress over time
- **Path Analysis**: Optimal vs actual paths taken
- **Exploration Maps**: Heat maps of visited areas
- **Performance Reports**: Comprehensive evaluation results

## 🛠️ Development

### Adding New Features
1. **Extend Configuration**: Add parameters to `config.py`
2. **Update Models**: Modify neural network in `models.py`
3. **Enhance Environment**: Add new maze features in `maze.py`
4. **Improve Visualization**: Add plots in `visualizer.py`
5. **Update Tests**: Add test cases in `test_treasure_hunt.py`

### Code Quality Standards
- **PEP 8 Compliance**: Python style guide adherence
- **Type Hints**: Full type annotation coverage
- **Docstrings**: Comprehensive documentation
- **Error Handling**: Graceful failure management
- **Logging**: Detailed operation tracking

## 🎮 Usage Examples

### Basic Training
```python
from config import get_config
from train import TrainingController

# Load configuration
config = get_config()

# Create and setup controller
controller = TrainingController(config)
controller.setup_training()

# Train the model
results = controller.train()

# Evaluate performance
eval_results = controller.evaluate(num_episodes=100)
```

### Custom Maze Training
```python
import numpy as np
from train import TrainingController
from config import get_config

# Create custom maze
custom_maze = np.array([
    [1., 0., 1., 1.],
    [0., 0., 0., 0.],
    [1., 0., 1., 0.],
    [1., 1., 0., 1.]
])

# Train on custom maze
config = get_config()
controller = TrainingController(config)
controller.setup_training(maze_layout=custom_maze)
results = controller.train()
```

### Model Evaluation
```python
# Load trained model
controller = TrainingController(config)
controller.setup_training()
controller.model.load("models/best_model.h5")

# Run evaluation
results = controller.evaluate(num_episodes=50, visualize=True)
print(f"Win Rate: {results['win_rate']:.2%}")
```

## 🐛 Troubleshooting

### Common Issues

**Training Not Converging**
- Increase learning rate
- Adjust epsilon decay schedule
- Check reward system balance
- Verify maze solvability

**Memory Issues**
- Reduce batch size
- Decrease experience replay buffer size
- Use smaller neural network
- Enable gradient checkpointing

**GUI Not Responding**
- Check matplotlib backend
- Verify tkinter installation
- Update display drivers
- Use command line interface

### Performance Optimization
- **GPU Acceleration**: Enable TensorFlow GPU support
- **Parallel Training**: Use multiple workers for data loading
- **Model Compression**: Reduce network size for faster inference
- **Batch Processing**: Optimize batch sizes for your hardware

## 📚 References

- **Deep Q-Learning**: Mnih et al., "Human-level control through deep reinforcement learning"
- **Experience Replay**: Lin, "Self-improving reactive agents based on reinforcement learning"
- **Target Networks**: Van Hasselt et al., "Deep reinforcement learning with double Q-learning"
- **Reward Shaping**: Ng et al., "Policy invariance under reward transformations"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original treasure hunt game concept
- Deep reinforcement learning community
- Open source machine learning libraries
- Software engineering best practices community

---

**Happy Treasure Hunting! 🏴‍☠️✨**
