"""
Main entry point for the Enhanced Treasure Hunt Game.
This provides a user-friendly interface to run training, evaluation, and testing.
"""

import argparse
import sys
import os
import time
from typing import Optional

from config import GameConfig, get_config, save_config, load_config
from train import TrainingController
from test_treasure_hunt import run_tests


def print_banner():
    """Print a nice banner for the application."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🏴‍☠️  ENHANCED TREASURE HUNT GAME  🏴‍☠️              ║
    ║                                                              ║
    ║              Deep Q-Learning AI Training System              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config_summary(config: GameConfig):
    """Print a summary of the current configuration."""
    print("📋 Current Configuration:")
    print(f"   • Maze Size: {config.maze_size[0]}x{config.maze_size[1]}")
    print(f"   • Training Epochs: {config.epochs}")
    print(f"   • Learning Rate: {config.learning_rate}")
    print(f"   • Initial Epsilon: {config.initial_epsilon}")
    print(f"   • Final Epsilon: {config.final_epsilon}")
    print(f"   • Hidden Layer Size: {config.hidden_layer_size}")
    print(f"   • Target Win Rate: {config.target_win_rate}")
    print(f"   • Model Save Path: {config.model_save_path}")
    print(f"   • Plots Save Path: {config.plots_save_path}")
    print()


def train_model(config: GameConfig, epochs: Optional[int] = None, 
                config_file: Optional[str] = None) -> tuple:
    """
    Train the treasure hunt AI model.
    
    Args:
        config: Game configuration
        epochs: Optional number of epochs to train
        config_file: Optional config file to save
        
    Returns:
        Tuple of (controller, results)
    """
    print("🚀 Starting AI Training...")
    print("=" * 60)
    
    # Override epochs if specified
    if epochs is not None:
        config.epochs = epochs
        print(f"Training for {epochs} epochs (overriding config)")
    
    # Print configuration
    print_config_summary(config)
    
    # Save configuration if requested
    if config_file:
        save_config(config, config_file)
        print(f"Configuration saved to {config_file}")
    
    # Create training controller
    controller = TrainingController(config)
    
    # Set up training
    print("🔧 Setting up training components...")
    controller.setup_training()
    
    # Train the model
    print("🧠 Training the AI model...")
    start_time = time.time()
    results = controller.train()
    training_time = time.time() - start_time
    
    # Print results
    print("\n📊 Training Results:")
    print(f"   • Final Win Rate: {results['final_win_rate']:.2%}")
    print(f"   • Best Win Rate: {results['best_win_rate']:.2%}")
    print(f"   • Total Episodes: {results['total_episodes']}")
    print(f"   • Training Time: {training_time:.1f} seconds")
    print(f"   • Final Epsilon: {results['final_epsilon']:.3f}")
    print(f"   • Average Reward: {results['avg_reward']:.2f}")
    
    # Evaluate the model
    print("\n🔍 Evaluating trained model...")
    eval_results = controller.evaluate(num_episodes=50, visualize=True)
    
    print("\n📈 Evaluation Results:")
    print(f"   • Win Rate: {eval_results['win_rate']:.2%}")
    print(f"   • Average Reward: {eval_results['avg_reward']:.2f}")
    print(f"   • Average Episode Length: {eval_results['avg_episode_length']:.1f}")
    
    # Save results
    controller.save_training_results(results)
    
    return controller, results


def run_tests_suite():
    """Run the comprehensive test suite."""
    print("🧪 Running Test Suite...")
    print("=" * 60)
    
    result = run_tests()
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return True
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return False


def interactive_mode():
    """Run the application in interactive mode."""
    print("🎮 Interactive Mode")
    print("=" * 60)
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Train a new model")
        print("2. Load and continue training")
        print("3. Evaluate an existing model")
        print("4. Run tests")
        print("5. View configuration")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            # Train new model
            config = get_config()
            
            # Ask for custom parameters
            print("\nCustomize training parameters (press Enter for defaults):")
            epochs = input(f"Epochs (default: {config.epochs}): ").strip()
            if epochs:
                config.epochs = int(epochs)
            
            learning_rate = input(f"Learning rate (default: {config.learning_rate}): ").strip()
            if learning_rate:
                config.learning_rate = float(learning_rate)
            
            epsilon = input(f"Initial epsilon (default: {config.initial_epsilon}): ").strip()
            if epsilon:
                config.initial_epsilon = float(epsilon)
            
            train_model(config)
            
        elif choice == '2':
            # Load and continue training
            model_path = input("Enter path to existing model: ").strip()
            if os.path.exists(model_path):
                print("Loading existing model...")
                # TODO: Implement model loading and continued training
                print("Feature coming soon!")
            else:
                print("Model file not found!")
                
        elif choice == '3':
            # Evaluate existing model
            model_path = input("Enter path to model: ").strip()
            if os.path.exists(model_path):
                print("Evaluating model...")
                # TODO: Implement model evaluation
                print("Feature coming soon!")
            else:
                print("Model file not found!")
                
        elif choice == '4':
            # Run tests
            run_tests_suite()
            
        elif choice == '5':
            # View configuration
            config = get_config()
            print_config_summary(config)
            
        elif choice == '6':
            # Exit
            print("👋 Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Enhanced Treasure Hunt Game - Deep Q-Learning AI Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                    # Train with default settings
  python main.py --train --epochs 500      # Train for 500 epochs
  python main.py --test                     # Run test suite
  python main.py --interactive              # Interactive mode
  python main.py --config config.json      # Load custom config
        """
    )
    
    # Main actions
    parser.add_argument('--train', action='store_true', 
                       help='Train a new AI model')
    parser.add_argument('--test', action='store_true', 
                       help='Run the test suite')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode')
    
    # Training options
    parser.add_argument('--epochs', type=int, 
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, 
                       help='Learning rate for the neural network')
    parser.add_argument('--epsilon', type=float, 
                       help='Initial exploration rate (epsilon)')
    
    # Configuration options
    parser.add_argument('--config', type=str, 
                       help='Path to configuration file')
    parser.add_argument('--save-config', type=str, 
                       help='Save current configuration to file')
    
    # Output options
    parser.add_argument('--verbose', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        print(f"📁 Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Override config with command line arguments
    if args.epochs:
        config.epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.epsilon:
        config.initial_epsilon = args.epsilon
    if args.no_plots:
        config.save_plots = False
    
    # Execute requested action
    if args.train:
        train_model(config, args.epochs, args.save_config)
        
    elif args.test:
        success = run_tests_suite()
        sys.exit(0 if success else 1)
        
    elif args.interactive:
        interactive_mode()
        
    else:
        # Default: show help and run interactive mode
        if not any([args.train, args.test, args.interactive]):
            print("No action specified. Running in interactive mode...")
            interactive_mode()
        else:
            parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Training interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        if '--verbose' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)
