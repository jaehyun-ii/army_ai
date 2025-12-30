# advcar/core/cli.py
import argparse
import yaml
from .config import AttackConfig
from .trainer import Trainer

def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Car Attack Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Primary Argument ---
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True, 
        help="Path to the attack YAML config file."
    )
    
    # --- Common Override Arguments ---
    # We set default=None to easily check if the user provided this argument.
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None, 
        help="Override the number of training epochs."
    )
    parser.add_argument(
        "--lr", "--learning-rate", 
        dest='learning_rate', 
        type=float, 
        default=None, 
        help="Override the learning rate."
    )
    parser.add_argument(
        "--bs", "--batch-size", 
        dest='batch_size',
        type=int, 
        default=None, 
        help="Override the batch size."
    )
    parser.add_argument(
        "-o", "--output-path", 
        type=str, 
        default=None, 
        help="Override the output path for the final artifact."
    )

    args = parser.parse_args()

    # 1. Load configuration from YAML file
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # 2. Merge/Override with command-line arguments
    # This is the key logic. We only override if the argument was actually provided.
    if args.output_path is not None:
        config_dict['output_path'] = args.output_path
        
    if args.epochs is not None:
        config_dict['training']['epochs'] = args.epochs
        
    if args.learning_rate is not None:
        config_dict['training']['learning_rate'] = args.learning_rate
        
    if args.batch_size is not None:
        config_dict['training']['batch_size'] = args.batch_size

    # 3. Validate the final configuration with Pydantic
    try:
        config = AttackConfig(**config_dict)
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return

    # 4. Launch the trainer with the final, validated config
    print("--- Final Configuration ---")
    print(config.model_dump_json(indent=2))
    print("-------------------------")
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()