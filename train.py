"""
Main training script for GIL-DDI model
Usage:
    python train.py --task 1 --epochs 120 --learning_rate 0.001 --batch_size 1024
"""
import os
import sys
import logging
from datetime import datetime

from config import get_config, print_config


def setup_logging(log_dir, task):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"task{task}_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Main training function"""
    # Get configuration
    args = get_config()

    # Setup logging
    logger = setup_logging(args.log_dir, args.task)
    logger.info("Starting GIL-DDI Training")
    print_config(args)

    # Import task-specific training script
    if args.task == 1:
        logger.info("Loading Task 1: Known Drug-Drug Interaction Prediction")
        from code.GILIP_task1 import train_task1
        train_task1(args, logger)
    elif args.task == 2:
        logger.info("Loading Task 2: Known-New Drug Interaction Prediction")
        from code.GILIP_task2 import train_task2
        train_task2(args, logger)
    elif args.task == 3:
        logger.info("Loading Task 3: New Drug-Drug Interaction Prediction")
        from code.GILIP_task3 import train_task3
        train_task3(args, logger)
    else:
        logger.error(f"Invalid task number: {args.task}")
        sys.exit(1)

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

