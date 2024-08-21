
import argparse

from logger.wandb_logger import WandBLogger
from models.utils_models import ModelType



def initialize_wandb(args : argparse.Namespace) -> WandBLogger:
    
    if ModelType.RESNET.value in args.model_type:
        logger = WandBLogger(run_name = args.model_type, group_name=ModelType.RESNET.value) if WandBLogger else None
    
    elif ModelType.SEGFORMER.value in args.model_type:
        logger = WandBLogger(run_name = args.model_type, group_name=ModelType.SEGFORMER.value) if WandBLogger else None
        
    elif ModelType.UNET.value in args.model_type:
        logger = WandBLogger(run_name = args.model_type, group_name=ModelType.UNET.value) if WandBLogger else None
    
    else:
        raise NotImplementedError(f"Model type {args.model_type} is not implemented, so logger cannot be created.")
    
    logger.log({
        f"learning_rate": args.learning_rate,
        f"weight_decay": args.weight_decay,
        f"gamma": args.gamma
    })
    
    return logger