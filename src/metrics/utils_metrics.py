import utils_main

from metrics.parent_metric import MetricTracker
from metrics.dice_score import DiceScore
from metrics.intersect_over_union import IntersectOverUnion



def create_metric_trackers():
    
    # Create the metrics
    train_metrics = [DiceScore(utils_main.N_CLASSES), IntersectOverUnion(utils_main.N_CLASSES)]
    val_metrics = [DiceScore(utils_main.N_CLASSES), IntersectOverUnion(utils_main.N_CLASSES)]
    test_metrics = [DiceScore(utils_main.N_CLASSES), IntersectOverUnion(utils_main.N_CLASSES)]
    
    # Create the metric trackers
    train_metric_tracker = MetricTracker(train_metrics)
    val_metric_tracker = MetricTracker(val_metrics)
    test_metric_tracker = MetricTracker(test_metrics)
    
    return train_metric_tracker, val_metric_tracker, test_metric_tracker