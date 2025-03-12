from pfl.context import CentralContext
import torch

from typing import Optional
from pfl.aggregate.base import Aggregator
from pfl.stats import TrainingStatistics

class MaxMagAggregator(Aggregator):

    def accumulate(self, accumulated: Optional[TrainingStatistics], user_stats: TrainingStatistics) -> TrainingStatistics:
        if accumulated is None:
            return user_stats

        for name in accumulated.keys():
            accumulated_weight = accumulated[name]
            user_weight = user_stats[name]
            accumulated[name] = torch.where(
                accumulated_weight > user_weight,
                accumulated_weight,
                user_weight
            )   
        accumulated.weight = 1
        return accumulated


    def worker_reduce(self, 
        aggregated_worker_stats: TrainingStatistics,
        central_context: CentralContext,
        aggregated_worker_metrics):

        raise NotImplementedError