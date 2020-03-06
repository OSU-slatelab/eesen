#                                                                                                                                                                                                                                             # LRScheduler Base Class
#                                                                                                                                                                                                                                             
# Author: Eric Fosler-Lussier                                                                                                                                                                                                                 

# Provides base definition for class, common auxiliary functions
 
class LRScheduler:

    def __init__(self, config):
        pass

    def initialize_training(self):
        pass

    def update_lr_rate(self, cv_ters):
        pass

    def set_epoch(self, epoch):
        pass

    def resume_from_log(self):
        pass

    def compute_avg_ters(self, stats):
        total_ters, nters = 0.0, 0.0
        for language_id, target_scheme in stats.items():
            for target_id, stat in target_scheme.items():

                # Skip the number stat
                if target_id == 'n':
                    continue

                #if stat['ter'] > 0:
                total_ters += stat['ter']
                nters += 1

        return total_ters / nters

