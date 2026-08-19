## A module dedicated to computing the true population statistics
## after the training is done following the original Batch norm paper

# See: https://discuss.pytorch.org/t/population-statistics-for-batchnorm-instead-of-running-average/20559

# Example of usage:

# Note: you might want to traverse the dataset a couple of times to get
# a better estimate of the population statistics
# Make sure your trainloader has shuffle=True and drop_last=True

# net.apply(adjust_bn_layers_to_compute_population_stats)
# for i in range(10):
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(trainloader):
#             _ = net(inputs.cuda())
# net.apply(restore_original_settings_of_bn_layers)


# Why this works --
# https://pytorch.org/docs/master/torch.nn.html#torch.nn.BatchNorm1d
# if you set momentum property in batchnorm layer, it will
# compute cumulutive average or just simple average of observed
# values

import torch

def adjust_bn_layers_to_compute_population_stats(module):

    #JL: Modification of original to BN1d and BN3d as well
    if isinstance(module, torch.nn.BatchNorm1d | torch.nn.BatchNorm2d | torch.nn.BatchNorm3d):

        # Removing the stats computed using exponential running average
        # and resetting count
        module.reset_running_stats()

        # Doing this so that we can restore it later
        module._old_momentum = module.momentum

        # Switching to cumulutive running average
        module.momentum = None

        # This is necessary -- because otherwise the
        # newly observed batches will not be considered
        module._old_training = module.training
        module._old_track_running_stats = module.track_running_stats

        module.training = True
        module.track_running_stats = True


def restore_original_settings_of_bn_layers(module):

    #JL: Modification of original to BN1d and BN3d as well
    if isinstance(module, torch.nn.BatchNorm1d | torch.nn.BatchNorm2d | torch.nn.BatchNorm3d):

        # Restoring old settings
        module.momentum = module._old_momentum
        module.training = module._old_training
        module.track_running_stats = module._old_track_running_stats

