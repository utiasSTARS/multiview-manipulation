import socket
import os

TREVOR_MACHINES = ['trevor-GS60-6QE', 'monolith', 'Obelisk', 'megalith', 'graham', 'trevor-home-desktop']


def default_arg_parser():
    machine = socket.gethostname()

    if machine.startswith('gra'): machine = 'graham'

    if machine == 'trevor-GS60-6QE':
        main_data_dir = '/media/trevor/Data/corrective-interactive-data/dac/testing-data'
        bc_models_dir = '/media/trevor/Data/corrective-interactive-data/dac/testing-data/bc_models'
        expert_data_dir = main_data_dir + '/../expert_data'
    elif machine == 'monolith':
        main_data_dir = '/media/raid5-array/experiments/bc-view-agnostic'
        bc_models_dir = main_data_dir + '/bc_models'
        expert_data_dir = main_data_dir + '/demonstrations'
    elif machine == 'Obelisk':
        main_data_dir = '/media/HDD1/experiments/fire-interaction/testing_data'
        bc_models_dir = '/media/HDD1/experiments/fire-interaction/bc_models'
        expert_data_dir = main_data_dir + '/../expert_data'
    elif machine == 'megalith':
        main_data_dir = '/home/tablett/projects/dac-interaction/testing-data'
        bc_models_dir = '/home/tablett/projects/dac-interaction/testing-data/bc_models'
        expert_data_dir = main_data_dir + '/../expert_data'
    elif machine == 'graham':
        main_data_dir = '/home/abletttr/scratch/experiments/bc-view-agnostic'
        bc_models_dir = main_data_dir + '/bc_models'
        compute_home = os.environ["SLURM_TMPDIR"]
        expert_data_dir = compute_home + '/data/bc-view-agnostic'
    elif machine == 'trevor-home-desktop':
        main_data_dir = '/home/trevor/data/paper-data/bc-view-agnostic'
        bc_models_dir = main_data_dir + '/bc_models'
        expert_data_dir = main_data_dir + '/demonstrations'

    return None, main_data_dir, bc_models_dir, expert_data_dir
