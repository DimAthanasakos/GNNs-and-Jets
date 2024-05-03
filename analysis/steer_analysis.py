#! /usr/bin/env python
'''
Main script to steer graph construction and GNN training
'''

import argparse
import os
import sys
import yaml
import time

import data_IO
import graph_constructor
import ml_analysis
import plot_results

# Base class
sys.path.append('.')
from base import common_base

####################################################################################################################
class SteerAnalysis(common_base.CommonBase):

    #---------------------------------------------------------------
    # Constructor
    #---------------------------------------------------------------
    def __init__(self, input_file='', config_file='', output_dir='', regenerate_graphs=False, use_precomputed_graphs=False, gpu_mode = 'single', **kwargs):

        self.config_file = config_file
        self.input_file = input_file
        self.output_dir = output_dir
        self.regenerate_graphs = regenerate_graphs
        self.use_precomputed_graphs = use_precomputed_graphs
        self.gpu_mode = gpu_mode
        self.rank = int(os.getenv("LOCAL_RANK", "0"))

        self.initialize(config_file)
        if self.rank == 0:
            print()
            print(self)


    #---------------------------------------------------------------
    # Initialize config
    #---------------------------------------------------------------
    def initialize(self, config_file):
        if self.rank == 0:
            print('Initializing class objects')

        with open(config_file, 'r') as stream:
            self.config = yaml.safe_load(stream)
        self.models = self.config['models']


    #---------------------------------------------------------------
    # Main function
    #---------------------------------------------------------------
    def run_analysis(self):
        '''
        For now, we will assume that MC samples have been used to
        generate a dataset of particles and subjets (subjets_unshuffled.h5).
        
        Existing datasets are listed here:
        https://docs.google.com/spreadsheets/d/1DI_GWwZO8sYDB9FS-rFzitoDk3SjfHfgoKVVGzG1j90/edit#gid=0
        
        The graph_constructor module constructs the input graphs to the ML analysis:
          - graphs_numpy_subjet.h5: builds graphs from JFN output subjets_unshuffled.h5
          - graphs_pyg_subjet__{graph_key}.pt: builds PyG graphs from subjet_graphs_numpy.h5
          - graphs_pyg_particle__{graph_key}.pt: builds PyG graphs from energyflow dataset
        '''

        # If you want to use subjets instead of hadrons, we also need an input file to read the subjets from.
        # If so, the input file must be one of the files in the google spreadsheet: https://docs.google.com/spreadsheets/d/1DI_GWwZO8sYDB9FS-rFzitoDk3SjfHfgoKVVGzG1j90/edit#gid=0
        

        for model in self.models:
            if model in ['subjet_gcn_pytorch', 'subjet_gat_pytorch'] and self.input_file: 
                # Check whether the graphs file has already been generated, and if not, generate it
                # TODO: This is a bit of a hack, but it works for now. We also need to add a check for the pyg graphs.
                graph_numpy_subjet_file = os.path.join(self.output_dir, 'graphs_numpy_subjet.h5') 
                print('========================================================================')
                if self.regenerate_graphs or not os.path.exists(graph_numpy_subjet_file):
                    input_data = data_IO.read_data(self.input_file)
                    graph_constructor.construct_graphs(self.config_file, self.output_dir, self.use_precomputed_graphs, \
                                                       sub_or_part='subjet', input_data = input_data)
                else:
                    print(f'Subjet numpy graphs found: {graph_numpy_subjet_file}')

            elif model in ['particle_gcn_pytorch', 'particle_gat_pytorch']:
                # Check whether the graphs file has already been generated, and if not, generate it
                for graph_structure in self.config[model]['graph_types']:
                    graph_key = f'particle__{graph_structure}'
                    graph_pyg_particle_file = os.path.join(self.output_dir, f"graphs_pyg_{graph_key}.pt")

                    print('========================================================================')
                    if self.regenerate_graphs or not os.path.exists(graph_pyg_particle_file):\
                        graph_constructor.construct_graphs(self.config_file, self.output_dir, self.use_precomputed_graphs, \
                                                           sub_or_part='particle', graph_structure = graph_structure)
                    else:
                        print(f'Particle pyg graphs found: {graph_pyg_particle_file}')

            else: # for particle_net and transformer the graph structure is dynamically generated at each layer which dominates the training time.
                continue
        # Perform ML analysis, and write results to file
        if self.rank == 0:
            print()
            print('========================================================================')
            print('Running ML analysis...')

        start_time = time.time()

        analysis = ml_analysis.MLAnalysis(self.config_file, self.output_dir, self.gpu_mode)
        analysis.train_models()
        # check if the code is running in parallel, and only plot results for one process (same results for all processes)

        # Plot results
        if self.rank == 0:
            print('========================================================================')
            print('Plotting results...')
            plot = plot_results.PlotResults(self.config_file, self.output_dir)
            plot.plot_results()
            print('--- {} minutes ---'.format((time.time() - start_time)/60.))

####################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Hadronization Analysis')
    parser.add_argument('-c', '--config_file', 
                        help='Path of config file for analysis',
                        action='store', type=str,
                        default='config.yaml', )
    parser.add_argument('-i' ,'--input_file', 
                        help='Path to subjets_unshuffled.h5 file with ndarrays for ML input',
                        action='store', type=str,
                        default='', )
    parser.add_argument('-o', '--output_dir',
                        help='Output directory for output to be written to',
                        action='store', type=str,
                        default='./TestOutput', )
    parser.add_argument('--regenerate_graphs', 
                        help='construct graphs from subjets_unshuffled.h5', 
                        action='store_true', default=False)
    parser.add_argument('--use_precomputed_graphs', 
                        help='use graphs from subjets_unshuffled.h5', 
                        action='store_true', default=False)
    
    # Argument for running on a single GPU or multiple GPUs
    parser.add_argument('-g', '--gpu_mode',
                    help='Specify "single" for single GPU or "multi" for multiple GPUs',
                    choices=['single', 'multi'],
                    default='single',
                    action='store')
    
    args = parser.parse_args()

    # If invalid config_file or input_file is given, exit
    if not os.path.exists(args.config_file):
        print(f'Config file {args.config_file} does not exist! Exiting!')
        sys.exit(0)

    # If output dir does not exist, create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


    analysis = SteerAnalysis(input_file= '' if not os.path.exists(args.input_file) else args.input_file,
                             config_file=args.config_file, 
                             output_dir=args.output_dir, 
                             regenerate_graphs=args.regenerate_graphs,
                             use_precomputed_graphs=args.use_precomputed_graphs,
                             gpu_mode=args.gpu_mode, 
                             )
    analysis.run_analysis()
    
    
