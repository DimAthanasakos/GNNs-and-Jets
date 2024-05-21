'''
Script to create subjets and nsubjettiness.
'''


import os
import time
import numpy as np
import math 
import sys
import glob
from collections import defaultdict

import functools
import socket 
import matplotlib.pyplot as plt

from dataloader import read_file

# Fastjet via python (from external library heppy)
import fastjet as fj
import fjcontrib
import fjext

import logging
from pathlib import Path
import h5py
import awkward as ak

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import uproot
import pandas as pd

# Energy flow package
import energyflow

# Base class
sys.path.append('.')
from base import common_base

################################################################
class Process(common_base.CommonBase):

    def __init__(self, output_dir='', n_total = 1000, N_cluster_list = [20], K = 4, classification_task = 'qvsg', **kwargs):
        super(common_base.CommonBase, self).__init__(**kwargs)

        self.start_time = time.time()

        self.output_dir = output_dir  # Path to store the output. Be careful with overwriting.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.n_total = n_total

        self.n_total = n_total 
        self.K = K 
        self.N_cluster_list = N_cluster_list
        self.classification_task = classification_task

        # Create two-layer nested defaultdict of lists to store jet observables
        self.output = defaultdict(lambda: defaultdict(list))

        #self.R_jet = 0.4
        
        self.N_list = []
        self.beta_list = []

        # construct the nsub basis 
        for i in range(K-2):
            self.N_list += [i+1] * 3
            self.beta_list += [0.5,1,2]

        self.N_list += [K-1, K-1] 
        self.beta_list += [1, 2]
    

        self.init_data()

        # Transform the dictionary of lists into a dictionary of numpy arrays
        self.output_numpy = {}
        for key,value in self.output.items():
            self.output_numpy[key] = self.transform_to_numpy(value)

        # Currently this is very sketchy and not modular. It should be improved. I apologize to anyone trying to understand this.
        # we've now filled the subjets and nsubjettiness into self.output. We can now calculate the nsubs on the clustered jets (subjets and not hadrons) 
        for N_cluster in self.N_cluster_list:
            x_subjet_input = []
            for n in range(N_cluster):
                x_subjet_input.append(np.array(self.output_numpy[f'subjet'][f'N{N_cluster}_sub_pt'][:,n]))
                x_subjet_input.append(np.array(self.output_numpy[f'subjet'][f'N{N_cluster}_sub_rap'][:,n]))
                x_subjet_input.append(np.array(self.output_numpy[f'subjet'][f'N{N_cluster}_sub_phi'][:,n]))
                
            x_subjet_input = np.array(x_subjet_input).T
            X = x_subjet_input.reshape(-1, N_cluster, 3)  # To bring it to the form (n_total, n_particles, dof of each particle)

            X = np.array(X)
            self.output[f'subjet'][f'N{N_cluster}_sub_X'] = X

            columns = ['pt', 'y', 'phi']

            df_subjet = pd.DataFrame(X.reshape(-1, 3), columns=columns)
            df_subjet.index = np.repeat(np.arange(X.shape[0]), X.shape[1]) + 1
            df_subjet.index.name = 'jet_id'
                    
            # (i) Group the subjet dataframe by jet id
            #     df_particles_grouped is a DataFrameGroupBy object with one particle dataframe per jet
            df_fjsubjet_grouped = df_subjet.groupby('jet_id')


            # (ii) Transform the DataFrameGroupBy object to a SeriesGroupBy of fastjet::PseudoJets
            print(f'Converting subjet N:{N_cluster} dataframe to fastjet::PseudoJets...')
            self.df_fjsubjet = df_fjsubjet_grouped.apply(self.get_fjparticles)
            print('Done.')
            print()

            # Fill each of the jet_variables into a list
            fj.ClusterSequence.print_banner()
            print('Computing N-subjettiness based on subjets and not hadrons...')
            result = [self.analyze_event(df_fjsubjet, subjet=True, N_cluster=N_cluster) for df_fjsubjet in self.df_fjsubjet]
          

        for key,value in self.output.items():
            self.output_numpy[key] = self.transform_to_numpy(value)


        # Reformat output for ML algorithms (array with 1 array per jet which contain all N-subjettiness values)
        self.output_final = {}
        self.output_final['nsub'] = np.array([list(self.output_numpy['nsub'].values())])[0].T

        # do this for subjets as well
        for N_cluster in self.N_cluster_list:
            self.output_final[f'nsub_subjet_N{N_cluster}'] = np.array([list(self.output_numpy[f'nsub_subjet_N{N_cluster}'].values())])[0].T
        
        for key,val in self.output_numpy['subjet'].items():
            self.output_final[f'subjet_{key}'] = val


        # Write jet arrays to file
        with h5py.File(os.path.join(self.output_dir, 'subjets_unshuffled.h5'), 'w') as hf:
            print('-------------------------------------')

            # Write labels: gluon 0, quark 1
            hf.create_dataset(f'y', data=self.y)
            print(f'labels: {self.y.shape}')
            
            # Write numpy arrays
            for key,val in self.output_final.items():
                hf.create_dataset(key, data=val)
                print(f'{key}: {val.shape}')

                # Check whether any training entries are empty
                [print(f'WARNING: input entry {i} is empty') for i,x in enumerate(val) if not x.any()]

            hf.create_dataset('N_clustering', data = self.N_cluster_list)   
            

        #with h5py.File(self.output_dir, 'w') as f:
        #    f.create_dataset('X_nsub', data=self.X_nsub)
        #    f.create_dataset('Y', data=self.Y)
        #    print('Data written to file')
        #    print()

    #---------------------------------------------------------------
    def init_data(self):
        '''
        Load the pp and AA data
        '''
        
        if self.classification_task == 'qvsg':
        # Load the four-vectors directly from the quark vs gluon data set
            X, self.y = energyflow.datasets.qg_jets.load(num_data=self.n_total, pad=True, 
                                                                generator='pythia',  # Herwig is also available
                                                                with_bc=False        # Turn on to enable heavy quarks
                                                            )                        # X_PFN.shape = (n_jets, n_particles per jet, n_variables)  
        
        elif self.classification_task == 'ZvsQCD':
            # Each file contains 100k jets for each class
            n_signal = n_bckg = self.n_total // 2 
            #/pscratch/sd/d/dimathan/JetClass_Dataset/t_jets/bqq
            directory_path = '/pscratch/sd/d/dimathan/JetClass_Dataset'
            if self.classification_task == 'ZvsQCD':   
                signal_jet_filepattern=  f"{directory_path}/Z_jets/ZToQQ*"
                label_signal = 'label_Zqq'
            elif self.classification_task == 'TvsQCD': 
                signal_jet_filepattern=  f"{directory_path}/t_jets/bqq/TTBar*"
                label_signal = 'label_Tbqq'

            bckg_jet_filepattern = f"{directory_path}/qg_jets/ZJetsToNuNu*"

            # Getting the list of files that match the patterns
            signal_jet_files = glob.glob(signal_jet_filepattern)
            bckg_jet_files = glob.glob(bckg_jet_filepattern)
            
            x_particles_signal, x_jet_signal, y_signal = np.array([]), np.array([]), np.array([]) 
            for file in signal_jet_files:
                x_particles, x_jet, y = read_file(filepath = file, labels = [label_signal, 'label_QCD'])
                x_particles_signal = np.concatenate((x_particles_signal, x_particles), axis = 0) if x_particles_signal.size else x_particles
                x_jet_signal = np.concatenate((x_jet_signal, x_jet), axis = 0) if x_jet_signal.size else x_jet
                y_signal = np.concatenate((y_signal, y), axis = 0) if y_signal.size else y
                if x_particles_signal.shape[0] >= n_signal: 
                    x_particles_signal = x_particles_signal[:n_signal]
                    x_jet_signal = x_jet_signal[:n_signal]
                    y_signal = y_signal[:n_signal]
                    break # Stop reading files if we have enough jets
                
            x_particles_bckg, x_jet_bckg, y_bckg = np.array([]), np.array([]), np.array([])
            for file in bckg_jet_files:
                x_particles, x_jet, y = read_file(filepath = file, labels = [label_signal, 'label_QCD'])
                x_particles_bckg = np.concatenate((x_particles_bckg, x_particles), axis = 0) if x_particles_bckg.size else x_particles
                x_jet_bckg = np.concatenate((x_jet_bckg, x_jet), axis = 0) if x_jet_bckg.size else x_jet
                y_bckg = np.concatenate((y_bckg, y), axis = 0) if y_bckg.size else y
                if x_particles_bckg.shape[0] >= n_bckg: 
                    x_particles_bckg = x_particles_bckg[:n_bckg]
                    x_jet_bckg = x_jet_bckg[:n_bckg]
                    y_bckg = y_bckg[:n_bckg]
                    break
            
            # concatenate the two datasets 
            X_ParT = np.concatenate((x_particles_signal, x_particles_bckg), axis = 0)
            Y_ParT = np.concatenate((y_signal, y_bckg), axis = 0)
            x_jet = np.concatenate((x_jet_signal, x_jet_bckg), axis = 0)

            print()
            print(f"Found {len(signal_jet_files)} files matching {self.classification_task} pattern.")
            print(f"Found {len(bckg_jet_files)} files matching ZJetsToNuNu pattern.")
            print()
            print(f"Loaded {X_ParT.shape[0]} jets for the {self.classification_task} classification task.")
            print()
            

            # match the shape of the data to the shape of the energyflow data for consistency
            self.y = Y_ParT[:, 0] # one-hot encoding, where 0: Background (QCD) and 1: Signal (Z) 
            X = np.transpose(X_ParT, (0, 2, 1))

            # shuffle the data 
            indices = np.random.permutation(self.n_total)
            self.y = self.y[indices]
            X = X[indices]


        print(f'X_ParT.shape: {X.shape}')
        X = X[:, :, :3]
        print(f'X_ParT.shape: {X.shape}')

        columns = ['pt', 'y', 'phi']
        df_particles = pd.DataFrame(X.reshape(-1, 3), columns=columns)
        df_particles.index = np.repeat(np.arange(X.shape[0]), X.shape[1]) + 1
        df_particles.index.name = 'jet_id'
                
        # (i) Group the particle dataframe by jet id
        #     df_particles_grouped is a DataFrameGroupBy object with one particle dataframe per jet
        df_fjparticles_grouped = df_particles.groupby('jet_id')


        # (ii) Transform the DataFrameGroupBy object to a SeriesGroupBy of fastjet::PseudoJets
        print('Converting particle dataframe to fastjet::PseudoJets...')
        self.df_fjparticles = df_fjparticles_grouped.apply(self.get_fjparticles)
        print('Done.')
        print()

        
        # Fill each of the jet_variables into a list
        fj.ClusterSequence.print_banner()
        print('Finding jets and computing N-subjettiness and subjets...')
        result = [self.analyze_event(fj_particles) for fj_particles in self.df_fjparticles]


        # make self.output into a numpy array 
        #self.output_np = {key: np.array(value) for key, value in self.output.items()}

        #print(self.output_np.keys())

        # create a self.X_nsub that is a numpy array of shape (n_jets, n_features) where the n_features are the Nsubjettiness features
        #X_nsub = np.column_stack([self.output_np[key] for key in self.output_np.keys()])
        
        #return X_nsub, Y
        

    #---------------------------------------------------------------
    # Transform particles to fastjet::PseudoJets
    #---------------------------------------------------------------
    def get_fjparticles(self, df_particles_grouped):

        return fjext.vectorize_pt_eta_phi(df_particles_grouped['pt'].values,
                                          df_particles_grouped['y'].values,
                                          df_particles_grouped['phi'].values,
                                          )


    #---------------------------------------------------------------
    # Process an event
    #---------------------------------------------------------------
    def analyze_event(self, fj_particles, subjet = False, N_cluster = -1):
    
        # Check that the entries exist appropriately
        if fj_particles and type(fj_particles) != fj.vectorPJ:
            print('fj_particles type mismatch -- skipping event')
            return

        # Find jets -- one jet per "event".  We only use antikt for the Jet Clustering
        jetR = fj.JetDefinition.max_allowable_R
        jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)

        cs = fj.ClusterSequence(fj_particles, jet_def)
        jet_selected = fj.sorted_by_pt(cs.inclusive_jets())[0]
  
        # Compute jet quantities and store in our data structures
        self.analyze_jets(jet_selected, subjet, N_cluster)
        

    #---------------------------------------------------------------
    # Analyze jets of a given event.
    #---------------------------------------------------------------
    def analyze_jets(self, jet_selected, subjet = False, N_cluster = -1):
        self.fill_nsubjettiness(jet_selected, subjet, N_cluster)
        if not subjet:
            self.fill_subjets(jet_selected)


    #---------------------------------------------------------------
    # Compute Nsubjettiness of jet
    #---------------------------------------------------------------
    def fill_nsubjettiness(self, jet, subjet = False, N_cluster = -1):
        #print(f'fill_nsubjettiness: subjet={subjet}, N_cluster={N_cluster}')
        #time.sleep(0.1)
        axis_definition = fjcontrib.OnePass_KT_Axes()

        for i,N in enumerate(self.N_list):
            
            beta = self.beta_list[i]
            measure_definition = fjcontrib.UnnormalizedMeasure(beta)
            n_subjettiness_calculator = fjcontrib.Nsubjettiness(N, axis_definition, measure_definition)
            n_subjettiness = n_subjettiness_calculator.result(jet)/jet.pt()

            if subjet:
                self.output[f'nsub_subjet_N{N_cluster}'][f'N{N}_beta{beta}'].append(n_subjettiness)
            else:
                self.output['nsub'][f'N{N}_beta{beta}'].append(n_subjettiness)


    #---------------------------------------------------------------
    # Compute subjet kinematics...
    #---------------------------------------------------------------
    def fill_subjets(self, jet):
        
        hadrons_aux  = fj.sorted_by_pt(jet.constituents())

        jetR = fj.JetDefinition.max_allowable_R
        subjet_def = fj.JetDefinition(fj.kt_algorithm, jetR)
        cs_subjet = fj.ClusterSequence(jet.constituents(), subjet_def)

        for N_cluster in self.N_cluster_list:

            subjets = fj.sorted_by_pt(cs_subjet.exclusive_jets_up_to(N_cluster))

            n_subs = len(subjets)
            subjet_pt_list  = [subjets[n].pt() for n in range(n_subs)]  + [0]*(N_cluster - n_subs)
            subjet_phi_list = [subjets[n].phi() for n in range(n_subs)] + [0]*(N_cluster - n_subs)
            subjet_rap_list = [subjets[n].rap() for n in range(n_subs)] + [0]*(N_cluster - n_subs)

            self.output[f'subjet'][f'N{N_cluster}_sub_pt'].append(np.array(subjet_pt_list))
            self.output[f'subjet'][f'N{N_cluster}_sub_rap'].append(np.array(subjet_rap_list))
            self.output[f'subjet'][f'N{N_cluster}_sub_phi'].append(np.array(subjet_phi_list))

  

    #---------------------------------------------------------------
    # Transform dictionary of lists into a dictionary of numpy arrays
    #---------------------------------------------------------------
    def transform_to_numpy(self, jet_variables_list):

        jet_variables_numpy = {}
        for key,val in jet_variables_list.items():
            jet_variables_numpy[key] = np.array(val)
        
        return jet_variables_numpy


if __name__ == '__main__':
    # Path to store the nsubs file. Be careful with overwriting.
    
    dir = '/pscratch/sd/d/dimathan/GNN/exclusive_subjets_ZvsQCD_200k_N203040506080100'
    Process(dir, n_total = 200000, N_cluster_list = [20, 30, 40, 50, 60, 80, 100], K = 21, classification_task = 'ZvsQCD')
    print('done')