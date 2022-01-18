#!/usr/bin/env python3

import os

# limit numpy threads for NERZ
os.environ['OMP_NUM_THREADS'] = '20'
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"

from numpy.random import default_rng
from fitting.crossvalidation import frrsa
from memory_profiler import memory_usage
import pandas as pd

rng = default_rng(seed=4)
hyperparams = None
distance = 'pearson'
outer_k = 5
outer_reps = 10
splitter = 'random'
score_type = 'pearson'
betas_wanted = True
predictions_wanted=True
parallel = True
rng_state = 1

import time

df = pd.read_csv('results_parallel_on_off.csv')

for n_objects in [50, 100, 150]:
    for n_units in [1000, 10000, 100000]:
        for n_outputs in [2, 10, 100]:
            for parallel in [True, False]:
                cond = ((df['n_units'] == n_units) & 
                        (df['parallel'] == parallel)  & 
                        (df['n_objects'] == n_objects) &
                        (df['n_outputs'] == n_outputs)
                ).any()
                
                if not cond:
                    target = rng.integers(low=0, high=100, size=(n_objects,n_objects,n_outputs))
                    predictor = rng.integers(low=0, high=100, size=(n_units,n_objects))
                    s = time.time()
                    ram_usage = memory_usage((frrsa, (target,
                                                                predictor, 
                                                                distance,
                                                                outer_k, 
                                                                outer_reps, 
                                                                splitter, 
                                                                hyperparams, 
                                                                score_type, 
                                                                betas_wanted,
                                                                predictions_wanted,
                                                                parallel,
                                                                rng_state)))
                    print(ram_usage)
                    break
                    e = time.time()
                    duration = e - s
                    string = '%s,%s,%s,%s,%s,%s,%s,%s,8\n' % (n_objects, n_units, n_outputs, duration, outer_k, outer_reps, max(ram_usage),parallel)
                    with open('results_parallel_on_off.csv', 'a') as result:
                        result.write(string)

    
