#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 09:31:34 2018

@author: thomas.gibon@list.lu

Based on a MATLAB version written by Yasushi Kondo, initially built on a first
version by Glen Peters.
Reference: Peters and Hertwich (2006, ESR)
Algorithm available publicly at https://www.tandfonline.com/doi/full/10.1080/09535310600653008
"""

import numpy as np
import pandas as pd


def SPA(F, A, y, Tmax, threshold, filename=None, M=None, max_npaths=1000,
        index=None, direct=None):
    '''
    Performs a structural path analysis on a given linear algebra system.
    The system can either be a life cycle or an input-output system.

    The algorithm extracts [emissions-intermediate consumption-final demand]
    pathways with the highest contribution to a given emission.

    Arguments (valid for EXIOBASE)
    ---------
    - F: 1-by-p (or n) matrix containing stressors, mixed units per M€
    - A: p × p (or n × n) matrix, containing intermediate consumption, M€/M€,
    - y: p × 1 (or n × 1) total final demand vector, in M€,
    - Tmax: integer, maximum number of upstream tiers to search,
    - threshold: float between 0 and 1, cutoff under which a contribution
        is discarded,
    - filename: where to store the results,
    - M: multiplier (F*L), providing M saves memory,
    - direct: in case there are direct emissions (as with households, e.g.)

    ...where p is the number of products, and n the number of industries
    '''

    if type(A) == pd.core.frame.DataFrame:
        if not index:
            index = A.index
        A = A.values

    if 'M' not in locals():
        M = np.linalg.solve(np.eye(A.shape[0]) - A.T, F)

    # Calculate total emissions and tolerance
    e = M.dot(y)
    tolerance = threshold * e

    # Start extracting the paths
    paths, _ = extract_paths(F, A, y, M, Tmax, tolerance, max_npaths)
    
    if direct:
        paths['DIRECT'] = {'value': direct, 'sequence': []}
        
    coverage = sum(v['value'] for v in paths.values())
    paths['REST'] = {'value': e - coverage, 'sequence': []}
    paths['TOTAL'] = {'value': e, 'sequence': []}

    df_paths = pd.DataFrame(paths).T
    df_paths.sort_values('value', ascending=False, inplace=True)
    df_paths['contribution'] = df_paths['value']/e

    if type(index) in [list, pd.core.indexes.multi.MultiIndex, pd.core.indexes.base.Index]:
        df_paths['path'] = [[index[ss] for ss in s]
                            for s in df_paths['sequence']]

    if 'filename' in locals():
        df_paths.to_csv(filename)

    return df_paths


def extract_paths(F, A, y, M, Tmax, tolerance, max_npaths):
    '''
    Initialize the recursion
    '''
    paths = {}

    paths, count = extract_paths_rc(paths, 0, [], np.nan, 0, F, A, y, M, Tmax,
                                    tolerance)

    return paths, count


def extract_paths_rc(paths, count, sequence, val_wo_F, T, F, A, y, M, Tmax,
                     tolerance):
    '''
    Recursion
    
    paths,
    count,
    sequence,
    intermediate consumption vector,
    tier,
    stressor vector,
    intermediate consumption matrix
    '''

    if T > 0:
        count += 1
        paths[count] = {'sequence': sequence,
                        'value': F[sequence[-1]] * val_wo_F}

    if T <= Tmax:
        if T == 0:
            next_val_wo_F = y
        else:
            next_val_wo_F = A[:, sequence[-1]] * val_wo_F

    next_subtree_val = M * next_val_wo_F
    tofind = np.where(next_subtree_val > tolerance)[0].tolist()

    for i in tofind:
        paths, count = extract_paths_rc(
            paths, count, sequence + [i], next_val_wo_F[i], T+1, F, A, y, M, Tmax, tolerance)

    return paths, count


if __name__ == '__main__':
    '''
    test with EXIOBASE3
    1 M€ of French wheat
    '''

    if not {'A', 'F', 'M'}.issubset(set(locals())):

        import pymrio as mr

        # Load the system
        folder = 'IOT_2011_pxp'
        pxp = mr.load_all(path=folder)

        A = pxp.A

        # Calculate the IO system
        pxp.calc_system()

        # Calculate only the extensions we need
        # Choose stressor here
        stressor = 'CO2 - combustion - air'
        
        F = mr.calc_S(pxp.satellite.F, pxp.x).loc[stressor]
        M = mr.calc_M(F, pxp.L)

    y = pd.Series(np.zeros_like(F), index=A.index)
    y[('CN', 'Construction work (45)')] = 1  # Choose sector here
    y = y.values
    
    # One sector
    paths = SPA(F, A, y, Tmax=30, threshold=.0001, filename='test_single_sector.csv', M=M, direct=None)
    
    # Luxembourg households
    country  = 'LU'
    category = 'Final consumption expenditure by households'    
    
    paths = SPA(F, A, pxp.Y.LU[category],
                Tmax=30, threshold=.0001, filename='test_LU_hh.csv', M=M,
                direct=pxp.satellite.F_hh[country][category][stressor])
    
    # Luxembourg all
    paths = SPA(F, A, pxp.Y.LU.sum(1),
                Tmax=30, threshold=.0001, filename='test_LU_all.csv', M=M,
                direct=pxp.satellite.F_hh[country].sum(1)[stressor])
