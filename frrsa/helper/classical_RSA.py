#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spyder 4.2.5 | Python 3.8.8 64-bit | Qt 5.9.7 | PyQt5 5.9.2 | Darwin 18.7.0
"""
Contains function to perform classical Representational Similarity Analysis.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

from functools import partial
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist


def make_RDM(activity_pattern_matrix, distance='pearson'):
    # TODO: implement for multiple matrices.
    """Compute dissimilarity matrix.

    Parameters
    ----------
    activity_pattern_matrix : ndarray
        Matrix which holds activity patterns for several conditions. Each
        column is one condition, each row one measurement channel.
    distance : {'pearson', 'sqeuclidean'}, optional
        The desired distance measure (defaults to `pearson`).

    Returns
    -------
    rdm : ndarray
        Dissimilarity matrix indicating dissimilarity for all conditions in
        `activity_pattern_matrix`.
    """
    if distance == 'pearson':
        rdm = 1 - np.corrcoef(activity_pattern_matrix, rowvar=False)
    elif distance == 'sqeuclidean':
        rdm = squareform(pdist(activity_pattern_matrix.T, 'sqeuclidean'))
    return rdm


def flatten_RDM(rdm: np.ndarray) -> np.ndarray:
    """Flatten the upper half of a dissimilarity matrix to a vector.

     Multiple dissimilarity matrices can be fed at once.

    Parameters
    ----------
    rdm : ndarray
        The RDM)s) which shall be flattened. Expected shape is
        (n_conditions, n_conditions, n_targets), where `n_targets` denotes the
        number of RDMs. If `n_targets == 1`, `rdm` can be of shape
        (n_conditions, n_conditions).

    Returns
    -------
    rdv : ndarray
        The unique upper half of `rdm`.
    """
    if rdm.ndim == 3:
        mapfunc = partial(squareform, checks=False)
        rdv = np.array(list(map(mapfunc, np.moveaxis(rdm, -1, 0)))).T
    elif rdm.ndim == 2:
        rdv = rdm[np.triu_indices(rdm.shape[0], k=1)].reshape(-1, 1)
    return rdv


def noise_ceiling(reference_rdms, correlation='pearson'):
    '''Compute noise ceilings for represntational similarity analysis.
    
    Parameters
    ----------
    reference_rdms : ndarray
        Several dissimilarity matrices based on which the noise ceilings shall
        be computed. The shape (n,n,m) is mandatory, where m denotes
        different matrices of shape (n,n).
    correlation : {'pearson', 'spearman'}, optional
        The correlation coefficient which should be used to compute ceilings
        (defaults to `pearson`).
        
    Returns
    -------
        ceilings : ndarray
            Upper and lower noise ceiling value.
        
    Notes
    -----
    This implementation is inspired by the MATLAB implementation presented in [1]_.
    
    References
    ----------
    .. [1] Nili, H., Wingfield, C., Walther, A., Su, L., Marslen-Wilson, W.,
       & Kriegeskorte, N. (2014). A Toolbox for Representational Similarity
       Analysis. PLoS Computational Biology, 10(4), e1003553.
       https://doi.org/10.1371/journal.pcbi.1003553
    '''
    n_subjects = reference_rdms.shape[2]
    reference_rdms = flatten_RDM(reference_rdms)
    if correlation=='pearson':
        reference_rdms = (reference_rdms - reference_rdms.mean(0)) / reference_rdms.std(0)
    elif correlation=='spearman':
        reference_rdms = rankdata(reference_rdms, axis=0)
    #TODO: maybe implement Kendall's tau_a
    
    reference_rdm_average = np.mean(reference_rdms, axis=1)
    upper_bound = 0
    lower_bound = 0
    
    for n in range(n_subjects):
        index = list(range(n_subjects))
        index.remove(n)
        rdm_n = reference_rdms[:,n]
        reference_rdm_average_loo = np.mean(reference_rdms[:,index], axis=1)
        upper_bound += np.corrcoef(reference_rdm_average, rdm_n)[0][1]
        lower_bound += np.corrcoef(reference_rdm_average_loo, rdm_n)[0][1]
        #TODO: maybe implement Kendall's tau_a
        
    upper_bound /= n_subjects
    lower_bound /= n_subjects
    return np.array([upper_bound, lower_bound])
