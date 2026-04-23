# from dataclasses import dataclass

import numpy as np
import pandas as pd
# import scipy.sparse as sp
from scipy import stats
from math import radians, sin, cos, sqrt, atan2

# from ferm.distance import wrap_geodist


class FERM:
    """
    Implement the Feature-Enriched Radiation Model (FERM).
    
    """    
              
    def __init__(
        self,
        path_niche_array: str,
        path_pop: str,
    ) -> None:
        """
        Parameters
        ----------
            
        """ 

        self.path_niche_array = path_niche_array
        self.path_pop = path_pop

    
    @staticmethod
    def build_distance_matrix(nodes):
        codes = nodes["code"].tolist()
        D = pd.DataFrame(0.0, index=codes, columns=codes)
        coord = nodes.set_index("code")[["lat", "lon"]]
        for i in codes:
            for j in codes:
                if i != j:
                    D.loc[i, j] = haversine_km(coord.loc[i, "lat"], coord.loc[i, "lon"], coord.loc[j, "lat"], coord.loc[j, "lon"])
        return D


    # -------------------------------------------------------------------------
    # Run FERM - Compute fluxes
    # -------------------------------------------------------------------------
        
    def run(
            self,
            nodes:pd.DataFrame, 
            num_particles:int = 300, 
            sigma:float = 0.15, 
            niche_col:str = "niche",
            verbose: bool = False,            
            ) -> pd.DataFrame:
        """
        Run the FERM model on a node table.
    
        Parameters
        ----------
        nodes : pd.DataFrame
            Must contain:
            - 'code': unique node identifier
            - 'population': node population
            - niche_col: niche variable
        num_particles : int, default=300
            Number of particles sampled per origin.
        sigma : float, default=0.15
            Standard deviation used in Gaussian max sampling.
        niche_col : str, default='gdp_per_capita_2018'
            Column used as niche mean.
        verbose : bool, default=False
            If True, print current origin code.
    
        Returns
        -------
        pd.DataFrame
            origin-destination flux matrix.
        """
        

        # Distance matrix
        D = FERM.build_distance_matrix(nodes)                
        
        nodes = nodes.set_index("code")

        populations = nodes["population"].round().clip(lower=1).astype(int)
        
        niche = nodes[niche_col].astype(float).fillna(0.0)
        
        res = pd.DataFrame(0.0, index=nodes.index, columns=nodes.index)
        
        for i in nodes.index:
            if verbose: print(f"{i}")
            
            m_i = populations[i]
            mu_i = niche[i]
            absorption_i = gaussian_max_sample_vec(
                mu=mu_i, 
                sigma=sigma, 
                n=m_i, 
                size=num_particles
            )
            
            dests_sorted = [j for j in D.loc[i].sort_values().index if j != i]
            assigned = np.zeros(num_particles, dtype=bool)
            counts = pd.Series(0.0, index=nodes.index)
            
            for j in dests_sorted:
                if assigned.all():
                    break
                n_j = populations[j]
                mu_j = niche[j] 
                
                n_remaining = (~assigned).sum()
                absorbance_j = gaussian_max_sample_vec(
                    mu=mu_j, 
                    sigma=sigma, 
                    n=n_j, 
                    size=n_remaining
                    )
                win_mask_local = absorbance_j > absorption_i[~assigned]
                if np.any(win_mask_local):
                    idx_global = np.where(~assigned)[0][win_mask_local]
                    counts[j] += len(idx_global)
                    assigned[idx_global] = True
            if counts.sum() > 0:
                res.loc[i] = counts / counts.sum()
        return res



def gaussian_max_sample_vec(mu, sigma, n, size):
    """
    Sample the maximum of n Gaussian samples
    """
    # assert isinstance(n,int) and n >= 1
    n = max(int(n), 1)
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0:
        return np.full(size, float(mu))
    u = np.random.random(size=size)
    q = np.exp(np.log(u) / n)
    return mu + sigma * stats.norm.ppf(q)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))





