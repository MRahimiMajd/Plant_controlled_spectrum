import numpy as np
import pandas as pd
import itertools
from scipy.stats import pearsonr
import pingouin as pg

# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_Carmona_et_al_data():
    root_traits0 = pd.read_csv("data/Root_traits.txt", sep=' ')
    root_traits = root_traits0[['SRL', 'D', 'RTD', 'N']].dropna()
    ############################
    above_traits0 = pd.read_csv('data/Above_traits.txt', sep=' ')
    above_traits = above_traits0[['la', 'ln', 'ph', 'sla', 'ssd', 'sm']].dropna()
    above_traits['lma'] = 1/above_traits['sla']
    above_traits = above_traits.drop(['sla'],axis=1)
    above_traits = np.log10(above_traits)
    ############################
    common_species = root_traits.index.intersection(above_traits.index)
    root_traits_common    = root_traits.loc[common_species]
    above_traits_common   = above_traits.loc[common_species]
    data_df = pd.concat([above_traits_common, root_traits_common], axis=1)
    ############################
    return data_df 

def get_significance_pattern(pear_p, cov_p, alpha=0.05):
    pear_sig = pear_p < alpha
    cov_sig = cov_p < alpha
    
    if pear_sig and cov_sig:
        return "both"
    elif pear_sig and not cov_sig:
        return "only_pear"
    elif cov_sig and not pear_sig:
        return "only_cov"
    else:
        return "neither"

def build_pairwise_df(df, alpha=0.05, print_stat=True):
    traits = df.columns
    
    trait_combinations = list(itertools.combinations(traits, 2))
    
    rows = []
    non_signidicant = 0
    for t1, t2 in trait_combinations: 
        # Direct Pearson and Spearman correlations
        pear_r, pear_p = pearsonr(df[t1], df[t2])
        
        base_row = {
            'trait1': t1,
            'trait2': t2,
            'pearson_r': pear_r,
            'pearson_pval': pear_p,
        }

        # Partial correlations of pair (t1, t2), controlling for each covariate.
        covariates = [c for c in traits if c not in (t1, t2)]
        for cov in covariates:
            partial = pg.partial_corr(data=df, x=t1, y=t2, covar=cov, method='pearson').iloc[0]
            cov_r = partial['r'] 
            cov_p = partial['p-val'] 

            pattern = get_significance_pattern(pear_p, cov_p, alpha)
            sign_flip = np.sign(pear_r) != np.sign(cov_r)
            abs_diff = abs(pear_r - cov_r)
            square_diff = abs_diff ** 2
            
            #### Siginificance constriants
            if pattern == "neither": 
                non_signidicant +=1
                continue

            if pattern == "only_cov" and sign_flip:
                abs_diff = abs(cov_r)
                square_diff = cov_r**2
                
            elif pattern == "only_pear" and sign_flip:
                abs_diff = abs(pear_r)
                square_diff = pear_r**2
                 
            
            # Combine base and partial correlation results
            row = {
                **base_row,
                'covariate': cov,
                'cov_r': cov_r,
                'cov_pval': cov_p,
                'abs_diff': abs_diff,
                'square_diff': square_diff,
                'sign_flip': int(sign_flip),
                'pattern': pattern,
            }
            rows.append(row)
            
    res_df = pd.DataFrame(rows)
    if print_stat:
        print(f"Total rows (trait pairs + covariates): {len(res_df)}")
        print(f"  Non-significant: {(len(trait_combinations) * len( covariates))-len(res_df)}")
        print(f"  Both Pear & Partial significant: {len(res_df[res_df['pattern']=='both'])}")
        print(f"  Only Pearson significant (only_pear): {len(res_df[res_df['pattern']=='only_pear'])}")
        print(f"  Only Partial significant (only_cov): {len(res_df[res_df['pattern']=='only_cov'])}")

        flip_mask = res_df['sign_flip'] == 1
        sig_patterns = res_df['pattern'].isin(['both','only_pear','only_cov'])

        print(f"  Sign-flips only in one-sided significant cases: {len(res_df[res_df['pattern'].isin(['only_pear','only_cov']) & flip_mask])}")
        print(f"  Sign-flips in any significant case: {len(res_df[sig_patterns & flip_mask])}")
        print("---------------------------\n")

    return res_df

def maximum_cover(df_corr, k=3, weight_col='square_diff',top_n=10,print_stat=True):
    """
    Test all combinations of k traits to find the set that maximizes the sum of weighted 
    differences (square_diff) between Pearson and partial correlations, using a brute-force approach.
    """
    traits = sorted(df_corr['trait1'].unique()) 
    total_weight = df_corr[weight_col].sum()
    
    columns_to_check = ['trait1', 'trait2', 'covariate']
    results = []
    for subset in itertools.combinations(traits, k):
        vals = list(subset)
        is_covered = df_corr[columns_to_check].isin(vals).any(axis=1)
        covered_weight = df_corr.loc[is_covered, weight_col].sum()
        results.append({
            'traits_subset': subset,
            'covered_weight': covered_weight,
            'fraction': covered_weight / total_weight,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='covered_weight', ascending=False).reset_index(drop=True)
    
    if print_stat:
        print(f"\n--- Top {top_n} Maximum Cover Solutions (k={k}, maximizing 'square_diff') ---")
        
        for i, row in results_df.head(top_n).iterrows():
            print(f"#{i+1}: Set: {row['traits_subset']}")
            print(f"  Fraction of Total Weight: {row['fraction']:.4f}")
            print("-" * 25)
    
    return results_df
###############################################################################
# 1. Load Data
data = load_Carmona_et_al_data()

################ To remove the outliers
z_data = (data - data.mean()) / data.std()
mask = ~(np.abs(z_data) > 4).any(axis=1)
data = data[mask]
################

partial_res = build_pairwise_df(data, print_stat = True)
###############################################################################
solutions = maximum_cover(partial_res, k=3, weight_col='square_diff')
