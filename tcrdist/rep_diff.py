import pandas as pd
import numpy as np
import itertools
import warnings

import statsmodels.api as sm
import patsy

from scipy.stats import chi2_contingency
from scipy.stats.contingency import expected_freq
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance

import fishersapi

from .pvalue_adjustment import adjustnonnan

__all__ = ['neighborhood_diff',
           'hcluster_diff',
           'member_summ']

"""TODO:
 * Add useful marginal frequencies to NNdiff output (like hierdiff)
 * Make sure RR and OR make sense. Maybe include only one for fisher test?"""
 

def neighborhood_diff(clone_df, pwmat, x_cols, count_col='count', test='chi2', knn_neighbors=50, knn_radius=None, test_only=None, **kwargs):
    """Tests for association of categorical variables in x_cols with the neighborhood
    around each TCR in clone_df. The neighborhood is defined by the K closest neighbors
    using pairwise distances in pwmat, or defined by a distance radius.

    Use Fisher's exact test (test='fishers') to detect enrichment/association of the neighborhood
    with one variable.

    Tests the 2 x 2 table for each clone:

    +----+----+-------+--------+
    |         |  Neighborhood  |
    |         +-------+--------+
    |         | Y     |    N   |
    +----+----+-------+--------+
    |VAR |  1 | a     |    b   |
    |    +----+-------+--------+
    |    |  0 | c     |    d   |
    +----+----+-------+--------+

    Use the chi-squared test (test='chi2') or logistic regression (test='logistic') to detect association across multiple variables.
    Note that with sparse neighborhoods Chi-squared tests and logistic regression are unreliable. It is possible
    to pass an L2 penalty to the logistic regression using l2_alpha in kwargs, howevere this requires a permutation
    test (nperms also in kwargs) to compute a value.

    Use the Cochran-Mantel-Haenszel test (test='chm') to test stratified 2 x 2 tables: one VAR vs. neighborhood, over sever strata
    defined in other variables. Use x_cols[0] as the primary (binary) variable and other x_cols for the categorical
    strata-defining variables. This tests the overall null that OR = 1 for x_cols[0]. A test is also performed
    for homogeneity of the ORs among the strata (Breslow-Day test).

    Params
    ------
    clone_df : pd.DataFrame [nclones x metadata]
        Contains metadata for each clone.
    pwmat : np.ndarray [nclones x nclones]
        Square distance matrix for defining neighborhoods
    x_cols : list
        List of columns to be tested for association with the neighborhood
    count_col : str
        Column in clone_df that specifies counts.
        Default none assumes count of 1 cell for each row.
    test : str
        Specifies Fisher's exact test ("fishers"), Chi-squared ("chi2") or
        logistic regression ("glm") for testing the association.
        Also "chi2+fishers" tests the global null using Chi2 and all pairwise
        combinations of variable values using Fisher's.
    knn_neighbors : int
        Number of neighbors to include in the neighborhood.
    knn_radius : float
        Radius for inclusion of neighbors within the neighborhood.
        Specify K or R but not both.
    test_only : None or np.ndarray
        Indices into clone_df specifying the neighborhoods for testing.
    kwargs : dict
        Arguments for the various test functions (currently only logistic
        regression, which takes l2_alpha and nperms)

    Returns
    -------
    res_df : pd.DataFrame [nclones x results]
        Results from testing the neighborhood around each clone."""
    if knn_neighbors is None and knn_radius is None:
        raise(ValueError('Must specify K or radius'))
    if not knn_neighbors is None and not knn_radius is None:
        raise(ValueError('Must specify K or radius (not both)'))

    if test == 'fishers':
        test_func = _fisherNBR
        assert len(x_cols) == 1
    elif test in ['chisq', 'chi2']:
        test_func = _chi2NBR
    elif test == 'glm':
        test_func = _glmNBR

    n = clone_df.shape[0]
    assert n == pwmat.shape[0]
    assert n == pwmat.shape[1]
    ycol = 'NBR'

    if count_col is None:
        clone_df = clone_df.assign(count_col=1)
        count_col = 'count_col'

    if test_only is None:
        test_only = clone_df.index
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = []
        for clonei in test_only:
            ii = np.nonzero(clone_df.index == clonei)[0][0]
            if not knn_neighbors is None:
                if knn_neighbors < 1:
                    frac = knn_neighbors
                    K = int(knn_neighbors * n)
                    print('Using K = %d (%1.0f%% of %d)' % (K, 100*frac, n))
                else:
                    K = knn_neighbors
                R = np.partition(pwmat[ii, :], knn_neighbors)[knn_neighbors]
            else:
                R = knn_radius
            y = (pwmat[ii, :] <= R).astype(float)
            K = np.sum(y)

            cdf = clone_df.assign(**{ycol:y})[[ycol, count_col] + x_cols]
            counts = _prep_counts(cdf, x_cols, ycol, count_col)

            out = {'CTS%d' % i:v for i,v in enumerate(counts.values.ravel())}

            uY = [1, 0]
            out.update({'x_col_%d' % i:v for i,v in enumerate(x_cols)})
            for i,xvals in enumerate(counts.index.tolist()):
                if type(xvals) is tuple:
                    val = '|'.join(xvals)
                else:
                    val = xvals
                out.update({'x_val_%d' % i:val,
                            'x_freq_%d' % i: counts.loc[xvals, 1] / counts.loc[xvals].sum()})

            out.update({'index':clonei,
                        'neighbors':list(clone_df.index[np.nonzero(y)[0]]),
                        'K_neighbors':K,
                        'R_radius':R})

            if test == 'logistic':
                glm_res = _glmCatNBR(cdf, x_cols, y_col=ycol, count_col=count_col, **kwargs)
                out.update(glm_res)
            elif test == 'chi2+fishers':
                comb_res = _chi2_fishersNBR(counts)
                out.update(comb_res)
            elif test == 'chm':
                comb_res = _CMH_NBR(counts)
                out.update(comb_res)
            res.append(out)

        res_df = pd.DataFrame(res)
        if test in ['fishers', 'chi2']:
            out = test_func(res_df, count_cols=[c for c in res_df.columns if c.startswith('CTS')])
            res_df = res_df.assign(**out)

    for c in [c for c in res_df.columns if 'pvalue' in c]:
        res_df = res_df.assign(**{c.replace('pvalue', 'FWERp'):adjustnonnan(res_df[c].values, method='holm'),
                                  c.replace('pvalue', 'FDRq'):adjustnonnan(res_df[c].values, method='fdr_bh')})
    return res_df

def hcluster_diff(clone_df, pwmat, x_cols, count_col='count', test_within=[], test='chi2', min_n=20, method='complete', **kwargs):
    """Tests for association of categorical variables in x_cols with each cluster/node
    in a hierarchical clustering of clones with distances in pwmat.

    Use Fisher's exact test (test='fishers') to detect enrichment/association of the neighborhood/cluster
    with one variable.

    Tests the 2 x 2 table for each clone:

    +----+----+-------+--------+
    |         |    Cluster     |
    |         +-------+--------+
    |         | Y     |    N   |
    +----+----+-------+--------+
    |VAR |  1 | a     |    b   |
    |    +----+-------+--------+
    |    |  0 | c     |    d   |
    +----+----+-------+--------+

    Use the chi-squared test (test='chi2') or logistic regression (test='logistic') to detect association across multiple variables.
    Note that with small clusters Chi-squared tests and logistic regression are unreliable. It is possible
    to pass an L2 penalty to the logistic regression using l2_alpha in kwargs, howevere this requires a permutation
    test (nperms also in kwargs) to compute a value.

    Use the Cochran-Mantel-Haenszel test (test='chm') to test stratified 2 x 2 tables: one VAR vs. cluster, over sever strata
    defined in other variables. Use x_cols[0] as the primary (binary) variable and other x_cols for the categorical
    strata-defining variables. This tests the overall null that OR = 1 for x_cols[0]. A test is also performed
    for homogeneity of the ORs among the strata (Breslow-Day test).

    Params
    ------
    clone_df : pd.DataFrame [nclones x metadata]
        Contains metadata for each clone.
    pwmat : np.ndarray [nclones x nclones]
        Square distance matrix for defining neighborhoods
    x_cols : list
        List of columns to be tested for association with the neighborhood
    count_col : str
        Column in clone_df that specifies counts.
        Default none assumes count of 1 cell for each row.
    test_within : list of columns
        Provides option to test within groups using a pd.DataFrame.GroupBy. Allows for one clustering
        of pooled TCRs, but testing within groups (e.g. participants or conditions)
    test : str
        Specifies Fisher's exact test ("fishers"), Chi-squared ("chi2") or
        logistic regression ("glm") for testing the association.
        Also "chi2+fishers" tests the global null using Chi2 and all pairwise
        combinations of variable values using Fisher's.
    min_n : int
        Minimum size of a cluster for it to be tested.
    kwargs : dict
        Arguments for the various test functions (currently only logistic
        regression, which takes l2_alpha and nperms)

    Returns
    -------
    res_df : pd.DataFrame [nclusters x results]
        Results from testing each cluster.
    Z : linkage matrix [clusters, 4]
        Clustering result returned from scipy.cluster.hierarchy.linkage"""
    if test == 'fishers':
        test_func = _fisherNBR
        assert len(x_cols) == 1
    elif test in ['chisq', 'chi2']:
        test_func = _chi2NBR
    elif test == 'glm':
        test_func = _glmNBR

    n = clone_df.shape[0]
    assert n == pwmat.shape[0]
    assert n == pwmat.shape[1]
    ycol = 'NBR'

    compressedDmat = distance.squareform(pwmat)
    Z = sch.linkage(compressedDmat, method=method)

    if len(test_within) == 0:
        """When a Series of ones is passed to Groupby, one group with all rows will be analyzed"""
        test_within = pd.Series(np.ones(clone_df.shape[0]), index=clone_df.index)
        no_groups = True
    else:
        no_groups = False

    clusters = {}
    for i, merge in enumerate(Z):
        cid = 1 + i + Z.shape[0]
        clusters[cid] = [merge[0], merge[1]]

    def _get_indices(clusters, i):
        if i <= Z.shape[0]:
            return [int(i)]
        else:
            return _get_indices(clusters, clusters[i][0]) + _get_indices(clusters, clusters[i][1])

    def _get_cluster_indices(clusters, i):
        if i <= Z.shape[0]:
            return []
        else:
            return [int(i)] + _get_cluster_indices(clusters, clusters[i][0]) + _get_cluster_indices(clusters, clusters[i][1])

    members = {i:_get_indices(clusters, i) for i in range(Z.shape[0] + 1, max(clusters.keys()) + 1)}
    """Note that the list of clusters within each cluster includes the current cluster"""
    cluster_members = {i:_get_cluster_indices(clusters, i) for i in range(Z.shape[0] + 1, max(clusters.keys()) + 1)}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = []
        for ind, gby in clone_df.groupby(test_within):
            """Use groupby to generate indices, but continue to analyze whole clone_df
            setting non-group counts to zero"""
            if not no_groups:
                if len(test_within) == 1:
                    """Workaround since groupby returns ind as a value if only one groupby level is provided"""
                    gby_info = {test_within[0]: ind}    
                else:
                    gby_info = {k:v for k,v in zip(test_within, ind)}
            else:
                gby_info = {}
            clone_tmp = clone_df.copy()
            """Set counts to zero for all clones that are not in the group being tested"""
            not_gby = [ii for ii in clone_df.index if not ii in gby.index]
            clone_tmp.loc[not_gby, count_col] = 0
            for cid, m in members.items():
                not_m = [i for i in range(n) if not i in m]
                y = np.zeros(n)
                y[m] = 1

                K = np.sum(y)
                R = np.max(pwmat[m, :][:, m])

                cdf = clone_tmp.assign(**{ycol:y})[[ycol, count_col] + x_cols]
                counts = _prep_counts(cdf, x_cols, ycol, count_col)

                out = {'CTS%d' % i:v for i,v in enumerate(counts.values.ravel())}

                uY = [1, 0]
                out.update({'x_col_%d' % i:v for i,v in enumerate(x_cols)})
                for i,xvals in enumerate(counts.index.tolist()):
                    if type(xvals) is tuple:
                        val = '|'.join(xvals)
                    else:
                        val = xvals
                    out.update({'x_val_%d' % i:val,
                                'x_freq_%d' % i: counts.loc[xvals, 1] / counts.loc[xvals].sum()})
                
                out.update({'cid':cid,
                            'members_index':list(clone_tmp.index[m]),
                            'members_i':m,
                            'cid_members':cluster_members[cid],
                            'K_neighbors':K,
                            'R_radius':R})
                if not no_groups:
                    m_within = [mi for mi in m if clone_tmp[count_col].iloc[mi] > 0]
                    out.update({'members_within_index':list(clone_tmp.index[m_within]),
                                'members_within_i':m_within})

                if K >= min_n and K < (n-min_n):
                    if test == 'logistic':
                        glm_res = _glmCatNBR(cdf, x_cols, y_col=ycol, count_col=count_col, **kwargs)
                        out.update(glm_res)
                    elif test == 'chi2+fishers':
                        comb_res = _chi2_fishersNBR(counts)
                        out.update(comb_res)
                    elif test == 'chm':
                        comb_res = _CMH_NBR(counts)
                        out.update(comb_res)
                    out.update({'tested':True})
                else:
                    out.update({'tested':False})
                out.update(gby_info)
                res.append(out)

        res_df = pd.DataFrame(res)
        if test in ['fishers', 'chi2']:
            tmp = res_df.loc[res_df['tested']]
            out = test_func(tmp, count_cols=[c for c in res_df.columns if c.startswith('CTS')])
            res_df = res_df.assign(**{k:np.nan*np.ones(res_df.shape[0]) for k in out.keys()})
            for k in out.keys():
                res_df.loc[res_df['tested'], k] = out[k]

    for c in [c for c in res_df.columns if 'pvalue' in c]:
        res_df = res_df.assign(**{c.replace('pvalue', 'FWERp'):adjustnonnan(res_df[c].values, method='holm'),
                                  c.replace('pvalue', 'FDRq'):adjustnonnan(res_df[c].values, method='fdr_bh')})
    return res_df, Z


def member_summ(res_df, clone_df, summ_within=False, count_col='count', addl_cols=[], addl_n=1):
    """Return additional summary info about each result (row)) based on the members of the cluster.
        
    By default, summary will include all rows of clone_df even when results are based on "test_within" grouping.
    To get a summary within each group use summ_within=True.

    summ_df = member_summ(res_df, clone_df)
    res_df = res_df.join(summ_df, how='left')
    """
    def _top_N_str(m, col, count_col, N):
        gby = m.groupby(col)[count_col].agg(np.sum)
        gby = 100 * gby / gby.sum()
        gby = gby.sort_values(ascending=False)
        out = ', '.join(['%s (%2.1f%%)' % (idx, v) for idx,v in gby.iteritems()][:N])
        return out
    
    split = []
    for resi, res_row in res_df.iterrows():
        if summ_within:
            m = clone_df.iloc[res_row['members_within_i']]
        else:
            m = clone_df.iloc[res_row['members_i']]

        mode_i = m[count_col].idxmax()
        summ = {}
        for c in [c for c in clone_df.columns if 'cdr3' in c]:
            summ[c] = _top_N_str(m, c, count_col, 1)
        for c in [c for c in clone_df.columns if 'gene' in c]:
            summ[c] = _top_N_str(m, c, count_col, 3)

        x_val_cols = [c for c in res_df.columns if 'x_val_' in c]
        x_freq_cols = [c for c in res_df.columns if 'x_freq_' in c]
        
        for label_col, freq_col in zip(x_val_cols, x_freq_cols):
            summ[res_row[label_col]] = np.round(res_row[freq_col], 3)

        for c in [c for c in addl_cols]:
            summ[c] = _top_N_str(m, col, count_col, summ_n)
        summ = pd.Series(summ, name=resi)
        split.append(summ)
    summ = pd.DataFrame(split)
    return summ

def _prep_counts(cdf, xcols, ycol='NBR', count_col=None):
    if count_col is None:
        cdf = cdf.assign(Count=1)
        count_col = 'Count'
    counts = cdf.groupby(xcols + [ycol], sort=True)[count_col].agg(np.sum).unstack(ycol).fillna(0)
    for i in [0, 1]:
        if not i in counts.columns:
            counts.loc[:, i] = np.zeros(counts.shape[0])

    counts = counts[[0, 1]]
    return counts

def _chi2NBR(res_df, count_cols):
    """Applies a chi2 test to every row of res_df using the columns provided
    in count_cols. For each row, the vector of counts in count_cols can
    be reshaped into a 2 x n table for providing to scipy.stats.chi2_contingency

    Parameters
    ----------
    res_df : pd.DataFrame [ntests x 2*nsquares of a contingency table]
        Each row contains a set of counts to be tested.
    count_cols : list
        Columns containing the counts in a "flattened" order such that
        it can be reshaped into a 2 x n contingency table

    Returns
    -------
    res : dict
        A dict of two numpy vectors containing the chisq statistic and associated
        p-value for each test. Vectors will have length ntests, same as res_df.shape[0]"""
    res = {'chisq':np.nan * np.zeros(res_df.shape[0]),
            'pvalue':np.nan * np.zeros(res_df.shape[0])}
    for i in range(res_df.shape[0]):
        tab = res_df[count_cols].iloc[i].values.reshape((len(count_cols) // 2, 2))
        """Squeeze out rows where there were no instances inside or outside the cluster
        (happens with test_within)"""
        both_zero_ind = np.all(tab==0, axis=1)
        tab = tab[~both_zero_ind, :]
        try:
            res['chisq'][i], res['pvalue'][i], dof, expected = chi2_contingency(tab)
        except ValueError:
            res['chisq'][i], res['pvalue'][i] = np.nan, np.nan
    return res

def _CMH_NBR(counts, continuity_correction=True):
    """Applies a Cochran-Mantel-Haenszel test to a set of 2 x 2 tables,
    where each table in the set has cluster membership as one variable
    and the first x_col as the other variable (requires that x_cols[0] is binary).
    Each table in the set is from a different strata defined by the categorical
    variables in x_cols[1:]. This test only applies to tests pf more than one variable.

    Parameters
    ----------
    counts : pd.DataFrame
        Generated by _prep_counts(), it has columns 0 and 1 for cluster membership
        and rows for all combinations of values in x_cols (effectively a result of
        grouping-by xcols)
    continuity_correction : bool
        Whether to use a continuity correct in the CMH test.

    Returns
    -------
    out : dict
        Results from the test including a pooled OR, RR and pvalues
        for testing the overall null of OR = 1 for x_cols[0] and
        null of OR_1 = OR_2 = OR_3 ... (Breslow-Day test of homogeneity)"""
    tables = []
    for i, gby in counts.groupby(level=counts.index.names[1:]):
        if gby.shape == (2, 2):
            tables.append(gby.values)

    st = sm.stats.StratifiedTable(tables)
    out = {'equal_odds_pvalue':st.test_equal_odds().pvalue,
           'null_odds_pvalue':st.test_null_odds(correction=continuity_correction).pvalue,
           'OR_pooled':st.oddsratio_pooled,
           'RR_pooled':st.riskratio_pooled}
    return out

def _chi2_fishersNBR(counts):
    """Applies a Chi2 test to a 2 x n contingency table to identify
    a deviation from the null hypothesis of observed = expected frequencies
    inside and outside the cluster. Then applies a series of Fisher's exact
    tests to all the 2 x 2 combinations of tables to identify which variables
    are associated with cluster membership. Each table has cluster membership
    as one variable and one or more variables from x_cols as the others.

    Parameters
    ----------
    counts : pd.DataFrame
        Generated by _prep_counts(), it has columns 0 and 1 for cluster membership
        and rows for all combinations of values in x_cols (effectively a result of
        grouping-by xcols)

    Returns
    -------
    out : dict
        Results from the tests including: chi2 and pvalue for the Chi2 test
        and RR, OR and a pvalue for each Fisher's exact test."""
    labels = []
    for rowi in counts.index.tolist():
        if type(rowi) is tuple:
            labels.append('|'.join(rowi))
        else:
            labels.append(rowi)
    res = {}
    res['chisq'], res['pvalue'], dof, expected = chi2_contingency(counts.values)
    for rowi, rowj in itertools.combinations(range(counts.shape[0]), 2):
        lab = '%s vs %s' % (labels[rowi], labels[rowj])
        # OR = ((a/b) / (c/d)) or a*d/b*c
        """It is assumed here that the number clones in the neighborhood is in col_j = 1"""
        OR, p = fishersapi.fishers_vec(counts.iloc[rowi, 1],
                                       counts.iloc[rowi, 0],
                                       counts.iloc[rowj, 1],
                                       counts.iloc[rowj, 0],
                                       alternative='two-sided')
        RR = (counts.iloc[rowi, 1] / (counts.iloc[rowi, 1] + counts.iloc[rowi, 0])) / (counts.iloc[rowj, 1] / (counts.iloc[rowj, 1] + counts.iloc[rowj, 0]))
        res.update({'RR %s' % lab: RR,
                    'OR %s' % lab: OR,
                    'pvalue %s' % lab: p})
    return res

def _fisherNBR(res_df, count_cols):
    """Applies a Fisher's exact test to every row of res_df using the 4 columns provided
    in count_cols. For each row, the vector of counts in count_cols can
    be reshaped into a 2 x 2 contingency table.

    Parameters
    ----------
    res_df : pd.DataFrame [ntests x 4]
        Each row contains a set of 4 counts to be tested.
    count_cols : list
        Columns containing the counts in a "flattened" order such that
        it can be reshaped into a 2 x 2 contingency table

    Returns
    -------
    res : dict
        A dict of three numpy vectors containing the OR, the RR and the p-value.
        Vectors will have length ntests, same as res_df.shape[0]"""
    a = res_df[count_cols[0]].values
    b = res_df[count_cols[1]].values
    c = res_df[count_cols[2]].values
    d = res_df[count_cols[3]].values

    OR, p = fishersapi.fishers_vec(a, b, c, d, alternative='two-sided')
    """It is assumed here that the number clones in the neighborhood is in col_j = 1 (i.e. b, d)"""
    RR = (b / (a + b)) / (d / (c + d))
    return {'RR':RR, 'OR':OR, 'pvalue':p}

def _glmCatNBR(df, x_cols, y_col='NBR', count_col=None, l2_alpha=0, nperms=100):
    """Applies a logisitic regression with cluster membership as the outcome and
    other variables in x_cols as predictors. The major advantage of this method
    is the ability to test for multiple associations simultaneously while adjusting
    for covariates. The major problem with this method is that if the cluster
    is small and/or covariates are sparsely populated from the data then the
    model will not properly converge or may be a "perfect seperation".
    Worse, it may appear to converge but will not produce valid
    confidence intervals. Penalized regression is one way to handle these
    circumstances, however valid inference is a challenge and its not obvious
    how best to decide on the magnitude of the penalty. Bootstrapping is one way
    to compute a valid p-value, but it is slow unless parallelized. Therefore,
    this is a work in progress.

    Parameters
    ----------
    df : pd.DataFrame [ncells x ycols and xcols]
        A raw data matrix to be used in logistic regression.
    x_cols : list
        Predictors in the regression model.
    y_col : str
        Column in df to be used as the regression outcome,
        typically NBR representing cluster membership [1, 0]
    count_col : str
        Optionally provide weights for each covariate/outcome combination
        instead of having one observation per row.
    l2_alpha : float
        Magnitude of the L2-penalty
    nperms : int
        Number of permutations for the permutation test that is required by
        penalized regression to get a p-value

    Returns
    -------
    out : dict
        A dict of numpy vectors, one value per parameter providing: OR, coef, pvalue"""
    if count_col is None:
        freq_weights = None
    else:
        freq_weights = df[count_col]

    formula = ' + '.join(['C(%s)' % c for c in x_cols])
    X = patsy.dmatrix(formula, df, return_type='dataframe')
    glmParams = dict(exog=X,
                     family=sm.families.Binomial(link=sm.families.links.logit),
                     freq_weights=freq_weights,
                     hasconst=True)
    mod = sm.GLM(endog=df[y_col].values, **glmParams)
    if l2_alpha == 0:
        res = mod.fit()
        out = {'%s_pvalue' % c:res.pvalues[c] for c in X.columns if not 'Intercept' in c}
    else:
        res = mod.fit_regularized(L1_wt=0, alpha=l2_alpha)
        rparams = np.zeros((len(res.params), nperms))
        for permi in range(nperms):
            randy = df[y_col].sample(frac=1, replace=False).values
            rres = sm.GLM(endog=randy, **glmParams).fit_regularized(L1_wt=0, alpha=l2_alpha)
            rparams[:, permi] = rres.params

        perm_values = ((np.abs(res.params[:, None]) < np.abs(rparams)).sum(axis=1) + 1) / (nperms + 1)
        out = {'%s_pvalue' % c:v for c,v in zip(X.columns, perm_values) if not 'Intercept' in c}

    out.update({'%s_coef' % c:res.params[c] for c in X.columns if not 'Intercept' in c})
    out.update({'%s_OR' % c:np.exp(res.params[c]) for c in X.columns if not 'Intercept' in c})
    return out