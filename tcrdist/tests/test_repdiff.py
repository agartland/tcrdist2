import unittest
import os.path as op
import inspect
import pandas as pd
import numpy as np
import warnings

import tcrdist as td
from tcrdist.repertoire import TCRrep

class test_stats(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        filename = op.join(td.__path__[0], 'datasets', 'vdjDB_PMID28636592.tsv')
        pd_df = pd.read_csv(filename, sep='\t')
        t_df = td.mappers.vdjdb_to_tcrdist2(pd_df=pd_df)

        t_df = t_df.loc[(t_df.organism == 'HomoSapiens') & (t_df.epitope == 'M1')]

        tr = TCRrep(cell_df=t_df, organism='human')
        tr.infer_cdrs_from_v_gene(chain='alpha')
        tr.infer_cdrs_from_v_gene(chain='beta')
        tr.index_cols =['subject',
                        'cdr3_b_aa']
        tr.deduplicate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr.compute_pairwise_all(chain='beta', metric='nw', proceses=1)
        self.pw = tr.cdr3_b_aa_pw

        np.random.seed(110820)
        self.clone_df = tr.clone_df.assign(Visit=np.random.choice(['Pre', 'Post'], size=tr.clone_df.shape[0], p=[0.4, 0.6]),
                                            Stim=np.random.choice(['A', 'B', 'C'], size=tr.clone_df.shape[0], p=[0.4, 0.1, 0.5]))

    def test_chm_NN(self):
        res = td.stats.neighborhoodDiff(self.clone_df, self.pw, x_cols=['Visit', 'Stim'], test='chm')
        self.assertTrue(res.shape[0] == self.clone_df.shape[0])

    def test_fishers_NN(self):
        res = td.stats.neighborhoodDiff(self.clone_df, self.pw, x_cols=['Visit'], test='fishers')
        self.assertTrue(res.shape[0] == self.clone_df.shape[0])

    def test_chi2_NN(self):
        res = td.stats.neighborhoodDiff(self.clone_df, self.pw, x_cols=['Visit'], test='chi2')
        res = td.stats.neighborhoodDiff(self.clone_df, self.pw, x_cols=['Visit'], test='chi2+fishers')
        self.assertTrue(res.shape[0] == self.clone_df.shape[0])

    def test_fishers_HC(self):
        res = td.stats.hclusterDiff(self.clone_df, self.pw, x_cols=['Visit'], test='fishers')

def _generate_peptide_data(L=5, n=1000, seed=110820):
    """Attempt to generate some random peptide data with a
    phenotype enrichment associated with a motif"""
    np.random.seed(seed)
    alphabet = 'ARNDCQEGHILKMFPSTWYVBZ'
    probs = np.random.rand(len(alphabet))
    probs = probs / np.sum(probs)

    seqs = [''.join(np.random.choice(list(alphabet), size=5, p=probs)) for i in range(n)]
    """
    def _assign_trait(seq):
        if seq[1] in 'KRQ' or seq[3] in 'KRQ':
            pr = 0.99
        elif seq[0] in 'QA':
            pr = 0.01
        else:
            pr = 0.03
        return np.random.choice([1, 0], p=[pr, 1-pr])
    """
    def _assign_trait(seq):
        d = np.sum([i for i in map(operator.__ne__, seq, seqs[0])])
        return int(d <= 2)
    trait = np.array([_assign_trait(p) for p in seqs])

    pw = pwsd.apply_pairwise_sq(seqs, metric=pwsd.metrics.hamming_distance)

    Z = sch.linkage(pw, method='complete')
    labels = sch.fcluster(Z, 50, criterion='maxclust')

    dat = pd.DataFrame({'seq':seqs,
                        'trait':trait,
                        'cluster':labels,
                        'count':np.random.randint(4, 10, size=n)})
    return dat, pw

class TestHierTest(unittest.TestCase):

    def test_hierdiff_test(self):
        dat, pw = _generate_peptide_data()
        res = hierdiff.neighborhood_diff(dat,
                          pwmat=scipy.spatial.distance.squareform(pw),
                          x_cols=['trait'],
                          count_col='count',
                          test='fishers',
                          knn_neighbors=None, knn_radius=3)
        res = dat.join(res)
        self.assertTrue(res.shape[0] == dat.shape[0])
    
if __name__ == '__main__':
    unittest.main()
