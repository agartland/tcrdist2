import pandas as pd
import sys
import numpy as np
import logging

logger = logging.getLogger('find_clones.py')

from . import util
from . import logo_tools

__all__ = ['findClones']

def getAllTCRs(psDf):
    """
    Parameters
    ----------
    psDF : pd.DataFrame

    Returns
    -------

    all_tcrs : dict
        a mess of a dictionary of dictionaries keyed like as follows
        key0: (epitope, subject)
            key1: ('TRAV35*01', 'TRAJ42*01', 'TRBV12-3*01', 'TRBJ1-2*01', 'tgtgctgggcaagcaagccaaggaaatctcatcttt', 'tgtgccagcagtatacaggccctattgaccttc')
                list[{key3}]
        {tuple:{ hexduple: [[{}],[{}],  ] } }


        [('pp65', 'human_subject0010')]
            [('TRAV35*01', 'TRAJ42*01', 'TRBV12-3*01', 'TRBJ1-2*01', 'tgtgctgggcaagcaagccaaggaaatctcatcttt', 'tgtgccagcagtatacaggccctattgaccttc')]

        dict_keys(['id', 'epitope', 'subject', 'va_gene', 'va_rep', 'va_mm', 'ja_gene', 'ja_rep', 'ja_mm', 'cdr3a_plus', 'va_evalue', 'ja_evalue', 'va_bitscore_gap', 'ja_bitscore_gap', 'a_status', 'a_good_hits', 'cdr3a', 'cdr3a_nucseq', 'cdr3a_quals', 'va_mismatches', 'ja_mismatches', 'va_alignlen', 'ja_alignlen', 'va_blast_hits', 'ja_blast_hits', 'va_genes', 'ja_genes', 'va_reps', 'ja_reps', 'va_countreps', 'ja_countreps', 'vb_gene', 'vb_rep', 'vb_mm', 'jb_gene', 'jb_rep', 'jb_mm', 'cdr3b_plus', 'vb_evalue', 'jb_evalue', 'vb_bitscore_gap', 'jb_bitscore_gap', 'b_status', 'b_good_hits', 'cdr3b', 'cdr3b_nucseq', 'cdr3b_quals', 'vb_mismatches', 'jb_mismatches', 'vb_alignlen', 'jb_alignlen', 'vb_blast_hits', 'jb_blast_hits', 'vb_genes', 'jb_genes', 'vb_reps', 'jb_reps', 'vb_countreps', 'jb_countreps', 'organism', 'TCRID', 'a_indels', 'a_nucseq_prob', 'a_protseq_prob', 'b_indels', 'b_nucseq_prob', 'b_protseq_prob', 'cdr3a_new_nucseq', 'cdr3a_protseq_masked', 'cdr3a_protseq_prob', 'cdr3b_new_nucseq', 'cdr3b_protseq_masked', 'cdr3b_protseq_prob', 'ja_rep_prob', 'jb_rep_prob', 'va_rep_prob', 'vb_rep_prob', 'cdr3a_min_qual', 'cdr3b_min_qual', 'cdr3_min_qual'])



    Notes
    -----
    one of the reason this code is so cryptic is objects returned by functions like this!!!!!
    all_tcrs could contain objects with attributed

    """

    all_tcrs = {}
    for rowi, row in psDf.iterrows():
        l = row.to_dict()

        epitope = l['epitope']
        mouse = l['subject']
        va_gene = l['va_gene']
        ja_gene = l['ja_gene']
        cdr3a_nucseq = l['cdr3a_nucseq']

        vb_gene = l['vb_gene']
        jb_gene = l['jb_gene']
        cdr3b_nucseq = l['cdr3b_nucseq']

        l['cdr3a_min_qual'] = min( [int(x) for x in l['cdr3a_quals'].split('.') ] )
        l['cdr3b_min_qual'] = min( [int(x) for x in l['cdr3b_quals'].split('.') ] )
        l['cdr3_min_qual'] = min( l['cdr3a_min_qual'], l['cdr3b_min_qual'] )

        genesets = []
        for ab in 'ab':
            for vj in 'vj':
                genesets.append( set( l[vj+ab+'_genes'].split(';')))

        em = (epitope, mouse)

        if em not in all_tcrs:
            all_tcrs[em] = {}

        tcrseq = (va_gene, ja_gene, vb_gene, jb_gene, cdr3a_nucseq, cdr3b_nucseq)

        if tcrseq not in all_tcrs[em]:
            all_tcrs[em][tcrseq] = []

        """Note that the entire row "l" is included in this object"""
        all_tcrs[em][tcrseq].append( [l, genesets] )
    return all_tcrs

def count_mismatches( a, b):
    assert len(a) == len(b)
    mismatches =0
    for x, y in zip(a, b):
        if not logo_tools.nucleotide_symbols_match(x, y):
            mismatches += 1
    return mismatches

def get_common_genes( tcrs ):
    """
    Parameters
    ----------
    tcrs :

    Returns
    -------
    all_genesets

    Notes
    -----
    Causing trouble with parasail: attempting to document
    """
    all_genesets = []
    first_genesets = tcrs[0][1]
    for ii in range(4):
        genes = []
        for g in first_genesets[ii]:
            allfound=True
            for (l, genesets) in tcrs:
                if g not in genesets[ii]:
                    allfound=False
            if allfound:
                genes.append( g )
        all_genesets.append( set(genes) )
    return all_genesets

def findClones(psDf, min_quality_for_singletons=20, average_clone_scores=False, none_score_for_averaging=9.6):
    segtypes_lowercase = ['va', 'ja', 'vb', 'jb']
    organism = psDf.organism.iloc[0]

    all_tcrs = getAllTCRs(psDf)

    total_clones, skipcount = (0, 0)
    for em in all_tcrs:
        nbrs = {}
        for t1 in all_tcrs[em]:
            nbrs[t1] = [t1]

        quals={}
        for t1 in all_tcrs[em]:
            tcrs1 = all_tcrs[em][t1]

            qa1 = max( [x[0][ 'cdr3a_min_qual' ] for x in tcrs1 ] )
            qb1 = max( [x[0][ 'cdr3b_min_qual' ] for x in tcrs1 ] )

            quals[t1] = (qa1, qb1)

            cdr3a_new_nucseqs = list( { x[0]['cdr3a_new_nucseq'] for x in tcrs1 } )
            cdr3b_new_nucseqs = list( { x[0]['cdr3b_new_nucseq'] for x in tcrs1 } )

            cdr3_nucseq_prob1  = float( tcrs1[0][0][ 'a_nucseq_prob'  ] ) * float( tcrs1[0][0][ 'b_nucseq_prob' ] )
            cdr3_protseq_prob1 = float( tcrs1[0][0][ 'a_protseq_prob' ] ) * float( tcrs1[0][0][ 'b_protseq_prob' ] )

            cdr3_nucseq_prob1  = int( np.log10( cdr3_nucseq_prob1  ) ) if cdr3_nucseq_prob1 >0 else -99
            cdr3_protseq_prob1 = int( np.log10( cdr3_protseq_prob1 ) ) if cdr3_protseq_prob1>0 else -99

            assert len(cdr3a_new_nucseqs)==1
            assert len(cdr3b_new_nucseqs)==1

            cdr3a_new_nucseq1 = cdr3a_new_nucseqs[0]
            cdr3b_new_nucseq1 = cdr3b_new_nucseqs[0]

            ## what genes are present for all of this guys hits?
            genesets1 = get_common_genes( tcrs1 )

            for ii in range(4):
                assert t1[ii] in genesets1[ii]

            ## look for nearby guys
            for t2 in all_tcrs[em]:
                if t2<=t1:continue

                tcrs2 = all_tcrs[em][t2]
                genesets2 = get_common_genes( tcrs2 )

                qa2 = max( [x[0][ 'cdr3a_min_qual' ] for x in tcrs2 ] )
                qb2 = max( [x[0][ 'cdr3b_min_qual' ] for x in tcrs2 ] )

                cdr3a_new_nucseq2 = list( { x[0]['cdr3a_new_nucseq'] for x in tcrs2 } )[0]
                cdr3b_new_nucseq2 = list( { x[0]['cdr3b_new_nucseq'] for x in tcrs2 } )[0]

                cdr3_nucseq_prob2  = float( tcrs2[0][0][ 'a_nucseq_prob'  ] ) * float( tcrs2[0][0][ 'b_nucseq_prob' ] )
                cdr3_protseq_prob2 = float( tcrs2[0][0][ 'a_protseq_prob' ] ) * float( tcrs2[0][0][ 'b_protseq_prob' ] )

                cdr3_nucseq_prob2  = int( np.log10( cdr3_nucseq_prob2  ) ) if cdr3_nucseq_prob2 >0 else -99
                cdr3_protseq_prob2 = int( np.log10( cdr3_protseq_prob2 ) ) if cdr3_protseq_prob2>0 else -99

                mismatches=0
                samelen = True
                for ii in [4, 5]:
                    if len(t1[ii] ) != len(t2[ii] ):
                        samelen = False
                        break
                    mismatches += count_mismatches( t1[ii], t2[ii] )

                s1, s2 = ( cdr3a_new_nucseq1 + ' ' + cdr3b_new_nucseq1,
                          cdr3a_new_nucseq2 + ' ' + cdr3b_new_nucseq2 )
                new_mismatches = count_mismatches(s1, s2) if len(s1)==len(s2) else 9

                common_genes = []
                for ii, genes1 in enumerate( genesets1 ):
                    genes=[]
                    for g in genes1:
                        if g in genesets2[ii]:
                            genes.append( g )
                    common_genes.append( genes )
                clones_have_common_genes = ( sum( [len(x)>0 for x in common_genes ] )==4 )


                if samelen and mismatches<3 and clones_have_common_genes:
                    logger.debug('close by1: {:2d} {:2d} {} {} {:2d} {} {} {} {}'.format( qa1,
                                                                                         qb1,
                                                                                         mismatches,
                                                                                         new_mismatches,
                                                                                         len(all_tcrs[em][t1]),
                                                                                         cdr3_nucseq_prob1,
                                                                                         cdr3_protseq_prob1,
                                                                                         s1,
                                                                                         ' '.join(t1[:4]) ))
                    logger.debug('close by2: {:2d} {:2d} {} {} {:2d} {} {} {} {}'.format( qa2,
                                                                                          qb2,
                                                                                         mismatches,
                                                                                         new_mismatches,
                                                                                         len(all_tcrs[em][t2]),
                                                                                         cdr3_nucseq_prob2,
                                                                                         cdr3_protseq_prob2,
                                                                                         s2,
                                                                                         ' '.join(t2[:4]) ))

                    ## plan: only merge if one gene is perfect
                    if t1[4] == t2[4] or t1[5] == t2[5]:
                        mmgene = 'A' if t1[4] != t2[4] else 'B'
                        q1 = qa1 if mmgene=='A' else qb1
                        q2 = qa2 if mmgene=='A' else qb2
                        new_nucseq1 = cdr3a_new_nucseq1 if mmgene=='A' else cdr3b_new_nucseq1
                        new_nucseq2 = cdr3a_new_nucseq2 if mmgene=='A' else cdr3b_new_nucseq2

                        minq_size = len(all_tcrs[em][t1]) if q1<q2 else len(all_tcrs[em][t2])
                        min_size = min( len(all_tcrs[em][t1]), len(all_tcrs[em][t2]) )


                        tmp = 'merge1: {:2d} {} {} {:2d} {} {} {}'\
                                 .format( q1, mismatches, new_mismatches, len(all_tcrs[em][t1]),
                                         cdr3_nucseq_prob1, new_nucseq1, ' '.join(t1[:4]) )
                        logger.debug(tmp)
                        tmp = 'merge2: {:2d} {} {} {:2d} {} {} {}'\
                                 .format( q2, mismatches, new_mismatches, len(all_tcrs[em][t2]),
                                         cdr3_nucseq_prob2, new_nucseq2, ' '.join(t2[:4]) )
                        logger.debug(tmp)


                        do_merge = ( mismatches==0 or
                                     ( mismatches==1 and min(q1, q2) < 20 and minq_size ==1 and
                                       ( new_mismatches<=1 or min(cdr3_nucseq_prob1, cdr3_nucseq_prob2)<-15 ) ) )

                        if do_merge:
                            logger.info('domerge1: {:2d} {} {} {:2d} {} {} {} {} {}'\
                                .format( q1, mismatches, new_mismatches, len(all_tcrs[em][t1]),
                                         cdr3_nucseq_prob1, new_nucseq1, em[0], em[1], ' '.join(t1[:4]) ))
                            logger.info('domerge2: {:2d} {} {} {:2d} {} {} {} {} {}'\
                                .format( q2, mismatches, new_mismatches, len(all_tcrs[em][t2]),
                                         cdr3_nucseq_prob2, new_nucseq2, em[0], em[1], ' '.join(t2[:4]) ))

                            ## which should we take
                            nbrs[ t1 ].append( t2 )
                            nbrs[ t2 ].append( t1 )


        seen = []
        outrows = []
        for t1 in all_tcrs[em]:
            if t1 in seen: continue

            ## get the big set-- single linkage...
            all_nbrs = [t1]
            while True:
                old = all_nbrs[:]
                new = all_nbrs[:]
                for t in old:
                    for nbr in nbrs[t]:
                        if nbr not in new:
                            new.append(nbr)
                all_nbrs = new[:]
                if len(new) == len(old):
                    break

            ## now which one should be the representative?
            ## now we are doing quality filtering here
            sizel = []
            clone_size = 0
            members = []
            member_tcrs = []
            for t in all_nbrs:
                size = len(all_tcrs[em][t])
                clone_size += size
                sizel.append( ( size, min(quals[t]), t ) )
                members.extend( [x[0]['id'] for x in all_tcrs[em][t] ] )
                member_tcrs.extend( all_tcrs[em][t] )
                assert t not in seen

            sizel.sort()
            sizel.reverse()
            assert len(members) == clone_size and len(member_tcrs) == clone_size


            if len(sizel)>1:
                logger.info('sizel: %s', [(x[0], x[1]) for x in sizel])

            aq, bq = quals[t1]

            if clone_size==1 and ( aq < min_quality_for_singletons or bq<min_quality_for_singletons ):
                logger.info('skipping singleton because min_quality lower than %s: %s %s %s' %(min_quality_for_singletonsaq, bq, t1[:4]))
                skipcount+=1
                continue

            trep = sizel[0][-1]
            if t1 != trep:
                logger.debug('nonrep: %s %s %s %s %s', aq, bq, t1[:4], 'rep:', trep[:4])
                continue

            ## ok, we are taking this guy as the rep, so mark all members as seen
            for t in all_nbrs:
                assert t not in seen
                seen.append( t )


            ## write out a new line
            l = all_tcrs[em][t1]

            outl = dict( l[0][0] )## copy

            clone_id = outl['id']+'.clone'
            #members = ';'.join( [ x[0]['id'] for x in l ] )

            outl['clone_id'] = clone_id
            del outl['id']

            outl['members'] = ';'.join(members)
            outl['clone_size'] = clone_size

            if average_clone_scores:
                for tsvtag in average_clone_scores:
                    scores = []
                    for t in all_nbrs:
                        for t_l, t_genesets in all_tcrs[em][t]:
                            score = float( t_l[tsvtag] )
                            if none_score_for_averaging == None or abs(score-none_score_for_averaging)>1e-3:
                                scores.append( score )
                    if none_score_for_averaging==None:
                        assert len(scores) == clone_size
                    if scores:
                        outl[tsvtag] = sum(scores) / len(scores)
                    else:
                        assert none_score_for_averaging != None
                        outl[tsvtag] = none_score_for_averaging

            genesets = get_common_genes( member_tcrs )

            for ii in range(4):
                if not genesets[ii]: ## whoah-- no overlap??
                    counts = {}
                    for (l, gsets) in member_tcrs:
                        for g in gsets[ii]:
                            counts[g] = counts.get(g, 0)+1
                    mx = max(counts.values())
                    genesets[ii] = { x for x, y in counts.items() if y==mx }
                    logger.info('empty common genes: %s, clone_size: %s, mx-genecount: %s, newgeneset: %s %s' % (ii, clone_size, mx, genesets[ii], em))

            for genes, segtype in zip( genesets, segtypes_lowercase ):
                assert genes
                tag = segtype+'_genes'
                assert tag in outl # should already be there, now over-writing
                outl[tag] = ';'.join(sorted(genes))

                ## update reps
                reps = sorted( set( ( util.get_rep(x, organism) for x in genes ) ) )
                tag = segtype+'_reps'
                assert tag in outl # should already be there, now over-writing
                outl[tag] = ';'.join(reps)

                ## update countreps
                countreps = sorted( set( ( util.get_mm1_rep_gene_for_counting(x, organism) for x in genes ) ) )
                tag = segtype+'_countreps'
                assert tag in outl # should already be there, now over-writing
                outl[tag] = ';'.join(countreps)

            total_clones += 1

            outrows.append(outl)

    clonesDf = pd.DataFrame(outrows)
    logger.info('skipcount: %s, total_clones: %s' % (skipcount, total_clones))
    return clonesDf
