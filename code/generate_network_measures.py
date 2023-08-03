import os
import threading

import random
import numpy as np
import pandas as pd
import networkx as nx
import argparse
from multiprocessing import Pool


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs



def square_rooted(x):
    return round(np.sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)



def single_network(wdirs,p_id,fp_class,SimMatrix,threshold,n):
    g = np.array(SimMatrix>=threshold,dtype=int)
    row, col = np.diag_indices_from(g)
    g[row, col] = 0
    G = nx.from_numpy_array(g)

    ########################################################
    ### generate network parameter metrics #################
    ########################################################

    ### Centrality in networks
    #3.7.1 Degree
    dc = nx.degree_centrality(G)
    dc = dict(sorted(dc.items()))

#     #3.7.2 Eigenvector
    try:
        eic = nx.eigenvector_centrality(G,max_iter=100)
        eic = dict(sorted(eic.items()))
    except:
        eic = dict(zip(range(n),np.repeat(np.nan,n)))
    #3.7.3 Closeness
    cc = nx.closeness_centrality(G)
    cc = dict(sorted(cc.items()))


    #3.7.5 (Shortest Path) Betweenness
    bc = nx.betweenness_centrality(G)
    bc = dict(sorted(bc.items()))

    #3.7.9 Load
    lc = nx.load_centrality(G)
    lc = dict(sorted(lc.items()))

    #3.7.10 Subgraph
    # sc = nx.subgraph_centrality(G)
    # sc = dict(sorted(sc.items()))

    #3.7.11 Harmonic Centrality
    hc = nx.harmonic_centrality(G)
    hc = dict(sorted(hc.items()))

    # k-core
    cn = nx.core_number(G)
    cn = dict(sorted(cn.items()))

    
    ol = nx.onion_layers(G)
    ol = dict(sorted(ol.items()))

    try:
        pr = nx.pagerank(G)
        pr = dict(sorted(pr.items()))
    except:
        pr = dict(zip(range(n),np.repeat(np.nan,n)))

    
    #clustering for each nodes
    clu = nx.clustering(G)
    clu = dict(sorted(clu.items()))
    
    #################################
    #network info
    #desity for the whole network
    density = nx.density(G)
    d_density = dict(zip(range(n),np.repeat(density,n)))

    ########################################################
    ### generate network info matrix #######################
    ########################################################

    network_matrix = pd.DataFrame({f'{fp_class}_{str(int(threshold*100))}_DC':dc,
                                   f'{fp_class}_{str(int(threshold*100))}_EIC':eic,
                                   f'{fp_class}_{str(int(threshold*100))}_CC':cc,
                                   f'{fp_class}_{str(int(threshold*100))}_BC':bc,
                                   f'{fp_class}_{str(int(threshold*100))}_LC':lc,
                                   f'{fp_class}_{str(int(threshold*100))}_HC':hc,
                                   f'{fp_class}_{str(int(threshold*100))}_CN':cn,
                                   f'{fp_class}_{str(int(threshold*100))}_OL':ol,
                                   f'{fp_class}_{str(int(threshold*100))}_PR':pr,
                                   f'{fp_class}_{str(int(threshold*100))}_Clustering':clu,
                                   f'{fp_class}_{str(int(threshold*100))}_Density':d_density,
                                   })

    network_matrix.to_csv(f'{wdirs}/networkinfo_{p_id}_{str(int(threshold*100))}.csv',index=False)
    return network_matrix

def single_patent_network_generation(wdirs,p_id,fps,fp_class,tlist):

    ########################################################
    ### generate similarity matrix #########################
    ########################################################
    n = len(fps)

    similarity_matrix = []

    if fp_class == 'mol2vec':
        for i in range(n):
            tsims = [cosine_similarity(fps[i], fps[j]) for j in range(n)]
            similarity_matrix.append(tsims)
    else:
        fpstr = [list(map(lambda x:str(x),np.int64(f))) for f in fps]
        fpstring = [''.join(f) for f in fpstr]
        fps_rdkit = [cDataStructs.CreateFromBitString(f) for f in fpstring]

        for i in range(n):
            tsims = Chem.DataStructs.BulkTanimotoSimilarity(fps_rdkit[i], fps_rdkit)
            similarity_matrix.append(tsims)

    SimMatrix = pd.DataFrame(similarity_matrix)

    ########################################################
    ### generate graph of network ##########################
    ########################################################

    NetworkMatrix = pd.DataFrame()
    for t in tlist:
        print(t)
        flist = os.listdir(wdirs)
        if f'networkinfo_{p_id}_{str(int(t*100))}.csv' in flist:
            network_matrix = pd.read_csv(f'{wdirs}/networkinfo_{p_id}_{str(int(t*100))}.csv')
        else:
            try:
                network_matrix = single_network(wdirs,p_id,fp_class,SimMatrix,t,n)
            except Exception as e:
                print(e.args)
                print(str(e))
                print(repr(e))
        NetworkMatrix = pd.concat([NetworkMatrix,network_matrix],axis=1)

    network_matrix = pd.DataFrame()
    SimMatrix = pd.DataFrame()
    return NetworkMatrix


def Network_Generation(wdirs,df,fp_class,tlist,plist):

    df_aimed = df.copy()
    df = pd.DataFrame()
    p_id = df_aimed.PATENT_ID[0]

    flag = True
    if len(plist) == 0:
        pass
    else:
        for i in plist:
            if p_id in i:
                flag = False
                break
            else:
                pass

    print(p_id,df_aimed.shape,flag)

    if flag == True:
        try:
            print(p_id,df_aimed.shape)
            df_aimed_fps = df_aimed.drop(columns=['PATENT_ID','P_Ca_SMILES','Target'])
            
            fps = np.array(df_aimed_fps)
            patent_id = df_aimed.PATENT_ID
            smis = df_aimed.P_Ca_SMILES
            target = df_aimed.Target
            df_aimed_fps = pd.DataFrame()

            print('start')
            NetworkMatrix = single_patent_network_generation(wdirs=wdirs,p_id=p_id,fps=fps,fp_class=fp_class,tlist=tlist)
            df_aimed_network = NetworkMatrix.copy()
            df_aimed_network.insert(0,'PATENT_ID',patent_id)
            df_aimed_network.insert(1,'P_Ca_SMILES',smis)
            df_aimed_network.insert(2,'Target',target)
            df_aimed_network.to_csv(f'{wdirs}/networkinfo_{p_id}.csv',index=False)
            for t in tlist:
                os.remove(f'{wdirs}/networkinfo_{p_id}_{str(int(t*100))}.csv')

            NetworkMatrix = pd.DataFrame()
            print(f'New:{p_id}')
        except Exception as e:
            print(e.args)
            print(str(e))
    else:
        print('Exist')
        pass
    return df_aimed_network



def main():

    parser = argparse.ArgumentParser(description='network construction')
    parser.add_argument('--p',default=4,type=int,help="number of pools to use")
    parser.add_argument('--fp_class',default='ecfp4',type=str,help="fingerprint type")
    # parser.add_argument('--threshold',default=0.7,type=float,help="threshold for network construction")
    
    args = parser.parse_args()
    pool = args.p
    fp_class = args.fp_class
    f_class = [fp_class]

    path = '../'
    for f in f_class:
        ### generate network info for each fp with different threshold
        wdirs = path+ f'results/network_{f}'
        if os.path.exists(f'{wdirs}'):
            ll = os.listdir(wdirs)
            plist = []
            for l in ll:
                df_temp = pd.read_csv(f'{wdirs}/{l}')
                c = 0
                try:
                    plist_temp = df_temp.PATENT_ID.unique().tolist()
                    plist.extend(plist_temp)
                except:
                    pass
            print(len(plist))
        else:
            plist = []
            os.makedirs(f'{wdirs}')

        ### data for each fingerprint
        df_total = pd.read_csv(f'../data/fp/{f}.csv')
        ff_df = []
        c = 0
        for patent in df_total.PATENT_ID.unique():
            if patent not in plist:
                df_temp = df_total[df_total.PATENT_ID == patent].reset_index(drop=True)
                ff_df.append(df_temp)
                c +=1

        df_total = pd.DataFrame()

        p = Pool(pool)
        tlist = np.linspace(0.4,0.9,11)  # threshold for each graph/network
        results = []
        for f_df in ff_df:
            try:
                r_ = p.apply_async(Network_Generation,args=(wdirs,f_df,f,tlist,plist))
            except Exception as e:
                print(e.args)
                print(str(e))
                print(repr(e))
            results.append(r_)
        p.close()
        p.join()

        print('start')
        l3fp = os.listdir(f'../data/fp')

        print('start combine')
        if f'networkinfo_{f}.csv' in l3fp:
            break
        else:
            ll = os.listdir(wdirs)
            df_all = pd.DataFrame()
            c = 0
            for l in ll:
                if 'networkinfo_' in l:
                    c +=1
                    print(c)
                    df_temp = pd.read_csv(f'{wdirs}/{l}')
                    df_temp['PATENT_NUM'] = np.repeat(df_temp.shape[0],df_temp.shape[0])
                    df_all = pd.concat([df_all,df_temp]).reset_index(drop=True)
                else:
                    pass
            df_all.to_csv(f'../data/fp/networkinfo_{f}.csv',index=False)
            df_all =  pd.DataFrame()


if __name__ == "__main__":
    main()