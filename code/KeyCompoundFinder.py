import os,sys
import time
import numpy as np
import pandas as pd
import warnings
import argparse
import json


import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
from multiprocessing import Pool

import FindKeyCompound


def FindKC(wdirs,methods:list,params:list,df:pd.DataFrame(),plist=[]):
    ###############################################################
    # load data
    df = df.copy(True)
    patents = df.PATENT_ID.unique().tolist()
    total = len(patents)
    # prepare for later use
    df_result = pd.DataFrame()

    # for each patent find their key compound by different method
    name = patents[0]

    for pid in patents:

        dict_p_result = {}
        ###############################################################
        # load a single patent data
        df_p = df[df.PATENT_ID == pid].reset_index(drop=True)

        # prepare for the results file
        dict_p_result['DrugBank_ID'] = df_p.DrugBank_ID[0]
        dict_p_result['Name'] = df_p.Name[0]
        dict_p_result['Drug_Groups'] = df_p.Drug_Groups[0]
        dict_p_result['SMILES'] = df_p.SMILES[0]
        dict_p_result['Ca_SMILES'] = df_p.Ca_SMILES[0]
        dict_p_result['PATENT_ID'] = df_p.PATENT_ID[0]
        dict_p_result['HOMOGENEITY'] = df_p.HOMOGENEITY[0]
        dict_p_result['PCT_FILTERED'] = df_p.PCT_FILTERED[0]

        # candidate smiles
        candidate = df_p.Ca_SMILES[0]
        
        # clean smiles
        smiles_column_name = 'P_SMILES'
        standard_smiles_column_name = 'P_Ca_SMILES'

        ###############################################################
        # method1 csa
        if methods[0] == True:
            csa_param = params[0]
            fp_class = csa_param.get('fp_class')
            fp_r = csa_param.get('fp_r')
            fp_b = csa_param.get('fp_b')
            threshold = csa_param.get('threshold')

            rounds = 5

            df_csa_key,df_csa_res = FindKeyCompound.CSA(df_p,standard_smiles_column_name,fp_class=fp_class,fp_r=fp_r,fp_b=fp_b,threshold=threshold,rounds=rounds)
            df_csa_key.to_csv(f'{wdirs}/{pid}_csa_key.csv',index=False)
            df_csa_res.to_csv(f'{wdirs}/{pid}_csa_res.csv',index=False)

            stop = False
            n_key = df_csa_key.shape[0]
            n_res  = df_csa_res.shape[0]
            for i in range(n_key):
                if df_csa_key[standard_smiles_column_name][i] == candidate:
                    rank = i+1
                    score = df_csa_key.csa_score[i]
                    dict_p_result['CSA_in_Round5'] = 1
                    dict_p_result['CSA_Rank'] = rank
                    dict_p_result['CSA_Score'] = score
                    stop = True
                    break
                else:
                    pass

            if stop == False:
                for i in range(n_res):
                    if df_csa_res[standard_smiles_column_name][i] == candidate:
                        rank = i+n_key+1
                        score = df_csa_res.csa_score[i]
                        dict_p_result['CSA_in_Round5'] = 0
                        dict_p_result['CSA_Rank'] = rank
                        dict_p_result['CSA_Score'] = score
                        stop = True
                        break
                    else:
                        pass

            if stop == False:
                dict_p_result['CSA_in_Round5'] = 0
                dict_p_result['CSA_Rank'] = None
                dict_p_result['CSA_Score'] = None
            else:
                pass

            if dict_p_result.get('CSA_Rank') == None:
                dict_p_result['CSA_in_Top5'] = None
            elif dict_p_result.get('CSA_Rank') <= 5:
                dict_p_result['CSA_in_Top5'] = 1
            else:
                dict_p_result['CSA_in_Top5'] = 0

        else:
            pass

        ###############################################################
        # method2 midol
        try:
            if methods[1] == True:
                if pid not in plist:
                    midol_param = params[1]
                    timeout = midol_param.get('timeout')
                    threshold = midol_param.get('threshold')

                    df_midol = FindKeyCompound.MIdol(df_p,standard_smiles_column_name,
                                                     timeout=timeout,threshold=threshold)
                    df_midol.to_csv(f'{wdirs}/{pid}_midol.csv',index=False)
                else:
                    df_midol = pd.read_csv(f'{wdirs}/{pid}_midol.csv')

                stop = False
                for i in range(df_midol.shape[0]):
                    if df_midol[standard_smiles_column_name][i] == candidate:
                        rank = i+1
                        neighbor = df_midol.midol_neighbor[i]
                        score = df_midol.midol_score[i]

                        dict_p_result['MIdol_Rank'] = rank
                        dict_p_result['MIdol_Neighbor'] = neighbor
                        dict_p_result['MIdol_Score'] = score

                        stop = True
                        break
                    else:
                        pass
                if stop == False:
                    dict_p_result['MIdol_Rank'] = np.nan
                    dict_p_result['MIdol_Neighbor'] = np.nan
                    dict_p_result['MIdol_Score'] = np.nan

                if dict_p_result.get('MIdol_Rank') <= 5:
                    dict_p_result['MIdol_in_Top5'] = 1
                else:
                    dict_p_result['MIdol_in_Top5'] = 0

            else:
                pass
        except Exception as e:
            # 访问异常的错误编号和详细信息
            print(e.args)
            print(str(e))
            print(repr(e))
        ###############################################################
        # method3 fog auto
        if methods[2] ==True:

                fog_param = params[2]
                timeout = fog_param.get('timeout')
                cluster = fog_param.get('cluster')
                fp_class =fog_param.get('fp_class')
                fp_r = fog_param.get('fp_r')
                fp_b = fog_param.get('fp_b')
                threshold = fog_param.get('threshold')

                rounds = 5
                try:
                    df_fog_auto_list,cores = FindKeyCompound.FOG_Based_AutoCluster(df_p,standard_smiles_column_name,
                                                                                   rank=rounds,timeout=timeout,cluster=cluster,
                                                                                   fp_class=fp_class,fp_r=fp_r,fp_b=fp_b,
                                                                                   threshold=threshold)
                except Exception as e:
                    print (e)

                # display(rdkit.Chem.Draw.MolsToGridImage(cores, subImgSize=(180,180),molsPerRow=5))
                n = len(df_fog_auto_list)

                if n < rounds:
                    cluster = n
                else:
                    cluster = rounds

                df_fog_auto_results = pd.DataFrame()
                for i in range(cluster):
                    
                    df_temp = df_fog_auto_list[i]
                    df_temp.to_csv(f'{wdirs}/{pid}_autofog_cluster{i+1}.csv',index=False)
                    stop = False
                    df_score = df_temp.Score.value_counts().to_frame()
                    df_score.sort_index(ascending=False,inplace=True)

                    if df_score.shape[0] > 1:
                        n1 = df_score.iloc[0,].tolist()[0]
                        n2 = df_score.iloc[1,].tolist()[0]
                        if n1 > 1:
                            end = n1
                            rank_list = np.repeat(1,end).tolist()
                            rank_share_list = np.repeat(end,end).tolist()
                        else:
                            end = n1 + n2
                            rank_list = np.repeat(1,n1).tolist() + np.repeat(2,n2).tolist()
                            rank_share_list = np.repeat(n1,n1).tolist() + np.repeat(n2,n2).tolist()
                    else:
                        n1 = df_score.iloc[0,].tolist()[0]
                        end = n1
                        rank_list = np.repeat(1,end).tolist()
                        rank_share_list = np.repeat(end,end).tolist()

                    df_fog_auto_temp_results = df_temp.iloc[:end,]
                    df_fog_auto_temp_results['FOG_Cluster'] = np.repeat(i+1,end).tolist()
                    df_fog_auto_temp_results['FOG_Rank'] = rank_list
                    df_fog_auto_temp_results['FOG_Rank_Share'] = rank_share_list

                    df_fog_auto_results = pd.concat([df_fog_auto_results,df_fog_auto_temp_results],ignore_index=True)

                df_fog_auto_results.to_csv(f'{wdirs}/{pid}_FOG_auto_Top5Top2.csv',index=False)

                stop = False
                for i in range(df_fog_auto_results.shape[0]):
                    if df_fog_auto_results[standard_smiles_column_name][i] == candidate:
                        dict_p_result['FOG_auto_in_Top5&Top2'] = 1
                        dict_p_result['FOG_Cluster'] = df_fog_auto_results.FOG_Cluster[i]
                        dict_p_result['FOG_Rank'] = df_fog_auto_results.FOG_Rank[i]
                        dict_p_result['FOG_Rank_Share'] = df_fog_auto_results.FOG_Rank_Share[i]
                        dict_p_result['FOG_Score'] = df_fog_auto_results.Score[i]
                        stop = True
                        break
                    else:
                        pass
                    
                if stop == False:
                    dict_p_result['FOG_auto_in_Top5&Top2'] = 0
                    dict_p_result['FOG_Cluster'] = None
                    dict_p_result['FOG_Rank'] = None
                    dict_p_result['FOG_Rank_Share'] = None
                    dict_p_result['FOG_Score'] = None
        else:
            pass


        # the result for each patent
        df_p_result = pd.DataFrame.from_dict(dict_p_result,orient='index').T
        # print('This',df_p_result)
        df_result = pd.concat([df_result,df_p_result],ignore_index=True)

    df_result.to_csv(f'{wdirs}/results_{name}.csv',index=False)
    return df_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='choose different method to find the key compound')
    parser.add_argument('--p',default=2,type=int,help="number of pools to use")
    # csa
    parser.add_argument("--csa",default=True,type=bool,
        help="whether use csa method")
    parser.add_argument("--csa_param",default={'fp_class':'ECFP','fp_r':2,'fp_b':1024,'threshold':0.7},type=json.loads,
        help="param for csa method, example:{'fp_class':'ECFP','fp_r':2,'fp_b':1024,'threshold':0.7}")
    #midol
    parser.add_argument("--midol",default=False,type=bool,
        help="whether use midol method")
    parser.add_argument("--midol_param",default={'timeout':1,'threshold':0.9},type=json.loads,
        help="param for midol method,example:{'timeout':1,'threshold':0.9}")
    #fog
    parser.add_argument("--fog",default=False,type=bool,
        help="whether use fog method")
    parser.add_argument("--fog_param",default={'timeout':60,'cluster':'a','fp_class':'ECFP','fp_r':2,'fp_b':1024,'threshold':0.6},type=json.loads,
        help="param for fog method,example:{'timeout':60,'cluster':'a','fp_class':'ECFP','fp_r':2,'fp_b':1024,'threshold':0.7}")


    args = parser.parse_args()

    pool = args.p

    csa_flag = args.csa
    midol_flag = args.midol
    fog_flag = args.fog

    methods = [False,False,False]
    params = []
    if csa_flag == True:
        csa_param = args.csa_param
        name_list = [str(i) for i in list(csa_param.values())[:-1]]
        name_list.insert(0,'CSA')
        name_list.append(str(csa_param.get('threshold')).split('.')[-1])
        print(name_list) 
        name_string = '-'.join(name_list)
        print(name_string)
        methods[0] = True
        params.append(csa_param)
    else:
        csa_param = {}
        params.append(csa_param)

    if midol_flag == True:
        midol_param = args.midol_param
        name_list = [str(i) for i in list(midol_param.values())[:-1]]
        name_list.append(str(midol_param.get('threshold')).split('.')[-1])
        name_list.insert(0,'MIdol')
        name_string = '-'.join(name_list)
        print(name_string)
        methods[1] = True
        params.append(midol_param)
    else:
        midol_param = {}
        params.append(midol_param)

    if fog_flag == True:
        fog_param = args.fog_param
        name_list = [str(i) for i in list(fog_param.values())[:-1]]
        name_list.insert(0,'FOG')
        name_list.append(str(fog_param.get('threshold')).split('.')[-1])
        name_string = '-'.join(name_list)
        print(name_string)
        methods[2] = True
        params.append(fog_param)
    else:
        fog_param = {}
        params.append(fog_param)

    p = Pool(pool)
    path = '../'
    ###############################################################
    # create wdir and log file
    wdirs = path + 'results/' + name_string
    if os.path.exists(f'{wdirs}'):
        flist = os.listdir(wdirs)
        plist = []
        for f in flist:
            if 'results' in f:
                plist.append(f.split('_')[1][:-4])   # patents done
        pass
    else:
        plist = []
        os.makedirs(f'{wdirs}')

    df_total = pd.read_csv(f'../data/data.csv')
    ff_df = []
    c = 0
    for patent in df_total.PATENT_ID.unique():
        if patent not in plist:
            df_temp = df_total[df_total.PATENT_ID == patent].reset_index(drop=True)
            ff_df.append(df_temp)
            c +=1

    results = []
    for f_df in ff_df:
        r = p.apply_async(FindKC,args=(wdirs,methods,params,f_df,plist))
        results.append(r)
    p.close()
    p.join()

    ll = os.listdir(wdirs)
    df_all = pd.DataFrame()
    for l in ll:
        if 'results' in l:
            df_temp = pd.read_csv(f'{wdirs}/{l}')
            df_all = pd.concat([df_all,df_temp]).reset_index(drop=True)
        else:
            pass
    df_all.to_csv(f'{wdirs}/results_all.csv',index=False)

    df_results = pd.read_csv(f'{wdirs}/results_all.csv')
    print('Drugs   :',len(df_results.Name.unique()))
    n_total = df_results.shape[0]
    print('Patents :', n_total)
    print(n_total)


