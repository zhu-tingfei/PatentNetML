import os
import time
import logging
import argparse
import warnings
import linecache
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import deepchem as dc

import rdkit
from rdkit import Chem
from rdkit import RDLogger

from rdkit.Chem import AllChem 
from rdkit.Chem.rdFMCS import FindMCS
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.ChemUtils import SDFToCSV
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem import FragmentMatcher
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import cDataStructs
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS, cluster_optics_dbscan

################################################################
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))



def StandardizeSmi(smiles,clearCharge=True, clearFrag=True, canonTautomer=False,isomeric=False):
    try:
        clean_mol = Chem.MolFromSmiles(smiles)
        if clearFrag:
            clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        if clearCharge:
            uncharger = rdMolStandardize.Uncharger() 
            clean_mol = uncharger.uncharge(clean_mol)
        if canonTautomer:
            te = rdMolStandardize.TautomerEnumerator() # idem
            clean_mol = te.Canonicalize(clean_mol)

        stan_smiles=Chem.MolToSmiles(clean_mol, isomericSmiles=isomeric)
        if Chem.MolFromSmiles(stan_smiles) == None:
            stan_smiles = None
    except Exception as e:
        stan_smiles = None
        print (e)
    return stan_smiles

################################################################

def DataClean(df:pd.DataFrame(),in_column='SMILES',out_column='Ca_Smiles',
    basicClean=True,clearCharge=True, clearFrag=True, canonTautomer=True, isomeric=True):

    df_clean = df.copy(True)
    df_clean.dropna(subset=[in_column],inplace=True)

    smiles = []
    for smi in df_clean[in_column]:
        ca_smi = StandardizeSmi(smi,basicClean=basicClean,
                                clearCharge=clearCharge,clearFrag=clearFrag,
                                canonTautomer=canonTautomer,isomeric=isomeric)
        smiles.append(ca_smi)
    df_clean[out_column] = smiles
    df_clean.dropna(subset=[out_column],inplace=True)
    df_clean.drop_duplicates(subset=[out_column],inplace=True,ignore_index=True)
    return df_clean

################################################################

def group_generation(df:pd.DataFrame(),in_column:str,out_column:str,cluster:str,fp_class='ECFP',fp_r=2,fp_b=1024,d=0.6):

    df_g = df.copy(True)

    mols = [Chem.MolFromSmiles(smi) for smi in df_g[in_column].tolist()]
    try:
        if fp_class == 'ECFP':
            fp = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, fp_r, fp_b) for mol in mols]
        elif fp_class == 'MACCS':
            fp = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        elif fp_class == 'FCFP':
            fp = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, fp_r, fp_b,useFeatures=True) for mol in mols]
    except:
        PrintException()
    similarity_matrix = []
    for i in range(len(fp)):
        tsims = Chem.DataStructs.BulkTanimotoSimilarity(fp[i], fp)
        similarity_matrix.append(tsims)
    distance_matrix = 1 - np.array(similarity_matrix)

    ### different cluster method
    if cluster == 'a':
        clustering = AgglomerativeClustering(n_clusters=None,
                                             affinity='precomputed',
                                             linkage='complete',
                                             distance_threshold=d).fit(distance_matrix)
    elif cluster == 'b':
        clustering = OPTICS(min_samples=2).fit(fp)

    groups = clustering.labels_
    df_g[out_column] = groups
    # print(groups)
    return df_g

################################################################

def BulkTanimotoMCSS(mol_a,mol_list,timeout):

    TanimotoMCSS = []
    N_a = mol_a.GetNumAtoms()
    for mol_b in mol_list:
        if mol_b == mol_a:
            tanimotoMCSS = 1
        else:
            N_b = mol_b.GetNumAtoms()
            N_ab = FindMCS([mol_a,mol_b],maximizeBonds=True,matchValences=False,timeout=timeout).numAtoms
            if N_ab == -1:
                N_ab = 0
            tanimotoMCSS = N_ab/(N_a+N_b-N_ab)

        TanimotoMCSS.append(tanimotoMCSS)
        
    return TanimotoMCSS

################################################################

def MapCompoundToSmarts(framework:list,smiles:list,timeout=1):

    map_compound_to_mcs = {}
    mols = [Chem.MolFromSmiles(smi) for smi in framework]
    for threshold in np.linspace(1,0.1,num=10):
        sma = FindMCS(mols,maximizeBonds=True,matchValences=False,timeout=timeout,threshold=threshold).smartsString
        if type(sma) == str:
            break
        else:
            continue


    if type(sma) == str:

        p = FragmentMatcher.FragmentMatcher()
        p.Init(sma)

        res_fra = []
        res_smi = []
        for fra,smi,mol in zip(framework,smiles,mols):
            if p.HasMatch(mol) == 1:
                map_compound_to_mcs[smi] = sma
            else:
                res_smi.append(smi)
                res_fra.append(fra)
    else:
        map_compound_to_mcs.update(dict(zip(smiles,smiles)))
        res_fra = []
        res_smi = []

    return map_compound_to_mcs,res_fra,res_smi

################################################################
def RGroupAnalysis(df:pd.DataFrame(),column:str,sma:str):
    try:
        df_r = df.copy(True)
        num_compound = df_r.shape[0]
        df_r.to_csv('./df_debug.csv')
        df_r['Score'] = np.repeat(0,num_compound)
        mols = [Chem.MolFromSmiles(smi) for smi in df_r[column]]
        gs, _ = rdRGroupDecomposition.RGroupDecompose([Chem.MolFromSmarts(sma)], mols)

        if len(gs) != 0:

            num_Rgroup = len(gs[0])-1

            core = gs[0].get('Core')
            df_r['Core'] = np.repeat(Chem.MolToSmarts(core),num_compound)
            
            # print('For SMARTS:%s, there are %s R-group(s) for %s compounds.'%(sma,num_Rgroup,num_compound))

            for n in range(1,num_Rgroup+1):
                r_list = []
                for j in gs:
                    try:
                        r_g = j.get('R%s'%n)
                        try:
                            r_list.append(Chem.MolToSmarts(r_g))
                        except:
                            r_list.append(Chem.MolToSmiles(r_g))
                    except:
                        r_list.append('Error None')
                r_dict = {}
                for r in r_list:
                    if r not in r_dict.keys():
                        r_dict[r] = r_list.count(r)

                df_r['R%s'%n] = r_list
                df_r['R%s_count'%n] = [r_dict.get(r) for r in r_list]


                df_r['Score'] = np.array(df_r['Score'].tolist()) + np.array(df_r['R%s_count'%n].tolist())

            df_r.sort_values(by='Score',inplace=True,ascending=False,ignore_index=True)
        else:
            core = None
    except:
        PrintException()
    # df_r.to_csv('./RG_debug.csv')
    return df_r,core

################################################################
def CsaMaxNeighbor(df:pd.DataFrame(),column:str,fp_class:str,fp_r=2,fp_b=1024,threshold=0.7):

    df_cas_temp = df.copy(True)
    smis = df_cas_temp[column].tolist()
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    if fp_class == 'ECFP':
        fp = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, fp_r, fp_b) for mol in mols]
    elif fp_class == 'MACCS':
        fp = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
    elif fp_class == 'FCFP':
        fp = [Chem.AllChem.GetMorganFingerprintAsBitVect(mol, fp_r, fp_b,useFeatures=True) for mol in mols]
    # elif fp_class == 'PubChem':
    #     featurizer = dc.feat.PubChemFingerprint()
    #     features = featurizer.featurize(smis)
    #     fp = [cDataStructs.CreateFromBitString(np.array_str(np.array(f))) for f in features]
        

    Neighbor = []
    Score = []
    for i in range(len(fp)):
        tsims = Chem.DataStructs.BulkTanimotoSimilarity(fp[i], fp)
        n_neighbor = -1
        tscore = -1
        for j in tsims:
            if j >= threshold:
                n_neighbor += 1
                tscore += j
        Neighbor.append(n_neighbor)
        Score.append(tscore)
    df_cas_temp['csa_neighbor'] = Neighbor
    df_cas_temp['csa_score'] = Score

    max_neighbor = max(Neighbor)
    min_neighbor = min(Neighbor)

    df_top = df_cas_temp[df_cas_temp.csa_neighbor == max_neighbor]
    df_top.sort_values(by=['csa_neighbor','csa_score'],
        inplace=True,ascending=False,ignore_index=True)

    if max_neighbor == min_neighbor:
        df_res = pd.DataFrame(columns=df_cas_temp.columns)
    else:
        df_res = df_cas_temp[df_cas_temp.csa_neighbor < max_neighbor]
        df_res.sort_values(by=['csa_neighbor','csa_score'],
            inplace=True,ascending=False,ignore_index=True)

    return df_top,df_res


################################################################
def find_max_index(d):
    max_list = []
    max_value = max(d.values())  
    for m, n in d.items():       
        if n == max_value:
            max_list.append(m)

    return max_list
################################################################


def CSA(df:pd.DataFrame(),column:str,fp_class:str,fp_r=2,fp_b=1024,threshold=0.7,rounds=5):

    df_csa_res = df.copy(True)
    df_csa_key = pd.DataFrame()

    for i in range(rounds):
        if df_csa_res.shape[0] > 1:
            df_top,df_res = CsaMaxNeighbor(df_csa_res,column,fp_class,fp_r,fp_b,threshold)
            df_csa_key = pd.concat([df_csa_key,df_top],ignore_index=True)
            df_csa_res = df_res.copy(True)
        elif df_csa_res.shape[0] == 1:
            df_csa_key = pd.concat([df_csa_key,df_csa_res],ignore_index=True)
            df_csa_res = pd.DataFrame()
        else:
            break

    return df_csa_key,df_csa_res
################################################################

################################################################
def MIdol(df:pd.DataFrame(),column:str,timeout:1,threshold:0.9):

    df_midol = df.copy(True)
    mols = [Chem.MolFromSmiles(smi) for smi in df_midol[column]]
    n = len(mols)

    Neighbor = []
    Score = []
    similarity_matrix = np.ones((n,n),dtype=np.float)

    for i in range(n):
        tsim = BulkTanimotoMCSS(mols[i], mols[i:],timeout)
        similarity_matrix[i,:][i:] = tsim
        similarity_matrix[:,i][i:] = tsim

        n_neighbor = -1
        score = -1

        for j in similarity_matrix[i,:]:

            if j >= threshold:
                n_neighbor += 1
                score += j
        Neighbor.append(n_neighbor)
        Score.append(score)

        if i%10 == 0:
            print('Molecular Idol is calculating: %s in %s.'%(i,n))


    df_midol['midol_neighbor'] = Neighbor
    df_midol['midol_score'] = Score
    df_midol.sort_values(by=['midol_neighbor','midol_score'],inplace=True,ascending=False,ignore_index=True)

    return df_midol
################################################################


def FOG_Based_AutoCluster(df:pd.DataFrame(),column:str,rank:int,timeout:int,cluster='a',fp_class='ECFP',fp_r=3,fp_b=1024,threshold=0.6):

    try:
        output = []

        # create molecular frameworks
        df_autofog = df.copy(True)
        scaffolds = []
        for smi in df_autofog[column]:
            scaffold_smi = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi), includeChirality=False)
            if len(scaffold_smi) == 0 :
                scaffolds.append(smi)
            else:
                scaffolds.append(scaffold_smi)

        df_autofog['Murcko_Framework'] = scaffolds

        # cluster molecular frameworks
        df_autofog = group_generation(df_autofog,'Murcko_Framework','Group_Murcko_Framework',cluster,fp_class,fp_r,fp_b,threshold)

        # for each cluster determine MCS and rank
        dic_smi_sma = {}
        for c in range(1+max(df_autofog['Group_Murcko_Framework'])):
            df_cluster = df_autofog[df_autofog['Group_Murcko_Framework'] == c]
            framework = df_cluster['Murcko_Framework'].tolist()
            smiles = df_cluster[column].tolist()

            # for each MCS
            # map compounds to MCS
            while len(framework) >1: 
                map_compound_to_mcs,res_fra,res_smi = MapCompoundToSmarts(framework,smiles,timeout)
                framework = res_fra
                smiles = res_smi
                dic_smi_sma.update(map_compound_to_mcs)
            if len(smiles) == 1:
                smi = smiles[0]
                dic_smi_sma[smi] = smi

            # identify R-Group
            # calculate FOG score

        smarts = [dic_smi_sma.get(smi) for smi in df_autofog[column]]
        df_autofog['Smarts'] = smarts

        smarts_num = {}
        for sma in smarts:
            if sma not in smarts_num.keys():
                smarts_num[sma] = smarts.count(sma)

        df_autofog['Smarts_count'] = [smarts_num.get(sma) for sma in smarts]

        df_autofog.sort_values(by='Smarts_count',inplace=True,ascending=False,ignore_index=True)
        df_autofog.dropna(subset=['Smarts'],inplace=True)
        df_drop_smarts_duplicates = df_autofog.drop_duplicates(['Smarts'],inplace=False,ignore_index=True)

        smarts_ordered = df_drop_smarts_duplicates.Smarts.tolist()

        if rank <= len(smarts_ordered):
            end = rank
        else:
            end = len(smarts_ordered)
        wrong = 0
        cores = []
        for sma in smarts_ordered[:end]:
            df_serie = df_autofog[df_autofog['Smarts'] == sma].reset_index(drop=True)
            if df_serie.shape[0] > 1:
                # df_serie.to_csv('./debug.csv')
                df_serie,core = RGroupAnalysis(df_serie,column,sma)
                if core != None:
                    cores.append(core)
                    output.append(df_serie)
                else:
                    wrong = +1
            else:
                pass
        if wrong >0:
            if end+wrong <= len(smarts_ordered):
                for sma in smarts_ordered[end:end+wrong:]:
                    df_serie = df_autofog[df_autofog['Smarts'] == sma].reset_index(drop=True)
                    if df_serie.shape[0] > 1:
                        df_serie,core = RGroupAnalysis(df_serie,column,sma)
                    else:
                        pass
            else:
                pass
        else:
            pass

    except Exception as e:
        print (e)
        return None

    return output,cores


