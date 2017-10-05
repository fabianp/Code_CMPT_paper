# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 15:15:39 2016

@author: elena


Created on Thu Jul 21 17:50:29 2016

@author: elena
"""


import nilearn
#from mySMasker import myNiftiSpheresMasker
import nilearn.image as image
from nilearn.image import *
import numpy as np
import nibabel as nib
import os
from nilearn import input_data
import re
import CMPT
from CMPT import *
import random
from joblib import Parallel, delayed
import fnmatch


def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest    
    
   

def betamaps2arrays(SubjID, Mod, datapath1, main_masker):
    n_of_subj=len(SubjID)
    labels_all=[0]*n_of_subj

    subj_data_all=[0]*n_of_subj

    SubjName=[0]*n_of_subj
    
    for subj in range(0, n_of_subj): 
        SubjName[subj]=SubjID[subj][-4:]
        print SubjName[subj]
        datapath2=os.path.join(datapath1, SubjName[subj]+'_' + Mod) 
        os.chdir(datapath2)
        labels1=[]
        for file1 in os.listdir(datapath2):
    
            try:
                labels1.append(int(re.findall('\d+', file1)[2]))
            except:
                IndexError
                pass
  
        labels_all[subj]=labels1
    
        temp_list=concat_imgs(nib.load(os.listdir(datapath2)[k]) for k in range(0, len(os.listdir(datapath2))))
 
        subj_data_all[subj]=main_masker.fit_transform(temp_list)
        
        filename1=SubjName[subj]+'_'+Mod
        np.save(os.path.join(datapath1, filename1), subj_data_all[subj])
          
        filename_lab='labels'+'_'+SubjName[subj]+'_'+Mod
        np.save(os.path.join(datapath1, filename_lab), labels1)
    return SubjName, subj_data_all, labels_all

def make_permuted_maps(n_perm, SubjName, subj_data_all, labels_all, permdatapath, Mod, main_masker):
    n_of_subj=len(SubjName)
   
    if os.path.isdir(permdatapath)==False:
        os.mkdir(permdatapath)
        
    if os.path.isdir(os.path.join(permdatapath, Mod))==False:
        os.mkdir(os.path.join(permdatapath, Mod))
    for perm in range(0, n_perm):
        labels1_perm=scrambled(labels_all[0])
     #   print len(labels1_perm)

        conditions=np.unique(labels1_perm)

        for subj in range(0, n_of_subj): 
            
                temp_array=subj_data_all[subj]
              #  print len(temp_array)
                if os.path.isdir(permdatapath+'/'+Mod +'/'+SubjName[subj])==False:
                    os.mkdir(permdatapath+'/'+Mod +'/'+SubjName[subj])

                for cond in conditions:
                    subj_cond_dir=permdatapath +'/'+ Mod +'/'+ SubjName[subj] +'/'+'cond' + str(cond)
 
                    if os.path.isdir(subj_cond_dir)==False:
                        os.mkdir(subj_cond_dir)
             
                    temp_array_0=np.array((temp_array[labels1_perm==cond]).mean(axis=0), dtype='f4')
  
                    nii1=main_masker.inverse_transform(temp_array_0)
   
                    nib.save(nii1, os.path.join(subj_cond_dir, 'beta_'+SubjName[subj]+'_'+str(perm)+'_'+Mod+'_cond'+str(cond)+'.nii.gz'))
    return conditions
    
def make_permuted_arrays(permdatapath, n_perm, SubjID, Mod, conditions, main_masker):
    n_of_subj=len(SubjID)
    n_splits=n_perm/100
    all_perms=np.arange(0, n_perm) 
    all_perms=np.array_split(all_perms, n_splits)  
    
    
    
    os.chdir(permdatapath)
    print "Starting permutation computations"
    for mysplit in all_perms:
        print mysplit[0]
        
        for cond in conditions:
            subj_data_perm=[0]*n_of_subj
            for s in range(0, n_of_subj): 
                SubjName=SubjID[s][-4:]
                print SubjName
                subj_cond_dir=permdatapath+'/'+Mod +'/'+SubjName+'/'+'cond'+str(cond)
            
            
                os.chdir(subj_cond_dir)
                subj_data_perm[s]=main_masker.fit_transform(concat_imgs(nib.load(os.listdir(subj_cond_dir)[k]) for k in mysplit))
                print "Subject done !"
            filename_data=('100_perm_'+str(mysplit[0])+'_'+Mod+'_'+'cond'+str(cond))
            np.save(os.path.join(permdatapath,filename_data), subj_data_perm)
        


if __name__ == "__main__":
    
    """List of Subject IDs """
    SubjID=["19881016MCBL", "19890126ANPS", "19901103GBTE", "19900422ADDL", "19850630IAAD", "19851030DNGL", "19750827RNPL", "19830905RBMS", "19861104GGBR"]
    
    #["19901103GBTE", "19851030DNGL", "19750827RNPL"] #["19901103GBTE", "19851030DNGL"] # MAKE SURE TO CHANGE TO YOUR SUBJECT ID LIST
    

    """Paths 
    
    Naming conventions: datapath=folder named data in your cwd. Inside the datapath subjects' 
    betamaps are in     the folders named with SubjName_Mod(ality):SubjID_Mod 
    
    Betamap filename: SubjID_run_volume(=HRF peak in the block)_condition(1 or 0)
    Run and volume are just for numbering while condition is important because 
    it will be extracted as label"""
    
    cwd="/home/elena/CMPT/to_share/" #MAKE SURE TO CHANGE TO YOUR CWD
 
    datapath=os.path.join(cwd, 'data')
  
    permdatapath=os.path.join(cwd, 'permutations') #, 'all_delays') #'all_delays'
    maskfile=os.path.join(cwd, 'mynewgreymask.nii.gz')
    
    """Label for the congnitive modality and conditions according to the naming conventions """
    
    Mod='VS' # MAKE SURE TO CHANGE TO YOUR MODALITIES
    #conditions=[0, 1]  #MAKE SURE TO CHANGE TO YOUR CONDITION LIST
    
    
    """N of permutations to create """
    n_perm=10000
                
                
    """here we load the grey matter mask and create a masker that will transform
    the resulting significance map into a whole brain image"""

    gm_mask = nib.load(maskfile)
    print gm_mask.shape
    main_masker=input_data.NiftiMasker(mask_img=gm_mask)
    

    """This function transforms beta maps into arrays  """
    SubjName, all_data, all_labels=betamaps2arrays(SubjID, Mod, datapath, main_masker)
    
    """This function creates permuted beta maps  """
    conditions=make_permuted_maps(n_perm, SubjName, all_data, all_labels, permdatapath, Mod, main_masker)
    
    """This function creates permuted arrays with 100 permutations in each  """
    make_permuted_arrays(permdatapath, n_perm, SubjID, Mod, conditions, main_masker)



print "We are done !!"