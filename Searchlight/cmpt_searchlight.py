#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:14:35 2017

@author: elena
"""

import time

#from mySMasker import myNiftiSpheresMasker
import nilearn.image as image
from nilearn.image import *
import numpy as np
import nibabel as nib
import os
from nilearn import input_data
import re
import CMPT
from CMPT import test_stat
import random
from joblib import Parallel, delayed
import fnmatch
from pearson_vectorized import pearsonr_vectorized
import numpy as np
import sklearn
from sklearn import neighbors
from sklearn.neighbors import KDTree
from sklearn.externals.joblib import Memory
from distutils.version import LooseVersion

import multiprocessing
import itertools

from nilearn.image.resampling import coord_transform
from nilearn._utils import CacheMixin
from nilearn._utils.niimg_conversions import check_niimg_4d, check_niimg_3d
from nilearn._utils.class_inspect import get_params
from nilearn import image
from nilearn import masking
from nilearn import input_data
from nilearn.input_data.base_masker import filter_and_extract, BaseMasker
import nibabel as nib
#from get_sphere import apply_mask_and_gettest

print "all imported"

def scrambled(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest    
    
    
    
def get_sphere_on_array(radius, seeds, kdtree):
                                  
    seeds=seeds.reshape(1, -1)                                 
                                 
    sphere=kdtree.query_radius(seeds, radius)
    sphere=sphere[0].tolist()
  
    return sphere 
    
def my_cmpt_searchlight(subj_data, subj_labels, mask_index, radius, mask_tree, i):
 
        if i%10000==0:
            print "index", i
        alert=0
        stat=0
        coord=np.asarray([mask_index[0][i], mask_index[1][i], mask_index[2][i]])
 
        mysphere=get_sphere_on_array(radius,coord, mask_tree)
  
        if len(mysphere)==0:            
            alert=1
            print "sphere empty, will skip"
  
        if len(mysphere)<25:
                    print "sphere too small, will skip"
                    alert=1
  
        else:
            
   
             for subj in range(0, len(subj_data[0])): 
      
                img_cond_modality=[0]*len(subj_data)
                for mod in range(0, len(subj_data)):
                    temp_mod=[]
  
                    maps=[]

                    maps=subj_data_all[mod][subj][:, mysphere]
                      
                    for cond in np.unique(subj_labels[mod][subj]):
   
                        ind=np.array(subj_labels[mod][subj]==cond).flatten()
   
                        temp_mod.append(np.mean(maps[ind], axis=0))
       
                    img_cond_modality[mod]=temp_mod
  
                stat+=test_stat(np.asarray(img_cond_modality[0]), np.asarray(img_cond_modality[1]))
             #   print stat
                    
        return alert, mysphere, stat
       

def sl_cmpt_permutations_routine(subj_data_all, labels_all, subj_data_perm, mask_index, mask_tree, radius, vox_group, stat_map, sign_map):
  #  print "permutations started 1"    
    for vox in vox_group:
     #   print vox#range(0, 20): # len(mask_index[0])):
        os.chdir(cwd)
        alert, sphere, stat_map[vox] = my_cmpt_searchlight(subj_data_all, labels_all, mask_index, radius, mask_tree, vox)
    #    print stat_map[vox]
    
        if alert==0:
            
            sign_map[vox]=sl_cmpt_permutations(sphere, subj_data_perm,  stat_map[vox], vox)
         #   print sign_map.shape
        #    print sign_map[vox]
    return sign_map
        
def sl_cmpt_permutations(mysphere, subj_data_perm, true_stat, vox):
  #  print "permutations started 2" 
    n_of_subj=len(subj_data_perm[0][0])
  #  print n_of_subj
  #  print len(subj_data_perm), len(subj_data_perm[1]) #0 modality condition subject
    stat_perm=[0]*n_of_subj
  #  mysphere=apply_mask_and_get_affinity(coord, gm_mask, 8, allow_overlap=0, mask_img=None)
   # mysphere=get_sphere_on_array(8,coord, mask_tree, allow_overlap=0)
  #  print "2"
    #(coord, niimg, 8, allow_overlap=1, mask_img=gm_mask)
     #   perms=[0]*n_of_subj
      #  for perm in range(0, n_perm):
    for s in range(0, n_of_subj):
        img_cond_modality=[0]*len(subj_data_perm)
        for mod in range(0, len(subj_data_perm)):
            temp_subj=[]
            for cond in range(0, len(subj_data_perm[0])):
                temp_subj.append(subj_data_perm[mod][cond][s][:, mysphere])
            img_cond_modality[mod]=temp_subj
#                img_cond_modality_1=[0]*2
#                img_cond_modality_2=[0]*2
#                      
#                
#                img_cond_modality_1[0]=subj_data_perm_1_1[s][:, mysphere]
#                img_cond_modality_1[1]=subj_data_perm_1_2[s][:, mysphere]
#                img_cond_modality_2[0]=subj_data_perm_2_1[s][:, mysphere]
#                img_cond_modality_2[1]=subj_data_perm_2_2[s][:, mysphere]
#              #  print img_cond_modality_3_1[0].shape
                #upd_stat
      #  print len(img_cond_modality[0][0]), len(img_cond_modality[1][0])
        stat_perm[s]=pearsonr_vectorized(img_cond_modality[0][0], img_cond_modality[1][0])+pearsonr_vectorized(img_cond_modality[0][1], img_cond_modality[1][1]) - \
                    (pearsonr_vectorized(img_cond_modality[0][0], img_cond_modality[1][1])+pearsonr_vectorized(img_cond_modality[0][1], img_cond_modality[1][0]))
        
        
                    #    print stat_perm[s]
              #  print "stats computed"
#                if split_counter==0:
#                    stat_perm[s]=upd_stat
#                #    print len(stat_perm[s])
#            #    print stat_perm[s]
#                else:
#                    stat_perm[s]=np.hstack([stat_perm[s], upd_stat])
                 #   print stat_perm[s].shape
                
                
                
  #  print len(stat_perm),    len(stat_perm[0])                  
   # print np.sum(stat_perm, axis=1)
    sign_map[vox]=np.sum(np.sum(stat_perm, axis=0)< true_stat)
  #  print sign_map[vox]
    return   sign_map[vox]

if __name__ == '__main__':
    start = time.time()
    
    """List of Subject IDs """
    SubjID=["19881016MCBL", "19890126ANPS", "19901103GBTE", "19900422ADDL", "19850630IAAD", "19851030DNGL", "19750827RNPL", "19830905RBMS", "19861104GGBR"]
    #["19901103GBTE", "19851030DNGL", "19750827RNPL"]
    n_of_subj=len(SubjID)
    SubjName=[0]*n_of_subj
    
    """Paths """
    
    cwd="/home/elena/CMPT/to_share/"  #"MAKE SURE TO CHANGE TO YOUR CWD
    results_dir="./results/"  
    datapath=os.path.join(cwd, 'data')
    if os.path.isdir(results_dir)==False:
                os.mkdir(results_dir)
    
  
    counter=0
#n_perm=100
    n_perm=10000
 #   perm_count=np.zeros([n_perm])
 
    all_perms=np.arange(0, n_perm)  #np.arange(0, len(mask_index[0]))
    n_splits=n_perm/100
    all_perms=np.array_split(all_perms, n_splits)  

    maskfile='./mynewgreymask.nii.gz' #MAKE SURE TO HAVE YOUR MASK IN THE CWD
    gm_mask = nib.load(maskfile)  
    print "mask loaded"
    print gm_mask.shape
    main_masker=input_data.NiftiMasker(mask_img=gm_mask)
    mm=main_masker.fit_transform(gm_mask)
#print gm_mask.shape
#We also create an index through the mask 
    current_mask=np.array((gm_mask.get_data()==1))
    print current_mask.shape[0], current_mask.shape[1],current_mask.shape[2]  
    mask_index=np.array(np.where(current_mask))
    print mask_index.shape
    mask_tree=KDTree(mask_index.T) #np.unravel_index(current_mask, 3)
####mask_coords = np.asarray(list(zip(*mask_index)))
#This one will hold the real group statistic
#spheres=mask_tree.query_radius(mask_tree, 8)
#print len(spheres)
    stat_map=np.zeros_like(mm).T

#This one will hold significances. Both need to have the same shape
#as the transformed gm mask to be transformed back to .nii
    new_sign_map=np.zeros_like(mm).T
#new_sign_map=np.zeros_like(mm).T
    print stat_map.shape


#Next lists will hold the data from all subjects - first,
#for the permutations, next, as the overarching loop is going through voxels
  #  subj_maps_all_1=[0]*n_of_subj
    

#    subj_maps_all_2=[0]*n_of_subj
#    labels_all_2=[0]*n_of_subj
#
#    
#    subj_data_all_2=[0]*n_of_subj
#subj_data_perm_1_1=[0]*n_of_subj
#subj_data_perm_2_1=[0]*n_of_subj
#subj_data_perm_1_2=[0]*n_of_subj
#subj_data_perm_2_2=[0]*n_of_subj
    
    Mod=['Im', 'VS'] #MAKE SURE TO CHANGE TO YOUR MODALITIES
    labels_all=[0]*len(Mod)
    subj_data_all=[0]*len(Mod)
    radius=8
  #  datapath="./data" #"/nilab0/kalinina/ATTEND/for_cluster/"
    permdatapath=os.path.join(cwd, 'permutations') 
#Here, we have to stack together each subjects beta maps into a 
#single 4D nifti image; then, each subject's 4D goes into the list
#Labels - because I included them into the file name, here you are free to change
    
    for mod in range(0, len(Mod)):
        
        subj_temp=[]
        labels_temp=[]
        for subj in range(0, n_of_subj): 
            SubjName[subj]=SubjID[subj][-4:]
            labels_temp.append(np.load(os.path.join(datapath, 'labels'+'_'+SubjName[subj]+'_'+Mod[mod]+'.npy'))) 
            subj_temp.append(np.load(os.path.join(datapath, SubjName[subj]+'_'+Mod[mod]+'.npy')))
        #    print len(subj_temp)
        subj_data_all[mod]=subj_temp
        labels_all[mod]=labels_temp
       # print len(subj_data_all), len(subj_data_all[0])
     #   subj_data_all_2[subj]=np.load(os.path.join(datapath, SubjName[subj]+'_'+Mod2+'.npy'))

   # '/nilab0/kalinina/ATTEND/for_cluster/'
   # labels_all_2=np.load('./cmpt_testing_data/labelsVS.npy')       
        
#    labels_all_1=np.load(os.path.join("/home/elena/CMPT/DATA", 'labelsIm.npy'))  #'/nilab0/kalinina/ATTEND/for_cluster/'
#    labels_all_2=np.load(os.path.join("/home/elena/CMPT/DATA", 'labelsVS.npy'))  #'/nilab0/kalinina/ATTEND/for_cluster/'
#print len(mask_index[0])
  #  mylabels1=np.load(os.path.join("/home/elena/CMPT/DATA", 'labelsIm.npy')) #'/nilab0/kalinina/ATTEND/for_cluster/'
   # mylabels2=np.load(os.path.join("/home/elena/CMPT/DATA", 'labelsVS.npy'))
    n_jobs=-1 #-1
  #  niimg = nib.load('/home/elena/CMPT/permutations/beta_ANPS_2_Im_cond1.nii.gz') #'/nilab0/kalinina/ATTEND/test_4/beta_ANPS_2_Im_cond1.nii.gz'
    all_voxels=np.arange(0, len(mask_index[0]))  #np.arange(0, len(mask_index[0]))
    n_splits=multiprocessing.cpu_count() #n_jobs #multiprocessing.cpu_count()  #n_jobs
    print "N of splits", n_splits
#if n_splits<0:
#     n_splits=cpu.count
    all_voxels=np.array_split(all_voxels, n_splits)  
    conditions=np.unique(labels_all[0])



    split_counter=0
    for mysplit in all_perms:
        print "split_counter=", split_counter
        result=[]
    
    
        sign_map=np.zeros_like(mm).T
        subj_data_perm=[0]*len(Mod) #*len(conditions) #[] #[0]*n_of_subj
      #  subj_data_perm_3_1=[] #[0]*n_of_subj
      #  subj_data_perm_1_2=[] #[0]*n_of_subj
      #  subj_data_perm_3_2=[] #[0]*n_of_subj
        print "maps created"
 ########################################################################   
   
   # print subj_data_all_2[subj].shape
#    
#        subj_data_perm_1_1[subj]=np.load(os.path.join(permdatapath, '100_perm_'+ SubjName[subj]+str(asplit[0])+'_1_1'+'.npy'))
#        subj_data_perm_1_2[subj]=np.load(os.path.join(permdatapath, '100_perm_'+str(asplit[0])+ SubjName[subj]+'_1_2'+'.npy'))
#        subj_data_perm_2_1[subj]=np.load(os.path.join(permdatapath, '100_perm_'+str(asplit[0])+ SubjName[subj]+'_2_1'+'.npy'))
#        subj_data_perm_2_2[subj]=np.load(os.path.join(permdatapath, '100_perm_'+str(asplit[0])+ SubjName[subj]+'_2_2'+'.npy'))
#    
###################################################################
        for mod in range(0, len(Mod)):
            temp_cond=[]
            for cond in conditions:
                temp_cond.append(np.load(os.path.join(permdatapath, '100_perm_'+str(mysplit[0])+'_'+Mod[mod]+'_' + 'cond'+str(cond)+'.npy')))
            subj_data_perm[mod]=temp_cond
     #   print len(subj_data_perm), len(subj_data_perm[0])
#        subj_data_perm_2_1=np.load(os.path.join(permdatapath,'100_perm_'+str(mysplit[0])+'_3_1.npy'))
#        subj_data_perm_1_2=np.load(os.path.join(permdatapath,'100_perm_'+str(mysplit[0])+'_2_2.npy'))
#        subj_data_perm_2_2=np.load(os.path.join(permdatapath,'100_perm_'+str(mysplit[0])+'_3_2.npy'))
#    
        print "maps loaded"

        result = Parallel(n_jobs=n_jobs)(delayed(sl_cmpt_permutations_routine)(subj_data_all, labels_all, subj_data_perm, mask_index, mask_tree, radius, mysplit,stat_map, sign_map) for mysplit in all_voxels) # len(mask_index[0]))) #  20)) len(mask_index[0])))
    
        print result[0].shape
 #   print len(result)
        for z in range(0, len(result)):
      #  print z
            xxx=result[z].nonzero()[0]
       # print len(xxx)
       # print result[z][xxx]
      #  print xxx
            new_sign_map[xxx]+=result[z][xxx]
        
    
    
        split_counter+=1
#print len(result)
    sign_niimg=main_masker.inverse_transform(new_sign_map.T)
    os.chdir(results_dir)        
    nib.save(sign_niimg, 'sl_VS_Im_delay_6.nii')
    print "yeah, done !"
    end = time.time()
    print (end - start)

#if __name__ == "__main__":
    # execute only if run as a script
 #   main()