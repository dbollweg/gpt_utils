#!/usr/bin/env python3
#
# GPT inversion sources selection
#
import gpt as g
import os
import h5py
import numpy as np

def get_c2pt_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".c2pt"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/c2pt/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag

def get_qTMDWF_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qTMDWF"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qTMDWF/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag

def get_qDA_file_tag(data_dir, lat, cfg, ama, src, sm):

    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".qDA"
    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    return data_dir + "/qDA/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag

def get_sample_log_tag(ama, src, sm):

    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    log_sample = ama_tag + "_" + src_tag + "_" + sm_tag

    return log_sample

def save_proton_c2pt_hdf5(corr, tag, gammalist, plist):

    roll = -int(tag.split(".")[4].split('t')[1])

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')
    sm = f.create_group("SS")
    for ig, gm in enumerate(gammalist):
        g = sm.create_group(gm)
        for ip, p in enumerate(plist):
            dataset_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g.create_dataset(dataset_tag, data=np.roll(corr[ig][ip], roll, axis=0))
    f.close()

def save_c2pt_hdf5(corr, tag, gammalist, plist):

    roll = -int(tag.split(".")[4].split('t')[1])

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')
    sm = f.create_group("SS")
    for ig, gm in enumerate(gammalist):
        g = sm.create_group(gm)
        for ip, p in enumerate(plist):
            dataset_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g.create_dataset(dataset_tag, data=np.roll(corr[0][ip][ig], roll, axis=0))
    f.close()

def save_qTMDWF_hdf5_subset(corr, tag, gammalist, plist, W_index_list, i_sub):

    roll = -int(tag.split(".")[4].split('t')[1])
    bT_list = ['b_X', 'b_Y']

    if g.rank() == 0:
        print("-->>",W_index_list)

    save_h5 = tag + ".h5"
    if i_sub == 0:
        f = h5py.File(save_h5, 'w')
    else:
        f = h5py.File(save_h5, 'a')
    sm = f.require_group("SP")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.require_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.require_group(p_tag)
            for i, idx in enumerate(W_index_list):
                path = bT_list[idx[3]] + '/' + 'eta'+str(idx[2]) + '/' + 'bT'+str(idx[0])
                g_data = g_p.require_group(path)
                if g.rank() == 0 and ig == 0 and ip == 0:
                    #g_p.keys()
                    #g_data.keys()
                    print("Want to save", path+'bz'+str(idx[1]))
                g_data.create_dataset('bz'+str(idx[1]), data=np.roll(corr[i][ip][ig], roll, axis=0))
    f.close()

def save_qTMDWF_hdf5(corr, tag, gammalist, plist, eta, b_T, b_z):

    roll = -int(tag.split(".")[4].split('t')[1])
    td_offset = b_T*b_z*len(eta)
    eta_offset = b_T*b_z
    bz_offset = b_T
    bT_list = ['b_X', 'b_Y']

    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'a')
    sm = f.create_group("SP")
    for ig, gm in enumerate(gammalist):
        g_gm = sm.create_group(gm)
        for ip, p in enumerate(plist):
            p_tag = "PX"+str(p[0])+"PY"+str(p[1])+"PZ"+str(p[2])
            g_p = g_gm.create_group(p_tag)
            for transverse_direction in [0,1]:
                g_T = g_p.create_group(bT_list[transverse_direction])
                for eta_idx, current_eta in enumerate(eta):
                    g_eta = g_T.create_group('eta'+str(current_eta))
                    for current_b_T in range(0, b_T):
                        g_bT = g_eta.create_group('bT'+str(current_b_T))
                        for current_bz in range(0, b_z):
                            bz_tag = 'bz'+str(current_bz)
                            W_index = current_b_T + bz_offset*current_bz + eta_offset*eta_idx + td_offset*transverse_direction
                            g_bT.create_dataset(bz_tag, data=np.roll(corr[W_index][ip][ig], roll, axis=0))
    f.close() 
