from cmath import phase
from math import gamma
import gpt as g
import numpy as np
from utils.io_corr import *
import gpt_qpdf_utils

class TMD_WF_measurement(gpt_qpdf_utils.pion_measurement):
    def __init__(self,parameters):
        self.eta = parameters["eta"]
        self.b_z = parameters["b_z"]
        self.b_T = parameters["b_T"]
        self.pzmin = parameters["pzmin"]
        self.pzmax = parameters["pzmax"]
        self.plist = [ [0,0, pz, 0] for pz in range(self.pzmin,self.pzmax)]
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.save_propagators = parameters["save_propagators"]

    def contract_TMD(self, prop_f, prop_b, phases, tag, W_index_list, i_sub):

        corr = g.slice_trDA(prop_b,prop_f,phases, 3)
        if g.rank() == 0:
            #self.save_qTMDWF_hdf5(corr, tag, my_gammas)
            save_qTMDWF_hdf5_subset(corr, tag, my_gammas, self.plist, W_index_list, i_sub)
        del corr

    def create_src_TMD(self, pos, trafo, grid):
        
        srcD = g.mspincolor(grid)
        srcD[:] = 0
        
        g.create.point(srcD, pos)
        
        srcDm = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.neg_boost)
        srcDp = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.pos_boost)
        
        del srcD
        return srcDp, srcDm

    def constr_TMD_bprop(self, prop_b, W, W_index_list):
        # prop_list = [prop_b,]
        # FIXME: temporarily remove the no Wilson line contraction
        prop_list = []
        # W_index_list[i] = [bT, bz, eta, Tdir]
        for i, idx in enumerate(W_index_list):
            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]
            #g.mem_report(details=False)
            prop_list.append(g.eval(g.gamma[5]*g.adj(g.gamma[5]*g.eval(W[i] * g.cshift(g.cshift(prop_b,transverse_direction,current_b_T),2,2*current_bz))*g.gamma[5])))
            #g.mem_report(details=False)
        return prop_list

    def create_TMD_WL(self, U):

        W = []
        index_list = []

        # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
        for transverse_direction in [0,1]:
            for current_eta in self.eta:
                for current_bz in range(0, self.b_z):
                    for current_b_T in range (0, self.b_T):
                        prv_link = g.qcd.gauge.unit(U[2].grid)[0]
                        current_link = prv_link
                        # FIXME: phase need to be corrected due to source position
                        for dz in range(0, current_eta+current_bz):
                            current_link=g.eval(prv_link * g.cshift(U[2],2, dz))
                            prv_link=current_link

                        for dx in range(0, current_b_T):
                            current_link=g.eval(prv_link * g.cshift(g.cshift(U[transverse_direction], 2, current_eta+current_bz),transverse_direction, dx))
                            prv_link=current_link


                        for dz in range(0, current_eta-current_bz):
                            current_link=g.eval(prv_link * g.adj(g.cshift(g.cshift(g.cshift(U[2], 2, current_eta+current_bz-1), transverse_direction, current_b_T),2,-dz)))
                            prv_link=current_link

                        W.append(current_link)
                        index_list.append([current_b_T, current_bz, current_eta, transverse_direction])
                        if current_eta==4 and current_bz==1 and current_b_T==0:
                            g.message(f"WL, {index_list[-1]}, {W[-1]}")
                            #g.message(f"Udz0, {U[2]}")
                            g.message(f"Ushift, {g.eval(g.cshift(U[2],2, 0) * g.cshift(U[2],2, 1))}")
        return W, index_list
