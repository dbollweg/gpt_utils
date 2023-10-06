from cmath import phase
from math import gamma
import gpt as g

import numpy as np
from qTMD.gpt_proton_qTMD_utils import proton_measurement

class proton_TMD(proton_measurement):
    def __init__(self, parameters):
        self.eta = parameters["eta"]
        self.b_z = parameters["b_z"]
        self.b_T = parameters["b_T"]
        self.pzmin = parameters["pzmin"]
        self.pzmax = parameters["pzmax"]
        self.pf = parameters["pf"]
        self.plist = [[0,0,pz,0] for pz in range(self.pzmin,self.pzmax)]
        self.width = parameters["width"]
        self.boost_in = parameters["boost_in"]
        self.boost_out = parameters["boost_out"]
        self.pos_boost = self.boost_in
        self.pol_list = ["P+_Sz+","P+_Sx+","P+_Sx-"]
        self.save_propagators = parameters["save_propagators"]
        self.t_insert = parameters["t_insert"]
        
    
    
        
    def create_fw_prop_TMD(self, prop_f, W, W_index_list):
        g.message("Creating list of W*prop_f")
        prop_list = [prop_f,]
        
        for i, idx in enumerate(W_index_list):
        
            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]
            #prop_list.append(g.eval(g.gamma[5]*g.adj(g.gamma[5]*g.eval(W[i] * g.cshift(g.cshift(prop_f,transverse_direction,current_b_T),2,round(2*current_bz)))*g.gamma[5])))
            prop_list.append(g.eval(W[i] * g.cshift(g.cshift(prop_f,transverse_direction,current_b_T,),2,round(2*current_bz))))
        
        return prop_list

    def create_bw_seq(self, inverter, prop, trafo, flavor):
        tmp_trafo = g.convert(trafo, prop.grid.precision) #Need later for mixed precision solver
        
        prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)
        
        pp = 2.0 * np.pi * np.array(self.pf) / prop.grid.fdimensions
        P = g.exp_ixp(pp)

        
        src_seq = [g.mspincolor(prop.grid) for i in range(3)]
        dst_seq = []
        
        g.message("starting diquark contractions")
        g.qcd.baryon.proton_seq_src(prop, src_seq, self.t_insert, flavor)
        g.message("diquark contractions done")
        
        dst_tmp = g.mspincolor(prop.grid)
        for i in range(3):
            smearing_input = g.eval(g.gamma[5]*P*g.conj(src_seq[i]))

            tmp_prop = g.create.smear.boosted_smearing(trafo, smearing_input,w=self.width, boost=self.boost_out)

            dst_tmp = g.eval(inverter * tmp_prop)           
            dst_seq.append(g.eval(g.gamma[5] * g.conj(dst_tmp)))
        
        g.message("bw. seq propagator done")
        return dst_seq
    
    def contract_TMD(self, prop_f, prop_bw_seq, phases, tag):
        corr = []
        for fixed_pol_bwprop in prop_bw_seq:
            corr.append(g.slice_trQPDF(prop_f,fixed_pol_bwprop,phases,3))
    
        return corr
    
    def create_TMD_WL(self,U):
        W = []
        index_list = []

        # positive direction
        # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
        for transverse_direction in [0,1]:
            for current_eta in self.eta:

                prv_link = g.qcd.gauge.unit(U[2].grid)[0]
                current_link = prv_link
                # FIXME: phase need to be corrected due to source position
                for dz in range(0, current_eta-1):
                    current_link=g.eval(prv_link * g.cshift(U[2],2, dz))
                    prv_link=current_link
                current_bz_link = current_link

                for current_bz in range(0, min([self.b_z, current_eta])):

                    current_bz_link=g.eval(current_bz_link * g.cshift(U[2],2, current_eta+current_bz-1))
                    current_bT_link = current_bz_link

                    for current_b_T in range (0, self.b_T):
                        if current_b_T != 0:
                            current_bT_link=g.eval(current_bT_link * g.cshift(g.cshift(U[transverse_direction], 2, current_eta+current_bz),transverse_direction, current_b_T-1))

                        prv_link = current_bT_link
                        for dz in range(0, current_eta-current_bz):
                            current_link=g.eval(prv_link * g.adj(g.cshift(g.cshift(g.cshift(U[2], 2, current_eta+current_bz-1), transverse_direction, current_b_T),2,-dz)))
                            prv_link=current_link

                        W.append(current_link)
                        index_list.append([current_b_T, current_bz, current_eta, transverse_direction])

                        # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z+1)
                        current_link=g.eval(prv_link * g.adj(g.cshift(g.cshift(g.cshift(U[2], 2, current_eta+current_bz-1), transverse_direction, current_b_T),2,-(current_eta-current_bz))))
                        W.append(current_link)
                        index_list.append([current_b_T, current_bz-0.5, current_eta+0.5, transverse_direction])

                # create Wilson lines from all to all + (eta+1+0) + b_perp - (eta+1-0)
                current_eta += 1
                current_bz = 0
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

        # negative direction
        # create Wilson lines from all to all - (eta+bz) + b_perp + (eta-b_z)
        for transverse_direction in [0,1]:
            for current_eta in self.eta:

                prv_link = g.qcd.gauge.unit(U[2].grid)[0]
                current_link = prv_link
                # FIXME: phase need to be corrected due to source position
                for dz in range(0, current_eta-1):
                    current_link=g.eval(prv_link * g.adj(g.cshift(U[2],2, -dz-1)))
                    prv_link=current_link
                current_bz_link = current_link

                for current_bz in range(0, min([self.b_z, current_eta])):

                    current_bz_link=g.eval(current_bz_link * g.adj(g.cshift(U[2],2, -current_eta-current_bz)))
                    current_bT_link = current_bz_link

                    for current_b_T in range (0, self.b_T):
                        if current_b_T != 0:
                            current_bT_link=g.eval(current_bT_link * g.cshift(g.cshift(U[transverse_direction], 2, -current_eta-current_bz),transverse_direction, current_b_T-1))

                        prv_link = current_bT_link
                        for dz in range(0, current_eta-current_bz):
                            current_link=g.eval(prv_link * g.cshift(g.cshift(g.cshift(U[2], 2, -current_eta-current_bz), transverse_direction, current_b_T),2,dz))
                            prv_link=current_link

                        W.append(current_link)
                        index_list.append([current_b_T, -current_bz, -current_eta, transverse_direction])

                        # create Wilson lines from all to all - (eta+bz) + b_perp + (eta-b_z+1)
                        current_link=g.eval(prv_link * g.cshift(g.cshift(g.cshift(U[2], 2, -current_eta-current_bz), transverse_direction, current_b_T),2,(current_eta-current_bz)))
                        W.append(current_link)
                        index_list.append([current_b_T, -(current_bz-0.5), -(current_eta+0.5), transverse_direction])

                # create Wilson lines from all to all - (eta+1+0) + b_perp + (eta+1-0)
                current_eta += 1
                current_bz = 0
                for current_b_T in range (0, self.b_T):

                    prv_link = g.qcd.gauge.unit(U[2].grid)[0]
                    current_link = prv_link
                    # FIXME: phase need to be corrected due to source position
                    for dz in range(0, current_eta+current_bz):
                        current_link=g.eval(prv_link * g.adj(g.cshift(U[2],2, -dz-1)))
                        prv_link=current_link

                    for dx in range(0, current_b_T):
                        current_link=g.eval(prv_link * g.cshift(g.cshift(U[transverse_direction], 2, -current_eta-current_bz),transverse_direction, dx))
                        prv_link=current_link

                    for dz in range(0, current_eta-current_bz):
                        current_link=g.eval(prv_link * g.cshift(g.cshift(g.cshift(U[2], 2, -current_eta-current_bz), transverse_direction, current_b_T),2,dz))
                        prv_link=current_link

                    W.append(current_link)
                    index_list.append([current_b_T, -current_bz, -current_eta, transverse_direction])
        return W, index_list