from cmath import phase
from math import gamma
import gpt as g
from io_corr import *
import numpy as np
from qTMD.gpt_proton_qTMD_utils import proton_measurement

"""
================================================================================
                Gamma structures and Projection of nucleon states
================================================================================
"""
### Gamma structures
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]

### Projection of nucleon states
Cg5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma[5].tensor()
Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
Szp = (g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Szm = (g.gamma["I"].tensor() + 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
Sxp = (g.gamma["I"].tensor() - 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
Sxm = (g.gamma["I"].tensor() + 1j*g.gamma[1].tensor()*g.gamma[2].tensor())
PpSzp = Pp * Szp
PpSzm = Pp * Szm
PpSxp = Pp * Sxp
PpSxm = Pp * Sxm
#my_projections=["PpSzp", "PpSxp", "PpSxm"]
#my_projections=["PpSzp", "PpSzm", "PpSxp"]
#PolProjections = [PpSzp, PpSxp, PpSxm]
#PolProjections = [PpSzp, PpSzm, PpSxp]
PolProjections = {
    "PpSzp": PpSzp,
    "PpSzm": PpSzm,
    "PpSxp": PpSxp,
    "PpSxm": PpSxm,  
}

"""
================================================================================
                                proton_TMD
================================================================================
"""
class proton_TMD(proton_measurement):

    def __init__(self, parameters):

        self.eta = parameters["eta"] # list of eta
        self.b_z = parameters["b_z"] # largest b_z
        self.b_T = parameters["b_T"] # largest b_T

        self.pf = parameters["pf"] # momentum of final nucleon state; pf - q = pi ??
        self.plist = [[x,y,z,0] for x in parameters["qext"] for y in parameters["qext"] for z in parameters["qext"]] # generating momentum transfers

        self.width = parameters["width"] # Gaussian smearing width
        self.boost_in = parameters["boost_in"] # ?? Forward propagator boost smearing
        self.boost_out = parameters["boost_out"] # ?? Backward propagator boost smearing
        self.pos_boost = self.boost_in # Forward propagator boost smearing for 2pt

        self.pol_list = parameters["pol"] # projection of nucleon state
        self.t_insert = parameters["t_insert"] # time separation of three point function

        self.save_propagators = parameters["save_propagators"] # if save propagators
    
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
        
        src_seq = [g.mspincolor(prop.grid) for i in range(len(self.pol_list))]
        dst_seq = []
        dst_tmp = g.mspincolor(prop.grid)
        
        #g.qcd.baryon.proton_seq_src(prop, src_seq, self.t_insert, flavor)
        for i, pol in enumerate(self.pol_list):

            if (flavor == 1): 
                g.message("starting diquark contractions for up quark insertion and Polarization ", i)

                src_seq[i] = self.up_quark_insertion(prop, prop, Cg5, PolProjections[pol])
            elif (flavor == 2):
                g.message("starting diquark contractions for down quark insertion and Polarization ", i)

                src_seq[i] = self.down_quark_insertion(prop, Cg5, PolProjections[pol])
            else: 
                raise Exception("Unknown flavor for backward sequential src construction")
        
            # sequential solve through t=t_insert
            src_seq_t = g.lattice(src_seq[i])
            src_seq_t[:] = 0
            src_seq_t[:, :, :, self.t_insert] = src_seq[i][:, :, :, self.t_insert]

            g.message("diquark contractions for Polarization ", i, " done")
        
            smearing_input = g.eval(g.gamma[5]*P*g.conj(src_seq_t))

            tmp_prop = g.create.smear.boosted_smearing(trafo, smearing_input,w=self.width, boost=self.boost_out)

            dst_tmp = g.eval(inverter * tmp_prop)           
            dst_seq.append(g.eval(g.gamma[5] * g.conj(dst_tmp)))
        
        g.message("bw. seq propagator done")
        return dst_seq
    
    def contract_TMD(self, prop_f, prop_bw_seq, phases, W_index, tag, iW):
        
        for pol_index, fixed_pol_bwprop in enumerate(prop_bw_seq):
            corr = g.slice_trQPDF(prop_f,fixed_pol_bwprop,phases,3)
            pol_tag = tag + self.pol_list[pol_index]
            #save_qTMD_proton_hdf5(corr, tag, my_gammas, self.plist, W_index[2], W_index[0], W_index[1], W_index[3])
            if g.rank() == 0:
                print('g.rank():',g.rank(), ', pol_tag:', pol_tag)
                save_qTMD_proton_hdf5_subset(corr, pol_tag, my_gammas, self.plist, [W_index], iW)

        #return corr
    
    def create_TMD_Wilsonline_index_list(self):
        index_list = []
        
        for transverse_direction in [0,1]:
            for current_eta in self.eta:
                
                for current_bz in range(0, min([self.b_z, current_eta])):
                    for current_b_T in range(0, self.b_T):
                        
                        # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
                        index_list.append([current_b_T, current_bz, current_eta, transverse_direction])
                        
                        # create Wilson lines from all to all - (eta+bz) + b_perp - (eta-b_z)
                        index_list.append([current_b_T, -current_bz, -current_eta, transverse_direction])
                        
                        # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z+1)
                        #index_list.append([current_b_T, current_bz-0.5, current_eta+0.5, transverse_direction])
                        
                        # create Wilson lines from all to all - (eta+bz) + b_perp + (eta-b_z+1)
                        #index_list.append([current_b_T, -(current_bz-0.5), -(current_eta+0.5), transverse_direction])
                    
        return index_list
    
    def create_TMD_Wilsonline(self, U, index_set):

        assert len(index_set) == 4
        bt_index = index_set[0]
        bz_index = index_set[1]
        eta_index = index_set[2]
        transverse_dir = index_set[3]
        
        prv_link = g.qcd.gauge.unit(U[2].grid)[0]
        WL = prv_link
        
        for dz in range(0, eta_index+bz_index):
            WL = g.eval(prv_link * g.cshift(U[2], 2, dz))
            prv_link = WL
            
        for dx in range(0, bt_index):
            WL=g.eval(prv_link * g.cshift(g.cshift(U[transverse_dir], 2, eta_index+bz_index),transverse_dir, dx))
            prv_link=WL

        for dz in range(0, eta_index-bz_index):
            WL=g.eval(prv_link * g.adj(g.cshift(g.cshift(g.cshift(U[2], 2, eta_index+bz_index-1), transverse_dir, bt_index),2,-dz)))
            prv_link=WL

        return WL
            
    def down_quark_insertion(self, Q, Gamma, P):
        #eps_abc eps_a'b'c'Gamma_{beta alpha}Gamma_{beta'alpha'}P_{gamma gamma'}
        # * ( Q^beta'beta_b'b Q^gamma'gamma_{c'c} -  Q^beta'gamma_b'c Q^gamma'beta_{c'b} )
        
        eps = g.epsilon(Q.otype.shape[2])
        
        R = g.lattice(Q)
        
        PDu = g(g.spin_trace(P*Q))

        GtDG = g.eval(g.transpose(Gamma)*Q*Gamma)

        GtDG = g.separate_color(GtDG)
        PDu = g.separate_color(PDu)
        
        GtD = g.eval(g.transpose(Gamma)*Q)
        PDG = g.eval(P*Q*Gamma)
        
        GtD = g.separate_color(GtD)
        PDG = g.separate_color(PDG)
        
        D = {x: g.lattice(GtDG[x]) for x in GtDG}

        for d in D:
            D[d][:] = 0
            
        for i1, sign1 in eps:
            for i2, sign2 in eps:
                D[i1[0], i2[0]] += sign1 * sign2 * g.transpose((PDu[i2[2], i1[2]] * GtDG[i2[1], i1[1]] - GtD[i2[1],i1[2]] * PDG[i2[2], i1[1]]))
                
        g.merge_color(R, D)
        return R

    def up_quark_insertion(self, Qu, Qd, Gamma, P):

        eps = g.epsilon(Qu.otype.shape[2])
        R = g.lattice(Qu)
        Dut = g.lattice(Qu)

        #first term    
        GDdGt = g.eval(Gamma * Qd * g.transpose(Gamma))
        GDdGt = g.separate_color(GDdGt)
        
        D = {x: g.lattice(GDdGt[x]) for x in GDdGt}
        for d in D:
            D[d][:] = 0
            
        Du_sep = g.separate_color(Qu)
        Du_spintransposed = {x: g.lattice(Du_sep[x]) for x in Du_sep}
        for d in Du_spintransposed:
            Du_spintransposed[d] = g(g.transpose(Du_sep[d]))
        g.merge_color(Dut,Du_spintransposed)
        
        DuP = g(Dut*P)
        DuP = g.separate_color(DuP)
        
        #second term
        PDuG = g(P * Dut * Gamma)
        PDuG = g.separate_color(PDuG)
        DdGt = g(Qd * g.transpose(Gamma))
        DdGt = g.separate_color(DdGt)    
        
        
        #third term
        GDd = g.eval(Gamma * Qd)
        GDd = g.separate_color(GDd)

        GtDut = g.eval(g.transpose(Gamma) * Dut)
        GtDut = g.separate_color(GtDut)
            
        #fourth term
        PDu_trace = g(g.spin_trace(P * Dut))
        PDu_trace = g.separate_color(PDu_trace)

        #sum color indices
        for i1, sign1 in eps:
            for i2, sign2 in eps:
                D[i2[2], i1[2]] += sign1 * sign2 * (GDdGt[i1[1],i2[1]] * DuP[i1[0],i2[0]]
                                                    + PDuG[i1[0],i2[0]] * DdGt[i1[1],i2[1]] 
                                                    + P * g.spin_trace(GDd[i1[1],i2[1]] * GtDut[i1[0],i2[0]]) 
                                                    +  GDdGt[i1[1],i2[1]] * PDu_trace[i1[0],i2[0]])
                
        g.merge_color(R, D)
        return R
