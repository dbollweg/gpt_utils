from cmath import phase
from math import gamma
import gpt as g
import numpy as np
from utils.io_corr import *

#ordered list of gamma matrix identifiers, needed for the tag in the correlator output
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]

ordered_list_of_gammas = [g.gamma[5], g.gamma["T"], g.gamma["T"]*g.gamma[5],
                                      g.gamma["X"], g.gamma["X"]*g.gamma[5], 
                                      g.gamma["Y"], g.gamma["Y"]*g.gamma[5],
                                      g.gamma["Z"], g.gamma["Z"]*g.gamma[5], 
                                      g.gamma["I"], g.gamma["SigmaXT"], 
                                      g.gamma["SigmaXY"], g.gamma["SigmaXZ"], 
                                      g.gamma["SigmaZT"]
                            ]

'''---------------------------------------------'''
'''-------------------- pion -------------------'''
'''---------------------------------------------'''

class pion_measurement:
    def __init__(self, parameters):
        self.plist = parameters["plist"]
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.save_propagators = parameters["save_propagators"]

    def set_input_facilities(self, corr_file):
        self.input_correlator = g.corr_io.reader(corr_file)

    def set_output_facilities(self, corr_file, prop_file):
        self.output_correlator = g.corr_io.writer(corr_file)
        
        if(self.save_propagators):
            self.output = g.gpt_io.writer(prop_file)

    def propagator_input(self, prop_file):
        g.message(f"Reading propagator file {prop_file}")
        read_props = g.load(prop_file)
        return read_props

    def propagator_output_k0(self, tag, prop_f):

        g.message("Saving forward propagator")
        prop_f_tag = "%s/%s" % (tag, str(self.pos_boost))
        self.output.write({prop_f_tag: prop_f})
        self.output.flush()
        g.message("Propagator IO done")

    def propagator_output(self, tag, prop_f, prop_b):

        g.message("Saving forward propagator")
        prop_f_tag = "%s/%s" % (tag, str(self.pos_boost)) 
        self.output.write({prop_f_tag: prop_f})
        self.output.flush()
        g.message("Saving backward propagator")
        prop_b_tag = "%s/%s" % (tag, str(self.neg_boost))
        self.output.write({prop_b_tag: prop_b})
        self.output.flush()
        g.message("Propagator IO done")

    def make_24D_inverter(self, U, evec_file):

        l_exact = g.qcd.fermion.zmobius(
            #g.convert(U, g.single),
            U,
            {
                "mass": 0.00107,
                "M5": 1.8,
                "b": 1.0,
                "c": 0.0,
                "omega": [
                    1.0903256131299373,
                    0.9570283702230611,
                    0.7048886040934104,
                    0.48979921782791747,
                    0.328608311201356,
                    0.21664245377015995,
                    0.14121112711957107,
                    0.0907785101745156,
                    0.05608303440064219 - 0.007537158177840385j,
                    0.05608303440064219 + 0.007537158177840385j,
                    0.0365221637144842 - 0.03343945161367745j,
                    0.0365221637144842 + 0.03343945161367745j,
                ],
                "boundary_phases": [1.0, 1.0, 1.0, -1.0],
            },
        )

        l_sloppy = l_exact.converted(g.single)
        g.message(f"Loading eigenvectors from {evec_file}")
        g.mem_report(details=False)
        eig = g.load(evec_file, grids=l_sloppy.F_grid_eo)

        g.mem_report(details=False)
        pin = g.pin(eig[1], g.accelerator)
        g.message("creating deflated solvers")

        g.message("creating deflated solvers")
        light_innerL_inverter = g.algorithms.inverter.preconditioned(
           g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
           g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        light_innerH_inverter = g.algorithms.inverter.preconditioned(
            g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
            g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-4, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        g.mem_report(details=False)
        light_exact_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=12,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
            eps=1e-4,
            maxiter=12,
        )

        ############### final inverter definitions
        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, prop_l_sloppy, pin
    
    def make_64I_inverter(self, U, evec_file):
        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #64I params
                "mass": 0.000678,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 12,
                "boundary_phases": [1.0, 1.0, 1.0, 1.0],
                },

        )

        l_sloppy = l_exact.converted(g.single)
        g.message(f"Loading eigenvectors from {evec_file}")
        g.mem_report(details=False)
        eig = g.load(evec_file, grids=l_sloppy.F_grid_eo)

        g.mem_report(details=False)
        pin = g.pin(eig[1], g.accelerator)
        g.message("creating deflated solvers")

        light_innerL_inverter = g.algorithms.inverter.preconditioned(
           g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
           g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        light_innerH_inverter = g.algorithms.inverter.preconditioned(
            g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
            g.algorithms.inverter.sequence(
               g.algorithms.inverter.coarse_deflate(
                   eig[1],
                   eig[0],
                   eig[2],
                   block=400,
                   fine_block=4,
                   linear_combination_block=32,
               ),
               g.algorithms.inverter.split(
                   g.algorithms.inverter.cg({"eps": 1e-4, "maxiter": 200}),
                   mpi_split=g.default.get_ivec("--mpi_split", None, 4),
               ),
           ),
        )

        g.mem_report(details=False)
        light_exact_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=12,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
            eps=1e-4,
            maxiter=12,
        )


        ############### final inverter definitions
        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, prop_l_sloppy, pin

    def make_debugging_inverter(self, U):

        '''
        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #96I params
                #"mass": 0.00054,
                #"M5": 1.8,
                #"b": 1.5,
                #"c": 0.5,
                #"Ls": 12,
                #"boundary_phases": [1.0, 1.0, 1.0, -1.0],},
        #MDWF_2+1f_64nt128_IWASAKI_b2.25_ls12b+c2_M1.8_ms0.02661_mu0.000678_rhmc_HR_G
                #64I params
                #"mass": 0.0006203,
                #"M5": 1.8,
                #"b": 1.5,
                #"c": 0.5,
                #"Ls": 12,
                #"boundary_phases": [1.0, 1.0, 1.0, 1.0],},
                #48I params
                #"mass": 0.00078,
                #"M5": 1.8,
                #"b": 1.5,
                #"c": 0.5,
                #"Ls": 24,
                #"boundary_phases": [1.0, 1.0, 1.0, -1.0],},
        )
        '''

        l_exact = g.qcd.fermion.zmobius(
            #g.convert(U, g.single),
            U,
            {
                "mass": 0.00107,
                "M5": 1.8,
                "b": 1.0,
                "c": 0.0,
                "omega": [
                    1.0903256131299373,
                    0.9570283702230611,
                    0.7048886040934104,
                    0.48979921782791747,
                    0.328608311201356,
                    0.21664245377015995,
                    0.14121112711957107,
                    0.0907785101745156,
                    0.05608303440064219 - 0.007537158177840385j,
                    0.05608303440064219 + 0.007537158177840385j,
                    0.0365221637144842 - 0.03343945161367745j,
                    0.0365221637144842 + 0.03343945161367745j,
                ],
                "boundary_phases": [1.0, 1.0, 1.0, -1.0],
            },
        )

        l_sloppy = l_exact.converted(g.single)

        light_innerL_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-2, maxiter = 10000))
        light_innerH_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-8, maxiter = 200))

        prop_l_sloppy = l_exact.propagator(light_innerH_inverter).grouped(6)
        prop_l_exact = l_exact.propagator(light_innerL_inverter).grouped(6)
        return prop_l_exact, prop_l_sloppy

    ############## make list of complex phases for momentum proj.
    def make_mom_phases(self, grid, origin=None):    
        one = g.identity(g.complex(grid))
        pp = [-2 * np.pi * np.array(p) / grid.fdimensions for p in self.plist]
       
        P = g.exp_ixp(pp, origin)
       
        mom = [g.eval(pp*one) for pp in P]
        return mom

    # create Wilson lines from all --> all + dz for all dz in 0,zmax
    def create_WL(self, U):
        W = []
        W.append(g.qcd.gauge.unit(U[2].grid)[0])
        for dz in range(0, self.zmax):
            W.append(g.eval(W[dz] * g.cshift(U[2], 2, dz)))
                
        return W

    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt(self, prop_f, prop_b, phases, trafo, tag):

        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        prop_b = g.create.smear.boosted_smearing(tmp_trafo, prop_b, w=self.width, boost=self.neg_boost)

        #corr = g.slice_trDA(prop_f,g.gamma[5]*g.adj(g.gamma[5]*prop_b*g.gamma[5]),phases, 3) 
        corr = g.slice_trDA(g.gamma[5]*g.adj(g.gamma[5]*prop_b*g.gamma[5]), prop_f, phases, 3) 
        #corr = g.slice_trDA(prop_f,g.adj(prop_b),phases, 3)
        if g.rank() == 0:
            save_c2pt_hdf5(corr, tag, my_gammas, self.plist)
        del corr 

    #function that creates boosted, smeared src.
    def create_src_2pt(self, pos, trafo, grid):
        
        srcD = g.mspincolor(grid)
        
        
        g.create.point(srcD, pos)
        g.message("point src set")

        srcDm = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.neg_boost)
        g.message("pos. boosted src done")
        
        srcDp = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.pos_boost)
        g.message("neg. boosted src done")
        
        return srcDp, srcDm


'''---------------------------------------------'''
'''-------------------pion DA-------------------'''
'''---------------------------------------------'''

class pion_DA_measurement(pion_measurement):
    def __init__(self,parameters):
        self.zmax = parameters["zmax"]
        self.pzmin = parameters["pzmin"]
        self.pzmax = parameters["pzmax"]
        self.plist = [ [0,0, pz, 0] for pz in range(self.pzmin,self.pzmax)]
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.save_propagators = parameters["save_propagators"]

    # create Wilson lines from all --> all +- dz for all dz in 0,zmax
    def create_DA_WL(self, U):

        index_list = []

        # z in [0, zmax-1]
        W = []
        W.append(g.qcd.gauge.unit(U[2].grid)[0]) # z = 0
        for dz in range(0, self.zmax-1):
            W.append(g.eval(W[dz] * g.cshift(U[2], 2, dz))) # z from 1 to zmax-1
        index_list = [[0,i,0,0] for i in range(0, self.zmax)]
        # z in [-1, -(zmax-1)] NOTE: V[dz] is just corrected, may need double check
        V = []
        V.append(g.qcd.gauge.unit(U[2].grid)[0])
        for dz in range(-1, -self.zmax, -1):
            V.append(g.eval(V[-dz-1] * g.adj(g.cshift(U[2], 2, dz))))
        index_list += [[0,i,0,0] for i in range(-1, -self.zmax, -1)]
                
        return W+V[1:], index_list

    # Creating list of W*prop_b for all z
    def constr_DA_bprop(self, prop_b, W, W_index_list):

        prop_list = []
        # W_index_list[i] = bz
        for i, idx in enumerate(W_index_list):
            current_bz = idx[1]
            prop_list.append(g.eval(g.gamma[5]*g.adj(g.gamma[5]*g.eval(W[i] * g.cshift(prop_b,2,current_bz))*g.gamma[5])))
        return prop_list

    def contract_DA(self, prop_f, prop_b, phases, tag, W_index_list, i_sub):

        corr = g.slice_trDA(prop_b,prop_f,phases, 3)
        if g.rank() == 0:
            #self.save_qTMDWF_hdf5(corr, tag, my_gammas)
            save_qTMDWF_hdf5_subset(corr, tag, my_gammas, self.plist, W_index_list, i_sub)
        del corr



'''---------------------------------------------'''
'''-----------------pion TMDWF------------------'''
'''---------------------------------------------'''

class pion_TMDWF_measurement(pion_measurement):
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

    # if i_sub=0, will create a new .h5 file, elif i_sub != 0, will add data into exist .h5 file
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

        prop_list = []
        # W_index_list[i] = [bT, bz, eta, Tdir]
        for i, idx in enumerate(W_index_list):
            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]
            prop_list.append(g.eval(g.gamma[5]*g.adj(g.gamma[5]*g.eval(W[i] * g.cshift(g.cshift(prop_b,transverse_direction,current_b_T),2,round(2*current_bz)))*g.gamma[5])))
        return prop_list

    def constr_TMD_bprop_TEST(self, prop_b, W, W_index_list):

        prop_list = []
        # W_index_list[i] = [bT, bz, eta, Tdir]
        for i, idx in enumerate(W_index_list):
            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]
            g.message(f"index: {idx}, step 1")
            g.eval(W[i] * g.cshift(g.cshift(prop_b,transverse_direction,current_b_T),2,round(2*current_bz)) * prop_b)
            g.message(f"index: {idx}, step 2")
            g.eval(W[i] * g.cshift(g.cshift(prop_b,transverse_direction,current_b_T),2,round(2*current_bz)))
            g.message(f"index: {idx}, step 3")
            prop_list.append(g.eval(g.gamma[5]*g.adj(g.gamma[5]*g.eval(W[i] * g.cshift(g.cshift(prop_b,transverse_direction,current_b_T),2,round(2*current_bz)))*g.gamma[5])))
            g.message(f"index: {idx}, step 4")
        return prop_list

    def create_TMD_WL(self, U):

        W = []
        index_list = []

        # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
        for transverse_direction in [0,1]:
            for current_eta in self.eta:
                if current_eta == 12:
                    #b_z_min, b_z_max = 0, self.b_z
                    b_T_min, b_T_max = 0, self.b_T
                    bzlist = [i for i in range(0, self.b_z)]
                else:
                    #b_z_min, b_z_max = 8, 9
                    b_T_min, b_T_max = 8, 9
                    bzlist = [0, 8]
                #for current_bz in range(0, self.b_z):
                #for current_bz in range(b_z_min, b_z_max):
                for current_bz in bzlist:
                    #for current_b_T in range (0, self.b_T):
                    for current_b_T in range (b_T_min, b_T_max):
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
                        if current_eta == 12:
                            W.append(g.qcd.gauge.unit(U[2].grid)[0])
                            index_list.append([current_b_T, current_bz, 0, transverse_direction])
        return W, index_list
    
    # This one include the odd z beyond z=2b_z. Also include a eta'=eta+1 with b_z=0.
    def create_TMD_WL_odd(self, U):

        W = []
        index_list = []

        # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
        for transverse_direction in [0,1]:
            for current_eta in self.eta:
                for current_bz in range(0, min([self.b_z, current_eta])):
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

        return W, index_list
    
    # This one include the odd z beyond z=2b_z. Also include a eta'=eta+1 with b_z=0.
    # reduced the number of loops
    # even & odd,  plus, adding minus
    def create_TMD_WL_eo_pm(self, U):

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
