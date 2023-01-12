from cmath import phase
from math import gamma
import gpt as g
import numpy as np
from io_corr import *

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

class pion_measurement:
    def __init__(self, parameters):
        self.plist = parameters["plist"]
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.save_propagators = parameters["save_propagators"]

    def set_output_facilities(self, corr_file, prop_file):
        self.output_correlator = g.corr_io.writer(corr_file)
        
        if(self.save_propagators):
            self.output = g.gpt_io.writer(prop_file)


    def set_input_facilities(self, corr_file):
        self.input_correlator = g.corr_io.reader(corr_file)

        
    def propagator_input(self, prop_file):
        g.message(f"Reading propagator file {prop_file}")
        read_props = g.load(prop_file)
        return read_props

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
        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #24D params
                "mass": 0.00107,
                "M5": 1.8,
                "b": 2.5,
                "c": 1.5,
                "Ls": 24,
                "boundary_phases": [1.0, 1.0, 1.0, -1.0],
                },
                    
        )

        l_sloppy = l_exact.converted(g.single)
        g.message(f"Loading eigenvectors from {evec_file}")
        g.mem_report(details=False)
        eig = g.load(evec_file, grids=l_sloppy.F_grid_eo)
        #g.message("pinning coarse evs to device memory")
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
                    g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 10000}),
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
                    g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 10000}),
                    mpi_split=g.default.get_ivec("--mpi_split", None, 4),
                ),
            ),
        )
        g.mem_report(details=False)
#        light_exact_inverter = g.algorithms.inverter.defect_correcting(
#            g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
#            eps=1e-8,
#            maxiter=10,
#        )

#        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(
#            g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
#            eps=1e-8,
#            maxiter=2,#
#        )


        ############### final inverter definitions
        prop_l_sloppy = l_sloppy.propagator(light_innerL_inverter).grouped(4)#light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_innerH_inverter).grouped(4)

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
                   g.algorithms.inverter.cg({"eps": 1e-7, "maxiter": 300}),
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
                    g.algorithms.inverter.cg({"eps": 1e-7, "maxiter": 300}),
                    mpi_split=g.default.get_ivec("--mpi_split", None, 4),
                ),
            ),
        )
        g.mem_report(details=False)
        light_exact_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-7,
            maxiter=12,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-7,
            maxiter=2,
        )


        ############### final inverter definitions
        prop_l_sloppy = l_sloppy.propagator(light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, prop_l_sloppy, pin

    def make_48I_inverter(self, U, evec_file):
        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #48I params
                "mass": 0.00078,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 14,
                "boundary_phases": [1.0, 1.0, 1.0, -1.0],
                },
                    
        )

        l_sloppy = l_exact.converted(g.single)
        g.message(f"Loading eigenvectors from {evec_file}")
        g.mem_report(details=False)
        eig = g.load(evec_file, grids=l_sloppy.F_grid_eo)
        g.message("pinning coarse evs to device memory")
        g.mem_report(details=False)
        pin = g.pin(eig[1], g.accelerator)
        g.message("creating deflated solvers")
        light_innerL_inverter = g.algorithms.inverter.preconditioned(
            g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd),
            g.algorithms.inverter.sequence(
                g.algorithms.inverter.coarse_deflate(
                    eig[1],
                    eig[0],
                    eig[2]
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
                    eig[2]
                ),
                g.algorithms.inverter.split(
                    g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 300}),
                    mpi_split=g.default.get_ivec("--mpi_split", None, 4),
                ),
            ),
        )
        g.mem_report(details=False)
        light_exact_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=10,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=2,
        )


        ############### final inverter definitions
        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, prop_l_sloppy, pin


    #make the inverters needed for the 96I lattices
    def make_96I_inverter(self, U, evec_file):

        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #96I params
                "mass": 0.00054,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 12,
                "boundary_phases": [1.0, 1.0, 1.0, -1.0],
            },

        )

        l_sloppy = l_exact.converted(g.single)

        eig = g.load(evec_file, grids=l_sloppy.F_grid_eo)
        # ## pin coarse eigenvectors to GPU memory
        pin = g.pin(eig[1], g.accelerator)


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
                    g.algorithms.inverter.cg({"eps": 1e-8, "maxiter": 300}),
                    mpi_split=g.default.get_ivec("--mpi_split", None, 4),
                ),
            ),
        )

        light_exact_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=10,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=2,
        )


        ############### final inverter definitions
        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, prop_l_sloppy, pin

    def make_debugging_inverter_mixed(self, U):

        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #48I params
                "mass": 0.00078,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 24,
                "boundary_phases": [1.0, 1.0, 1.0, 1.0],},
        )

        l_sloppy = l_exact.converted(g.single)

        light_innerL_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-4, maxiter = 5000))
        light_innerH_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-3, maxiter = 5000))

        light_exact_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
            eps=1e-4,
            maxiter=1000,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-4,
            maxiter=200,
        )

        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(6)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(6)
        return prop_l_exact, prop_l_sloppy

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

        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #64I params
                "mass": 0.0006203,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 12,
                "boundary_phases": [1.0, 1.0, 1.0, 1.0],},
                #48I params
                #"mass": 0.00078,
                #"M5": 1.8,
                #"b": 1.5,
                #"c": 0.5,
                #"Ls": 24,
                #"boundary_phases": [1.0, 1.0, 1.0, -1.0],},
        )

        l_sloppy = l_exact.converted(g.single)

        light_innerL_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-2, maxiter = 10000))
        light_innerH_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-2, maxiter = 200))

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
            W.append(g.eval(W[dz-1] * g.cshift(U[2], 2, dz)))
                
        return W


    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt(self, prop_f, prop_b, phases, trafo, tag):

        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        prop_b = g.create.smear.boosted_smearing(tmp_trafo, prop_b, w=self.width, boost=self.neg_boost)
     
        #corr = g.slice_trDA(prop_f,g.gamma[5]*g.adj(g.gamma[5]*prop_b*g.gamma[5]),phases, 3) 
        corr = g.slice_trDA(g.gamma[5]*g.adj(g.gamma[5]*prop_b*g.gamma[5]),prop_f,phases, 3) 

        if g.rank() == 0:
            save_c2pt_hdf5(corr, tag, my_gammas, self.plist)

    def contract_2pt_test(self, prop_f, prop_b, phases, trafo, tag):
        
        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        prop_b = g.create.smear.boosted_smearing(tmp_trafo, prop_b, w=self.width, boost=self.neg_boost)
     
        #corr = g.slice_trDA(prop_f,g.gamma[5]*g.adj(g.gamma[5]*prop_b*g.gamma[5]),phases, 3) 
        corr = g.slice_trDA(g.gamma[5]*g.adj(g.gamma[5]*prop_b*g.gamma[5]),prop_f,phases, 3) 

        corr_tag = "./2pt"
        corr_p = corr[0]
        for i, corr_mu in enumerate(corr_p):
            out_tag = f"{corr_tag}/p{self.plist[i]}"
            for j, corr_t in enumerate(corr_mu):
                g_tag = f"{out_tag}/{my_gammas[j]}"
                self.output_correlator.write(g_tag, corr_t)
                
                if "exact" in tag:
                    self.set_input_facilities("./correlators_exact_reference")
                elif "sloppy1" in tag:
                    self.set_input_facilities("./correlators_sloppy1_reference")
                else:
                    self.set_input_facilities("./correlators_sloppy2_reference")

                test_list = self.input_correlator.tags[g_tag]
                deviations = test_list-corr_t
               # g.message(test_list)
                assert (abs(deviations) < 1e-12).all()
                
        g.message(f"\033[1;32m Correlator {tag} is correct! \033[1;37;0m")
    

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



class pion_DA_measurement(pion_measurement):
    def __init__(self, parameters):
        self.zmax = parameters["zmax"]
        self.pzmin = parameters["pzmin"]
        self.pzmax = parameters["pzmax"]
        self.plist = [ [0,0, pz, 0] for pz in range(self.pzmin, self.pzmax)]
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]
        self.neg_boost = parameters["neg_boost"]
        self.save_propagators = parameters["save_propagators"]

    # Creating list of W*prop_b for all z
    def constr_backw_prop_for_DA(self, prop_b, W):
        g.message("Creating list of W*prop_b for all z")
        prop_list = [prop_b,]

        for z in range(1,self.zmax):
            prop_list.append(g.eval(g.gamma[5]*g.adj(g.gamma[5]*g.eval(W[z] * g.cshift(prop_b,2,z))*g.gamma[5])))
        
        return prop_list

    # Function that essentially defines our version of DA
    def contract_DA(self, prop_f, prop_b, phases, tag):

        # create and save correlators
        corr = g.slice_trDA(prop_b,prop_f,phases, 3)

        # corr = g.slice(
        #      g.trace(g.adj(prop_b) * W * g.gamma["Z"] * P * prop_f), 3)

        g.message("Starting IO")       
        for z, corr_p in enumerate(corr):
            corr_tag = "%s/DA/z%s" % (tag, str(z))
            for i, corr_mu in enumerate(corr_p):
                p_tag = f"{corr_tag}/p{self.plist[i]}"
                for j, corr_t in enumerate(corr_mu):
                    out_tag = f"{p_tag}/{my_gammas[j]}"
                    self.output_correlator.write(out_tag, corr_t)
                    g.message("Correlator %s\n" % out_tag, corr_t)

class TMD_WF_measurement(pion_measurement):
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

    def contract_TMD(self, prop_f, prop_b, phases, tag):

        corr = g.slice_trDA(prop_b,prop_f,phases, 3)

        #TODO ad io function from utils/tools

        g.message("Starting IO")       
        for z, corr_p in enumerate(corr):
            corr_tag = "%s/TMD/z%s" % (tag, str(z))
            for i, corr_mu in enumerate(corr_p):
                p_tag = f"{corr_tag}/p{self.plist[i]}"
                for j, corr_t in enumerate(corr_mu):
                    out_tag = f"{p_tag}/{my_gammas[j]}"
                    self.output_correlator.write(out_tag, corr_t)
                    g.message("Correlator %s\n" % out_tag, corr_t)
                    
    def create_src_TMD(self, pos, trafo, grid):
        
        srcD = g.mspincolor(grid)
        srcD[:] = 0
        
        g.create.point(srcD, pos)
        
        srcDm = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.neg_boost)
        srcDp = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.pos_boost)
        
        del srcD
        return srcDp, srcDm

    # def constr_backw_prop_for_TMD(self, prop_b, W):
    #     g.message("Creating list of W*prop_b for all z+b_perp-z+b_z")
    #     prop_list = [prop_b,]

    #     for z in range(1,self.eta+self.b_z//2):
    #         prop_list.append(g.eval(g.adj(W[z] * g.cshift(prop_b,2,z))))
    #     for x in range(0,self.b_T):
    #         prop_list.append(g.eval(g.adj(W[self.eta + self.b_z//2 + x] * g.cshift(prop_b,0,x))))
    #     for z in range(0,self.eta-self.b_z//2):
    #         prop_list.append(g.eval(g.adj(W[self.eta + self.b_z//2 + self.b_T + z] * g.cshift(prop_b,2,-z))))
        
    #     return prop_list


    def constr_TMD_bprop(self, prop_b, W):
        #prop_list = [prop_b,]
        prop_list = []

        td_offset = self.b_T*self.b_z*len(self.eta)
        eta_offset = self.b_T*self.b_z
        bz_offset = self.b_T
        
        for transverse_direction in [0,1]: 
            for eta_idx, current_eta in enumerate(self.eta):
                for current_bz in range(0, self.b_z):
                    for current_b_T in range(0, self.b_T):

                        W_index = current_b_T + bz_offset*current_bz + eta_offset*eta_idx + td_offset*transverse_direction
                        
                        #g_{src}*adj(g5*WL*bprop*g5), to be contracted with fprop and phases
                        
                        prop_list.append(g.eval(g.gamma[5]*g.adj(g.gamma[5]*g.eval(W[W_index] * g.cshift(g.cshift(prop_b,transverse_direction,current_b_T),2,2*current_bz))*g.gamma[5])))

        return prop_list
        
    def create_TMD_WL(self, U):

        W = []

        for transverse_direction in [0,1]:
            for current_eta in self.eta:
                for current_bz in range(0, self.b_z):
                    for current_b_T in range (0, self.b_T):
                        
                        prv_link = g.qcd.gauge.unit(U[2].grid[0])
                        current_link = prv_link
                        for dz in range(0, current_eta+current_bz):
                            current_link=g.eval(prv_link * g.cshift(U[2],2, dz))
                            prv_link=current_link
                            
                        for dx in range(0, current_b_T):
                            current_link=g.eval(prv_link * g.cshift(U[transverse_direction],transverse_direction, dx))
                            prv_link=current_link

                        
                        for dz in range(0, current_eta-current_bz):
                            current_link=g.eval(prv_link * g.cshift(U[2],2,-dz))
                            prv_link=current_link
                            
                        W.append(current_link)
        return W

    
class pion_ff_measurement(pion_measurement):
    def __init__(self, parameters):
        self.p = parameters["pf"]
        self.q = parameters["q"]
        self.plist = [self.q,] 
        self.t_insert = parameters["t_insert"]
        self.width = parameters["width"]
        self.boost_in = parameters["boost_in"]
        self.boost_out = parameters["boost_out"]
        self.pos_boost = self.boost_in
        self.neg_boost = [-self.pos_boost[0], -self.pos_boost[1], -self.pos_boost[2]]
        self.save_propagators = parameters["save_propagators"]

    def contract_FF(self, prop_f, prop_b, phases, tag):

        #This should work, as both prop are not lists! 
        corr = g.slice_trQPDF(prop_b, prop_f, phases, 3)


        g.message("Starting IO")    
        for z, corr_p in enumerate(corr):
            corr_tag = "%s/FF" % (tag)
            for i, corr_mu in enumerate(corr_p):
                p_tag = f"{corr_tag}/pf{self.p}/q{self.q}"
                for j, corr_t in enumerate(corr_mu):
                    out_tag = f"{p_tag}/{my_gammas[j]}"
                    self.output_correlator.write(out_tag, corr_t)
                    #g.message("Correlator %s\n" % out_tag, corr_t)

    def create_bw_seq(self, inverter, prop, trafo):

        tmp_trafo = g.convert(trafo, prop.grid.precision)

        ss_prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)

        del prop

        pp = 2.0 * np.pi * np.array(self.p) / ss_prop.grid.fdimensions
        P = g.exp_ixp(pp)

        # sequential solve through t=insertion_time
        t_op = self.t_insert
        src_seq = g.lattice(ss_prop)
        src_seq[:] = 0
        src_seq[:, :, :, t_op] = ss_prop[:, :, :, t_op]

        del ss_prop

        src_seq @=  g.gamma[5]* src_seq
        #multiply with complex conjugate phase because it's a backwards prop.
        src_seq @= g.adj(P)* src_seq 


        #does overwriting on the fly work?
        bw_boost = [-self.boost_out[0], -self.boost_out[1], -self.boost_out[2]]
        src_seq = g.create.smear.boosted_smearing(tmp_trafo, src_seq, w=self.width, boost=bw_boost)

        dst_seq = g.lattice(src_seq)
        dst_seq @= inverter * src_seq

        #This is now in principle B^dagger_zx but with the complex conj phase and a missing factor of gamma5 

        return (g.adj(dst_seq)*g.gamma[5])

class pion_qpdf_measurement(pion_measurement):
    def __init__(self, parameters):
        self.zmax = parameters["zmax"]
        self.p = parameters["pf"]
        self.q = parameters["q"]
        self.plist = [self.q,]
        self.t_insert = parameters["t_insert"]
        self.width = parameters["width"]
        self.boost_in = parameters["boost_in"]
        self.boost_out = parameters["boost_out"]
        self.pos_boost = self.boost_in
        self.neg_boost = [-self.pos_boost[0], -self.pos_boost[1], -self.pos_boost[2]]
        self.save_propagators = parameters["save_propagators"]

    def contract_QPDF(self, prop_f, prop_b, phases, tag):
 
        corr = g.slice_trQPDF(prop_b, prop_f, phases, 3)


        g.message("Starting IO")
        for z, corr_p in enumerate(corr):
            corr_tag = "%s/QPDF" % (tag)
            for i, corr_mu in enumerate(corr_p):
                p_tag = f"{corr_tag}/pf{self.p}/q{self.q}"
                for j, corr_t in enumerate(corr_mu):
                    out_tag = f"{p_tag}/{my_gammas[j]}"
                    self.output_correlator.write(out_tag, corr_t)
                    #g.message("Correlator %s\n" % out_tag, corr_t)


    def create_fw_prop_QPDF(self, prop_f, W):
        g.message("Creating list of W*prop_f for all z")
        prop_list = [prop_f,]

        for z in range(1,self.zmax):
            prop_list.append(g.eval(W[z]*g.cshift(prop_f,2,z)))
        
        return prop_list      

    def create_bw_seq(self, inverter, prop, trafo):

        tmp_trafo = g.convert(trafo, prop.grid.precision)

        ss_prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)

        del prop

        pp = 2.0 * np.pi * np.array(self.p) / ss_prop.grid.fdimensions
        P = g.exp_ixp(pp)

        # sequential solve through t=insertion_time
        t_op = self.t_insert
        src_seq = g.lattice(ss_prop)
        src_seq[:] = 0
        src_seq[:, :, :, t_op] = ss_prop[:, :, :, t_op]

        del ss_prop

        src_seq @=  g.gamma[5]* src_seq
        #multiply with complex conjugate phase because it's a backwards prop.
        src_seq @= g.adj(P)* src_seq


        #does overwriting on the fly work?
        bw_boost = [-self.boost_out[0], -self.boost_out[1], -self.boost_out[2]]
        src_seq = g.create.smear.boosted_smearing(tmp_trafo, src_seq, w=self.width, boost=bw_boost)

        dst_seq = g.lattice(src_seq)
        dst_seq @= inverter * src_seq

        #This is now in principle B^dagger_zx but with the complex conj phase and a missing factor of gamma5 

        return (g.adj(dst_seq)*g.gamma[5])

class proton_measurement:
    def __init__(self, parameters):
        self.plist = parameters["plist"]
        self.pol_list = ["P+_Sz+","P+_Sx+","P+_Sx-"]
        self.width = parameters["width"]
        self.pos_boost = parameters["pos_boost"]


    def set_output_facilites(self, corr_file, prop_file):
        self.output_correlator = g.corr_io.writer(corr_file)

        if(self.save_propagators):
            self.output = g.gpt_io.writer(prop_file)


    def propagator_output(self, tag, prop_f):

        g.message("Saving forward propagator")
        prop_f_tag = "%s/%s" % (tag, str(self.pos_boost))
        self.output.write({prop_f_tag: prop_f})
        self.output.flush()
        g.message("Propagator IO done")


    def make_debugging_inverter(self, U):

        l_exact = g.qcd.fermion.mobius(
            U,
            {
                "mass": 0.00054,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 12,
                "boundary_phases": [1.0, 1.0, 1.0, -1.0],},
        )

        l_sloppy = l_exact.converted(g.single)

        light_innerL_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd), g.algorithms.inverter.cg(eps = 1e-8, maxiter = 200))
        light_innerH_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo1_ne(parity=g.odd), g.algorithms.inverter.cg(eps = 1e-8, maxiter = 300))

        light_exact_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerH_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=10,
        )

        light_sloppy_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=2,
        )

        prop_l_sloppy = l_exact.propagator(light_sloppy_inverter).grouped(4)
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)
        return prop_l_exact, prop_l_sloppy
    
    # define transitions between grids (setup)
    def find_near_null_vectors(self, w, cgrid):
        rng = g.random("test_mg")
        slv = g.algorithms.inverter.fgmres(eps=1e-3, maxiter=50, restartlen=25, checkres=False)(w)
        basis = g.orthonormalize(rng.cnormal([w.vector_space[0].lattice() for i in range(15)]))
        null = g.lattice(basis[0])
        null[:] = 0
        for b in basis:
            slv(b, null)
    
        # TODO: apply open boundaries, e.g., in this function
        g.qcd.fermion.coarse.split_chiral(basis)
        bm = g.block.map(cgrid, basis)
        bm.orthonormalize()
        bm.check_orthogonality()
        return basis

    def make_Clover_MG_inverter(self, U):

        l_exact = g.qcd.fermion.wilson_clover(
        U,
        {
            "kappa": 0.12655,
            "csw_r": 1.0372,
            "csw_t": 1.0372,
            "xi_0": 1,
            "nu": 1,
            "isAnisotropic": False,
            "boundary_phases": [1.0, 1.0, 1.0, 1.0],
            },
        )

        # l_sloppy = l_exact.converted(g.single)

        # mg_setup_2lvl = g.algorithms.inverter.multi_grid_setup(block_size=[[2, 2, 2, 2]], projector=self.find_near_null_vectors)

        mg_setup_3lvl = g.algorithms.inverter.multi_grid_setup(
            block_size=[[2, 2, 2, 2], [2, 1, 1, 1]], projector=self.find_near_null_vectors
        )

        # mg_setup_2lvl_dp = mg_setup_2lvl(l_exact)
        mg_setup_3lvl_sp = mg_setup_3lvl(l_exact)

        # mg inner solvers
        wrapper_solver = g.algorithms.inverter.fgmres({"eps": 1e-1, "maxiter": 10, "restartlen": 5, "checkres": False})
        smooth_solver = g.algorithms.inverter.fgmres({"eps": 1e-14, "maxiter": 8, "restartlen": 4, "checkres": False})
        coarsest_solver = g.algorithms.inverter.fgmres({"eps": 5e-2, "maxiter": 50, "restartlen": 25, "checkres": False})

        # lvl 2 mg solver/preconditioner objects
        # mg_2lvl_vcycle_dp = g.algorithms.inverter.sequence(
        #     g.algorithms.inverter.coarse_grid(coarsest_solver, *mg_setup_2lvl_dp[0]),
        #     g.algorithms.inverter.calculate_residual(
        #         "before smoother"
        #     ),  # optional since it costs time but helps to tune MG solver
        #     smooth_solver,
        #     g.algorithms.inverter.calculate_residual("after smoother"),  # optional
        # )

        mg_3lvl_kcycle_sp = g.algorithms.inverter.sequence(
        g.algorithms.inverter.coarse_grid(
            wrapper_solver.modified(
                prec=g.algorithms.inverter.sequence(
                    g.algorithms.inverter.coarse_grid(coarsest_solver, *mg_setup_3lvl_sp[1]), smooth_solver
                )
            ),
            *mg_setup_3lvl_sp[0],
        ),
        smooth_solver,
        )

        # outer solver
        fgmres_params = {"eps": 1e-6, "maxiter": 1000, "restartlen": 20}

        # fgmres_outer = g.algorithms.inverter.fgmres(fgmres_params, prec=mg_2lvl_vcycle_dp)
        fgmres_outer = g.algorithms.inverter.fgmres(fgmres_params, prec=mg_3lvl_kcycle_sp)

        return fgmres_outer(l_exact)
        

    ############## make list of complex phases for momentum proj.
    def make_mom_phases(self, grid):    
        one = g.identity(g.complex(grid))
        pp = [-2 * np.pi * np.array(p) / grid.fdimensions for p in self.plist]
        P = g.exp_ixp(pp)
        mom = [g.eval(pp*one) for pp in P]
        return mom

    # create Wilson lines from all --> all + dz for all dz in 0,zmax
    def create_WL(self, U):
        W = []
        W.append(g.qcd.gauge.unit(U[2].grid)[0])
        for dz in range(0, self.zmax):
            W.append(g.eval(W[dz-1] * g.cshift(U[2], 2, dz)))
                
        return W


    #function that does the contractions for the smeared-smeared pion 2pt function
    def contract_2pt(self, prop_f, phases, trafo, tag):

        g.message("Begin sink smearing")
        tmp_trafo = g.convert(trafo, prop_f.grid.precision)

        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        g.message("Sink smearing completed")

        #This ans the IO still need work
        corr = g.slice_proton(prop_f, phases, 3) 

        #do correlator output 
        corr_tag = "%s/2pt" % (tag)
        for i, corr_pol in enumerate(corr):
            out_tag = f"{corr_tag}/Pol{self.pol_list[i]}"
            for j, corr_p in enumerate(corr_pol):
                out_tag = f"{corr_tag}/p{self.plist[j]}"
                self.output_correlator.write(out_tag, corr_p)
                #g.message("Correlator %s\n" % out_tag, corr_t)

    #function that creates boosted, smeared src.
    def create_src(self, pos, trafo, grid):
        
        srcD = g.mspincolor(grid)
        srcD[:] = 0
        
        g.create.point(srcD, pos)

        srcDp = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.pos_boost)

        del srcD

        return srcDp     


class proton_qpdf_measurement(proton_measurement):
   
    def __init__(self, parameters):
        self.zmax = parameters["zmax"]
        self.p = parameters["pf"]
        self.q = parameters["q"]
        self.plist = [self.q,]
        self.pol_list = ["P+_Sz+","P+_Sx+","P+_Sx-"]
        #self.Gamma = parameters["gamma"]
        self.t_insert = parameters["t_insert"]
        self.width = parameters["width"]
        self.boost_in = parameters["boost_in"]
        self.boost_out = parameters["boost_out"]
        self.pos_boost = self.boost_in
        self.save_propagators = parameters["save_propagators"]



    def create_fw_prop_QPDF(self, prop_f, W):
        g.message("Creating list of W*prop_f for all z")
        prop_list = [prop_f,]

        for z in range(1,self.zmax):
            prop_list.append(g.eval(W[z]*g.cshift(prop_f,2,z)))
        
        return prop_list  

    def create_bw_seq(self, inverter, prop, trafo):
        tmp_trafo = g.convert(trafo, prop.grid.precision)

        #Make SS propagator
        prop = g.create.smear.boosted_smearing(tmp_trafo, prop, w=self.width, boost=self.boost_out)

        pp = 2.0 * np.pi * np.array(self.p) / prop.grid.fdimensions
        P = g.exp_ixp(pp)

        # sequential solve through t=insertion_time for all 3 proton polarizations
        src_seq = [g.mspincolor(prop.grid) for i in range(3)]
        dst_seq = []
        #g.mem_report(details=True)
        g.message("starting diquark contractions")
        g.qcd.baryon.proton_seq_src(prop, src_seq, self.t_insert)
        g.message("diquark contractions done")
        dst_tmp = g.mspincolor(prop.grid)
        for i in range(3):

            dst_tmp @= inverter * g.create.smear.boosted_smearing(tmp_trafo, g.eval(g.gamma[5]* P* g.conj(src_seq[i])), w=self.width, boost=self.boost_out)
            #del src_seq[i]
            dst_seq.append(g.eval(g.gamma[5] * g.conj( dst_tmp)))
        g.message("bw. seq propagator done")
        return dst_seq            


    def contract_QPDF(self, prop_f, prop_bw, phases, tag):
 
        #This and the IO still need work

        for pol in self.pol_list:
            corr = g.slice_trQPDF(prop_bw, prop_f, phases, 3)

            corr_tag = f"{tag}/QPDF/Pol{pol}"
            for z, corr_p in enumerate(corr):
                for i, corr_mu in enumerate(corr_p):
                    p_tag = f"{corr_tag}/pf{self.p}/q{self.q}"
                    for j, corr_t in enumerate(corr_mu):
                        out_tag = f"{p_tag}/{my_gammas[j]}"
                        self.output_correlator.write(out_tag, corr_t)
