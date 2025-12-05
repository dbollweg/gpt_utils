import gpt as g
import gpt.create as creator
import numpy as np
from utils.io_corr import *
import time

my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]

ordered_list_of_gammas = [g.gamma[5], g.gamma["T"], g.gamma["T"]*g.gamma[5],
                                      g.gamma["X"], g.gamma["X"]*g.gamma[5], 
                                      g.gamma["Y"], g.gamma["Y"]*g.gamma[5],
                                      g.gamma["Z"], g.gamma["Z"]*g.gamma[5], 
                                      g.gamma["I"], g.gamma["SigmaXT"], 
                                      g.gamma["SigmaXY"], g.gamma["SigmaXZ"], 
                                      g.gamma["SigmaZT"]
                            ]
GEN_SIMD_WIDTH = 64

class pion_measurement:
    def __init__(self, parameters):
        self.save_propagators = parameters["save_propagators"]

    def set_output_facilities(self, prop_file):
        """Set correlator and propagator output filenames."""
        #self.output_correlator = g.corr_io.writer(corr_file)
        self.output = g.gpt_io.writer(prop_file)
        
        #if(self.save_propagators):
        #    self.output = g.gpt_io.writer(prop_file)

    def propagator_output(self, prop_tag, prop):
        """Write forward and backward propagators to disk."""

        self.output.write({prop_tag: prop})
        self.output.flush()

    def set_input_facilities(self, corr_file):
        self.input_correlator = g.corr_io.reader(corr_file)
        
    def propagator_input(self, prop_file):
        g.message(f"Reading propagator file {prop_file}")
        read_props = g.load(prop_file)
        return read_props

    def make_debugging_inverter_mixed_exact_mass(self, U, mass):

        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #64I params
                "mass": mass,
                "M5": 1.8,
                "b": 1.5,
                "c": 0.5,
                "Ls": 12,
                "boundary_phases": [1.0, 1.0, 1.0, 1.0],
                },
        )

        light_innerL_inverter = g.algorithms.inverter.preconditioned(g.qcd.fermion.preconditioner.eo2_ne(), g.algorithms.inverter.cg(eps = 1e-8, maxiter = 200))

        light_exact_inverter = g.algorithms.inverter.defect_correcting(
            g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=1000,
        )

        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(6)
        return prop_l_exact
        
    def make_64I_inverter_mass_exact(self, U, evec_file, mass):
        l_exact = g.qcd.fermion.mobius(
            U,
            {
                #64I params
                "mass": mass,
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

        g.mem_report(details=False)
        light_exact_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=12,
        )

        ############### final inverter definitions
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, pin

    def make_64I_inverter_exact(self, U, evec_file):
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

        g.mem_report(details=False)
        light_exact_inverter = g.algorithms.inverter.defect_correcting(g.algorithms.inverter.mixed_precision(light_innerL_inverter, g.single, g.double),
            eps=1e-8,
            maxiter=12,
        )

        ############### final inverter definitions
        prop_l_exact = l_exact.propagator(light_exact_inverter).grouped(4)

        return prop_l_exact, pin

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

class pion_TMDWF_measurement(pion_measurement):
    def __init__(self,parameters):
        self.eta = parameters["eta"]
        self.b_z = parameters["bz_length"]
        self.b_T = parameters["bT_length"]

    # if i_sub=0, will create a new .h5 file, elif i_sub != 0, will add data into exist .h5 file
    def contract_TMD(self, prop_f, prop_b, phases, tag, W_index_list, i_sub):

        corr = g.slice_trDA(prop_b,prop_f,phases, 3)
        if g.rank() == 0:
            #self.save_qTMDWF_hdf5(corr, tag, my_gammas)
            save_qTMDWF_hdf5_subset(corr, tag, my_gammas, self.plist, W_index_list, i_sub)
        del corr

    def constr_TMD_bprop(self, prop_b, W, W_index_list):

        prop_list = []
        # W_index_list[i] = [bT, bz, eta, Tdir]
        for i, idx in enumerate(W_index_list):
            current_b_T = idx[0]
            current_bz = idx[1]
            current_eta = idx[2]
            transverse_direction = idx[3]
            prop_list.append(g.eval(g.adj(g.gamma[5]*g.eval(W[i] * g.cshift(g.cshift(prop_b,transverse_direction,current_b_T),2,round(2*current_bz)))*g.gamma[5])))
        return prop_list

    def create_TMD_WL(self, U):

        W = []
        index_list = []

        # create Wilson lines from all to all + (eta+bz) + b_perp - (eta-b_z)
        for transverse_direction in [0,1]:
            for current_eta in self.eta:

                if current_eta == 12:
                    b_T_min, b_T_max = 0, self.b_T
                    bzlist = [i for i in range(0, self.b_z)]
                else:
                    b_T_min, b_T_max = 0, self.b_T
                    bzlist = [0]

                for current_bz in bzlist:

                    for current_b_T in range (b_T_min, b_T_max):
                        prv_link = g.qcd.gauge.unit(U[2].grid)[0]
                        current_link = prv_link

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



class current_current_correlator(pion_TMDWF_measurement):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.quark_mom = parameters["quark_mom"]

    def apply_phase(self, prop, mom, sign, grid):
        one = g.identity(g.complex(grid))
        pp = sign * 2 * np.pi * np.array(mom) / grid.fdimensions
       
        #P = g.exp_ixp(pp)
        P = g.exp_ixp(pp)
       
        mom = g.eval(P*one)
        prop = prop * mom
        return prop

    def make_mom_phases_wall(self, grid, mom):
        """Create list of complex phases for momentum projection."""    
        one = g.identity(g.complex(grid))
        pp = [-2 * np.pi * np.array(p) / grid.fdimensions for p in [mom]]
       
        P = g.exp_ixp(pp)
       
        phases = [g.eval(pp*one) for pp in P]
        
        return phases

    def contract_2pt_wall(self, prop_f, prop_b, phases, mom, trafo, tag):

        corr = g.slice_trDA(g.gamma[5]*g.adj(g.gamma[5]*prop_b*g.gamma[5]), prop_f, phases, 3) 
        
        if g.rank() == 0:
            save_c2pt_hdf5(corr, tag, my_gammas, [mom])
        del corr 

    def create_wall_src(self, tslice, momentum, grid):
        """Create wall source with phase exp(ipy) at given timeslice."""
        
        src = g.identity(g.mspincolor(grid))
        coors = g.coordinates(src)
        
        src = self.apply_phase(src,momentum, 1, grid)
        mask = g.complex(grid)
        g.coordinate_mask(mask,np.array([1 if i[3] == tslice else 0 for i in coors]))

        src = g(src * mask)
        return src