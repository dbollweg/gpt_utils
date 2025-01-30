import gpt as g
import gpt.create as creator
import numpy as np
from utils.io_corr import *
import time

from pyquda import init, LatticeInfo
from pyquda_utils import core, gpt, gamma

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

    def set_output_facilities(self, corr_file, prop_file):
        """Set correlator and propagator output filenames."""
        self.output_correlator = g.corr_io.writer(corr_file)
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

class current_current_correlator(pion_measurement):
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

    def create_wall_propagator(self, dirac, src, grid):
        start = time.time()

        src_pyquda = gpt.LatticePropagatorGPT(src, GEN_SIMD_WIDTH)
        src_pyquda.toDevice()
        prop_pyquda = core.invertPropagator(dirac, src_pyquda, 0)
        prop = g.mspincolor(grid)
        gpt.LatticePropagatorGPT(prop, GEN_SIMD_WIDTH, prop_pyquda)

        del src_pyquda, prop_pyquda
        g.message("TIME: fw prop inversion", time.time() - start)

        return prop