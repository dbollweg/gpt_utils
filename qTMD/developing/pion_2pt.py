import os
import gpt as g
import numpy as np
from pion_ff_class import current_current_correlator, ordered_list_of_gammas, my_gammas
from tools import *
from io_corr import *
import sys

data_dir = "/lustre1/pion3d/xgao/run/gpt_pion_ff/data"

# Configuration parameters
parameters = {
    # Parameters needed for pion_measurement parent class
    "plist": np.array([[0, 0, nz, 0] for nz in range(0, 3)]),     # List of momenta, each momentum must be a list of 4D arrays
    "width": 4.0,                # Width parameter
    "pos_boost": [0, 0, 0],      # Positive boost
    "neg_boost": [0, 0, 0],      # Negative boost
    "save_propagators": False,     # Whether to save propagators

    # Additional parameters needed for current_current_correlator class
    "t_separation": 4,          # Time separation
    "quark_mom": np.array([[0, 0, int(nz/2), 0] for nz in range(0, 3)]),   # Quark momentum, must be 4D array
    "meson_mom": np.array([[0, 0, nz, 0] for nz in range(0, 3)])    # Meson momentum, must be 4D array
}

if len(sys.argv) > 1:
    n_conf = int(sys.argv[1])
else:
    g.message("Please provide configuration number as command line argument")
    sys.exit(1)


# Configuration path and number
conf_path = "/lustre1/pion3d/xgao/run/gpt_pion_ff/gauge"
lat_tag = "l48c64a060"
sm_tag = '1HYP'
sample_log_file = data_dir + "/sample_log/" + str(n_conf) + '_' + sm_tag
if g.rank() == 0:
    f = open(sample_log_file, "a+")
    f.close()

# Load gauge field configuration
g.message("Loading gauge configuration")
U_fixed = g.convert(g.load(f"{conf_path}/conf.{n_conf}.coulomb.1e-08.it0"), g.double)
grid = U_fixed[0].grid
trafo = g.identity(g.mspincolor(grid))
g.message("Finished loading gauge config")
L = U_fixed[0].grid.fdimensions
Ls, Lt = L[0], L[3]

# Load gauge field configuration
#g.message("Loading gauge configuration")
#Ls = 4
#Lt = 16
#grid = g.grid([Ls,Ls,Ls,Lt], g.double)
#rng = g.random("seed text")
#U = g.qcd.gauge.random(grid, rng)
#L = U[0].grid.fdimensions
#U_fixed, trafo = g.gauge_fix(U, maxiter=50000, prec=1e-7)
#g.message("Finished loading gauge config")

U_smear = g.qcd.gauge.smear.hyp(U_fixed, alpha = np.array([0.75, 0.6, 0.3])) # csw and quark mass parameters were all after HYP smearing

# Initialize measurement class
Measurement = current_current_correlator(parameters)

# Set up Wilson-Clover fermion action
pset = {
        "kappa": 0.12623,
        "csw_r": 1.0336,
        "csw_t": 1.0336,
        "xi_0": 1,
        "nu": 1,
        "isAnisotropic": False,
        "boundary_phases": [1, 1, 1, -1],
    }

w = g.qcd.fermion.wilson_clover(U_smear, pset)

# Set up solver
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg_exact = inv.cg({"eps": 1e-7, "maxiter": 10000})
cg_sloppy = inv.cg({"eps": 1e-4, "maxiter": 5000})
prop_exact = w.propagator(inv.preconditioned(pc.eo1_ne(), cg_exact))
prop_sloppy = w.propagator(inv.preconditioned(pc.eo1_ne(), cg_sloppy))


# Create sources and propagators
tslice_list = [8*i for i in range(0, int(Lt/8))]
poslist = [[0,0,0,tslice] for tslice in tslice_list]
g.message(f"Sources: {poslist}\n")
for ipos, pos in enumerate(poslist):

    sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
    g.message(f"START SAMPLE {ipos}/{len(poslist)}: {sample_log_tag}")

    with open(sample_log_file, "a+") as f:
        if sample_log_tag in f.read():
            g.message("SKIP SAMPLE: " + sample_log_tag)
            # continue # if the job has been done, skip it; may disable this during test

    ''' ------------------------------------ '''
    '''         Guassian point source        '''
    ''' ------------------------------------ '''
    g.message("Creating sources")
    srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U_smear[0].grid)

    # Calculate forward and backward propagators
    g.message("Computing forward/backward props")
    prop_f = g.eval(prop_exact * srcDp)
    prop_b = g.eval(prop_exact * srcDm)

    # Calculate two-point functions
    g.message("Computing 2pt functions")
    phases = Measurement.make_mom_phases(U_smear[0].grid, pos)
    tag = get_c2pt_file_tag(data_dir, lat_tag, n_conf, "ex", pos, sm_tag)
    Measurement.contract_2pt(prop_f, prop_b, phases, trafo, tag)

    del prop_f
    del prop_b

    meson_mom_list = parameters["meson_mom"]
    quark_mom_list = parameters["quark_mom"]
    pion_TrSum = []
    pion_SumTr = []

    for imom in range(0, len(meson_mom_list)):

        quark_mom = quark_mom_list[imom]
        meson_mom = meson_mom_list[imom]

        ''' -------------------------------------- '''
        ''' Coulomb-gauge Wall source; {Sum Trace} '''
        ''' -------------------------------------- '''
        
        # Create wall source propagators
        g.message("Creating wall source propagators") 
        fw_src = Measurement.create_wall_src(pos[-1], quark_mom, grid) # \sum_{x} e^{-Pq*(y-x)}
        bw_src = Measurement.create_wall_src(pos[-1], -(meson_mom - quark_mom), grid) # (\sum_{x} e^{-(PH-Pq)*(y-x)})^*

        # Calculate forward and backward propagators
        g.message("Computing forward/backward props")
        prop_f = g.eval(prop_exact * fw_src)
        prop_b = g.eval(prop_exact * bw_src)

        # Calculate two-point functions with wall source
        g.message("Computing 2pt functions with wall source: Sum Tr") 

        prop_f_Pq = g.eval(g.slice(Measurement.apply_phase(prop_f, -quark_mom, 1, grid), 3))
        prop_b_Pqbar = g.eval(g.slice(Measurement.apply_phase(prop_b, (meson_mom - quark_mom), 1, grid), 3))

        corr = [g.eval(g.trace(g.adj(prop_b_Pqbar[i])*prop_f_Pq[i]) ) for i in range(0, Lt)]
        g.message(f"corr {np.shape(corr)}: {corr}")
        pion_SumTr += [[corr]]

        del prop_f
        del prop_b

    tag = get_c2pt_file_tag(data_dir, lat_tag, n_conf, "ex_wall_SumTr", pos, sm_tag)
    if g.rank() == 0:
        save_c2pt_hdf5([pion_SumTr], tag, ["5"], meson_mom_list)

    g.message(f"DONE SAMPLE {ipos}/{len(poslist)}: {sample_log_tag}")
