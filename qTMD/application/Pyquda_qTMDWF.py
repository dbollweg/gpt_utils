# load python modules
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
import os
import time
import math
from mpi4py import MPI

# load gpt modules
import gpt as g
from qTMD.gpt_qTMD_utils import pion_TMDWF_measurement
from utils.tools import *
from utils.io_corr import *

# load pyquda modules
from pyquda import init, LatticeInfo
from pyquda_utils import core, gpt, gamma

# Gobal parameters
data_dir = "/lustre/orion/nph158/proj-shared/xgao/l48c64a060/qTMDWF/data/"
sm_tag = "1HYP_M300_GSRC_W52_k5"
lat_tag = "HISQa060"
GEN_SIMD_WIDTH = 64
conf = g.default.get_int("--config_num", 0)
g.message(f"--config_num {conf}")



# --------------------------
# initiate quda
# --------------------------
mpi_geometry = [2, 2, 2, 4]
init(mpi_geometry, enable_mps=True)
G5 = gamma.gamma(15)





# --------------------------
# Setup parameters
# --------------------------
parameters = {
    "eta" : [16, 20, 24],
    "b_T": 19,
    "b_z" : 19,
    "pzmin" : 4,
    "pzmax" : 11,
    "width" : 5.2,
    "pos_boost" : [0,0,5],
    "neg_boost" : [0,0,-5],
    "save_propagators" : False
}
Measurement = pion_TMDWF_measurement(parameters)





# --------------------------
# Start measurements
# --------------------------

###################### load gauge ######################
Ls = 48
Lt = 64
sublattice = np.array([Ls,Ls,Ls,Lt]) // np.array(mpi_geometry)
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
U = g.convert( g.load(f"/lustre/orion/nph158/proj-shared/lattices/l48c64a060/fixed_GLU/l4864f21b7373m00125m0250a.{conf}.coulomb.1e-14"), g.double )
g.mem_report(details=False)
L = U[0].grid.fdimensions
U_prime, trafo = g.gauge_fix(U, maxiter=50000, prec=1e-10) # CG fix
del U_prime
U_hyp = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.75, 0.6, 0.3])) # hyp smearing
latt_info, gpt_latt, gpt_simd, gpt_prec = gpt.LatticeInfoGPT(U[0].grid, GEN_SIMD_WIDTH)
gauge = gpt.LatticeGaugeGPT(U_hyp, GEN_SIMD_WIDTH)
g.mem_report(details=False)

###################### setup source positions ######################
src_shift = np.array([0,0,0,0]) + np.array([7,11,13,23])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin) # create a list of source
src_production = src_positions[0 : 32] # take the number of sources needed for this project FIXME

###################### create multigrid inverter ######################
latt_info = LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
dirac = core.getDirac(latt_info, -0.038888, 1e-10, 10000, 1.0, 1.0336, 1.0336, [[6, 6, 6, 4]])
gauge = gpt.LatticeGaugeGPT(U_hyp, GEN_SIMD_WIDTH)
g.message("DEBUG plaquette U_hyp:", g.qcd.gauge.plaquette(U_hyp))
g.message("DEBUG plaquette gauge:", gauge.plaquette())
gauge.projectSU3(1e-15)
dirac.loadGauge(gauge)
g.message("Multigrid inverter ready.")
g.mem_report(details=False)


###################### prepare gauge links ######################
sample_log_file = data_dir + f"/sample_log/TMDWF_{sm_tag}_{conf}"
if g.rank() == 0:
    f = open(sample_log_file, "a+")
    f.close()

g.message("Wilson Link: Start")
g.mem_report(details=False)
W, W_index_list = Measurement.create_TMD_WL(U)
W_count = len(W_index_list)
W_subset_len = 50
W_subset_count = math.ceil(W_count/W_subset_len)
g.message("Wilson Link: W_count, W_subset_len, W_subset_count", W_count, W_subset_len, W_subset_count)
g.message("Wilson Link: W_index_list:",W_index_list)
g.mem_report(details=False)
g.message("Wilson Link: Done")

###################### looping over source ######################
for pos in src_production:

    sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
    g.message(f"Contraction START: {sample_log_tag}")
    with open(sample_log_file) as f:
        if sample_log_tag in f.read():
            g.message("Contraction SKIP: " + sample_log_tag)
            continue

    # Create momentum source and propagator
    g.message("START source position:", pos)
    g.mem_report(details=False)
    srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)
    phases = Measurement.make_mom_phases(grid, pos)

    # get propag
    bp = gpt.LatticePropagatorGPT(srcDp, GEN_SIMD_WIDTH)
    bm = gpt.LatticePropagatorGPT(srcDm, GEN_SIMD_WIDTH)
    g.message("Pass src to quda.")
    start = time.time()
    propag_f = core.invertPropagator(dirac, bp, 1, 0)
    propag_b = core.invertPropagator(dirac, bm, 1, 0)
    g.message("TIME: Pyquda inversion * 2", time.time() - start)
    prop_f = g.mspincolor(grid)
    prop_b = g.mspincolor(grid)
    gpt.LatticePropagatorGPT(prop_f, GEN_SIMD_WIDTH, propag_f)
    gpt.LatticePropagatorGPT(prop_b, GEN_SIMD_WIDTH, propag_b)
    del propag_f, propag_b

    # SS 2pt contraction
    start = time.time()
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    Measurement.contract_2pt(prop_f, prop_b, phases, trafo, tag)
    g.message("TIME: Contraction SS 2pt (includes sink smearing)", time.time() - start)

    # SP TMDWF contraction
    g.message(f"Contraction: Start TMDWF with N_W = {W_count}, divided into {W_subset_count} subsets.")
    qTMDWF_tag = get_qTMDWF_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    for i_sub in range(0, W_subset_count):
        g.message(f"Start TMDWF backward propagator subset of {i_sub} / {W_subset_count}")
        prop_b_W = Measurement.constr_TMD_bprop_Z5X5(prop_b,W[i_sub*W_subset_len:(i_sub+1)*W_subset_len], W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len])
        g.message("  Start TMDWF contractions")
        Measurement.contract_TMD(prop_f, prop_b_W, phases, qTMDWF_tag, W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len], i_sub)
        del prop_b_W
    g.message("Contraction: Done TMDWF")

    with open(sample_log_file, "a") as f:
        if g.rank() == 0:
            f.write(sample_log_tag+"\n")
    g.message("Contraction DONE: " + sample_log_tag)

    del prop_f, prop_b