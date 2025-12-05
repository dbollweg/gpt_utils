import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
import os
import time

import gpt as g
from qTMD.proton_qTMD_draft import proton_TMD
from qTMD.proton_qTMD_draft import Cg5
from utils import pyquda_gpt as pq

from pyquda import init, LatticeInfo, setGPUID
from pyquda.field import LatticeFermion, LatticePropagator, Ns, Nc
from pyquda_utils import core, io, source, gpt


import subprocess

def get_gpu_uuid():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        text=True,
    )
    gpu_uuids = result.stdout.strip().split("\n")
    return gpu_uuids

rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))  # 获取 MPI Rank
gpu_uuids = get_gpu_uuid()
gpu_id = rank % len(gpu_uuids)

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print(f"MPI Rank: {rank}, CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}, "
      f"GPU UUID: {gpu_uuids[gpu_id]}")

GEN_SIMD_WIDTH = 64

conf_n = 1014
init([1, 1, 1, 1], enable_mps=True)

parameters = {
    
    # NOTE: eta > 12 will only run bz=0: check qTMD.proton_qTMD_draft.create_TMD_Wilsonline_index_list
    "eta": [0],
    "b_z": 20,
    "b_T": 0,

    "qext": [0], # momentum transfer
    "qext_PDF": [0], # momentum transfer
    "pf": [0,0,0,0],
    "p_2pt": [[x,y,z,0] for x in [0] for y in [0] for z in [0]], # 2pt momentum

    "boost_in": [0,0,0],
    "boost_out": [0,0,0],
    "width" : 8.0,

    "pol": ["PpSzp", "PpSzm", "PpSxp"],
    "t_insert": 8, #starting with 10 source sink separation

    "save_propagators": False,
}
Measurement = proton_TMD(parameters)

##### small dummy used for testing
Ls = 8
Lt = 16
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.random(grid, rng)
U_prime, trafo = g.gauge_fix(U, maxiter=5000)
del U_prime

'''
gpt functions
'''

# Paths for configurations and dump
U_hyp = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.75, 0.6, 0.3]))

# Initialize the grid and Wilson fermion operator
grid = U_hyp[0].grid
p = {
    "kappa": 0.1256, # 0.12623 for 300 MeV pion; 0.1256 for 670 MeV pion
    "csw_r": 1.0336,
    "csw_t": 1.0336,
    "xi_0": 1,
    "nu": 1,
    "isAnisotropic": False,
    "boundary_phases": [1, 1, 1, -1],
}
w = g.qcd.fermion.wilson_clover(U_hyp, p)

# Create momentum source and propagator
pos = [0, 0, 0, 0]
src = Measurement.create_src_2pt(pos, trafo, U[0].grid)
print(np.shape(src), np.shape(src[0,0,0,0]))
print(np.shape(np.array(src[0,0,0,0])))

# Solver and propagator
inv = g.algorithms.inverter
pc = g.qcd.fermion.preconditioner
cg = inv.cg({"eps": 1e-8, "maxiter": 10000})
slv = w.propagator(inv.preconditioned(pc.eo2_ne(), cg))

dst = g.mspincolor(grid)
start = time.time()
dst @= slv * src
print("TIME: GPT CG", time.time() - start)

# Calculate pion correlation
corr_pion = g.slice(g.trace(g.adj(dst) * dst), 3)
g.message('GPT: pion',corr_pion)
'''
# Calculate proton correlation
phases = Measurement.make_mom_phases_2pt(U[0].grid, pos)
prop_f = dst
tmp_trafo = g.convert(trafo, prop_f.grid.precision)
#prop_f = g.create.smear.boosted_smearing(tmp_trafo, dst, w=parameters.width, boost=parameters.pos_boost)
dq = g.qcd.baryon.diquark(g(prop_f * Cg5), g(Cg5 * prop_f))
proton1 = g(g.spin_trace(dq) * prop_f + dq * prop_f)
prop_unit = g.mspincolor(prop_f.grid)
prop_unit = g.identity(prop_unit)
corr = g.slice_trDA([prop_unit], [proton1], phases,3)
corr = [[corr[0][i][j] for i in range(0, len(corr[0]))] for j in range(0, len(corr[0][0])) ]
g.message('GPT: proton', corr[1][0])
'''

print(">>> Shape of source in GPT", np.shape(src[:]))
print(U[0].grid)

'''
pyquda functions
'''

b = pq.LatticePropagatorGPT(src, GEN_SIMD_WIDTH)
b.toDevice()

latt_info = LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
#dirac = core.getDirac(latt_info, -0.0191, 1e-8, 1000, 1.0, 1.0336, 1.0336)
dirac = core.getDirac(latt_info, -0.0191, 1e-8, 1000, 1.0, 1.0336, 1.0336, [[2, 2, 2, 4], [4, 4, 4, 4]]) # remove the last two arguments for BiCGStab
#gauge = io.readMILCGauge(f"/global/cfs/cdirs/m4559/xgao/lattices/l48c64a060/gauge_a/l4864f21b7373m00125m0250a.{conf_n}")
gauge = pq.LatticeGaugeGPT(U, GEN_SIMD_WIDTH)
gauge.projectSU3(2e-14)
gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)
dirac.loadGauge(gauge)

t_src = (0, 0, 0, 0)
print(">>> Shape of source in pyquda", np.shape(b.lexico()), np.shape(b))

pion = cp.zeros((latt_info.Lt), "<f8")
start = time.time()
propag = core.invertPropagator(dirac, b, 0)
print("TIME: Pyquda multigrid", time.time() - start)
'''
propag = LatticePropagator(latt_info)
for spin in range(Ns):
    for color in range(Nc):
        b_sc = source.point(latt_info, t_src, spin, color)
        b_sc_gpt = LatticeFermion(latt_info, np.asarray(b.getFermion(spin, color)))
        print("Shape of source: ", np.shape(b_sc), np.shape(b_sc_gpt))
        x = dirac.invert(b_sc_gpt)
        print("Shape of x: ", np.shape(x))
        propag.setFermion(x, spin, color)
'''
#x = dirac.invertMultiSrc(b)
#for spin in range(latt_info.Ns):
#    for color in range(latt_info.Nc):
#        propag.setFermion(x[spin * latt_info.Nc + color], spin, color)
pion += contract("wtzyxjiba,wtzyxjiba->t", propag.data.conj(), propag.data).real
dirac.destroy()

print("Shape of pion.get():", pion.get().shape)
tmp = core.gatherLattice(pion.get().reshape(16, 1, 1, 1), [1, -1, -1, -1])
print(np.shape(tmp))
print(tmp[:,0,0,0])

'''
pyquda propag back to GPT contraction
'''
dst = g.mspincolor(grid)
prop_toGPT = pq.LatticePropagatorGPT(dst, GEN_SIMD_WIDTH, propag)
print(np.shape(prop_toGPT))
#print(np.shape(prop_toGPT), np.shape(prop_toGPT.lexico()))
prop_toGPT = g(prop_toGPT)
# Calculate pion correlation
corr_pion_toGPT = g.slice(g.trace(g.adj(prop_toGPT) * prop_toGPT), 3)
g.message('GPT-pyquda-GPT pion >>> GPT pion',np.array(corr_pion_toGPT)-np.array(corr_pion))
