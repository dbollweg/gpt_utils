# load python modules
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
import os
import time
from cupy.cuda import memory
import shutil

# load gpt modules
import gpt as g
from utils.tools import *
from utils.io_corr import *
from PyQUDA_pion_ff_class import current_current_correlator, my_gammas

# load pyquda modules
from pyquda import init, LatticeInfo
from pyquda_utils import core, gpt, gamma
import subprocess



''' --------------------------------- '''
'''          config and setup         '''
''' --------------------------------- '''
# data output dir
data_dir = "/lustre1/pion3d/xgao/run/qTMD_softFF/data"

# configuration number
conf = g.default.get_int("--config_num", 0)
g.message(f"--config_num {conf}")

# Configuration parameters
parameters = {
    # Parameters needed for pion_measurement parent class
    "save_propagators": False,     # Whether to save propagators

    # Additional parameters needed for current_current_correlator class
    "quark_mom": np.array([[0, 0, nz, 0] for nz in range(3, 4+1)]),   # Quark momentum, must be 4D array

    "bT_dir": [0],
    "bT_length": 20,

    "bz_length": 20,

    "pion_src": {
        "5": g.gamma[5],
        "Z5": g.gamma["Z"] * g.gamma[5],
        "Z5-X5": g.gamma["Z"] * g.gamma[5] - g.gamma["X"] * g.gamma[5],
    },
    "pion_sink": {
        "5": g.gamma[5],
        "Z5": g.gamma["Z"] * g.gamma[5],
        "Z5-X5": g.gamma["Z"] * g.gamma[5] - g.gamma["X"] * g.gamma[5],
    },

    "Gamma1": {
        "5": g.gamma[5],
        "I": g.gamma["I"],
        "X": g.gamma["X"],
        "Y": g.gamma["Y"],
        "X5": g.gamma["X"] * g.gamma[5],
        "Y5": g.gamma["Y"] * g.gamma[5],
    },
    "Gamma2": {
        "5": g.gamma[5],
        "I": g.gamma["I"],
        "X": g.gamma["X"],
        "Y": g.gamma["Y"],
        "X5": g.gamma["X"] * g.gamma[5],
        "Y5": g.gamma["Y"] * g.gamma[5],
    },
}
GEN_SIMD_WIDTH = 64

# Configuration path and number
conf_path = "/lustre1/pion3d/xgao/run/qTMD_softFF/S8T32"
lat_tag = "l8c32"
sm_tag = '1HYP'
sample_log_file_prop = data_dir + "/sample_log/" + str(conf) + '_' + sm_tag + '_prop'
sample_log_file_ff = data_dir + "/sample_log/" + str(conf) + '_' + sm_tag + '_ff'
if g.rank() == 0:
    f = open(sample_log_file_prop, "a+")
    f.close()
    f = open(sample_log_file_ff, "a+")
    f.close()


''' -------------------------------------- '''
'''     GPU information & initiate quda    '''
''' -------------------------------------- '''

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

mpi_geometry = [1, 1, 1, 4]
init(mpi_geometry, enable_mps=True)





''' ---------------------------------------------------- '''
'''     Load gauge field & init measurement class        '''
''' ---------------------------------------------------- '''
# Load gauge field configuration
#g.message("Loading gauge configuration")
#U = g.convert(g.load(f"{conf_path}/wilson_b6.{conf}"), g.double)
#grid = U[0].grid
#U_fixed, trafo = g.gauge_fix(U, maxiter=50000, prec=1e-7)
#g.message("Finished loading gauge config")
#L = U_fixed[0].grid.fdimensions
#Ls, Lt = L[0], L[3]

g.message("Loading gauge configuration")
Ls = 8
Lt = 32
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.random(grid, rng)
L = U[0].grid.fdimensions
U_fixed, trafo = g.gauge_fix(U, maxiter=50000, prec=1e-7)
g.message("Finished loading gauge config")

tslice_list = [i for i in range(0, int(Lt))]
poslist = [[0,0,0,tslice] for tslice in tslice_list]

U_smear = g.qcd.gauge.smear.hyp(U_fixed, alpha = np.array([0.75, 0.6, 0.3])) # csw and quark mass parameters were all after HYP smearing

# Initialize measurement class
Measurement = current_current_correlator(parameters)





''' ------------------------------------ '''
'''             Set up solver            '''
''' ------------------------------------ '''

# Set up Pyquda Wilson-Clover fermion action
latt_info = LatticeInfo([Ls, Ls, Ls, Lt], -1, 1.0)
dirac = core.getDirac(latt_info, 0.2, 1e-7, 1000, 1.0, 1.0372, 1.0372) # remove the last two arguments for BiCGStab; S mass -0.015
gauge = gpt.LatticeGaugeGPT(U_smear, GEN_SIMD_WIDTH)
gauge.projectSU3(2e-14)
dirac.loadGauge(gauge)
g.message("Multigrid inverter ready.")




''' ---------------------------------------------- '''
'''     Generate propagators on each time slice    '''
''' ---------------------------------------------- '''
# Create sources and propagators
g.message(f"Creating Wall-SRC propagators: {poslist}\n")
for ipos, pos in enumerate(poslist):

    g.mem_report(details=False)

    sample_log_tag = get_sample_log_tag("CG_wall_ex", pos, sm_tag)
    g.message(f"START SAMPLE {ipos}/{len(poslist)}: {sample_log_tag}")

    with open(sample_log_file_prop, "a+") as f:
        if sample_log_tag in f.read():
            g.message("SKIP SAMPLE: " + sample_log_tag)
            # continue # if the job has been done, skip it; may disable this during test

    corr_dir = data_dir + "/ff/" + str(conf)
    prop_dir = data_dir + "/prop/" + str(conf) + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
    Measurement.set_output_facilities(corr_dir, prop_dir)

    #-------------------------------------------
    #   Coulomb-gauge Wall-source propagators       
    #-------------------------------------------
    quark_mom_list = parameters["quark_mom"]
    for imom, quark_mom in enumerate(quark_mom_list):

        quark_mom = quark_mom_list[imom]
    
        # Calculate forward propagator
        fw_src = Measurement.create_wall_src(pos[-1], quark_mom, grid) 
        prop_f = Measurement.create_wall_propagator(dirac, fw_src, grid)
        prop_tag = f"CG_wall_ex/{lat_tag}/{sm_tag}/{conf}/{pos[0]}{pos[1]}{pos[2]}{pos[3]}/{str(quark_mom)}"
        Measurement.propagator_output(prop_tag, prop_f)
        g.message(f"Created wall source propagators: quark mom {quark_mom}") 

        del fw_src, prop_f

        # Calculate backward propagator
        bw_src = Measurement.create_wall_src(pos[-1], -quark_mom, grid)
        prop_b = Measurement.create_wall_propagator(dirac, bw_src, grid)
        prop_tag = f"CG_wall_ex/{lat_tag}/{sm_tag}/{conf}/{pos[0]}{pos[1]}{pos[2]}{pos[3]}/{str(-quark_mom)}"
        Measurement.propagator_output(prop_tag, prop_b)
        g.message(f"Created wall source propagators: quark mom {-quark_mom}") 

        del bw_src, prop_b

    with open(sample_log_file_prop, "a") as f:
        if g.rank() == 0:
            f.write(sample_log_tag+"\n")
    g.message(f"DONE SAMPLE propagator {ipos}/{len(poslist)}: {sample_log_tag}")





''' ------------------------------------ '''
'''           Measurement loop           '''
''' ------------------------------------ '''

quark_mom_list = parameters["quark_mom"]
pion_src = parameters["pion_src"]
pion_sink = parameters["pion_sink"]
Gamma1 = parameters["Gamma1"]
Gamma2 = parameters["Gamma2"]
g.message(f"Contraction of Wall-SRC propagators: location {poslist}\n")
g.message(f"Contraction of Wall-SRC propagators: quark momentum: {quark_mom_list}\n")

# looping over location of sources
for ipos, pos in enumerate(poslist):

    sample_log_tag = get_sample_log_tag("CG_wall_ex", pos, sm_tag)
    g.message(f"START SAMPLE {ipos}/{len(poslist)}: {sample_log_tag}")

    with open(sample_log_file_ff, "a+") as f:
        if sample_log_tag in f.read():
            g.message("SKIP SAMPLE: " + sample_log_tag)
            # continue # if the job has been done, skip it; may disable this during test

    # load propagators at the source
    prop_dir = data_dir + "/prop/" + str(conf) + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
    propagator_read = Measurement.propagator_input(prop_dir)
    prop_group_src = {}
    for p in propagator_read:
        prop_group_src.update(p)

    #-------------------------------------------
    #          2pt: quasi-TMDWF in CG          
    #-------------------------------------------
    # looping over fw propagator
    for ifw, quark_mom_fw in enumerate(quark_mom_list):
        # looping over bw propagator
        for ibw, quark_mom_bw in enumerate(quark_mom_list):

            meson_mom = quark_mom_fw - (-quark_mom_bw)
            meson_mom_tag = "PX"+str(meson_mom[0])+"PY"+str(meson_mom[1])+"PZ"+str(meson_mom[2])

            prop_tag = f"CG_wall_ex/{lat_tag}/{sm_tag}/{conf}/{pos[0]}{pos[1]}{pos[2]}{pos[3]}/{str(quark_mom_fw)}"
            prop_f = prop_group_src[prop_tag]
            g.message(f"Load propagator Gw {prop_tag}") 
            prop_tag = f"CG_wall_ex/{lat_tag}/{sm_tag}/{conf}/{pos[0]}{pos[1]}{pos[2]}{pos[3]}/{str(-quark_mom_bw)}"
            prop_b = prop_group_src[prop_tag]
            g.message(f"Load propagator Gw_bperp_dagger {prop_tag}") 
            g.mem_report(details=False)

            g.message(f"Contraction Start {meson_mom_tag}: SP quasi-TMDWF in CG") 

            prop_f_Pq = Measurement.apply_phase(prop_f, -meson_mom, 1, grid)

            for key in pion_src:
                qTMDWF_collection = []
                for bT_dir in parameters["bT_dir"]:
                    bz_dir = 2
                    for bT in range(0, parameters["bT_length"]+1):
                        for bz in range(0, parameters["bz_length"]+1):
                            corr = g.slice_trDA(pion_src[key]*g.adj(g.gamma[5]*g.eval(g.cshift(g.cshift(prop_b, bT_dir, bT), bz_dir, bz))*g.gamma[5]), prop_f_Pq, [g.identity(g.complex(grid))], 3)
                            qTMDWF_collection += [corr[0]]

                # this is to check the dispersion relation of the 2pt
                g.message("DEBUG", np.shape(qTMDWF_collection))
                check_2pt = np.roll(qTMDWF_collection[2][0][0], -pos[-1])
                g.message(f"{meson_mom_tag} pion TMDWF {key}: e^(m*a) {(check_2pt/np.roll(check_2pt, -1))[2:7]}")

                qTMDWF_tag = get_qTMDWF_file_tag(data_dir, lat_tag, conf, f"ex_wall_{meson_mom_tag}_src{key}_SP", pos, sm_tag)
                if g.rank() == 0:
                    save_qTMDWF_hdf5(qTMDWF_collection, qTMDWF_tag, my_gammas, [meson_mom], ['CG'], parameters["bT_length"]+1, parameters["bz_length"]+1, bT_dir=parameters["bT_dir"])

            g.message(f"Contraction Done {meson_mom_tag}: SP quasi-TMDWF in CG")
            del prop_f_Pq


            #-------------------------------------------
            #        4pt: ff for TMD soft factor          
            #-------------------------------------------
            # looping over propagators at the sink on all time slices
            keys_src = list(pion_src.keys())
            keys_sink = list(pion_sink.keys())
            keys_gm1 = list(Gamma1.keys())
            keys_gm2 = list(Gamma2.keys())

            corr_ff_list = []
            check_ff_list = []
            g.message(f"Contraction Start {meson_mom_tag}: ff for TMD soft factor")
            for tsep in range(0, Lt//2):
                
                # Load wall source propagators at the sink
                prop_dir = data_dir + "/prop/" + str(conf) + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str((tsep+pos[3])%Lt)
                propagator_read = Measurement.propagator_input(prop_dir)
                prop_group_sink = {}
                for p in propagator_read:
                    prop_group_sink.update(p)

                # now I have Gw, Gw_bperp_dagger, Gw_bperp, Gw_dagger
                prop_tag = f"CG_wall_ex/{lat_tag}/{sm_tag}/{conf}/{pos[0]}{pos[1]}{pos[2]}{(tsep+pos[3])%Lt}/{str(-quark_mom_fw)}"
                Gw_dagger = prop_group_sink[prop_tag]
                g.message(f"Load propagator Gw_dagger {prop_tag}") 
                prop_tag = f"CG_wall_ex/{lat_tag}/{sm_tag}/{conf}/{pos[0]}{pos[1]}{pos[2]}{(tsep+pos[3])%Lt}/{str(quark_mom_bw)}"
                Gw_bperp = prop_group_sink[prop_tag]
                g.message(f"Load propagator Gw_bperp {prop_tag}") 
                Gw = prop_f
                Gw_bperp_dagger = prop_b
                g.mem_report(details=False)

                # apply phase to the momentum transfer
                Gw_bperp = Measurement.apply_phase(Gw_bperp, -2*meson_mom, 1, grid)

                # compute the contraction
                shape = (len(keys_src), len(keys_gm1), len(parameters["bT_dir"]), parameters["bT_length"] + 1, Lt)
                tsep_ff_list = np.empty(shape, dtype=np.complex128)  
                for i in range(len(keys_src)):  
                    for j in range(len(keys_gm1)):
                        for k, bT_dir in enumerate(parameters["bT_dir"]):
                            for bT in range(parameters["bT_length"] + 1):
                                # Contraction: Gamma2 * Gw_bperp * pion_sink * Gw_dagger * Gamma1 * Gw * pion_src * Gw_bperp_dagger
                                temp_1 = pion_src[keys_src[i]] * g.gamma[5] * g.adj(g.cshift(Gw_bperp_dagger, bT_dir, bT)) * g.gamma[5] * Gamma2[keys_gm2[j]] * g.cshift(Gw_bperp, bT_dir, bT)
                                temp_2 = pion_sink[keys_sink[i]] * g.gamma[5] * g.adj(Gw_dagger) * g.gamma[5] * Gamma1[keys_gm1[j]] * Gw
                                #corr_ff = g.slice(g.trace(temp_1*temp_2), 3)
                                corr_ff = g.slice_trDA(temp_1, temp_2, [g.identity(g.complex(grid))], 3)[0][0][9]
                                tsep_ff_list[i, j, k, bT, :] = corr_ff
                corr_ff_list += [tsep_ff_list]

                check_ff_list += [np.roll(tsep_ff_list[0][0][0][2], -pos[-1])]
            # this is to check the dispersion relation of the 4pt
            for tsep in range(2, Lt//2):
                g.message(f"ff e^(m*a) tsep={tsep}: {check_ff_list[tsep-1][tsep//2].real/check_ff_list[tsep][tsep//2].real}")
            
            ff_tag = get_softFF_file_tag(data_dir, lat_tag, conf, f"ex_wall_{meson_mom_tag}_SP", pos, sm_tag, quark_mom_fw, -quark_mom_bw)
            if g.rank() == 0:
                save_softFF_hdf5(corr_ff_list, ff_tag, pion_src, pion_sink, Gamma1, Gamma2, parameters["bT_dir"], parameters["bT_length"], [tsep for tsep in range(0, Lt//2)])
            g.message(f"Contraction Done {meson_mom_tag}: ff for TMD soft factor")
    g.message(f"DONE SAMPLE {ipos}/{len(poslist)}: {sample_log_tag}")

    # delete all the propagators
    prop_dir = data_dir + "/prop/" + str(conf) + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
    shutil.rmtree(prop_dir) 
