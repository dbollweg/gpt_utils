# load python modules
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
import os
import time
from cupy.cuda import memory

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
data_dir = "/home/gaox/latwork/gpt-pyquda/TEST_local_all/qTMD/data"

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
conf_path = "/home/gaox/latwork/gpt-pyquda/TEST_local_all/qTMD/S8T32"
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

mpi_geometry = [1, 1, 1, 1]
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

    corr_dir = data_dir + "/corr/" + str(conf)
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

    # contraction of two-point function
    pion_src = parameters["pion_src"]
    pion_sink = parameters["pion_sink"]
    # looping over fw propagator
    for ifw, quark_mom_fw in enumerate(quark_mom_list):
        # looping over bw propagator
        for ibw, quark_mom_bw in enumerate(quark_mom_list):

            meson_mom = quark_mom_fw - (-quark_mom_bw)
            meson_mom_tag = "PX"+str(meson_mom[0])+"PY"+str(meson_mom[1])+"PZ"+str(meson_mom[2])

            prop_tag = f"CG_wall_ex/{lat_tag}/{sm_tag}/{conf}/{pos[0]}{pos[1]}{pos[2]}{pos[3]}/{str(quark_mom_fw)}"
            prop_f = prop_group_src[prop_tag]
            prop_tag = f"CG_wall_ex/{lat_tag}/{sm_tag}/{conf}/{pos[0]}{pos[1]}{pos[2]}{pos[3]}/{str(-quark_mom_bw)}"
            prop_b = prop_group_src[prop_tag]

            #-------------------------------------------
            #            SP: 2pt {Trace Sum}             
            #-------------------------------------------
            g.message("Computing SP 2pt functions: Sum Tr") 

            prop_f_Pq = Measurement.apply_phase(prop_f, -meson_mom, 1, grid)
            prop_b_Pqbar = prop_b

            for key in pion_src:
                #corr = g.slice(g.trace(pion_src[key1]*g.gamma[5]*g.adj(prop_b_Pqbar)*g.gamma[5]*pion_sink[key2]*prop_f_Pq), 3)
                corr = g.slice_trDA(pion_src[key]*g.adj(g.gamma[5]*prop_b*g.gamma[5]), prop_f_Pq, [g.identity(g.complex(grid))], 3) 
                pion = np.roll(corr[0][0][0], -pos[-1])
                g.message(f"{meson_mom_tag} pion 2pt {key}: e^(m*a) {(pion/np.roll(pion, -1))[2:11]}")

                tag = get_c2pt_file_tag(data_dir, lat_tag, conf, f"ex_wall_{meson_mom_tag}_src{key}_SP", pos, sm_tag)
                if g.rank() == 0:
                    save_c2pt_hdf5(corr, tag, my_gammas, [meson_mom], sm="SP")

            del prop_f_Pq, prop_b_Pqbar


            #-------------------------------------------
            #            SP: TMDWF {Trace Sum}             
            #-------------------------------------------
            g.message("Contraction Start: SP quasi-TMDWF in CG") 

            prop_f_Pq = Measurement.apply_phase(prop_f, -meson_mom, 1, grid)

            for key in pion_src:
                qTMDWF_collection = []
                for bT_dir in parameters["bT_dir"]:
                    bz_dir = 2
                    for bT in range(0, parameters["bT_length"]+1):
                        for bz in range(0, parameters["bz_length"]+1):
                            corr = g.slice_trDA(pion_src[key]*g.adj(g.gamma[5]*g.eval(g.cshift(g.cshift(prop_b, bT_dir, bT), bz_dir, bz))*g.gamma[5]), prop_f_Pq, [g.identity(g.complex(grid))], 3)
                            qTMDWF_collection += [corr[0]]
                pion = np.roll(qTMDWF_collection[0][0][0], -pos[-1])
                g.message(f"{meson_mom_tag} pion TMDWF {key}: e^(m*a) {(pion/np.roll(pion, -1))[2:11]}")

                qTMDWF_tag = get_qTMDWF_file_tag(data_dir, lat_tag, conf, f"ex_wall_{meson_mom_tag}_src{key}_SP", pos, sm_tag)
                if g.rank() == 0:
                    save_qTMDWF_hdf5(qTMDWF_collection, qTMDWF_tag, my_gammas, [meson_mom], ['CG'], parameters["bT_length"]+1, parameters["bz_length"]+1, bT_dir=parameters["bT_dir"])

            g.message("Contraction Done: SP quasi-TMDWF in CG")

'''
    # looping over propagators at the sink on all time slices
    for tsep in range(0, Lt//2):

        prop_dir = data_dir + "/prop/" + str(conf) + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(tsep+pos[3])
        propagator_read = Measurement.propagator_input(prop_dir)
        prop_group_sink = {}
        for p in propagator_read:
            prop_group_sink.update(p)
            
        for imom, quark_mom in enumerate(quark_mom_list):

            quark_mom = quark_mom_list[imom]

            # Load wall source propagators

            #-------------------------------------------
            #             2pt {Trace Sum}             
            #-------------------------------------------
            g.message("Computing 2pt functions: Sum Tr") 

            prop_f_Pq = Measurement.apply_phase(prop_f, -meson_mom, 1, grid)
            prop_b_Pqbar = prop_b

            corr = g.slice(g.trace(g.adj(prop_b_Pqbar)*g.gamma["T"]*prop_f_Pq), 3)
            corr = np.roll(corr, -pos[-1])
            g.message(f"Tr Sum -- Wall 2pt {np.shape(corr)}: {corr[:11]}")
            g.message(f"e^(m*a) {(corr/np.roll(corr, -1))[:11]}")
            pion_SP += [[corr]]

            #-------------------------------------------
            #             2pt {Sum Trace}             
            #-------------------------------------------
            g.message("Computing 2pt functions: Sum Tr") 

            prop_f_Pq = g.eval(g.slice(Measurement.apply_phase(prop_f, -(meson_mom - quark_mom), 1, grid), 3))
            prop_b_Pqbar = g.eval(g.slice(Measurement.apply_phase(prop_b, quark_mom, 1, grid), 3))

            corr = [g.eval(g.trace(g.adj(prop_b_Pqbar[i])*prop_f_Pq[i]) ) for i in range(0, Lt)]
            corr = np.roll(corr, -pos[-1]) 
            g.message(f"Sum Tr -- Wall 2pt {np.shape(corr)}: {corr[:11]}")
            g.message(f"e^(m*a) {(corr/np.roll(corr, -1))[:10]}")
            pion_SS += [[corr]]

            del prop_f_Pq, prop_b_Pqbar

            #-------------------------------------------
            #       soft function form factors            
            #-------------------------------------------
            Gw = prop_f
            Gw_bperp_dagger = prop_b

            # TODO
            pion_SRC = g.gamma[5]
            b_perp = 2
            perp_dir = 0
            Gamma_1, Gamma_2 = g.gamma["I"], g.gamma["I"]

            # loop over time separations
            tsep_list = parameters['t_separation']
            corr_ff_list = []
            for tsep in tsep_list:
                # Create wall source propagators
                g.message(f"Creating wall source propagators: quark mom {quark_mom}, meson mom {meson_mom}")
                Gw_dagger_src = Measurement.create_wall_src((pos[-1]+tsep)%Lt, -(meson_mom - quark_mom), grid) # (\sum_{x} e^{(PH-Pq)*(y-x)})^*
                Gw_bperp_src = Measurement.create_wall_src((pos[-1]+tsep)%Lt, quark_mom, grid) # \sum_{x} e^{Pq*(yt-x)}

                # Calculate Gw_bperp and Gw_dagger
                g.message("Computing forward/backward props with wall source")
                start = time.time()
                Gw_bperp_src_pyquda = gpt.LatticePropagatorGPT(Gw_bperp_src, GEN_SIMD_WIDTH)
                Gw_bperp_src_pyquda.toDevice()
                Gw_bperp_pyquda = core.invertPropagator(dirac, Gw_bperp_src_pyquda, 0)
                Gw_bperp = g.mspincolor(grid)
                gpt.LatticePropagatorGPT(Gw_bperp, GEN_SIMD_WIDTH, Gw_bperp_pyquda)
                g.message("TIME: Gw_bperp prop inversion", time.time() - start)
                start = time.time()
                Gw_dagger_src_pyquda = gpt.LatticePropagatorGPT(Gw_dagger_src, GEN_SIMD_WIDTH)
                Gw_dagger_src_pyquda.toDevice()
                Gw_dagger_pyquda = core.invertPropagator(dirac, Gw_dagger_src_pyquda, 0)
                Gw_dagger = g.mspincolor(grid)
                gpt.LatticePropagatorGPT(Gw_dagger, GEN_SIMD_WIDTH, Gw_dagger_pyquda)
                g.message("TIME: Gw_dagger prop inversion", time.time() - start)

                # now I have Gw, Gw_bperp_dagger, Gw_bperp, Gw_dagger
                # apply phase to the momentum transfer of each operator insersion
                Gw_bperp = Measurement.apply_phase(Gw_bperp, -2*meson_mom, 1, grid)

                temp_1 = g.adj(g.cshift(Gw_bperp_dagger, perp_dir, b_perp)) * g.gamma[5] * Gamma_2 * g.cshift(Gw_bperp, perp_dir, b_perp)
                temp_2 = g.adj(Gw_dagger) * g.gamma[5] * Gamma_1 * Gw
                corr_ff = g.slice(g.trace(temp_1*temp_2), 3)
                corr_ff_list += [np.roll(corr_ff, -pos[-1])]

            for i, tsep in enumerate(tsep_list):
                g.message(f"ff meson_mom={meson_mom} tsep={tsep}: {corr_ff_list[i].real}")
            for i, tsep in enumerate(tsep_list):
                if i == 0:
                    continue
                g.message(f"e^(m*a) tsep={tsep}: {np.sqrt(corr_ff_list[i-1][tsep//2].real/corr_ff_list[i][tsep//2].real)}")
            for i, tsep in enumerate(tsep_list):
                g.message(f"3pt/2pt_SS tsep={tsep}: {corr_ff_list[i][tsep//2].real/pion_SS[imom][0][tsep].real}")
            for i, tsep in enumerate(tsep_list):
                g.message(f"3pt/2pt_SP tsep={tsep}: {Ls**3*corr_ff_list[i][tsep//2].real/pion_SP[imom][0][tsep//2].real**2}")

        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex_wall_SumTr", pos, sm_tag)
        if g.rank() == 0:
            save_c2pt_hdf5([pion_SS], tag, ["5"], meson_mom_list)

        g.message(f"DONE SAMPLE {ipos}/{len(poslist)}: {sample_log_tag}")
'''