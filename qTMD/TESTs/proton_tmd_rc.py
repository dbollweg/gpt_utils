# This is a example of testing the gauge invariance
import gpt as g 
import os
import sys
import numpy as np
import math
from qTMD.proton_qTMD_draft import proton_TMD


from tools import *
from io_corr import *

root_output = "."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/ccs/home/xiangg/latwork/DWF/TEST/nucleon_TMD/data/"

#smearing
#smear_list = [['flow', '05eps01', 5], ['flow', '10eps01', 10], ['flow', '20eps01', 20], ['flow', '30eps01', 30]]
smear_list = [['flow', '01eps01', 1]]
# tags
sm_tag = "GSRC_W80_kz3"
lat_tag = "64I"

parameters = {
    
    #"eta": [8,12,16,20,24],
    #"b_z": 24,
    #"b_T": 24,
    "eta": [8],
    "b_z": 4,
    "b_T": 4,

    #"qext": [-2, -1, 0, 1, 2],
    #"pf": [0,0,6,0],
    "qext": [0],
    "pf": [0,0,1,0],

    #"boost_in": [0,0,3],
    #"boost_out": [0,0,3],
    #"width" : 8.0,
    "boost_in": [0,0,1],
    "boost_out": [0,0,1],
    "width" : 1.0,

    #"pol": ["PpSzp", "PpSzm", "PpSxp"],
    #"t_insert": 10, #starting with 10 source sink separation
    "pol": ["PpSzp"],
    "t_insert": 2, #starting with 10 source sink separation

    "save_propagators": False,
}
pf = parameters["pf"]
pf_tag = "PX"+str(pf[0]) + "PY"+str(pf[1]) + "PZ"+str(pf[2])

#ADD: Add coulomb gauge version: set longitudinal links to one
config_number = g.default.get_int("--config_num", 0)

groups = {
    "Frontier_batch_0": {
        "confs": [
            str(config_number),
        ],
#        "evec_fmt": "/lustre/orion/proj-shared/nph159/data/64I/%s.evecs/lanczos.output",
        "evec_fmt": "/lustre/orion/proj-shared/nph159/data/64I/job-0%s/lanczos.output",
        "conf_fmt": "/lustre/orion/proj-shared/nph159/data/64I/ckpoint_lat.Coulomb.%s",
    },
}


jobs = {
    "proton_TMD_AMA": {
        "exact": 1,
        "sloppy": 1,
        "low": 0,
    },
}


jobs_per_run = g.default.get_int("--gpt_jobs", 1)

# find jobs for this run
def get_job(only_on_conf=None):
    # statistics
    n = 0
    for group in groups:
        for job in jobs:
            for conf in groups[group]["confs"]:
                n += 1

    jid = -1
    for group in groups:
        for conf in groups[group]["confs"]:
            for job in jobs:
                jid += 1
                if only_on_conf is not None and only_on_conf != conf:
                    continue
                return group, job, conf, jid, n

    return None

if g.rank() == 0:
    first_job = get_job()
    run_jobs = str(
        list(
            filter(
                lambda x: x is not None,
                [first_job] + [get_job(first_job[2]) for i in range(1, jobs_per_run)],
            )
        )
    ).encode("utf-8")
else:
    run_jobs = bytes()
run_jobs = eval(g.broadcast(0, run_jobs).decode("utf-8"))

"""
================================================================================
            Every node now knows what to do -> Now initialization
================================================================================
"""

# configuration needs to be the same for all jobs, so load eigenvectors and configuration
conf = run_jobs[0][2]
group = run_jobs[0][0]


# loading gauge configuration
#g.message("Loading ")
#print(groups[group]["conf_fmt"] % conf)
#U = g.load(groups[group]["conf_fmt"] % conf)
#U_smear = g.copy(U)
#g.message("finished loading gauge config")

##### small dummy used for testing
Ls = 16
Lt = 16
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.random(grid, rng)

# do gauge fixing
U_prime, trafo = g.gauge_fix(U, maxiter=5000)
#del U_prime
L = U[0].grid.fdimensions

#print('DEBUG1', U[2])
#print('DEBUG2', U_prime[2])
#print('DEBUG3', g.qcd.gauge.transformed(U, trafo)[2])
#print('DEBUG4', g(trafo * U[2] * g.cshift(g.adj(trafo), 2, 1)))

Measurement = proton_TMD(parameters)
#prop_exact, prop_sloppy, pin = Measurement.make_64I_inverter(U, groups[group]["evec_fmt"] % conf)
prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)
pin = 0

W_index_list = Measurement.create_TMD_Wilsonline_index_list()
W_index_list_CG = Measurement.create_TMD_Wilsonline_index_list_CG(U[0].grid)

# show available memory
g.mem_report(details=False)
g.message(
"""
================================================================================
       Propagator generation run:
================================================================================
"""
)

for group, job, conf, jid, n in run_jobs:
    
    g.message(f"""Job {jid} / {n} :  configuration {conf}, job tag {job}""")
    
    src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
    source_positions = srcLoc_distri_eq(L, src_origin)
    
    source_positions_sloppy = source_positions[:jobs[job]["sloppy"]]
    source_positions_exact = source_positions[:jobs[job]["exact"]]

    g.message(f" positions_sloppy = {source_positions_sloppy}")
    g.message(f" positions_exact = {source_positions_exact}")
    
    sample_log_file = data_dir + "/sample_log_qtmd/" + conf + "_W80_k3"
    if g.rank() == 0:
        f = open(sample_log_file, "a+")
        f.close()

    # exact positions
    corr_dir = data_dir + "corr_ex/" + conf
    for pos in source_positions_exact:
        
        sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
        g.message(f"START: {sample_log_tag}")
        with open(sample_log_file) as f:
            if sample_log_tag in f.read():
                g.message("SKIP: " + sample_log_tag)
                continue

        #We aren't saving propagators in this run
        #prop_dir = data_dir + "prop_ex_W80_k3/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
        #if g.rank() == 0:
         #   if not prop_dir:
            #    os.makedirs(prop_dir)
        #Measurement.set_propagator_output_facilities(prop_dir)

        g.message("Generatring boosted src's")
        srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")
        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")

        g.message("Contraction: Starting 2pt (includes sink smearing)")
        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
        phases_2pt = Measurement.make_mom_phases_2pt(U[0].grid, pos)
        Measurement.contract_2pt_TMD(prop_exact_f, phases_2pt, trafo, tag)
        g.message("Contraction: Done 2pt (includes sink smearing)")

        sequential_bw_prop_down = Measurement.create_bw_seq(prop_exact, prop_exact_f, trafo, 2, pos)
        sequential_bw_prop_up = Measurement.create_bw_seq(prop_exact, prop_exact_f, trafo, 1, pos)
        g.message("backward prop done")

        phases_3pt = Measurement.make_mom_phases_3pt(U[0].grid, pos)
        for ism, smear in enumerate(smear_list):
            
            contract_tag, n_sm = smear[0]+smear[1], smear[2]
            if smear[0] == 'hyp':
                for i in range(n_sm):
                    U_smear = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.75, 0.6, 0.3]))
            if smear[0] == 'flow':
                for i in range(n_sm):
                    U_smear = g.qcd.gauge.smear.wilson_flow(U, epsilon=0.1)
            g.message("Gauge: Smearing/Flow finished")
            
            # gauge invariant/traditional stape gauge link
            for iW, WL_indices in enumerate(W_index_list):
                # FIXME test here

                W = Measurement.create_TMD_Wilsonline(U, WL_indices)
                tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_exact_f, [W], [WL_indices])

                '''
                # Check Wilson link
                if WL_indices[0] == 0 and WL_indices[1] == 1 and WL_indices[2] == 2:
                    print(iW, WL_indices)
                    print('WDEBUG0', g.eval(U[2][0,0,0,0] * U[2][0,0,1,0]))
                    print('WDEBUG1', W[0,0,0,0])
                    print('WDEBUG2', g.adj(trafo[0,0,0,0]) * W_prime[0,0,0,0] * trafo[0,0,2,0])
                '''

                #W_prime = Measurement.create_TMD_Wilsonline(U_prime, WL_indices)
                #prop_exact_f_prime = g.eval(trafo * prop_exact_f)
                #tmd_forward_prop_prime = Measurement.create_fw_prop_TMD(prop_exact_f_prime, [W_prime], [WL_indices])
                #sequential_bw_prop_down_prime = [g.eval(sequential_bw_prop_down[0]*g.adj(trafo))]
                #sequential_bw_prop_up_prime = [g.eval(sequential_bw_prop_up[0]*g.adj(trafo))]

                '''
                # Check Wilson propagator
                W_prime = Measurement.create_TMD_Wilsonline(U_prime,WL_indices)
                prop_exact_f_prime = g.eval(trafo * prop_exact_f)
                tmd_forward_prop_prime = Measurement.create_fw_prop_TMD(prop_exact_f_prime, [W_prime], [WL_indices])
                if WL_indices[0] == 0 and WL_indices[1] == 1 and WL_indices[2] == 2:
                    print(iW, WL_indices)
                    print('DEBUG0', tmd_forward_prop[0])
                    print('DEBUG1', g.eval(g.adj(trafo) * tmd_forward_prop_prime[0]))
                '''

                #prop_exact_f_prime = g.eval(trafo * prop_exact_f)
                #tmd_forward_prop_prime = Measurement.create_fw_prop_TMD(prop_exact_f_prime, [W_prime], [WL_indices])
                #g.message("TMD forward prop done")
                #print('Here2',g.eval(g.adj(trafo)*tmd_forward_prop_prime[0]))

                #qtmd_tag_exact = "%s/%s/%s/%s_%s" % ("qtmd_exact", lat_tag, sm_tag, str(conf), str(pos))
                qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI.D.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI.U.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                g.message("Starting TMD contractions")
                #proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop_prime, sequential_bw_prop_down_prime,phases_3pt, WL_indices, qtmd_tag_exact_D, iW)
                #proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop_prime, sequential_bw_prop_up_prime,phases_3pt, WL_indices, qtmd_tag_exact_U, iW)
                proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_exact_D, iW)
                proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_exact_U, iW)

                #del tmd_forward_prop_prime
                del tmd_forward_prop
                
            '''
            # CG gauge links with Transverse link only
            for iW, WL_indices in enumerate(W_index_list):
                W = Measurement.create_TMD_Wilsonline_CG_Tlink(U_smear,WL_indices)

                tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_exact_f, [W], [WL_indices])
                g.message("TMD forward prop done")
                #production tag should include more config/action details

               # qtmd_tag_exact = "%s/%s/%s/%s_%s" % ("qtmd_exact", lat_tag, sm_tag, str(conf), str(pos))
                qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_T.D.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_T.U.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)

                g.message("Starting TMD contractions")
                proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_exact_D, iW)
                proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_exact_U, iW)

                del tmd_forward_prop

            # CG without links
            for iW, WL_indices in enumerate(W_index_list_CG):
                W = Measurement.create_TMD_Wilsonline_CG(U_smear,WL_indices)

                tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_exact_f, [W], [WL_indices])
                g.message("TMD forward prop done")
                #production tag should include more config/action details

               # qtmd_tag_exact = "%s/%s/%s/%s_%s" % ("qtmd_exact", lat_tag, sm_tag, str(conf), str(pos))
                qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.D.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.U.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)

                g.message("Starting TMD contractions")
                proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_exact_D, iW)
                proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_exact_U, iW)

                del tmd_forward_prop
            '''
        with open(sample_log_file, "a+") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_exact_f

    del prop_exact

    '''
    # sloppy positions
    corr_dir = data_dir + "corr_sl/" + conf
    for pos in source_positions_sloppy:
        
        sample_log_tag = get_sample_log_tag("sl", pos, sm_tag)
        g.message(f"START: {sample_log_tag}")
        with open(sample_log_file) as f:
            if sample_log_tag in f.read():
                g.message("SKIP: " + sample_log_tag)
                continue

        # prop_dir = data_dir + "prop_sl_W80_k3/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
        # if g.rank() == 0:
        #     if not prop_dir:
        #         os.makedirs(prop_dir)
        # Measurement.set_propagator_output_facilities(prop_dir)


        g.message("Generatring boosted src's")
        srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop sloppy")
        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")

        g.message("Contraction: Starting 2pt (includes sink smearing)")
        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "sl", pos, sm_tag)
        phases_2pt = Measurement.make_mom_phases_2pt(U[0].grid, pos)
        Measurement.contract_2pt_TMD(prop_sloppy_f, phases_2pt, trafo, tag)
        g.message("Contraction: Done 2pt (includes sink smearing)")

        sequential_bw_prop_down = Measurement.create_bw_seq(prop_sloppy, prop_sloppy_f, trafo, 2, pos)
        sequential_bw_prop_up = Measurement.create_bw_seq(prop_sloppy, prop_sloppy_f, trafo, 1, pos)
        g.message("backward prop done")

        phases_3pt = Measurement.make_mom_phases_3pt(U[0].grid, pos)
        for ism, smear in enumerate(smear_list):
            
            contract_tag, n_sm = smear[0]+smear[1], smear[2]
            if smear[0] == 'hyp':
                for i in range(n_sm):
                    U_smear = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.75, 0.6, 0.3]))
            if smear[0] == 'flow':
                for i in range(n_sm):
                    U_smear = g.qcd.gauge.smear.wilson_flow(U, epsilon=0.1)
            g.message("Gauge: Smearing/Flow finished")
            
            # gauge invariant/traditional stape gauge link
            for iW, WL_indices in enumerate(W_index_list):
                W = Measurement.create_TMD_Wilsonline(U_smear,WL_indices)

                tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_sloppy_f, [W], [WL_indices])
                g.message("TMD forward prop done")
                #production tag should include more config/action details

               # qtmd_tag_sloppy = "%s/%s/%s/%s_%s" % ("qtmd_sloppy", lat_tag, sm_tag, str(conf), str(pos))
                qtmd_tag_sloppy_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI.D.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                qtmd_tag_sloppy_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI.U.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)

                g.message("Starting TMD contractions")
                proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_sloppy_D, iW)
                proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_sloppy_U, iW)

                del tmd_forward_prop

            # CG gauge links with Transverse link only
            for iW, WL_indices in enumerate(W_index_list):
                W = Measurement.create_TMD_Wilsonline_CG_Tlink(U_smear,WL_indices)

                tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_sloppy_f, [W], [WL_indices])
                g.message("TMD forward prop done")
                #production tag should include more config/action details

               # qtmd_tag_sloppy = "%s/%s/%s/%s_%s" % ("qtmd_sloppy", lat_tag, sm_tag, str(conf), str(pos))
                qtmd_tag_sloppy_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_T.D.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                qtmd_tag_sloppy_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_T.U.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)

                g.message("Starting TMD contractions")
                proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_sloppy_D, iW)
                proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_sloppy_U, iW)

                del tmd_forward_prop

            # CG without links
            for iW, WL_indices in enumerate(W_index_list_CG):
                W = Measurement.create_TMD_Wilsonline_CG(U_smear,WL_indices)

                tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_sloppy_f, [W], [WL_indices])
                g.message("TMD forward prop done")
                #production tag should include more config/action details

               # qtmd_tag_sloppy = "%s/%s/%s/%s_%s" % ("qtmd_sloppy", lat_tag, sm_tag, str(conf), str(pos))
                qtmd_tag_sloppy_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.D.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                qtmd_tag_sloppy_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.U.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)

                g.message("Starting TMD contractions")
                proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_sloppy_D, iW)
                proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_sloppy_U, iW)

                del tmd_forward_prop
                
        with open(sample_log_file, "a+") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_sloppy_f
        
    del prop_sloppy
    '''
del pin