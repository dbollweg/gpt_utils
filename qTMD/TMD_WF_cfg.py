#!/usr/bin/env python3

import gpt as g
import os
import math

from qTMD.gpt_qTMD_utils import TMD_WF_measurement
from utils.tools import *
from utils.io_corr import *

# configure
root_output ="."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/home/gaox/latwork/DWF/64I/prod/data/GSRC_W40_k0"

# configuration setup
groups = {
    "booster_batch_0": {
        "confs": [
            "cfg",
        ],
        #"evec_fmt": "/p/scratch/gm2dwf/evecs/96I/%s/lanczos.output",
        #"evec_fmt": "/home/gaox/latwork/DWF/64I/prod/gauge/%s.evecs/lanczos.output"
        "evec_fmt": "/home/gaox/latwork/DWF/64I/prod/gauge/%s.evecs/lanczos.output",
        "conf_fmt": "/home/gaox/latwork/DWF/64I/prod/gauge/Coulomb/ckpoint_lat.Coulomb.%s",
    },

}

# momenta setup
parameters = {
    "eta" : [15, 20, 25],
    "b_T": 16,
    "b_z" : 16,
    "pzmin" : 0,
    "pzmax" : 4,
    "width" : 4.0,
    "pos_boost" : [0,0,0],
    "neg_boost" : [0,0,0],
    "save_propagators" : False
}

# tags
sm_tag = "GSRC_W40_k0_TEST-new"
lat_tag = "64I"

# AMA setup
jobs = {
    "booster_exact_0": {
        "exact": 1,
        "sloppy": 1,
        "low": 0,
    },
}

"""
================================================================================
                                    Run setup
================================================================================
"""

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

#print(run_jobs)
# configuration needs to be the same for all jobs, so load eigenvectors and configuration
conf = run_jobs[0][2]
group = run_jobs[0][0]

##### small dummy used for testing
#grid = g.grid([8,8,8,8], g.double)
#rng = g.random("seed text")
#U = g.qcd.gauge.random(grid, rng)

# loading gauge configuration
U = g.load(groups[group]["conf_fmt"] % conf)
g.message("finished loading gauge config")

# do gauge fixing
U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime
L = U[0].grid.fdimensions

Measurement = TMD_WF_measurement(parameters)
prop_exact, prop_sloppy, pin = Measurement.make_64I_inverter(U, groups[group]["evec_fmt"] % conf)
#prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)

# show available memory
g.mem_report(details=False)
g.message(
"""
================================================================================
       TMD run on polaris ;  this run will attempt:
================================================================================
"""
)

# per job
for group, job, conf, jid, n in run_jobs:
    g.message(f"""Job {jid} / {n} :  configuration {conf}, job tag {job}"""
    )

    # the original point for source creation which shift by conf number
    src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
    source_positions = srcLoc_distri_eq(L, src_origin)
    #print(source_positions)
    source_positions_sloppy = source_positions[:jobs[job]["sloppy"]]
    source_positions_exact = source_positions[:jobs[job]["exact"]]

    g.message(f" positions_sloppy = {source_positions_sloppy}")
    g.message(f" positions_exact = {source_positions_exact}")

    #root_job = f"{root_output}/{conf}/{job}"
    #Measurement.set_output_facilites(f"{root_job}/correlators",f"{root_job}/propagators")

    sample_log_file = data_dir + "/sample_log/" + conf
    #if g.rank() == 0:
    f = open(sample_log_file, "w")
    f.close()

    g.message("Starting modified Wilson loops")
    W, W_index_list = Measurement.create_TMD_WL(U)
    W_count = len(W_index_list)
    W_subset_len = 50
    W_subset_count = math.ceil(W_count/W_subset_len)
    if g.rank() == 0:
        print("W_count, W_subset_len, W_subset_count", W_count, W_subset_len, W_subset_count)
        print("W_index_list:",W_index_list)

    # exact positions
    g.message(f" positions_exact = {source_positions_exact}")
    for pos in source_positions_exact:
        phases = Measurement.make_mom_phases(U[0].grid,pos)     
        sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
        g.message(f"START: {sample_log_tag}")
        with open(sample_log_file) as f:
            if sample_log_tag in f.read():
                g.message("SKIP: " + sample_log_tag)
                continue

        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")
        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")
        prop_exact_b = g.eval(prop_exact * srcDm)
        g.message("backward prop done")

        del srcDp
        del srcDm

        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
        g.message("Starting 2pt contraction (includes sink smearing)")
        Measurement.contract_2pt(prop_exact_f, prop_exact_b, phases, trafo, tag)
        g.message("2pt contraction done")

        g.message(f"Start TMD contraction with N_W = {W_count}, divided into {W_subset_count} subsets.")
        qTMDWF_tag = get_qTMDWF_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
        for i_sub in range(0, W_subset_count):
            g.message(f"Start TMD backward propagator subset of {i_sub} / {W_subset_count}")
            prop_b = Measurement.constr_TMD_bprop(prop_exact_b,W[i_sub*W_subset_len:(i_sub+1)*W_subset_len], W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len])
            g.message("  Start TMD contractions")
            Measurement.contract_TMD(prop_exact_f, prop_b, phases, qTMDWF_tag, W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len], i_sub)
            del prop_b
            g.message("  TMD contractions done")

        with open(sample_log_file, "a") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_exact_f
        del prop_exact_b
        
    g.message("exact positions done")
    del prop_exact

    # sloppy positions
    for pos in source_positions_sloppy:

        phases = Measurement.make_mom_phases(U[0].grid,pos)  
        sample_log_tag = get_sample_log_tag("sl", pos, sm_tag)
        g.message(f"START: {sample_log_tag}")
        with open(sample_log_file) as f:
            if sample_log_tag in f.read():
                g.message("SKIP: " + sample_log_tag)
                continue

        #g.message("STARTING SLOPPY MEASUREMENTS")
        g.message("Starting 2pt function")
        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")
        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")
        prop_sloppy_b = g.eval(prop_sloppy * srcDm)
        g.message("backward prop done")

        del srcDp
        del srcDm

        g.message("Starting pion 2pt function")
        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "sl", pos, sm_tag)
        g.message("Starting pion contraction (includes sink smearing)")
        Measurement.contract_2pt(prop_sloppy_f, prop_sloppy_b, phases, trafo, tag)
        g.message("pion contraction done")

        g.message(f"Start TMD contraction with N_W = {W_count}, divided into {W_subset_count} subsets.")
        qTMDWF_tag = get_qTMDWF_file_tag(data_dir, lat_tag, conf, "sl", pos, sm_tag)
        for i_sub in range(0, W_subset_count):
            g.message(f"Start TMD backward propagator subset of {i_sub} / {W_subset_count}")
            prop_b = Measurement.constr_TMD_bprop(prop_sloppy_b,W[i_sub*W_subset_len:(i_sub+1)*W_subset_len], W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len])
            g.message("  Start TMD contractions")
            Measurement.contract_TMD(prop_sloppy_f, prop_b, phases, qTMDWF_tag, W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len], i_sub)
            del prop_b
            g.message("  TMD contractions done")
       
        del prop_sloppy_f
        del prop_sloppy_b      
    
        with open(sample_log_file, "a") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

del prop_sloppy
del pin
