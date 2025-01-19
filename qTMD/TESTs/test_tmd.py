#!/usr/bin/env python3
import gpt as g
import os
import sys
import math
import numpy as np

from qTMD.gpt_qTMD_utils import TMD_WF_measurement
from tools import *
from io_corr import *
#from utils.tools import *
#from utils.io_corr import *

root_output="."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "."

# Wilson flow or HYP smearing
#smear_list = [['flow', '01eps01', 1], ['flow', '05eps01', 5], ['flow', '10eps01', 10], ['flow', '20eps01', 20]]
smear_list = [['flow', '05eps01', 5]]

# tags
sm_tag = "GSRC_W20_k4"
lat_tag = "64I"

# configuration setup
groups = {
    "Polaris_batch_0": {
        "confs": [
            "1000",
        ],
        "evec_fmt": "/home/gaox/latwork/DWF/64I/prod/gauge/%s.evecs/lanczos.output",
        "conf_fmt": "/home/gaox/latwork/DWF/64I/prod/gauge/Coulomb/ckpoint_lat.Coulomb.%s",
    },
}

# momenta setup
parameters = {
    "eta" : [4],
    "b_T": 6,
    "b_z" : 6,
    "pzmin" : 2,
    "pzmax" :4,
    "width" : 2.0,
    "pos_boost" : [0,0,4],
    "neg_boost" : [0,0,-4],
    "save_propagators" : True
}

# AMA setup
jobs = {
    "test_exact_0": {
        "exact": 1,
        "sloppy": 0,
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

#print(run_jobs)
# configuration needs to be the same for all jobs, so load eigenvectors and configuration
conf = run_jobs[0][2]
group = run_jobs[0][0]

Measurement = TMD_WF_measurement(parameters)

# show available memory
g.mem_report(details=False)
g.message(
"""
================================================================================
       contraction run:
================================================================================
"""
)

for ism, smear in enumerate(smear_list):

    g.message(f"Gauge: start {smear}")

    # loading gauge configuration
    grid = g.grid([16,16,16,16], g.double)
    rng = g.random("seed text")
    U = g.qcd.gauge.random(grid, rng)
    
    # do gauge smearing
    contract_tag, n_sm = smear[0]+smear[1], smear[2]
    if smear[0] == 'hyp':
        for i in range(n_sm):
            U = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.75, 0.6, 0.3]))
    if smear[0] == 'flow':
        for i in range(n_sm):
            U = g.qcd.gauge.smear.wilson_flow(U, epsilon=0.1)
    g.message("Gauge: Smearing/Flow finished")

    # do gauge fixing
    U_prime, trafo = g.gauge_fix(U, maxiter=1)
    del U_prime
    g.message("Gauge: No gauge fixing")
    L = U[0].grid.fdimensions

    # looping
    for group, job, conf, jid, n in run_jobs:
        
        g.message(f"""Job {jid} / {n} :  configuration {conf}, job tag {job}""")
        
        src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
        source_positions = srcLoc_distri_eq(L, src_origin)
        
        source_positions_sloppy = source_positions[:jobs[job]["sloppy"]]
        source_positions_exact = source_positions[:jobs[job]["exact"]]

        g.message(f" positions_sloppy = {source_positions_sloppy}")
        g.message(f" positions_exact = {source_positions_exact}")

        
        g.message("Wilson Link: Start")
        g.mem_report(details=False)
        W, W_index_list = Measurement.create_TMD_WL(U)
        W_count = len(W_index_list)
        W_subset_len = 3
        W_subset_count = math.ceil(W_count/W_subset_len)
        if g.rank() == 0:
            print("Wilson Link: W_count, W_subset_len, W_subset_count", W_count, W_subset_len, W_subset_count)
            print("Wilson Link: W_index_list:",W_index_list)
        g.mem_report(details=False)
        g.message("Wilson Link: Done")


        ################################ exact sources ###################################
        g.message("Contraction Start: exact")
        g.message(f"Contraction: positions_exact = {source_positions_exact}")

        for ipos, pos in enumerate(source_positions_exact):

            g.message(f"Contraction: exact[{ipos}] = {pos}")
            #props_exact = {}
            #prop_dir = data_dir + "prop_ex/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
           # propagator_read = Measurement.propagator_input(prop_dir)
            
            prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)
        

            phases = Measurement.make_mom_phases(U[0].grid, pos)
            
            prop_tag_exact = "%s/%s/%s/%s_%s" % ("exact", lat_tag, sm_tag, str(conf), str(pos))
            #g.message(f"Contraction Read Propagators: {prop_tag_exact}")
            #prop_f_tag = "%s/%s" % (prop_tag_exact, Measurement.pos_boost)
            #prop_b_tag = "%s/%s" % (prop_tag_exact, Measurement.neg_boost)
            
            g.message("Generatring boosted src's")
            srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

            # g.message("Starting prop exact")
            prop_f = g.eval(prop_exact * srcDp)
            g.message("forward prop done")
            prop_b = g.eval(prop_exact * srcDm)
            g.message("backward prop done")
            # prop_f = props_exact[prop_f_tag]
            # prop_b = props_exact[prop_b_tag]

            g.message("Contraction: Starting 2pt (includes sink smearing)")
            tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
            Measurement.contract_2pt(prop_f, prop_b, phases, trafo, tag)
            g.message("Contraction: Done 2pt (includes sink smearing)")

            g.message(f"Contraction: Start TMD with N_W = {W_count}, divided into {W_subset_count} subsets.")
            qTMDWF_tag = get_qTMDWF_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag+'_'+contract_tag)
            for i_sub in range(0, W_subset_count):
                g.message(f"Start TMD backward propagator subset of {i_sub} / {W_subset_count}")
                prop_b_W = Measurement.constr_TMD_bprop(prop_b,W[i_sub*W_subset_len:(i_sub+1)*W_subset_len], W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len])
                g.message("  Start TMD contractions")
                Measurement.contract_TMD(prop_f, prop_b_W, phases, qTMDWF_tag, W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len], i_sub)
                del prop_b_W
            g.message("Contraction: Done TMD")

            

            del prop_f, prop_b
        