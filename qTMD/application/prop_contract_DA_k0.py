#!/usr/bin/env python3
import gpt as g
import os
import sys
import math
import numpy as np

from qTMD.gpt_qTMD_utils import pion_DA_measurement
from tools import *
from io_corr import *
#from utils.tools import *
#from utils.io_corr import *

root_output="."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/home/gaox/latwork/DWF/64I/contraction_DA/data/GSRC_W40_k0/"

# Wilson flow or HYP smearing
#smear_list = [['flow', '01eps01', 1], ['flow', '05eps01', 5], ['flow', '10eps01', 10], ['flow', '20eps01', 20]]
smear_list = [['flow', '05eps01', 5], ['flow', '10eps01', 10]]
#smear_list = [['flow', '05eps01', 5]]

# tags
sm_tag = "GSRC_W40_k0"
lat_tag = "64I"

# configuration setup
groups = {
    "Polaris_batch_0": {
        "confs": [
            "cfg",
        ],
        "evec_fmt": "/home/gaox/latwork/DWF/64I/prod/gauge/%s.evecs/lanczos.output",
        "conf_fmt": "/home/gaox/latwork/DWF/64I/prod/gauge/Coulomb/ckpoint_lat.Coulomb.%s",
    },
}

# momenta setup
parameters = {
    "zmax" : 33,
    "pzmin" : 0,
    "pzmax" : 4,
    "width" : 4.0,
    "pos_boost" : [0,0,0],
    "neg_boost" : [0,0,0],
    "save_propagators" : True
}

# AMA setup
jobs = {
    "test_exact_0": {
        "exact": 1,
        "sloppy": 32,
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

Measurement = pion_DA_measurement(parameters)

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
    U = g.load(groups[group]["conf_fmt"] % conf)
    g.message("Gauge: finished loading gauge config")

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

        sample_log_file = data_dir + "/sample_log/" + contract_tag + conf
        if g.rank() == 0:
            f = open(sample_log_file, "a")
            f.close()

        g.message("Wilson Link: Start")
        g.mem_report(details=False)
        W, W_index_list = Measurement.create_DA_WL(U)
        W_count = len(W_index_list)
        W_subset_len = 20
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
            props_exact = {}
            prop_dir = data_dir + "prop_ex/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
            propagator_read = Measurement.propagator_input(prop_dir)
            for p in [propagator_read]:
                props_exact.update(p)
            #g.mem_report(details=False)

            sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
            g.message(f"Contraction START: {sample_log_tag}")
            with open(sample_log_file) as f:
                if sample_log_tag in f.read():
                    g.message("Contraction SKIP: " + sample_log_tag)
                    continue

            phases = Measurement.make_mom_phases(U[0].grid, pos)
            
            prop_tag_exact = "%s/%s/%s/%s_%s" % ("exact", lat_tag, sm_tag, str(conf), str(pos))
            g.message(f"Contraction Read Propagators: {prop_tag_exact}")
            prop_f_tag = "%s/%s" % (prop_tag_exact, Measurement.pos_boost)
            prop_b_tag = "%s/%s" % (prop_tag_exact, Measurement.neg_boost)
            prop_f = props_exact[prop_f_tag]
            prop_b = props_exact[prop_b_tag]

            g.message("Contraction: Starting 2pt (includes sink smearing)")
            tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
            Measurement.contract_2pt(prop_f, prop_b, phases, trafo, tag)
            g.message("Contraction: Done 2pt (includes sink smearing)")

            g.message(f"Contraction: Start DA with N_W = {W_count}, divided into {W_subset_count} subsets.")
            qDA_tag = get_qDA_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag+'_'+contract_tag)
            for i_sub in range(0, W_subset_count):
                g.message(f"Start DA backward propagator subset of {i_sub} / {W_subset_count}")
                prop_b_W = Measurement.constr_DA_bprop(prop_b,W[i_sub*W_subset_len:(i_sub+1)*W_subset_len], W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len])
                g.message("  Start DA contractions")
                Measurement.contract_DA(prop_f, prop_b_W, phases, qDA_tag, W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len], i_sub)
                del prop_b_W
            g.message("Contraction: Done DA")

            with open(sample_log_file, "a") as f:
                if g.rank() == 0:
                    f.write(sample_log_tag+"\n")
            g.message("Contraction DONE: " + sample_log_tag)

            del prop_f, prop_b
        del props_exact

        ################################ sloppy sources ###################################
        g.message("Contraction Start: sloppy")
        g.message(f"Contraction: positions_sloppy = {source_positions_sloppy}")

        for ipos, pos in enumerate(source_positions_sloppy):

            g.message(f"Contraction START: sloppy[{ipos}] = {pos}")
            props_sloppy = {}
            prop_dir = data_dir + "prop_sl/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
            propagator_read = Measurement.propagator_input(prop_dir)
            for p in propagator_read:
                props_sloppy.update(p)
            #g.mem_report(details=False)

            sample_log_tag = get_sample_log_tag("sl", pos, sm_tag)
            #g.message(f"Contraction do: {sample_log_tag}")
            with open(sample_log_file) as f:
                if sample_log_tag in f.read():
                    g.message("Contraction SKIP: " + sample_log_tag)
                    continue

            phases = Measurement.make_mom_phases(U[0].grid, pos)
            
            prop_tag_sloppy = "%s/%s/%s/%s_%s" % ("sloppy", lat_tag, sm_tag, str(conf), str(pos))
            g.message(f"Contraction Read Propagators: {prop_tag_sloppy}")
            prop_f_tag = "%s/%s" % (prop_tag_sloppy, Measurement.pos_boost)
            prop_b_tag = "%s/%s" % (prop_tag_sloppy, Measurement.neg_boost)
            prop_f = props_sloppy[prop_f_tag]
            prop_b = props_sloppy[prop_b_tag]

            g.message("Contraction: Starting 2pt (includes sink smearing)")
            tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "sl", pos, sm_tag)
            Measurement.contract_2pt(prop_f, prop_b, phases, trafo, tag)
            g.message("Contraction: Done 2pt (includes sink smearing)")

            g.message(f"Contraction: Start DA with N_W = {W_count}, divided into {W_subset_count} subsets.")
            qDA_tag = get_qDA_file_tag(data_dir, lat_tag, conf, "sl", pos, sm_tag+'_'+contract_tag)
            for i_sub in range(0, W_subset_count):
                g.message(f"Start DA backward propagator subset of {i_sub} / {W_subset_count}")
                prop_b_W = Measurement.constr_DA_bprop(prop_b,W[i_sub*W_subset_len:(i_sub+1)*W_subset_len], W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len])
                g.message("  Start DA contractions")
                Measurement.contract_DA(prop_f, prop_b_W, phases, qDA_tag, W_index_list[i_sub*W_subset_len:(i_sub+1)*W_subset_len], i_sub)
                del prop_b_W
            g.message("Contraction: Done DA")

            with open(sample_log_file, "a") as f:
                if g.rank() == 0:
                    f.write(sample_log_tag+"\n")
            g.message("Contraction DONE: " + sample_log_tag)

            del prop_f, prop_b
        del props_sloppy
