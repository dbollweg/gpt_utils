#!/usr/bin/env python3


import gpt as g
import os
import sys
import math

from qTMD.gpt_qTMD_utils import TMD_WF_measurement
from tools import *
from io_corr import *
#from utils.tools import *
#from utils.io_corr import *

root_output="."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/home/gaox/latwork/DWF/64I/contraction/data/GSRC_W40_k0/"

# tags
sm_tag = "GSRC_W40_k0"
lat_tag = "64I"
contract_tag = "hyp1"

hyp=True
flow=False
n_hyp=1
n_flow=0

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
    "eta" : [8],
    "b_T": 9,
    "b_z" : 9,
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

##### small dummy used for testing
#grid = g.grid([8,8,8,8], g.double)
#rng = g.random("seed text")
#U = g.qcd.gauge.random(grid, rng)

# loading gauge configuration
U = g.load(groups[group]["conf_fmt"] % conf)
g.message("finished loading gauge config")

g.message("Doing some kind of smearing")
if hyp:
    import numpy as np
    for i in range(n_hyp):
        U = g.qcd.gauge.smear.hyp(U, alpha = np.array([0.75, 0.6, 0.3]))

if flow:
    for i in range(n_flow):
        U = g.qcd.gauge.smear.wilson_flow(U, epsilon=0.1)
g.message("Smearing/Flow finishe")

# do gauge fixing
#U_prime, trafo = g.gauge_fix(U, maxiter=500)
#del U_prime
L = U[0].grid.fdimensions

Measurement = TMD_WF_measurement(parameters)
#prop_exact, prop_sloppy, pin = Measurement.make_64I_inverter(U, groups[group]["evec_fmt"] % conf)
#prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)

# show available memory
g.mem_report(details=False)
g.message(
"""
================================================================================
       contraction run:
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

    sample_log_file = data_dir + "/sample_log/" + contract_tag + conf
    if g.rank() == 0:
        f = open(sample_log_file, "a")
        f.close()


    ################################ exact sources ###################################
    g.message("Starting contraction: exact")
    g.message(f" positions_exact = {source_positions_exact}")
    # exact positions
    props_exact = {}
    prop_dir = data_dir + "prop_ex/" + conf
    #prop_dir = data_dir + "prop_ex"
    propagator_read = Measurement.propagator_input(prop_dir)
    g.mem_report(details=False)
    for p in [propagator_read]:
        props_exact.update(p)
    g.mem_report(details=False)

    corr_dir = data_dir + "corr_ex/" + conf
    for pos in source_positions_exact:

        sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
        g.message(f"START: {sample_log_tag}")
        with open(sample_log_file) as f:
            if sample_log_tag in f.read():
                g.message("SKIP: " + sample_log_tag)
                continue

        prop_src_dir = data_dir + "prop_ex_src/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
        if g.rank() == 0:
            if not prop_src_dir:
                os.makedirs(prop_src_dir)
        Measurement.set_output_facilities(corr_dir, prop_src_dir)

        prop_tag_exact = "%s/%s/%s/%s_%s" % ("exact", lat_tag, sm_tag, str(conf), str(pos))
        g.message(prop_tag_exact)
        prop_f_tag = "%s/%s" % (prop_tag_exact, Measurement.pos_boost)
        prop_b_tag = "%s/%s" % (prop_tag_exact, Measurement.neg_boost)
        prop_f = props_exact[prop_f_tag]
        prop_b = props_exact[prop_b_tag]
        Measurement.propagator_output_k0(prop_tag_exact, prop_f)

        with open(sample_log_file, "a") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_f, prop_b
    del props_exact

    ################################ sloppy sources ###################################
    g.message("Starting contraction: sloppy")
    g.message(f" positions_sloppy = {source_positions_sloppy}")
    # sloppy positions
    props_sloppy = {}
    prop_dir = data_dir + "prop_sl/" + conf
    #prop_dir = data_dir + "prop_sl"
    g.mem_report(details=False)
    for p in Measurement.propagator_input(prop_dir):
        props_sloppy.update(p)
    g.mem_report(details=False)

    corr_dir = data_dir + "corr_sl/" + conf
    for pos in source_positions_sloppy:

        sample_log_tag = get_sample_log_tag("sl", pos, sm_tag)
        g.message(f"START: {sample_log_tag}")
        with open(sample_log_file) as f:
            if sample_log_tag in f.read():
                g.message("SKIP: " + sample_log_tag)
                continue

        prop_src_dir = data_dir + "prop_sl_src/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
        if g.rank() == 0:
            if not prop_src_dir:
                os.makedirs(prop_src_dir)
        Measurement.set_output_facilities(corr_dir, prop_src_dir)

        prop_tag_sloppy = "%s/%s/%s/%s_%s" % ("sloppy", lat_tag, sm_tag, str(conf), str(pos))
        g.message(prop_tag_sloppy)
        prop_f_tag = "%s/%s" % (prop_tag_sloppy, Measurement.pos_boost)
        prop_b_tag = "%s/%s" % (prop_tag_sloppy, Measurement.neg_boost)
        prop_f = props_sloppy[prop_f_tag]
        prop_b = props_sloppy[prop_b_tag]
        Measurement.propagator_output_k0(prop_tag_sloppy, prop_f)

        with open(sample_log_file, "a") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_f, prop_b
    del props_sloppy
