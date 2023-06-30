#!/usr/bin/env python3


import gpt as g
import os
import sys

from qTMD.gpt_qTMD_utils import TMD_WF_measurement
from tools import *
from io_corr import *
#from utils.tools import *
#from utils.io_corr import *

# configure
root_output="."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/home/gaox/latwork/DWF/64I/propagator/data/GSRC_W40_k6/"

# tags
sm_tag = "GSRC_W40_k6"
lat_tag = "64I"

# configuration setup
groups = {
    "Polaris_batch_0": {
        "confs": [
            "cfg",
        ],
        "evec_fmt": "/home/gaox/latwork/DWF/64I/propagator/gauge/%s.evecs/lanczos.output",
        "conf_fmt": "/home/gaox/latwork/DWF/64I/propagator/gauge/Coulomb/ckpoint_lat.Coulomb.%s",
    },
}

# momenta setup
parameters = {
    "eta" : [8, 10, 12],
    "b_T": 9,
    "b_z" : 9,
    "pzmin" : 3,
    "pzmax" : 10,
    "width" : 4.0,
    "pos_boost" : [0,0,6],
    "neg_boost" : [0,0,-6],
    "save_propagators" : True
}

jobs = {
    "test_exact_0": {
        "exact": 2,
        "sloppy": 128,
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
    
    sample_log_file = data_dir + "/sample_log/" + conf
    if g.rank() == 0:
        f = open(sample_log_file, "a")
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

        prop_dir = data_dir + "prop_ex/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
        if g.rank() == 0:
            if not prop_dir:
                os.makedirs(prop_dir)
        Measurement.set_output_facilities(corr_dir, prop_dir)

        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")

        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")

        prop_exact_b = g.eval(prop_exact * srcDm)
        g.message("backward prop done")
        
        #production tag should include more config/action details
        prop_tag_exact = "%s/%s/%s/%s_%s" % ("exact", lat_tag, sm_tag, str(conf), str(pos))
        
        Measurement.propagator_output(prop_tag_exact, prop_exact_f, prop_exact_b)
        
        with open(sample_log_file, "a") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_exact_f
        del prop_exact_b

    del prop_exact

    # sloppy positions
    corr_dir = data_dir + "corr_sl/" + conf
    for pos in source_positions_sloppy:
        
        sample_log_tag = get_sample_log_tag("sl", pos, sm_tag)
        g.message(f"START: {sample_log_tag}")
        with open(sample_log_file) as f:
            if sample_log_tag in f.read():
                g.message("SKIP: " + sample_log_tag)
                continue
        
        prop_dir = data_dir + "prop_sl/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
        if g.rank() == 0:
            if not prop_dir:
                os.makedirs(prop_dir)
        Measurement.set_output_facilities(corr_dir, prop_dir)

        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop sloppy")

        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")

        prop_sloppy_b = g.eval(prop_sloppy * srcDm)
        g.message("backward prop done")
        
        #production tag should include more config/action details
        prop_tag_sloppy = "%s/%s/%s/%s_%s" % ("sloppy", lat_tag, sm_tag, str(conf), str(pos))        
        
        Measurement.propagator_output(prop_tag_sloppy, prop_sloppy_f, prop_sloppy_b)
        
        with open(sample_log_file, "a") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_sloppy_f
        del prop_sloppy_b
        
    del prop_sloppy
    
del pin
