#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020 / Philipp Scior 2021
#
# Calculate pion DA with A2A method
#
import gpt as g
import os
import numpy as np
from gpt_qpdf_utils import pion_DA_measurement
from tools import *
from io_corr import *


"""
================================================================================
                                Config setup
================================================================================
"""

# configure
root_output ="."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/home/gaox/latwork/DWF/64I/prod/data/"

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
    "zmax"  : 0,
    "pzmin" : -10,
    "pzmax" : 11,
    "width" : 4.0,
    "pos_boost" : [0,0,6],
    "neg_boost" : [0,0,-6],
    "save_propagators" : False
}
# tags
sm_tag = "GSRC_W40_kp6kn-6"
lat_tag = "64I"
# AMA setup
jobs = {
    "booster_exact_0": {
        "exact": 1,
        "sloppy": 64,
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

Measurement = pion_DA_measurement(parameters)
prop_exact, prop_sloppy, pin = Measurement.make_64I_inverter(U, groups[group]["evec_fmt"] % conf)
#prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)

# show available memory
g.mem_report(details=False)
g.message(
"""
================================================================================
       2pt run on booster ;  this run will attempt:
================================================================================
"""
)
# per job
for group, job, conf, jid, n in run_jobs:

    g.message(f"""Job {jid} / {n} :  configuration {conf}, job tag {job}""")

    # the original point for source creation which shift by conf number
    src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
    source_positions = srcLoc_distri_eq(L, src_origin)
    #print(source_positions)
    source_positions_sloppy = source_positions[:jobs[job]["sloppy"]]
    source_positions_exact = source_positions[:jobs[job]["exact"]]

    #root_job = data_dir + "/c2pt"
    #Measurement.set_output_facilites(f"{root_job}/correlators",f"{root_job}/propagators")
    
    sample_log_file = data_dir + "/sample_log/" + conf
    #if g.rank() == 0:
    f = open(sample_log_file, "w")
    f.close()

    # exact positions
    g.message(f" positions_exact = {source_positions_exact}")
    for pos in source_positions_exact:
        phases = Measurement.make_mom_phases(U[0].grid, pos)
        sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
        g.message(f"START: {sample_log_tag}")
        with open(sample_log_file) as f:
            if sample_log_tag in f.read():
                g.message("SKIP: " + sample_log_tag)
                continue
        
        #g.message("STARTING EXACT MEASUREMENTS")
        g.message("Starting 2pt function")
        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")
        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")
        prop_exact_b = g.eval(prop_exact * srcDm)
        g.message("backward prop done")

        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
        g.message("Starting 2pt contraction (includes sink smearing)")
        Measurement.contract_2pt(prop_exact_f, prop_exact_b, phases, trafo, tag)
        g.message("2pt contraction done")

        if(parameters["save_propagators"]):
            Measurement.propagator_output(tag, prop_exact_f, prop_exact_b)

        del prop_exact_f
        del prop_exact_b

        with open(sample_log_file, "a") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)
    
    # sloppy positions
    del prop_exact
    g.message(f" positions_sloppy = {source_positions_sloppy}")
    for pos in source_positions_sloppy:
        phases = Measurement.make_mom_phases(U[0].grid, pos)
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

        g.message("Starting prop sloppy")
        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")
        prop_sloppy_b = g.eval(prop_sloppy * srcDm)
        g.message("backward prop done")

        g.message("Starting pion 2pt function")
        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "sl", pos, sm_tag)
        g.message("Starting pion contraction (includes sink smearing)")
        Measurement.contract_2pt(prop_sloppy_f, prop_sloppy_b, phases, trafo, tag)
        g.message("pion contraction done")

        if(parameters["save_propagators"]):
            Measurement.propagator_output(tag, prop_sloppy_f, prop_sloppy_b)

        del prop_sloppy_f
        del prop_sloppy_b      

        with open(sample_log_file, "a") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)
        
del pin

