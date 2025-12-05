#!/usr/bin/env python3
import gpt as g
import os
import sys
import math
import numpy as np

from qTMD.gpt_proton_qTMD_utils import proton_qpdf_measurement
from tools import *
from io_corr import *
#from utils.tools import *
#from utils.io_corr import *

root_output="."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/home/gaox/latwork/DWF/64I/proton_2pt/data/"

# Wilson flow or HYP smearing
smear_list = [['flow', '00eps01', 5]]

# tags
sm_tag = "GSRC_W40_k0"
lat_tag = "64I"

# configuration setup
groups = {
    "Polaris_batch_0": {
        "confs": [
            "cfg",
        ],
        "evec_fmt": "/home/gaox/latwork/DWF/64I/contraction_DA/gauge/%s.evecs/lanczos.output",
        "conf_fmt": "/home/gaox/latwork/DWF/64I/contraction_DA/gauge/Coulomb/ckpoint_lat.Coulomb.%s",
    },
}

# momenta setup
parameters = {
    "pzmin" : 0,
    "pzmax" : 4,
    "q" : [0,1,0,0],
    "zmax" : 32,
    "t_insert" : 32,
    "width" : 4.0, #6, 7, 8, 9, 10
    "boost_in" : [0,0,0],
    "boost_out": [0,0,0],
    "save_propagators" : False
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

Measurement = proton_qpdf_measurement(parameters)
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
                    #continue

            phases = Measurement.make_mom_phases(U[0].grid, pos)
            
            prop_tag_exact = "%s/%s/%s/%s_%s" % ("exact", lat_tag, sm_tag, str(conf), str(pos))
            g.message(f"Contraction Read Propagators: {prop_tag_exact}")
            prop_f_tag = "%s/%s" % (prop_tag_exact, Measurement.boost_in)
            prop_f = props_exact[prop_f_tag]

            g.message("Contraction: Starting 2pt (includes sink smearing)")
            tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
            Measurement.contract_2pt_SRC(prop_f, phases, trafo, tag)
            g.message("Contraction: Done 2pt (includes sink smearing)")

            with open(sample_log_file, "a") as f:
                if g.rank() == 0:
                    f.write(sample_log_tag+"\n")
            g.message("Contraction DONE: " + sample_log_tag)

            del prop_f
        del props_exact

        for ipos, pos in enumerate(source_positions_sloppy):

            g.message(f"Contraction: sloppy[{ipos}] = {pos}")
            props_sloppy = {}
            prop_dir = data_dir + "prop_sl/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
            propagator_read = Measurement.propagator_input(prop_dir)
            for p in [propagator_read]:
                props_sloppy.update(p)
            #g.mem_report(details=False)

            sample_log_tag = get_sample_log_tag("sl", pos, sm_tag)
            g.message(f"Contraction START: {sample_log_tag}")
            with open(sample_log_file) as f:
                if sample_log_tag in f.read():
                    g.message("Contraction SKIP: " + sample_log_tag)
                    #continue

            phases = Measurement.make_mom_phases(U[0].grid, pos)

            prop_tag_sloppy = "%s/%s/%s/%s_%s" % ("sloppy", lat_tag, sm_tag, str(conf), str(pos))
            g.message(f"Contraction Read Propagators: {prop_tag_sloppy}")
            prop_f_tag = "%s/%s" % (prop_tag_sloppy, Measurement.boost_in)
            prop_f = props_sloppy[prop_f_tag]

            g.message("Contraction: Starting 2pt (includes sink smearing)")
            tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "sl", pos, sm_tag)
            Measurement.contract_2pt_SRC(prop_f, phases, trafo, tag)
            g.message("Contraction: Done 2pt (includes sink smearing)")

            with open(sample_log_file, "a") as f:
                if g.rank() == 0:
                    f.write(sample_log_tag+"\n")
            g.message("Contraction DONE: " + sample_log_tag)

            del prop_f
        del props_sloppy
