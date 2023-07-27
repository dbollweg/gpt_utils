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



"""
================================================================================
                                Config setup
================================================================================
"""

# configure
root_output ="."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/home/gaox/latwork/DWF/TEST/proton_2pt/data"

# configuration setup
groups = {
    "booster_batch_0": {
        "confs": [
            "cfg",
        ],
            "evec_fmt": "/home/gaox/latwork/DWF/24D/proton_2pt/gauge/job-0%s/lanczos.output",
        "conf_fmt": "/home/gaox/latwork/DWF/24D/proton_2pt/gauge/ckpoint_lat.%s",
    },

}

# momenta setup
parameters = {
    "zmax"  : 0,
    "pzmin" : 0,
    "pzmax" : 0,
    "width" : 1.0,
    "pos_boost" : [0,0,0],
    "neg_boost" : [0,0,0],
    "save_propagators" : False
}
parameters = {
    "pzmin" : 0,
    "pzmax" : 1,
    "q" : [0,0,0,0],
    "zmax" : 0,
    "t_insert" : 32,
    "width" : 4.0,
    "boost_in" : [0,0,0],
    "boost_out": [0,0,0],
    "save_propagators" : False
}

# tags
sm_tag = "GSRC_proton_W40_k0"
lat_tag = "24D"
# AMA setup
jobs = {
    "booster_exact_0": {
        "exact": 1,
        "sloppy": 0,
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

def uud_two_point(Q1, Q2, kernel):
    dq = g.qcd.baryon.diquark(g(Q1 * kernel), g(kernel * Q2))
    return g(g.color_trace(g.spin_trace(dq) * Q1 + dq * Q1))
def proton(Q1, Q2):
    C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[5].tensor()
    Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
    return g(g.trace(uud_two_point(Q1, Q2, Gamma) * Pp))

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
Ns=4
Nt=4
grid = g.grid([Ns,Ns,Ns,Nt], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.unit(grid)
#U = g.qcd.gauge.random(grid, rng)
#print(U)

# loading gauge configuration
#U = g.load(groups[group]["conf_fmt"] % conf)
#g.message("finished loading gauge config")

# do gauge fixing
U_prime, trafo = g.gauge_fix(U, maxiter=0)
del U_prime
L = U[0].grid.fdimensions

Measurement = proton_qpdf_measurement(parameters)
#prop_exact, prop_sloppy, pin = Measurement.make_24D_inverter(U, groups[group]["evec_fmt"] % conf)
prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)
phases = Measurement.make_mom_phases(U[0].grid)

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

    sample_log_file = data_dir + "/sample_log/" + sm_tag + conf
    if g.rank() == 0:
        f = open(sample_log_file, "a")
        f.close()
    source_positions_exact = [[0,0,0,0]]
    # exact positions
    g.message(f" positions_exact = {source_positions_exact}")
    for ipos, pos in enumerate(source_positions_exact):

        sample_log_tag = get_sample_log_tag("ex", pos, sm_tag)
        g.message(f"Contraction START: {sample_log_tag}")
        #with open(sample_log_file) as f:
        #    if sample_log_tag in f.read():
        #        g.message("Contraction SKIP: " + sample_log_tag)
        #        continue

        #g.message("STARTING EXACT MEASUREMENTS")
        g.message("Starting 2pt function")
        g.message("Generatring boosted src's")
        srcD = g.mspincolor(grid)
        g.create.point(srcD, pos)
        srcDp = srcD
        #srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        #print(srcDp)
        g.message("Starting prop exact")
        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")
        prop_exact_f = srcDp
        #print(prop_exact_f)

        g.message("Contraction: Starting 2pt (includes sink smearing)")
        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
        Measurement.contract_2pt(prop_exact_f, phases, trafo, tag)
        g.message("Contraction: Done 2pt (includes sink smearing)")

        proton1 = proton(prop_exact_f, prop_exact_f)
        if g.rank() == 0:
            print(proton1[0,0,0,0], proton1[0,0,0,1], proton1[0,0,0,2], proton1[0,0,0,3])
            #print(np.shape(proton1))
        del prop_exact_f

        #with open(sample_log_file, "a") as f:
        #    if g.rank() == 0:
        #        f.write(sample_log_tag+"\n")
        g.message("Contraction DONE: " + sample_log_tag)