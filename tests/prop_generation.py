#!/usr/bin/env python3


import gpt as g
import os
import sys

from gpt_qpdf_utils import pion_measurement
from utils.tools import *
from utils.io_corr import *


root_output="."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/lus/grand/projects/StructNGB/bollwegd/64I/Coulomb/"


groups = {
    "Polaris_batch_0": {
        "confs": [
            cfg,
        ],
        "evec_fmt": "/lus/grand/projects/StructNGB/bollwegd/64I/%.evecs/lanczos.output",
        "conf_fmt": "/lus/grand/projects/StructNGB/bollwegd/64I/Coulomb/ckpoint_lat.Coulomb.%s",        
    },
}

parameters = {
    "zmax"  : 0,
    "plist" : [[0,0, 0, 0],[0,0, 1, 0],[0,0, 2, 0],[0,0, 3, 0],[0,0, 4, 0],[0,0, 5, 0]],
    "width" : 4.0,
    "pos_boost" : [0,0,0],
    "neg_boost" : [0,0,0],
    "save_propagators" : True
}


jobs = {
    "test_exact_0": {
        "exact": 1,
        "sloppy": 2,
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

conf = run_jobs[0][2]
group = run_jobs[0][0]


U = g.load(groups[group]["conf_fmt"] % conf)
rng = g.random("seed text")



# do gauge fixing
U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime
L = U[0].grid.fdimensions

Measurement = pion_measurement(parameters)
prop_exact, prop_sloppy, pin = Measurement.make_64I_inverter(U, groups[group]["evec_fmt"] % conf)



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
    #if g.rank() == 0:
    f = open(sample_log_file, "w")
    f.close()
    
    Measurement.set_output_facilities("/lus/grand/projects/StructNGB/bollwegd/testrun/polaris_correlators_exact","/lus/grand/projects/StructNGB/bollwegd/testrun/polaris_propagators_exact")		    
    # exact positions
    
    for pos in source_positions_exact:
        
        g.message("Starting 2pt function")
        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")


        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")


        prop_exact_b = g.eval(prop_exact * srcDm)
        g.message("backward prop done")
        
        
        prop_tag_exact = "%s/%s" % ("polaris_test_exact", str(pos)) #production tag should include more config/action details

        
        Measurement.propagator_output(prop_tag_exact, prop_exact_f, prop_exact_b)
        
        del prop_exact_f
        del prop_exact_b

    del prop_exact
    
    for pos in source_positions_sloppy:
        
        g.message("Starting 2pt function")
        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop sloppy")


        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")


        prop_sloppy_b = g.eval(prop_sloppy * srcDm)
        g.message("backward prop done")
        
        
        prop_tag_sloppy = "%s/%s" % ("polaris_test_sloppy", str(pos)) #production tag should include more config/action details

        
        Measurement.propagator_output(prop_tag_sloppy, prop_sloppy_f, prop_sloppy_b)
        
        del prop_sloppy_f
        del prop_sloppy_b
        
    del prop_sloppy
    
del pin
        