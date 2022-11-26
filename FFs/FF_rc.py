#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020 / Philipp Scior 2021
#
# Calculate pion FF with A2A method
#
import gpt as g
import os
#import numpy as np
from gpt_qpdf_utils import pion_ff_measurement

# configure
# root_output = "/p/project/chbi21/gpt_test/DA"
root_output ="."

# 420, 500, 580
groups = {
    "booster_batch_0": {
        "confs": [
            "420",
            "1960",
            "2000",
        ],
        #"evec_fmt": "/p/scratch/gm2dwf/evecs/96I/%s/lanczos.output",
	    "evec_fmt": "/p/project/chbi21/gpt_test/96I/lanczos.output",
        "conf_fmt": "/p/project/chbi21/gpt_test/96I/ckpoint_lat.%s",
    },

}
parameters = {
    "pf" : [1,1,0,0],
    "q" : [0,1,0,0],
    "t_insert" : 4,
    "width" : 2.2,
    "boost_in" : [0,0,0],
    "boost_out" : [0,0,0],
    "save_propagators" : True
}


jobs = {
    "booster_exact_0": {
        "exact": 1,
        "sloppy": 0,
        "low": 0,
    },  
    "booster_sloppy_0": {
        "exact": 0,
        "sloppy": 10,
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
    # for job in jobs:
    #    for group in groups:
    #        for conf in groups[group]["confs"]:
    for group in groups:
        for conf in groups[group]["confs"]:
            for job in jobs:
                jid += 1
                if only_on_conf is not None and only_on_conf != conf:
                    continue
                root_job = f"{root_output}/{conf}/{job}"
                if not os.path.exists(root_job):
                    os.makedirs(root_job)
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

# every node now knows what to do



# configuration needs to be the same for all jobs, so load eigenvectors and configuration
conf = run_jobs[0][2]
group = run_jobs[0][0]


##### small dummy used for testing
grid = g.grid([8,8,8,8], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.random(grid, rng)

# loading gauge configuration
# U = g.load(groups[group]["conf_fmt"] % conf)
g.message("finished loading gauge config")


# do gauge fixing

U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime

L = U[0].grid.fdimensions

Measurement = pion_ff_measurement(parameters)

#prop_exact, prop_sloppy, pin = Measurement.make_96I_inverter(U, groups[group]["evec_fmt"])

prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)

phases = Measurement.make_mom_phases(U[0].grid)


# show available memory
g.mem_report(details=False)
g.message(
    """
================================================================================
       DA run on booster ;  this run will attempt:
================================================================================
"""
)
# per job
for group, job, conf, jid, n in run_jobs:
    g.message(
        f"""

    Job {jid} / {n} :  configuration {conf}, job tag {job}

"""
    )

    job_seed = job.split("_correlated")[0]
    rng = g.random(f"DA-ensemble-{conf}-{job_seed}")

    source_positions_sloppy = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["sloppy"])
    ]
    source_positions_exact = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["exact"])
    ]



    g.message(f" positions_sloppy = {source_positions_sloppy}")
    g.message(f" positions_exact = {source_positions_exact}")

    root_job = f"{root_output}/{conf}/{job}"

    Measurement.set_output_facilites(f"{root_job}/correlators",f"{root_job}/propagators")


    # exact positions
    for pos in source_positions_exact:

        g.message("STARTING EXACT MEASUREMENTS")

        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")
        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")
        prop_exact_b = g.eval(prop_exact * srcDm)
        g.message("backward prop done")

        tag = "%s/%s" % ("exact", str(pos)) 

        g.message("Starting 2pt contraction (includes sink smearing)")
        Measurement.contract_2pt(prop_exact_f, prop_exact_b, phases, trafo, tag)
        g.message("2pt contraction done")

        if(parameters["save_propagators"]):
            Measurement.propagator_output(tag, prop_exact_f, prop_exact_b)

        g.message("Create seq. backwards prop")
        prop_b = Measurement.create_bw_seq(prop_exact, prop_exact_b, trafo)

        g.message("Start FF contractions")
        Measurement.contract_FF(prop_exact_f, prop_b, phases, tag)
        g.message("FF done")

        del prop_exact_f
        del prop_b

        g.message("STARTING SLOPPY MEASUREMENTS")
    
        g.message("Starting prop sloppy")
        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")
        prop_sloppy_b = g.eval(prop_sloppy * srcDm)
        g.message("backward prop done")

        del srcDp
        del srcDm

        tag = "%s/%s" % ("sloppy", str(pos))

        g.message("Starting 2pt contraction (includes sink smearing)")
        Measurement.contract_2pt(prop_sloppy_f, prop_sloppy_b, phases, trafo, tag)
        g.message("2pt contraction done")

        if(parameters["save_propagators"]):
            Measurement.propagator_output(tag, prop_sloppy_f, prop_sloppy_b)

        g.message("Create seq. backwards prop")
        prop_b = Measurement.create_bw_seq(prop_sloppy, prop_sloppy_b, trafo)

        g.message("Start FF contractions")
        Measurement.contract_FF(prop_sloppy_f, prop_b, phases, tag)
        g.message("FF done")

        del prop_b
        del prop_sloppy_f
     
    g.message("exact positions done")

    # sloppy positions
    for pos in source_positions_sloppy:

        g.message("STARTING SLOPPY MEASUREMENTS")
        tag = "%s/%s" % ("sloppy", str(pos))

        g.message("Starting DA 2pt function")

        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)  

        g.message("Starting prop sloppy")
        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")
        prop_sloppy_b = g.eval(prop_sloppy * srcDm)
        g.message("backward prop done")

        del srcDp
        del srcDm

        tag = "%s/%s" % ("sloppy", str(pos))

        g.message("Starting 2pt contraction (includes sink smearing)")
        Measurement.contract_2pt(prop_sloppy_f, prop_sloppy_b, phases, trafo, tag)
        g.message("2pt contraction done")

        if(parameters["save_propagators"]):
            Measurement.propagator_output(tag, prop_sloppy_f, prop_sloppy_b)

        g.message("Create seq. backwards prop")
        prop_b = Measurement.create_bw_seq(prop_sloppy, prop_sloppy_b, trafo)

        g.message("Start FF contractions")
        Measurement.contract_FF(prop_sloppy_f, prop_b, phases, tag)
        g.message("FF done")

        del prop_b
        del prop_sloppy_f   
    
    g.message("sloppy positions done")
        
#del pin

