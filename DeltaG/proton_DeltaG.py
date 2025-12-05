#!/usr/bin/env python3
#
# Authors: Christoph Lehner 2020 / Philipp Scior 2021
#
# Calculate proton QPDF with A2A method
#
import gpt as g
import os
import numpy as np
from gpt_qpdf_utils import proton_qpdf_measurement

# configure
# root_output = "/p/project/chbi21/gpt_test/DA"
root_output ="."

# 420, 500, 580
groups = {
    "booster_batch_0": {
        "confs": [
            "420",
            "1260",
            "1272",
        ],
	"evec_fmt": "/p/project/chbi21/gpt_test/96I/lanczos.output",
        "conf_fmt": "/p/project/chbi21/gpt_test/96I/ckpoint_lat.%s",
    },

}
parameters = {
    "pf" : [1,1,0,0],
    "q" : [0,1,0,0],
    "zmax" : 4,
    "t_insert" : 4,
    "width" : 2.2,
    "boost_in" : [0,0,0],
    "boost_out": [0,0,0],
    "save_propagators" : True
}

hyp=False
flow=False
n_hyp=3
n_flow=3

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

g.message(f"conf = {conf}")
##### small dummy used for testing
#grid = g.grid([16,16,16,16], g.double)
rng = g.random("seed text")
#U = g.qcd.gauge.random(grid, rng)


# loading gauge configuration
path = groups[group]["conf_fmt"] % conf
g.message(f"configpath = {path}")
U = g.convert(g.load(groups[group]["conf_fmt"] % conf), g.double)
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
L = U[0].grid.fdimensions


#do gauge fixing
g.message("Starting gauge fixing to Coulomb gauge")
U_fixed, V = g.gauge_fix(U, maxiter=500)
V = g.project(V, "defect")

g.message(f"gauge fixing done")

import sys


Measurement = proton_qpdf_measurement(parameters)

inv_exact, inv_sloppy, pin = Measurement.make_DWF_inverter(U, groups[group]["evec_fmt"])

#inv_exact, inv_sloppy = Measurement.make_debugging_inverter(U)

#inv_exact = Measurement.make_Clover_MG_inverter(U)
#inv_sloppy = Measurement.make_Clover_MG_inverter(U)


phases = Measurement.make_mom_phases(U[0].grid)


# show available memory
g.mem_report(details=False)
g.message(
    """
================================================================================
       Delta G run on crusher ;  this run will attempt:
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
    rng = g.random(f"QPDF-ensemble-{conf}-{job_seed}")

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

    g.message("Starting Wilson loops")
    W = Measurement.create_WL(U)

    # exact positions
    for pos in source_positions_exact:

        g.message("STARTING EXACT MEASUREMENTS")

        g.message("Generatring boosted src's")
        srcDp = Measurement.create_src(pos, V, U[0].grid)

        g.message("Starting prop exact")
        prop_exact_f = g.eval(inv_exact * srcDp)
        g.message("forward prop done")

        tag = "%s/%s" % ("exact", str(pos)) 

        g.message("Starting 2pt contraction (includes sink smearing)")
        Measurement.contract_2pt(prop_exact_f, phases, V, tag)
        g.message("2pt contraction done")

        del prop_exact_f

        g.message("STARTING SLOPPY MEASUREMENTS")
    
        g.message("Starting prop sloppy")
        prop_sloppy_f = g.eval(inv_sloppy * srcDp)
        g.message("forward prop done")

        del srcDp

        tag = "%s/%s" % ("sloppy", str(pos))



        g.message("Starting 2pt contraction (includes sink smearing)")
        Measurement.contract_2pt(prop_sloppy_f, phases, V, tag)
        g.message("2pt contraction done")

        del prop_sloppy_f
     
    g.message("exact positions done")

    # sloppy positions
    for pos in source_positions_sloppy:

        g.message("STARTING SLOPPY MEASUREMENTS")
        tag = "%s/%s" % ("sloppy", str(pos))

        g.message("Starting 2pt function")

        g.message("Generatring boosted src's")
        srcDp = Measurement.create_src(pos, V, U[0].grid)  

        g.message("Starting prop sloppy")
        prop_sloppy_f = g.eval(inv_sloppy * srcDp)
        g.message("forward prop done")

        del srcDp

        tag = "%s/%s" % ("sloppy", str(pos))



        g.message("Starting 2pt contraction (includes sink smearing)")
        Measurement.contract_2pt(prop_sloppy_f, phases, V, tag)
        g.message("2pt contraction done")

    
    g.message("sloppy positions done")
   
    del pin

    #Definition of gauge potential A_mu is taken from arXiv:1609.05937. Factor 1/(4iga) is NOT included here!

    Ex = g.qcd.gauge.field_strength(U, 3, 0)
    Ey = g.qcd.gauge.field_strength(U, 3, 1)
    Ez = g.qcd.gauge.field_strength(U, 3, 2)

    Ax = g.eval(U[0] - g.adj(U[0]) + g.cshift(U[0],0,-1) + g.adj(g.cshift(U[0],0,-1)))
    Ax -= g.identity(Ax) * g.trace(Ax) / 3

    Ay = g.eval(U[1] - g.adj(U[1]) + g.cshift(U[1],1,-1) + g.adj(g.cshift(U[1],1,-1)))
    Ay -= g.identity(Ay) * g.trace(Ay) / 3

    Az = g.eval(U[2] - g.adj(U[2]) + g.cshift(U[2],2,-1) + g.adj(g.cshift(U[2],2,-1)))
    Az -= g.identity(Az) * g.trace(Az) / 3

    At = g.eval(U[3] - g.adj(U[3]) + g.cshift(U[3],3,-1) + g.adj(g.cshift(U[3],3,-1)))
    At -= g.identity(At) * g.trace(At) / 3


    Sg_x = g.slice(g.trace(Ey * Az - Ez * Ay), 3)
    Sg_y = g.slice(g.trace(Ez * Ax - Ex * Az), 3)
    Sg_z = g.slice(g.trace(Ex * Ay - Ey * Ax), 3)


    #O321 is the naming convention from Minkowski space. In the Euklidean space version of Grid t=3,
    # so Mikowski O321 = Euklidean O210!
    O321 = g.slice(g.trace(g.qcd.gauge.field_strength(U, 2, 1)* Ax - g.qcd.gauge.field_strength(U, 2, 0)* Ay ) , 3)
    O021 = g.slice(g.trace(g.qcd.gauge.field_strength(U, 3, 1)* Ax - g.qcd.gauge.field_strength(U, 3, 0)* Ay ) , 3)
    B_dot_A = np.array(O321) + np.array(g.slice(g.trace(g.qcd.gauge.field_strength(U, 1, 0) * Az) , 3))

    g.message(f"O321 = {O321}")
    g.message(f"O021 = {O021}")
    g.message(f"B dot A = {B_dot_A}")

