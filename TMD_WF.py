#!/usr/bin/env python3

import gpt as g
import os

from gpt_qpdf_utils import TMD_WF_measurement

root_output  = "."


groups = {
    "booster_batch_0": {
        "confs": [
            "1310",
        ],
        #"evec_fmt": "/p/scratch/gm2dwf/evecs/96I/%s/lanczos.output",
	    "evec_fmt": "/pscratch/sd/d/dbollweg/64I/%s.evecs/lanczos.output",
        #"evec_fmt": "/lus/grand/projects/StructNGB/bollwegd/64I/debug_%s_evecs",
        "conf_fmt": "/pscratch/sd/d/dbollweg/64I/ckpoint_lat.Coulomb.%s",
        
    },

}

parameters = {
    "eta" : 4,
    "b_T": 8,
    "b_z" : 8,
    "pzmin" : 0,
    "pzmax" : 5,
    "width" : 2.2,
    "pos_boost" : [1,1,1],
    "neg_boost" : [-1,-1,-1],
    "save_propagators" : True
}


jobs = {
    "polaris_exact_0": {
        "exact": 1,
        "sloppy": 0,
        "low": 0,
    },
    "polaris_sloppy_0": {
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

conf = run_jobs[0][2]
group = run_jobs[0][0]


##### small dummy used for testing
#grid = g.grid([16,16,16,16], g.double)
#rng = g.random("seed text")
#U = g.qcd.gauge.random(grid, rng)

# loading gauge configuration

print(groups[group]["conf_fmt"] % conf)
U = g.load(groups[group]["conf_fmt"] % conf)
rng = g.random("seed text")
g.message("finished loading gauge config")




# do gauge fixing

U_prime, trafo = g.gauge_fix(U, maxiter=180)
del U_prime

L = U[0].grid.fdimensions



Measurement = TMD_WF_measurement(parameters)

prop_exact, prop_sloppy, pin = Measurement.make_64I_inverter(U, groups[group]["evec_fmt"] % conf)

phases = Measurement.make_mom_phases(U[0].grid)

# show available memory
g.mem_report(details=False)
g.message(
    """
================================================================================
       TMD run on polaris ;  this run will attempt:
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
    rng = g.random(f"TMD-ensemble-{conf}-{job_seed}")

#    source_positions_sloppy = [[11,11,11,11],[0,1,0,14],[0,0,0,0]]
    source_positions_sloppy = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["sloppy"])
    ]
    source_positions_exact = [[0,0,0,0]]
    #source_positions_exact = [
    #    [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
    #    for j in range(jobs[job]["exact"])
    #]



    g.message(f" positions_sloppy = {source_positions_sloppy}")
    g.message(f" positions_exact = {source_positions_exact}")

    root_job = f"{root_output}/{conf}/{job}"

    Measurement.set_output_facilites(f"{root_job}/correlators",f"{root_job}/propagators")

    g.message("Starting modified Wilson loops")
    W = Measurement.create_TMD_WL(U)

    # exact positions
    for pos in source_positions_exact:

        g.message("STARTING EXACT MEASUREMENTS")

        g.message("Starting TMD wavefunction")

        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")
        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")
        prop_exact_b = g.eval(prop_exact * srcDm)
        g.message("backward prop done")

        tag = "%s/%s" % ("exact", str(pos)) 

        prop_b = Measurement.constr_TMD_bprop(prop_exact_b,W)
        g.message("Start TMD contractions")
        Measurement.contract_TMD(prop_exact_f, prop_b, phases, tag)
        del prop_b
        g.message("TMD done")


        del prop_exact_f
        del prop_exact_b

        
    g.message("exact positions done")

    # sloppy positions
    for pos in source_positions_sloppy:

        g.message("STARTING SLOPPY MEASUREMENTS")
        tag = "%s/%s" % ("sloppy", str(pos))

        g.message("Starting TMD wavefunction")

        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)  

        g.message("Starting prop exact")
        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")
        prop_sloppy_b = g.eval(prop_sloppy * srcDm)
        g.message("backward prop done")

        del srcDp
        del srcDm
    
        prop_b = Measurement.constr_TMD_bprop(prop_sloppy_b,W)

        g.message("Start TMD contractions")
        Measurement.contract_TMD(prop_sloppy_f, prop_b, phases, tag)
        g.message("TMD contractions done")
        del prop_b

       
        del prop_sloppy_f
        del prop_sloppy_b      
    
    g.message("sloppy positions done")
        
del pin
