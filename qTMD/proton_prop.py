import gpt as g 
import os
import sys
import numpy as np
from qTMD.gpt_proton_qTMD_utils import proton_measurement


from tools import *


root_output = "."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/lustre/orion/proj-shared/nph159/data/64I/propagator/GSRC_Proton/"


# tags
sm_tag = "GSRC_W80_k3"
lat_tag = "64I"


def get_sample_log_tag(ama, src, sm):

    ama_tag = str(ama)
    src_tag = "x"+str(src[0]) + "y"+str(src[1]) + "z"+str(src[2]) + "t"+str(src[3])
    sm_tag  = str(sm)

    log_sample = ama_tag + "_" + src_tag + "_" + sm_tag

    return log_sample

parameters = {
    "plist" : [[0,0,0,0]],
    "width" : 8.0,
    "pos_boost": [0,0,3],
    "save_propagators": True
}
config_number = g.default.get_int("--config_num", 0)

groups = {
    "Frontier_batch_0": {
        "confs": [
            str(config_number),
    #        "1430",
    #        "1450",
     #       "1470",
      #      "1490",
        ],
#        "evec_fmt": "/lustre/orion/proj-shared/nph159/data/64I/%s.evecs/lanczos.output",
        "evec_fmt": "/lustre/orion/proj-shared/nph159/data/64I/job-0%s/lanczos.output",
        "conf_fmt": "/lustre/orion/proj-shared/nph159/data/64I/ckpoint_lat.Coulomb.%s",
    },
}


jobs = {
    "proton_forward_prop_AMA": {
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

# configuration needs to be the same for all jobs, so load eigenvectors and configuration
conf = run_jobs[0][2]
group = run_jobs[0][0]


# loading gauge configuration
g.message("Loading ")
print(groups[group]["conf_fmt"] % conf)
U = g.load(groups[group]["conf_fmt"] % conf)
g.message("finished loading gauge config")

# do gauge fixing
U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime
L = U[0].grid.fdimensions

Measurement = proton_measurement(parameters)
prop_exact, prop_sloppy, pin = Measurement.make_64I_inverter(U, groups[group]["evec_fmt"] % conf)



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
    
    sample_log_file = data_dir + "/sample_log/" + conf + "_W80_k3"
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

        prop_dir = data_dir + "prop_ex_W80_k3/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
        if g.rank() == 0:
            if not prop_dir:
                os.makedirs(prop_dir)
        Measurement.set_propagator_output_facilities(prop_dir)

        g.message("Generatring boosted src's")
        srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")

        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")

        #production tag should include more config/action details
        prop_tag_exact = "%s/%s/%s/%s_%s" % ("exact", lat_tag, sm_tag, str(conf), str(pos))

        Measurement.propagator_output_k0(prop_tag_exact, prop_exact_f)

        with open(sample_log_file, "a") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_exact_f

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

        prop_dir = data_dir + "prop_sl_W80_k3/" + conf + "/" + "x"+str(pos[0]) + "y"+str(pos[1]) + "z"+str(pos[2]) + "t"+str(pos[3])
        if g.rank() == 0:
            if not prop_dir:
                os.makedirs(prop_dir)
        Measurement.set_propagator_output_facilities(prop_dir)

        g.message("Generatring boosted src's")
        srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop sloppy")

        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")


        #production tag should include more config/action details
        prop_tag_sloppy = "%s/%s/%s/%s_%s" % ("sloppy", lat_tag, sm_tag, str(conf), str(pos))

        Measurement.propagator_output_k0(prop_tag_sloppy, prop_sloppy_f)

        with open(sample_log_file, "a") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_sloppy_f
        
    del prop_sloppy
    
del pin
