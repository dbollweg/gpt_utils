import gpt as g
import numpy as np
from utils.tools import *
from utils.io_corr import *
from gpt_qpdf_utils import pion_measurement
import h5py
import sys


# momenta setup
parameters = {
    "zmax"  : 0,
    "plist" : [[0,0, 2, 0]],
    "width" : 4.0,
    "pos_boost" : [0,0,0],
    "neg_boost" : [0,0,0],
    "save_propagators" : False,
}

jobs = {
    "test_exact_0": {
        "exact": 1,
        "sloppy": 2,
        "low": 0,
    },  
}

groups = {
    "booster_batch_0": {
        "confs": [
            1890,
        ],
        #"evec_fmt": "/p/scratch/gm2dwf/evecs/96I/%s/lanczos.output",
        #"evec_fmt": "/home/gaox/latwork/DWF/64I/prod/gauge/%s.evecs/lanczos.output"
        "evec_fmt": "/lus/grand/projects/StructNGB/bollwegd/64I/%s.evecs/lanczos.output",
        "conf_fmt": "/lus/grand/projects/StructNGB/bollwegd/64I/Coulomb/ckpoint_lat.Coulomb.%s",
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

src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])

# loading gauge configuration
U = g.load(groups[group]["conf_fmt"] % conf)
g.message("finished loading gauge config")

rng = g.random("seed text")

g.message("finished creating gauge config")

# do gauge fixing
U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime
L = U[0].grid.fdimensions

Measurement = pion_measurement(parameters)

# show available memory
g.mem_report(details=False)
g.message(
"""
================================================================================
       2pt contraction run:
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

    

    Measurement.set_output_facilities("/lus/grand/projects/StructNGB/bollwegd/testrun/polaris_correlators_exact","/lus/grand/projects/StructNGB/bollwegd/testrun/polaris_propagators_exact")		    
    # exact positions

    props_exact = {}
    for p in Measurement.propagator_input("/lus/grand/projects/StructNGB/bollwegd/testrun/polaris_propagators_exact"):
        props_exact.update(p)

    

    g.message(f" positions_exact = {source_positions_exact}")
    for pos in source_positions_exact:
        phases = Measurement.make_mom_phases(U[0].grid, pos)
        
        g.message("Contracting propagators for 2pt function")

        tag = "%s/%s" % ("polaris_test_exact", str(pos))
        g.message(tag)

        prop_f_tag = "%s/%s" % (tag, Measurement.pos_boost)
        prop_b_tag = "%s/%s" % (tag, Measurement.neg_boost)

        prop_f = props_exact[prop_f_tag]
        prop_b = props_exact[prop_b_tag]

        Measurement.contract_2pt_test(prop_f, prop_b, phases, trafo, tag)
        del prop_f, prop_b
    del props_exact


    props_sloppy = {}
    for p in Measurement.propagator_input("/lus/grand/projects/StructNGB/bollwegd/testrun/polaris_propagators_sloppy"):
        props_sloppy.update(p)

    for count,pos in enumerate(source_positions_sloppy):
        phases = Measurement.make_mom_phases(U[0].grid, pos)
        
        g.message("Contracting propagators for 2pt function")
        
        tag = "%s/%s" % ("polaris_test_sloppy" + str(count+1), str(pos))

        prop_f_tag = "%s/%s" % (tag, Measurement.pos_boost)
        prop_b_tag = "%s/%s" % (tag, Measurement.neg_boost)

        prop_f = props_sloppy[prop_f_tag]
        prop_b = props_sloppy[prop_b_tag]

        Measurement.contract_2pt_test(prop_f, prop_b, phases, trafo, tag)

        del prop_f, prop_b
    del props_sloppy

