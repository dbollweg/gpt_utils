import gpt as g

from gpt_qpdf_utils import pion_measurement
from utils.tools import *
from utils.io_corr import *
import numpy as np

# momenta setup
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
##### small dummy used for testing
#grid = g.grid([8,8,8,8], g.double)
#rng = g.random("seed text")
#U = g.qcd.gauge.random(grid, rng)

# loading gauge configuration
U = g.load(groups[group]["conf_fmt"] % conf)
g.message("finished loading gauge config")
##### small dummy used for testing
#grid = g.grid([8,8,8,8], g.double)
rng = g.random("seed text")
#U = g.qcd.gauge.random(grid, rng)

g.message("finished creating gauge config")

# do gauge fixing
U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime
L = U[0].grid.fdimensions

Measurement = pion_measurement(parameters)
prop_exact, prop_sloppy, pin = Measurement.make_64I_inverter(U, groups[group]["evec_fmt"] % conf)

# show available memory
g.mem_report(details=False)
g.message(
"""
================================================================================
       2pt test run:
================================================================================
"""
)

for group, job, conf, jid, n in run_jobs:
    
    g.message(f"""Job {jid} / {n} :  configuration {conf}, job tag {job}""")
    
    src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
    source_positions = srcLoc_distri_eq(L, src_origin)
    #print(source_positions)
    source_positions_sloppy = source_positions[:jobs[job]["sloppy"]]
    source_positions_exact = source_positions[:jobs[job]["exact"]]

    g.message(f" positions_sloppy = {source_positions_sloppy}")
    g.message(f" positions_exact = {source_positions_exact}")

    #root_job = f"{root_output}/{conf}/{job}"
    #Measurement.set_output_facilites(f"{root_job}/correlators",f"{root_job}/propagators")

    # sample_log_file = data_dir + "/sample_log/" + conf
    # #if g.rank() == 0:
    # f = open(sample_log_file, "w")
    # f.close()


    # source_positions_exact = [src_origin]
    # source_positions_sloppy = [
    #     [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
    #     for j in range(jobs["test_exact_0"]["sloppy"])
    # ]

    Measurement.set_output_facilities("/lus/grand/projects/StructNGB/bollwegd/testrun/polaris_correlators_exact","/lus/grand/projects/StructNGB/bollwegd/testrun/polaris_propagators_exact")		    
    # exact positions
    g.message(f" positions_exact = {source_positions_exact}")
    for pos in source_positions_exact:
        phases = Measurement.make_mom_phases(U[0].grid, pos)
        
        g.message("Starting 2pt function")
        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")


        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")


        prop_exact_b = g.eval(prop_exact * srcDm)
        g.message("backward prop done")



        g.message("Starting 2pt contraction (includes sink smearing)")
        tag = "%s/%s" % ("polaris_test_exact", str(pos))
        g.message(tag)
        
        if(parameters["save_propagators"]):
            Measurement.propagator_output(tag, prop_exact_f, prop_exact_b)
        
        Measurement.contract_2pt_test(prop_exact_f, prop_exact_b, phases, trafo, tag)
        g.message("2pt contraction done")

        del prop_exact_f
        del prop_exact_b


    # sloppy positions
    del prop_exact
    Measurement.set_output_facilities("/lus/grand/projects/StructNGB/bollwegd/testrun/polaris_correlators_sloppy","/lus/grand/projects/StructNGB/bollwegd/testrun/polaris_propagators_sloppy")
    g.message(f" positions_sloppy = {source_positions_sloppy}")
    for count,pos in enumerate(source_positions_sloppy):
        phases = Measurement.make_mom_phases(U[0].grid, pos)

        g.message("Starting 2pt function")
        g.message("Generatring boosted src's")
        srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)  

        g.message("Starting prop sloppy")
        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")
        prop_sloppy_b = g.eval(prop_sloppy * srcDm)
        g.message("backward prop done")
        g.message("Starting pion contraction (includes sink smearing)")
        tag = "%s/%s" % ("test_sloppy" + str(count+1), str(pos))
        g.message(tag)


        if(parameters["save_propagators"]):
            Measurement.propagator_output(tag, prop_sloppy_f, prop_sloppy_b)

        Measurement.contract_2pt_test(prop_sloppy_f, prop_sloppy_b, phases, trafo, tag)
        g.message("pion contraction done")


        del prop_sloppy_f
        del prop_sloppy_b      

