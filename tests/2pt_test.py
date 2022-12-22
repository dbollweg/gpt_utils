import gpt as g

from gpt_qpdf_utils import pion_measurement


groups = {
    "test_batch_0": {
        "confs": [
            "cfg",
        ],
        "evec_fmt": "./lanczos.output",
        "conf_fmt": "./ckpoint_lat.Coulomb.%s",
    },

}
# momenta setup
parameters = {
    "zmax"  : 0,
    "plist" : [[0,0, 2, 0]],
    "width" : 2.0,
    "pos_boost" : [0,0,2],
    "neg_boost" : [0,0,-2],
    "save_propagators" : False
}

jobs = {
    "test_exact_0": {
        "exact": 1,
        "sloppy": 1,
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
grid = g.grid([8,8,8,8], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.random(grid, rng)

# loading gauge configuration
#U = g.load(groups[group]["conf_fmt"] % conf)
g.message("finished creating gauge config")

# do gauge fixing
U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime
L = U[0].grid.fdimensions

Measurement = pion_measurement(parameters)
prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)

# show available memory
g.mem_report(details=False)
g.message(
"""
================================================================================
       2pt test run ;  this run will attempt:
================================================================================
"""
)
# per job
for group, job, conf, jid, n in run_jobs:

    g.message(f"""Job {jid} / {n} :  configuration {conf}, job tag {job}""")

    # the original point for source creation which shift by conf number
    src_origin = [0,0,0,0] 
    
    source_positions_exact = [src_origin]
    source_positions_sloppy = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["sloppy"])
    ]
    
    Measurement.set_output_facilities("./correlators_exact","./propagators")		    
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
        tag = "%s/%s" % ("test_exact", str(pos))
        g.message(tag)
        Measurement.contract_2pt(prop_exact_f, prop_exact_b, phases, trafo, tag)
        g.message("2pt contraction done")

        del prop_exact_f
        del prop_exact_b

    
    # sloppy positions
    del prop_exact
    Measurement.set_output_facilities("./correlators_sloppy",".propagators")
    g.message(f" positions_sloppy = {source_positions_sloppy}")
    for pos in source_positions_sloppy:
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
        tag = "%s/%s" % ("test_sloppy", str(pos))
        g.message(tag)
        Measurement.contract_2pt(prop_sloppy_f, prop_sloppy_b, phases, trafo, tag)
        g.message("pion contraction done")


        del prop_sloppy_f
        del prop_sloppy_b      

