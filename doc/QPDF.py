import gpt as g
import os
from gpt_qpdf_utils import pion_qpdf_measurement

parameters = {
    "pf" : [1,1,0,0],
    "q" : [0,1,0,0],
    "zmax" : 4,
    "t_insert" : 4,
    "width" : 2.2,
    "boost_in" : [0,0,0],
    "boost_out" : [0,0,0],
    "save_propagators" : True
}

# loading gauge configuration
U = g.load(groups[group]["conf_fmt"] % conf)
g.message("finished loading gauge config")


# do gauge fixing

U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime

Measurement = pion_qpdf_measurement(parameters)

prop_exact, prop_sloppy, pin = Measurement.make_96I_inverter(U, groups[group]["evec_fmt"])

phases = Measurement.make_mom_phases(U[0].grid)

L = U[0].grid.fdimensions
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

    g.message("Create list of W * forward prop")
    prop_f = Measurement.create_fw_prop_QPDF(prop_exact_f, W)
    g.message("Start QPDF contractions")
    Measurement.contract_QPDF(prop_f, prop_b, phases, tag)
    g.message("PQDF done")

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

    g.message("Create list of W * forward prop")
    prop_f = Measurement.create_fw_prop_QPDF(prop_sloppy_f, W)
    g.message("Start QPDF contractions")
    Measurement.contract_QPDF(prop_f, prop_b, phases, tag)
    g.message("PQDF done")

    del prop_b
    del prop_sloppy_f
    
g.message("exact positions done")

# sloppy positions
for pos in source_positions_sloppy:

    g.message("STARTING SLOPPY MEASUREMENTS")
    
    # do the same as above

g.message("sloppy positions done")
    
del pin

