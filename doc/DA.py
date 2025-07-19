import gpt as g
import os
from gpt_qpdf_utils import pion_DA_measurement

parameters = {
    "zmax" : 24,
    "pzmin" : 0,
    "pzmax" : 5,
    "width" : 2.2,
    "pos_boost" : [0,0,0],
    "neg_boost" : [0,0,0],
    "save_propagators" : True
}

# loading gauge configuration
U = g.load(groups[group]["conf_fmt"] % conf)
g.message("finished loading gauge config")


# do gauge fixing

U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime

Measurement = pion_DA_measurement(parameters)

prop_exact, prop_sloppy, pin = Measurement.make_96I_inverter(U, groups[group]["evec_fmt"])

phases = Measurement.make_mom_phases(U[0].grid)

# set source

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

# set output path and open files
root_job = f"{root_output}/{conf}/{job}"
Measurement.set_output_facilites(f"{root_job}/correlators",f"{root_job}/propagators")

# make the Wilson lines
g.message("Starting Wilson lines")
W = Measurement.create_WL(U)

# exact positions
for pos in source_positions_exact:

    g.message("STARTING EXACT MEASUREMENTS")

    g.message("Starting DA 2pt function")

    g.message("Generatring boosted src's")
    srcDp, srcDm = Measurement.create_src_2pt(pos, trafo, U[0].grid)

    g.message("Starting prop exact")
    prop_exact_f = g.eval(prop_exact * srcDp)
    g.message("forward prop done")
    prop_exact_b = g.eval(prop_exact * srcDm)
    g.message("backward prop done")

    tag = "%s/%s" % ("exact", str(pos)) 

    #construct the list of 'backwards propagators' for DA: (W * prop b)^dagger
    prop_b = Measurement.constr_backw_prop_for_DA(prop_exact_b,W)
    g.message("Start DA contractions")
    Measurement.contract_DA(prop_exact_f, prop_b, phases, tag)
    del prop_b
    g.message("DA done")

    g.message("Starting 2pt contraction (includes sink smearing)")
    Measurement.contract_2pt(prop_exact_f, prop_exact_b, phases, trafo, tag)
    g.message("2pt contraction done")

    if(parameters["save_propagators"]):
        Measurement.propagator_output(tag, prop_exact_f, prop_exact_b)

    del prop_exact_f
    del prop_exact_b

    g.message("STARTING SLOPPY MEASUREMENTS")

    g.message("Starting DA 2pt function")

    g.message("Starting prop sloppy")
    prop_sloppy_f = g.eval(prop_sloppy * srcDp)
    g.message("forward prop done")
    prop_sloppy_b = g.eval(prop_sloppy * srcDm)
    g.message("backward prop done")

    del srcDp
    del srcDm

    tag = "%s/%s" % ("sloppy", str(pos))

    prop_b = Measurement.constr_backw_prop_for_DA(prop_sloppy_b,W)

    g.message("Start DA contractions")
    Measurement.contract_DA(prop_sloppy_f, prop_b, phases, tag)
    del prop_b
    g.message("DA done")

    g.message("Starting 2pt contraction (includes sink smearing)")
    Measurement.contract_2pt(prop_sloppy_f, prop_sloppy_b, phases, trafo, tag)
    g.message("2pt contraction done")

    if(parameters["save_propagators"]):
        Measurement.propagator_output(tag, prop_sloppy_f, prop_sloppy_b)

    del prop_sloppy_f
    del prop_sloppy_b
    
g.message("exact positions done")

# sloppy positions
for pos in source_positions_sloppy:

    g.message("STARTING SLOPPY MEASUREMENTS")
    
    # do stuff for sloppy sources

g.message("sloppy positions done")
        
del pin

