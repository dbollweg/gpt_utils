import gpt as g

from gpt_qpdf_utils import pion_measurement


# momenta setup
parameters = {
    "zmax"  : 0,
    "plist" : [[0,0, 2, 0]],
    "width" : 2.0,
    "pos_boost" : [0,0,2],
    "neg_boost" : [0,0,-2],
    "save_propagators" : True
}

jobs = {
    "test_exact_0": {
        "exact": 1,
        "sloppy": 2,
        "low": 0,
    },  
}

##### small dummy used for testing
grid = g.grid([8,8,8,8], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.random(grid, rng)

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
       2pt test run:
================================================================================
"""
)

src_origin = [0,0,0,0] 

source_positions_exact = [src_origin]
source_positions_sloppy = [
    [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
    for j in range(jobs["test_exact_0"]["sloppy"])
]

Measurement.set_output_facilities("./correlators_exact","./propagators_exact")		    
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
    
    if(parameters["save_propagators"]):
        Measurement.propagator_output(tag, prop_exact_f, prop_exact_b)
    
    Measurement.contract_2pt_test(prop_exact_f, prop_exact_b, phases, trafo, tag)
    g.message("2pt contraction done")

    del prop_exact_f
    del prop_exact_b


# sloppy positions
del prop_exact
Measurement.set_output_facilities("./correlators_sloppy","./propagators_sloppy")
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

