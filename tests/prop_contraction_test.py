import gpt as g

from gpt_qpdf_utils import pion_measurement


# momenta setup
parameters = {
    "zmax"  : 0,
    "plist" : [[0,0, 2, 0]],
    "width" : 2.0,
    "pos_boost" : [0,0,2],
    "neg_boost" : [0,0,-2],
    "save_propagators" : False,
}

jobs = {
    "test_exact_0": {
        "exact": 1,
        "sloppy": 1,
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
    
    g.message("Contracting propagators for 2pt function")

    tag = "%s/%s" % ("test_exact", str(pos))
    g.message(tag)

    prop_f, prop_b = Measurement.propagator_input("./propagators_exact", tag)

    Measurement.contract_2pt(prop_f, prop_b, phases, trafo, tag)
    del prop_f, prop_b

for pos in source_positions_sloppy:
    phases = Measurement.make_mom_phases(U[0].grid, pos)

    tag = "%s/%s" % ("test_sloppy", str(pos))

    prop_f, prop_b = Measurement.propagator_input("./propagators_sloppy", tag)

    Measurement.contract_2pt(prop_f, prop_b, phases, trafo, tag)

    del prop_f, prop_b


