import gpt as g

from gpt_qpdf_utils import pion_measurement
import h5py
import sys

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
props_exact = {}
for p in Measurement.propagator_input("./propagators_exact"):
    props_exact.update(p)

g.message(f" positions_exact = {source_positions_exact}")
for pos in source_positions_exact:
    phases = Measurement.make_mom_phases(U[0].grid, pos)
    
    g.message("Contracting propagators for 2pt function")

    tag = "%s/%s" % ("test_exact", str(pos))
    g.message(tag)

    prop_f_tag = "%s/%s" % (tag, Measurement.pos_boost)
    prop_b_tag = "%s/%s" % (tag, Measurement.neg_boost)

    prop_f = props_exact[prop_f_tag]
    prop_b = props_exact[prop_b_tag]

    Measurement.contract_2pt_test(prop_f, prop_b, phases, trafo, tag)
    del prop_f, prop_b
del props_exact

props_sloppy = {}
for p in Measurement.propagator_input("./propagators_sloppy"):
    props_sloppy.update(p)

for count,pos in enumerate(source_positions_sloppy):
    phases = Measurement.make_mom_phases(U[0].grid, pos)

    tag = "%s/%s" % ("test_sloppy" + str(count+1), str(pos))

    prop_f_tag = "%s/%s" % (tag, Measurement.pos_boost)
    prop_b_tag = "%s/%s" % (tag, Measurement.neg_boost)

    prop_f = props_sloppy[prop_f_tag]
    prop_b = props_sloppy[prop_b_tag]

    Measurement.contract_2pt_test(prop_f, prop_b, phases, trafo, tag)

    del prop_f, prop_b
del props_sloppy

