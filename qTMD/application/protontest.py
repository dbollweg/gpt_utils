#!/usr/bin/env python3
import gpt as g
import os
import sys
import math
import numpy as np

from qTMD.gpt_proton_qTMD_utils import proton_qpdf_measurement
parameters = {
    "pzmin" : 0,
    "pzmax" : 4,
    "q" : [0,1,0,0],
    "zmax" : 2,
    "t_insert" : 1,
    "width" : 1.0,
    "boost_in" : [0,0,0],
    "boost_out" : [0,0,0],
    "save_propagators": False
}

Measurement = proton_qpdf_measurement(parameters)

grid = g.grid([4,4,4,4], g.double)

U = g.qcd.gauge.unit(grid)

U_prime, trafo = g.gauge_fix(U, maxiter=1)

del U_prime

prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)

src_position = [0,0,0,0]
src = Measurement.create_src_2pt(src_position,trafo,grid)

prop_exact_f = g.eval(prop_exact * src)

phases = Measurement.make_mom_phases(U[0].grid, src_position)


correlator = Measurement.contract_proton_2pt(prop_exact_f, phases, trafo)

print(correlator)