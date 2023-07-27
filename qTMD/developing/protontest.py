#!/usr/bin/env python3

import gpt as g
import os
import sys
import math
import numpy as np

from qTMD.gpt_proton_qTMD_utils import proton_qpdf_measurement

parameters = {
    "pzmin" : 0,
    "pzmax" : 0,
    "q" : [0,0,0,0],
    "zmax" : 0,
    "t_insert" : 32,
    "width" :41.0,
    "boost_in": [0,0,0],
    "boost_out": [0,0,0],
    "save_propagators": False
}
def uud_two_point(Q1, Q2, kernel):
    dq = g.qcd.baryon.diquark(g(Q1 * kernel), g(kernel * Q2))
    return g(g.color_trace(g.spin_trace(dq) * Q1 + dq * Q1))

def proton(Q1, Q2):
    C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[5].tensor()
    Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
    return g(g.trace(uud_two_point(Q1,Q2,Gamma) * Pp))
    
Measurement = proton_qpdf_measurement(parameters)

grid = g.grid([4,4,4,4], g.double)

U = g.qcd.gauge.unit(grid)

U_prime, trafo = g.gauge_fix(U, maxiter=1)

del U_prime

prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)

src_position = [0,0,0,0]

src = Measurement.create_src_2pt(src_position,trafo, grid)

prop_exact_f = g.eval(prop_exact * src)

phases = Measurement.make_mom_phases(U[0].grid, src_position)

correlator = Measurement.contract_proton_2pt(prop_exact_f, phases, trafo)

print(correlator)