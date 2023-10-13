import gpt as g 

from proton_qTMD_draft import proton_TMD


def uud_two_point(Q1, Q2, kernel):
    dq = g.qcd.baryon.diquark(g(Q1 * kernel), g(kernel * Q2))
    return g(g.color_trace(g.spin_trace(dq) * Q1 + dq * Q1))

def proton_contr(Q1, Q2):
    C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[5].tensor()
    Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
    Szp = (g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
    PpSzp = Pp * Szp
    return g(g.trace(uud_two_point(Q1, Q2, Gamma) * PpSzp))

def proton_contr_alt(Q1, Q2):
    C  = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
    Gamma = C * g.gamma[5].tensor()
    Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
    Szp = (g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
    PpSzp = Pp * Szp

    dq = g.qcd.baryon.diquark(g(Q1 * Gamma), g(Gamma * Q2))

    return g(g.color_trace(g.spin_trace(dq) * g.spin_trace(PpSzp * Q1) + g.spin_trace(Q1 * PpSzp * dq)))


parameters = {
    "eta": [2],
    "b_z": 1,
    "b_T": 2,
    "pzmin": 0,
    "pzmax": 3,
    "width": 2,
    "boost_in": [0,0,0],
    "boost_out": [0,0,0],
    "pf": [1,1,0,0],
    "save_propagators": False,
    "t_insert": 2,
}

#1. load config, load eigenvectors
#U = g.load("/lustre/orion/proj-shared/nph159/data/64I/1410.evecs/lanczos.output")
grid = g.grid([8,8,8,8], g.double)
U = g.qcd.gauge.unit(grid)
L = U[0].grid.fdimensions

Measurement = proton_TMD(parameters)



U_prime, trafo = g.gauge_fix(U, maxiter=10000)

del U_prime

#prop_exact, prop_sloppy, pin = Measurement.make_64I_inverter(U, "/lustre/orion/proj-shared/nph159/data/64I/ckpoint_lat.Coulomb.1410")
prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)


src_position = [0,0,0,0]

src = Measurement.create_src_2pt(src_position, trafo, U[0].grid)

prop_exact_f = g.eval(prop_exact * src)
sequential_bw_prop_down = [g.mspincolor(prop_exact_f.grid) for i in range(3)]
g.qcd.baryon.proton_seq_src_full(prop_exact_f, sequential_bw_prop_down, 2)
        
phases = Measurement.make_mom_phases(U[0].grid, src_position)

protonctr = proton_contr_alt(prop_exact_f, prop_exact_f)
protonctr2 = proton_contr(prop_exact_f, prop_exact_f)

proton_test = g.eval(g.trace(prop_exact_f*sequential_bw_prop_down[0]))

#g.message(g(protonctr-proton_test))
correlator = [g.slice(g.eval(protonctr*pp),3) for pp in phases]

correlator_check = [g.slice(g.eval(g.trace(prop_exact_f*sequential_bw_prop_down[2])*pp),3) for pp in phases]

g.message(correlator[0])

g.message("Crosscheck") 
g.message(correlator_check[0])
