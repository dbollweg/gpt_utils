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

def tensortest(Q):
    C = g.separate_color(Q)
    g.message("Q after separate_color: ", C)
    CS = g.separate_spin(C[(0,0)])
    
    g.message("Q after sparate_spin,separate_color: ", CS)
    
    S = g.separate_spin(Q)
    g.message("Q after sparate_spin: ", S)
    
    
def down_quark_insertion(Q, Gamma, P):
    #eps_abc eps_a'b'c'Gamma_{beta alpha}Gamma_{beta'alpha'}P_{gamma gamma'}
    # * ( Q^beta'beta_b'b Q^gamma'gamma_{c'c} -  Q^beta'gamma_b'c Q^gamma'beta_{c'b} )
    
    eps = g.epsilon(Q.otype.shape[2])
    
    R = g.lattice(Q)
    
    PDu = g(g.spin_trace(P*Q))

    GtDG = g.eval(g.transpose(Gamma)*Q*Gamma)

    GtDG = g.separate_color(GtDG)
    PDu = g.separate_color(PDu)
    
    GtD = g.eval(g.transpose(Gamma)*Q)
    PDG = g.eval(P*Q*Gamma)
    
    GtD = g.separate_color(GtD)
    PDG = g.separate_color(PDG)
    
    D = {x: g.lattice(GtDG[x]) for x in GtDG}

    for d in D:
        D[d][:] = 0
        
    for i1, sign1 in eps:
        for i2, sign2 in eps:
            D[i1[0], i2[0]] += sign1 * sign2 * g.transpose((PDu[i2[2], i1[2]] * GtDG[i2[1], i1[1]] - GtD[i2[1],i1[2]] * PDG[i2[2], i1[1]]))
            
    g.merge_color(R, D)
    return R

def up_quark_insertion(Qu, Qd, Gamma, P):

    eps = g.epsilon(Qu.otype.shape[2])
    R = g.lattice(Qu)
    Dut = g.lattice(Qu)

    #first term    
    GDdGt = g.eval(Gamma * Qd * g.transpose(Gamma))
    GDdGt = g.separate_color(GDdGt)
    
    D = {x: g.lattice(GDdGt[x]) for x in GDdGt}
    for d in D:
        D[d][:] = 0
        
    Du_sep = g.separate_color(Qu)
    Du_spintransposed = {x: g.lattice(Du_sep[x]) for x in Du_sep}
    for d in Du_spintransposed:
        Du_spintransposed[d] = g(g.transpose(Du_sep[d]))
    g.merge_color(Dut,Du_spintransposed)
    
    DuP = g(Dut*P)
    DuP = g.separate_color(DuP)
    
    #second term
    PDuG = g(P * Dut * Gamma)
    PDuG = g.separate_color(PDuG)
    DdGt = g(Qd * g.transpose(Gamma))
    DdGt = g.separate_color(DdGt)    
    
    
    #third term
    GDd = g.eval(Gamma * Qd)
    GDd = g.separate_color(GDd)

    GtDut = g.eval(g.transpose(Gamma) * Dut)
    GtDut = g.separate_color(GtDut)
        
    #fourth term
    PDu_trace = g(g.spin_trace(P * Dut))
    PDu_trace = g.separate_color(PDu_trace)

    #sum color indices
    for i1, sign1 in eps:
        for i2, sign2 in eps:
            D[i2[2], i1[2]] += sign1 * sign2 * (GDdGt[i1[1],i2[1]] * DuP[i1[0],i2[0]]
                                                + PDuG[i1[0],i2[0]] * DdGt[i1[1],i2[1]] 
                                                + P * g.spin_trace(GDd[i1[1],i2[1]] * GtDut[i1[0],i2[0]]) 
                                                +  GDdGt[i1[1],i2[1]] * PDu_trace[i1[0],i2[0]])
            
    g.merge_color(R, D)
    return R
    
    

parameters = {
    "eta": [2],
    "b_z": 1,
    "b_T": 2,
    "pzmin": 0,
    "pzmax": 1,
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

correlator = [g.slice(g.eval(protonctr*pp),3) for pp in phases]

correlator_check = [g.slice(g.eval(g.trace(prop_exact_f*sequential_bw_prop_down[0])*pp),3) for pp in phases]

g.message("Crosscheck: Correlator - Correlator_cgpt") 

for j in range(len(correlator)):
    for i in range(len(correlator[0])):
        g.message(abs(correlator[j][i] - correlator_check[j][i]))

Cg5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma[5].tensor()

Pp = (g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25
Szp = (g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())
PpSzp = Pp * Szp

bw_prop_down_test = down_quark_insertion(prop_exact_f, Cg5, PpSzp)

correlator_check_second = [g.slice(g.eval(g.trace(prop_exact_f*bw_prop_down_test)*pp),3) for pp in phases]

g.message("Crosscheck: Correlator - Correlator_gpt") 

for j in range(len(correlator)):
    for i in range(len(correlator[0])):
        g.message(abs(correlator[j][i] - correlator_check_second[j][i]))
        
        
bw_prop_up_test = up_quark_insertion(prop_exact_f, prop_exact_f, Cg5, PpSzp)


w = g.qcd.wick()

x, y, z, zl = w.coordinate(4)
prop_zero = g.mspincolor(grid)
prop_zero *= 0.0
prop_unity = g.mspincolor(grid)
prop_unity *= 0.0
prop_unity += g.identity(prop_unity)

ud_propagators = {
    (x, y): prop_exact_f[0,0,0,0],
    (y, x): g(g.gamma[5] * g.adj(prop_exact_f[0,0,0,0]) * g.gamma[5]),
    (x, z): prop_unity[0,0,0,0],
    (z, x): prop_unity[0,0,0,0],
    (y, z): prop_unity[0,0,0,0],
    (z, y): prop_unity[0,0,0,0],
    (z, z): prop_zero[0,0,0,0],
}


u = w.fermion(ud_propagators)
d = w.fermion(ud_propagators)


na,nb = w.color_index(2)
nalpha, nbeta = w.spin_index(2)


C = 1j * g.gamma[1].tensor() * g.gamma[3].tensor()
Cg5 = w.spin_matrix(C * g.gamma[5].tensor())
PpSzp = w.spin_matrix((g.gamma["I"].tensor() + g.gamma[3].tensor()) * 0.25 * g.gamma["I"].tensor() - 1j*g.gamma[0].tensor()*g.gamma[1].tensor())


def nucleon_operator(w, u, d, x, alpha, matrix):
    a, b, c = w.color_index(3)
    beta, gamma = w.spin_index(2)
    return w.sum(
        u(x, alpha, a),
        w.sum(
            u(x, beta, b),
            w.epsilon(a,b,c),
            w.sum(matrix(beta, gamma), d(x, gamma, c), gamma),
            beta,
            b,
            c,
        ),
        a,
    )
    
def dummy_z_insertion(w, u1, u2, z):
    a = w.color_index()
    alpha = w.spin_index()
    return w.sum(u1(z,alpha,a) * u2(z,alpha,a),alpha,a)
    

O = nucleon_operator(w, u, d, x, nalpha, Cg5)

Obar = nucleon_operator(w, u.bar(), d.bar(), y, nbeta, Cg5)

proton_2pt = w.sum(Obar, PpSzp(nbeta, nalpha), O, nalpha, nbeta)


W_proton_2pt = w(proton_2pt, verbose=True)


C_proton_2pt = proton_contr(prop_exact_f, prop_exact_f)


eps1 = abs(C_proton_2pt[0,0,0,0] - W_proton_2pt) / abs(C_proton_2pt[0,0,0,0])

g.message(f"Proton 2pt test diquark vs wick: {eps1}")
g.message(W_proton_2pt)
g.message(C_proton_2pt[0,0,0,0])
TMD_proton_2pt = g.eval(g.trace(prop_exact_f*bw_prop_down_test))

TMD_proton_2pt_cgpt = g.eval(g.trace(prop_exact_f*sequential_bw_prop_down[0]))

eps2 = abs(TMD_proton_2pt[0,0,0,0] - W_proton_2pt) / abs(TMD_proton_2pt[0,0,0,0])
g.message(f"Proton 2pt test TMD bprop (down) gpt vs wick: {eps2}")

eps3= abs(TMD_proton_2pt_cgpt[0,0,0,0] - W_proton_2pt) / abs(TMD_proton_2pt_cgpt[0,0,0,0])
g.message(f"Proton 2pt test TMD bprop (down) cgpt vs wick: {eps3}")

TMD_down_insertion = w.sum(Obar,PpSzp(nbeta, nalpha), dummy_z_insertion(w,d.bar(),d,z),O, nalpha, nbeta)
TMD_up_insertion = w.sum(Obar,PpSzp(nbeta, nalpha), dummy_z_insertion(w,u.bar(),u,z),O, nalpha, nbeta)

test_down_wick = w(TMD_down_insertion, verbose = True, separate_diagrams=False)

test_down_gpt = g(g.trace(bw_prop_down_test))

eps4 = abs(test_down_wick - test_down_gpt[0,0,0,0]) / abs(test_down_gpt[0,0,0,0])
g.message(f"Test TMD bprop (down) gpt vs wick: {eps4}")
g.message(test_down_wick)
g.message(test_down_gpt[0,0,0,0])


test_up_wick = w(TMD_up_insertion, verbose = True, separate_diagrams=False)
test_up_gpt = g(g.trace(bw_prop_up_test))

eps5 = abs(test_up_wick - test_up_gpt[0,0,0,0]) / abs(test_up_gpt[0,0,0,0])
g.message(f"Test TMD bprop (up) gpt vs wick: {eps5}")
g.message(test_up_wick)
g.message(test_up_gpt[0,0,0,0])


if eps1 < 1e-7 and eps2 < 1e-7 and eps3 < 1e-7 and eps4 < 1e-7 and eps5 < 1e-7:
    g.message("ALL TESTS PASSED!")
else:
    g.message("TESTS FAILED!")