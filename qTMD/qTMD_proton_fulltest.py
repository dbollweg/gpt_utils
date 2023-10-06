import gpt as g

from proton_qTMD_draft import proton_TMD

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

#skip job and run nonsense (want to re-write the mess)


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

#Create list of staple shaped Wilson-lines
W, W_index_list = Measurement.create_TMD_WL(U)

#2. Create source apply boosted smearing
src_position = [0,0,0,0]

src = Measurement.create_src_2pt(src_position, trafo, U[0].grid)

prop_exact_f = g.eval(prop_exact * src)

sequential_bw_prop_down = Measurement.create_bw_seq(prop_exact, prop_exact_f, trafo, 2)
sequential_bw_prop_up = Measurement.create_bw_seq(prop_exact, prop_exact_f, trafo, 1)

tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_exact_f, W, W_index_list)

phases = Measurement.make_mom_phases(U[0].grid, src_position)

tag = "%s/%s" % ("tmd_test", str(src_position))

proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down, phases, tag)
proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up, phases, tag)

g.message(proton_TMDs_down[0][0][0][0])

#3. 