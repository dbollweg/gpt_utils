import gpt as g

grid = g.grid([10,10,10,10], g.double)

#create gauge field
U = g.qcd.gauge.unit(grid)

#get unit matrix
Umat = U[0][0,0,0,0]
#set it zero everywhere...
U[0][:] = 0
U[1][:] = 0
U[2][:] = 0
U[3][:] = 0

bz = 2
eta = 6
bT = 4
#...except on this coords (TMD wilson line shaped)
for i in range(0,eta+bz):
    U[0][i,0,0,0] = Umat
for j in range(0,bT):
    U[1][8,j,0,0] = Umat
for k in range(eta-bz,eta+bz):
    U[0][k,4,0,0] = Umat



prev_link = g.qcd.gauge.unit(U[0].grid)[0]
for dx in range(0,eta+bz):
    link = prev_link * g.cshift(U[0], 0, dx)
    prev_link = link

shifted_U = g.cshift(U[1], 0, eta+bz)
for dy in range(0,bT):
    link = prev_link * g.cshift(shifted_U,1, dy)
    prev_link = link

shifted_U = g.cshift(g.cshift(U[0], 0, eta+bz-1), 1, bT)
for dx in range(0,eta-bz):
    link = prev_link * g.adj(g.cshift(shifted_U, 0, -dx))
    prev_link = link






print("Printing link at origin (should be unity)")
print(g.eval(link)[0,0,0,0].array)

print("Printing link next to origin (should be zero)")
print(g.eval(link)[1,0,0,0].array)