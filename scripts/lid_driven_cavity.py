from pyls import Domain, Solver

# create a square domain
corners = [1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j]
dom = Domain(corners)
sol = Solver(dom, 10)
sol.solve()
# sol.construct_linear_system()
