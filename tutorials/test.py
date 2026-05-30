import scarabaeus as scb
km = scb.Units.get_units('km')

a = scb.ArrayWUnits([1, 2, 3], km)
b = scb.ArrayWUnits([4, 5, 6], km)

print(a + b)