import matplotlib.pyplot as plt

data = []
size = []

with open("errors.txt", "r") as file_errors:
  while string := file_errors.readline():
    current_size, current_data = map(float, string.split(" "))
    data.append(current_data)
    size.append(current_size)

plt.plot(size, data, "ob")
plt.suptitle('dimX = 128, values of the error varying nSteps')
plt.ylabel('Error')
plt.xlabel('Number of steps')

figure = plt.gcf() # get current figure
figure.set_size_inches(20, 10)

plt.savefig(f'errors.pdf', bbox_inches='tight', dpi=600)

plt.show()