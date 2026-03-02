from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

# Import matplotlib for displaying the plot
import matplotlib.pyplot as plt


# -------------------------------------------------
# 1. Create a quantum circuit with:
#    - 3 qubits (quantum register)
#    - 3 classical bits (for measurement results)
# -------------------------------------------------
qc = QuantumCircuit(3, 3)


# -------------------------------------------------
# 2. Create an entangled Bell pair between qubit 0 and qubit 1
#    Step 1: Apply Hadamard gate to qubit 0
#            This creates superposition |0> → (|0> + |1>) / √2
# -------------------------------------------------
qc.h(0)

# Step 2: Apply CNOT with qubit 0 as control and qubit 1 as target
#         This entangles qubit 0 and qubit 1 into a Bell state
qc.cx(0, 1)


# -------------------------------------------------
# 3. Perform Bell-state measurement preparation on qubits 1 and 2
#    This is the key step of entanglement swapping
# -------------------------------------------------

# Apply CNOT with qubit 1 as control and qubit 2 as target
# This correlates qubit 2 with the entangled system
qc.cx(1, 2)

# Apply Hadamard gate to qubit 1
# This completes the Bell-basis transformation
qc.h(1)


# -------------------------------------------------
# 4. Measure all qubits
#    Each qubit is measured into its corresponding classical bit
# -------------------------------------------------
qc.measure([0, 1, 2], [0, 1, 2])


# -------------------------------------------------
# 5. Create the Aer simulator backend
#    AerSimulator is the modern replacement for qiskit.Aer
# -------------------------------------------------
backend = AerSimulator()

# Transpile the circuit to match the backend's instruction set
qc = transpile(qc, backend)


# -------------------------------------------------
# 6. Run the circuit on the simulator
#    - shots=1024 means the circuit is executed 1024 times
# -------------------------------------------------
result = backend.run(qc, shots=1024).result()

# Retrieve measurement statistics
counts = result.get_counts()


# -------------------------------------------------
# 7. Plot the measurement results
#    The histogram shows correlated outcomes
#    indicating successful entanglement swapping
# -------------------------------------------------
plot_histogram(counts)
plt.title("Entanglement Swapping (3-Qubit Simulation)")
plt.show()