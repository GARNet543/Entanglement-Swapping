from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

# 1. Setup: 4 Qubits and 2 Classical bits for the middle measurement
qr = QuantumRegister(4, 'q')
cr = ClassicalRegister(2, 'c_bsm')
qc = QuantumCircuit(qr, cr)

# --- STEP 1: Create two Bell Pairs ---
# Pair A: q0 and q1
qc.h(qr[0])
qc.cx(qr[0], qr[1])

# Pair B: q2 and q3
qc.h(qr[2])
qc.cx(qr[2], qr[3])

qc.barrier()

# --- STEP 2: Bell State Measurement (BSM) on q1 and q2 ---
# We transform the q1-q2 basis back to the computational basis
qc.cx(qr[1], qr[2])
qc.h(qr[1])

# Measure the middle qubits
qc.measure(qr[1], cr[0])
qc.measure(qr[2], cr[1])

qc.barrier()

# --- STEP 3: Conditional Corrections ---
# Depending on the measurement of q1 and q2, q3 needs 
# specific rotations to ensure it is in a perfect Bell state with q0.
with qc.if_test((cr[0], 1)):
    qc.z(qr[3])
with qc.if_test((cr[1], 1)):
    qc.x(qr[3])

# --- Verification ---
# Let's see the state of the system
backend = Aer.get_backend('statevector_simulator')
job = backend.run(transpile(qc, backend))
final_state = job.result().get_statevector()

print("Circuit complete. The outer qubits (q0 and q3) are now entangled.")
print(qc.draw(output='text'))