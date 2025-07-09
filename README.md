# SubsystemEntropyCal
A respository of codes used in ArXiv:2501.06407

## Usage

- Create a instance of quantum code with parity-check matrix.

  code=code(H)

- Calculate the entanglement entropy of subsystem A, subsystem A contains qubits 1, 2, 3, 4.

  A=[1,2,3,4]
  entropy=code.entropy_cal_neo(A)

