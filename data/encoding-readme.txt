Here is a short example of how the current encoding works. Start with a four qubit device with the following two-qubit connectivity:

     (q0, q1), (q1, q0), (q1, q2), (q2, q1), (q2, q3), (q3, q2)

This is an example of a device with ''linear'' connectivity:

     0 <--> 1 <--> 2 <--> 3

Now assume that you run the following circuit:

q0 --- X --- * ----
             |
q1 --- Z --- + ----

q2 ----------------

q3 ----------------

This is a two layer, four qubit circuit in which only two of the qubits (q0 and q1) are active.
Because our device has four qubits and our encoding uses 10 color channels we will create a 3D
tensor of with the following sizes:

     (num_qubits, layers, 10) ---> (4, 2, 10)

The 10 color channels store gate and state information. We will ignore the state information for now
and instead focus on the first 7 gate channels. Each gate channel corresponds to a type of gate
that appears in our circuits:

[Z-channel, X-channel, X^(-1)-channel, CNOT(D), CNOT(U), CNOT(L), CNOT(R) ] 

Every (i,j,k) slot in the tensor is populated with a 0. We will modify these slots based on the
circuit itself.

First we change the (0, 0, 1) slot to a 1 as the first gate on qubit 0 is an X gate. We also change the 
(1, 0, 0) slot to equal \pi as the first gate applied to qubit 1 is a Z gate (there are four types of Z gates:
\pi/2, \pi, -\pi/2, 0). We then change the (0, 1, 6) slot to a 1, as the second layer uses qubit 0 as a control qubit
whose target qubit is to its right (0 ---> 1), while the (1, 1, 5) slot is set to -1 as the second layer uses qubit 1
as a target qubit whose control qubit is to its left (0 <--- 1). Remember, the target qubit is marked with 
a -1 and the control qubit with a 1. 

To complete the encoding, we pad our tensor with additional layers so that it has the same size as the
longest circuit in our dataset.   
