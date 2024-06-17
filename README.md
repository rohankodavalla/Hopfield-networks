# Hopfield-networks
The objective of this assignment is to design a Hopfield network to recognize the patterns as an associator


**What is the Project About**

This project involves designing a Hopfield network to recognize specific patterns. The primary goal is to test the performance of the network in recall mode using clean patterns and to evaluate its robustness by introducing 25% noise (randomly reversing 25% of the pixels). The project also includes the implementation of both synchronous and asynchronous learning procedures to compare their effectiveness in pattern recall.

-------------------------------------------------------------------------------------------------------------------------------------------------

**Tools and Technologies Used**

- Programming Languages: Python
- Frameworks and Libraries: NumPy, Matplotlib (for visualization)
- Machine Learning Models: Hopfield Network

-------------------------------------------------------------------------------------------------------------------------------------------------

**Solution Explanation**

Pattern Recognition with Hopfield Network:

- Designing the Network:
    Create a Hopfield network capable of recognizing a set of predefined patterns.

- Training the Network:
    Train the Hopfield network using clean patterns to store them as stable states.

- Recall Mode Testing:
    Test the network's performance by recalling the clean patterns.

- Introducing Noise:
    Corrupt the patterns by randomly reversing 25% of the pixels to evaluate the network's robustness.

- Learning Procedures:
    1. Implement both synchronous and asynchronous learning procedures.
    2. Compare the performance of synchronous and asynchronous learning in recalling both clean and noisy patterns.

-------------------------------------------------------------------------------------------------------------------------------------------------

**Task Details**

- Network Design and Training:
  1. Design a Hopfield network for pattern recognition.
  2. Train the network using a set of clean patterns.

- Performance Testing:
  1. Test the network's performance in recall mode with clean patterns.
  2. Introduce 25% noise to the patterns and test the network's performance in recalling these corrupted patterns.

- Learning Procedures Implementation:
  1. Implement synchronous learning, where all neurons are updated simultaneously.
  2. Implement asynchronous learning, where neurons are updated one at a time.
  3. Compare the performance of both learning methods in recalling the patterns correctly.

-------------------------------------------------------------------------------------------------------------------------------------------------

**Result**

The project successfully demonstrates the design and performance evaluation of a Hopfield network for pattern recognition.

- Key outcomes include:
  ~Successful recall of clean patterns by the Hopfield network.
  ~Evaluation of the network's robustness in recalling patterns with 25% noise.
  ~Comparison of synchronous and asynchronous learning procedures.
  ~Comments on the performance:
  ~Analysis of whether all patterns could be recalled correctly under both clean and noisy conditions.
  ~Observations on the differences in performance between synchronous and asynchronous learning procedures.



