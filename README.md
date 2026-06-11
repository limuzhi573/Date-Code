**1. Code Availability**

All codes used in this study are available from the corresponding author.

**2. Operating Environment**

Operating System: 64-bit Windows 10

Programming Language: Python 3.7.1, MATLAB R2024a

CPU: Intel i5-13420H

**3. Dependent Libraries and Tools**

Python:

numpy==1.21.5

pandas==1.3.5

matplotlib==3.5.3

scipy==1.7.3

torch==1.13.1+cpu

gymnasium==0.28.1

MATLAB:

No external dependencies

**4. File Structure**

data：

Clinical treatment data of 8 patients with prostate cancer

Patient Fitted Parameter Results Table

code：

(1).MATLAB

Parameter fitting for cancer patients — Output Patient Fitted Parameter Results Table

(2).Python

Figure 1-2. Reproduction of the fitting results — Output figure1, figure2

Figure 3. Global Optimization Based on Deep Reinforcement Learning — Output figure3

Figure 4. Local Optimization of Deep Reinforcement Learning Algorithm — Output figure4

Figure 5. Patient Group Optimization of Deep Reinforcement Learning Algorithm — Output figure5

**5. Operation Instructions**

(1).Install all required dependent libraries.

(2).Save the clinical treatment data for eight prostate cancer patients into the desktop new folder path prostate cancer clinical treatment data\\Bruchovsky\_et\_al,then modify the prefix of the file reading path in the code to match the local directory path of your current computer.

(3).Minor deviations may exist in repeated running results while the overall trend remains consistent. Increasing iteration times can optimize the performance.

(4).All output files will be automatically saved to the desktop.

**6. Input and Output**

(1).MATLAB

Input: Raw data in CSV format

Output: Parameter table

(2).Python

Input: Raw CSV data, parameter table

Output: Simulation diagrams of cancer treatment strategies

