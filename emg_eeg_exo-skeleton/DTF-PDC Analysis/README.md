# due to licensing I am not able to share the other files.


In order to run EEG_Analysis_TimeVarying.m, Simulation_TimeVarying.m, and Simulation_TimeInvariant.m please download dependencies and external toolbox.
<!-- blank line -->
----
<!-- blank line -->
-  Download the all functions [ here](https://se.mathworks.com/matlabcentral/fileexchange/45223-orthogonalized-partial-directed-coherence-measuring-time-varying-interactions-within-eeg-channels) and replace the EEG_Analysis_TimeVarying.m, Simulation_TimeVarying.m, and Simulation_TimeInvariant.m files in the destination

<!-- blank line -->
----
<!-- blank line -->

- ARFIT Toolbox, available at: http://www.clidyn.ethz.ch/arfit/index.html

Also remember to download arord, arqr, and covm functions through [ this link ](https://se.mathworks.com/matlabcentral/fileexchange/33721-time-varying-eeg-connectivity-a-time-frequency-approach) 

<!-- blank line -->
----
<!-- blank line -->
- {+ Make sure that you keep all the files in the same directory +}

<!-- blank line -->
----
<!-- blank line -->



- In order to perform the EEG analysis part, please run EEG_Analysis_TimeVarying.m
- In order to perform the time-varying simulation part, please run Simulation_TimeVarying.m
- In order to perform the time-invariant simulation part, please run Simulation_TimeInvariant.m


unizp the movement intention points and keep them all in same directory. An example of the matlab directory is as seen in below image

<!-- blank line -->
----
<!-- blank line -->



![Capture](/uploads/ca54052aa0cf4682fb0772e22d286618/Capture.PNG)

<!-- blank line -->
----
<!-- blank line -->



***Simulation_TimeVarying*** script is limited for 60 epochs. if you wish to run the script for all epochs, comment  the N_epochs =60; **line 69**


