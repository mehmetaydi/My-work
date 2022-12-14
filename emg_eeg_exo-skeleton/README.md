
During my work at Tampere University, I did most of the work on the analysis of brain signals. I have studied on the behavior of the brain signals received from the scalp of the patient for different modes of the exoskeleton device that supports the patient's walking. Besides that, I developed some algorithms to reduce artifacts that are coming from the heart and muscles. moreover, I classified brain signals according to the exoskeleton mode  with ML-based methods on brain signals.
## Readiness Potential (RP)
- The readiness potential (RP), a slow buildup of electrical potential recorded at the scalp using electroencephalography, has been associated with neural activity
involved in movement preparation. This [paper](https://www.sciencedirect.com/science/article/pii/S1053811919308778) explains pretty good what RP is. Acouple of images from my work is as below. The first image shows how brain signal behaves with different mode of [exoskeleton](https://ieeexplore.ieee.org/abstract/document/7419117) when the steps are taken by right leg. Green vertical line donates where the movement occurs. It can be seen that there is a huge reaction in brain signal when the movement starts. Second image depicts the the signal 1 second after move and 3 second before move. The last image illustrates the whole band pass filtered signal for C1 channel, where red lines represent the moves. We can see that the is huge response from brain when the patient starts moving. 

![1st walk Right legs](https://user-images.githubusercontent.com/101706254/193763420-f02b98de-fe57-4b63-aa97-18d7bfd85497.jpg)

![DigitMed_2018_4_2_84_239672_f1](https://user-images.githubusercontent.com/101706254/194016810-257726a4-7c8a-4421-a739-a26d23ca00e4.jpg)


![All segments of the C1 ](https://user-images.githubusercontent.com/101706254/193764098-dd7d2820-d0fc-46be-a5d0-a7c8f708c74a.png)

![C1 signal with filter](https://user-images.githubusercontent.com/101706254/193764122-9a8e4c38-3b57-4f1b-b97a-04c1b1f9cc68.png)

## Directed Transfer Function (DTF)
The Directed Transfer Function, the DTF, a causality measure used in  determination of brain connectivity patterns, characterizing  specific brain states/activities. Introduced by Kaminski and  Blinowska two decades ago, it was claimed that this measure is the proper measure for complexity involved in brain  activity, correcting incapacity and essential weakness of original Granger causality measures. This measure of brain connectivity reached broad application in very exciting brain activity modeling applied numerously by many research teams [1](https://ieeexplore.ieee.org/document/6339493). The best explanation of this work can be seen from this [article](https://braininformatics.springeropen.com/articles/10.1186/s40708-022-00154-8#:~:text=Directed%20transfer%20function%20%28DTF%29%20is%20good%20at%20characterizing,applied%20in%20discrimination%20of%20motor%20imagery%20%28MI%29%20tasks.). An example from my work is shown in the below figure. This image show the energy transfer from brain channels to heart (EMG1) and to leg muscles (EMG2). It can be seen that most of the energy transfer happens when the patient decide to move.
![090421 therapy plus Left Leg 3  segment-page-001](https://user-images.githubusercontent.com/101706254/193767991-93c8b3af-2abd-4a4d-8e76-a5e707aa4f6d.jpg)
