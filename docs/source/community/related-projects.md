# Related projects

Ecosystem overview, we will only list open-source projects here.
We will list some project that `movement` relates to, clarify what `movement` is not, so you can use one of the other options.

## Upstream of `movement`: motion tracking tools

There is a growing number of computer vision-based tools—such as [DeepLabCut](dlc:), [SLEAP](sleap:) and [LightningPose](lp:)—that enable researchers to track animal movements in video recordings without the need for physical markers.

__`movement` sits downstream of motion tracking tools: those tools produce motion tracks, while movement performs the analysis.__

`movement` provides convenient functions for loading data from files written by many of the popular motion tracking tools, see the [Input/Output](target-io) section for details.

There is a number of tools we don't yet support natively, but the list of supported tools in expanding. If there is a specific tool that you think is within our [scope](target-scope) and you would like us to support loading from, [get in touch](target-get-in-touch)!


## Related to `movement`: tool for analysing motion tracks

The extraction of motion tracks is often just the beginning of the analysis. Researchers use these tracks to investigate various aspects of behaviour, such as kinematics, social navigation and spatial navigation.

`movement` sits squarely within this category, but there are some other tools with overlapping functionalities that you might want to check out as well.


### Inspiration

Their >8k combined citations underline a fast-growing need for post-tracking software, and movement is currently the most versatile and actively maintained open-source option. DLC2Kinematics [1] and PyRat [2] offer some overlapping functionality but handle only DeepLabCut output and have been largely dormant for two years. A nascent R package, animovement [3], is beginning to address a similar remit; we collaborate with its developer on shared data standards, offering Python and R users complementary—not competing—tools.

The following projects cover related needs and served as inspiration for this project:
* [DLC2Kinematics](https://github.com/AdaptiveMotorControlLab/DLC2Kinematics)
* [PyRat](https://github.com/pyratlib/pyrat)
* [Kino](https://github.com/BrancoLab/Kino)
* [WAZP](https://github.com/SainsburyWellcomeCentre/WAZP)


https://simba-uw-tf-dev.readthedocs.io/en/latest/ : probably has the most overlap, very feature complete, does a bit of everything

https://github.com/benlansdell/ethome

https://github.com/mahan-hosseini/AutoGaitA

https://megabouts.ai/

https://rupertoverall.net/Rtrack/index.html
https://rupertoverall.net/ColonyTrack/
https://swarm-lab.github.io/swaRm/
https://github.com/JimMcL/trajr

Mostly classification?

https://github.com/neuroethology/MARS


### Non-video motion tracks

Motion tracks don't always come from video data.

If you track large-scale animal movements (e.g., migration patterns) using GPS or other geospatial technologies, you might want to explore specialized tools for analysing such data:

- [movingpandas](https://github.com/movingpandas/movingpandas)
- [traja](https://github.com/traja-team/traja)
- [scikit-mobility](https://scikit-mobility.github.io/scikit-mobility/)


If you work with marker-based motion capture systems and inertial measurement units (IMUs), you might find these packages useful:

- [pyomeca](https://pyomeca.github.io/)


## Behaviour classification

A separate category of behaviour classification tools use machine-learning frameworks to segment tracks into discrete actions. To maintain easy installation and a light dependency chain, movement deliberately focuses on I/O, cleaning and feature extraction—tasks these classifiers increasingly delegate to us. Two such tools, [VAME](https://roald-arboel.com/animovement
https://github.com/LINCellularNeuroscience/VAME) and [LISBET](https://github.com/BelloneLab/lisbet), already depend on movement. As more formats are added, movement will further strengthen its role as the hub between open-source trackers and classifiers.
