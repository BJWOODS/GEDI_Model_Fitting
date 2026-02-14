# GEDI Waveform Fitting (Cubic B-Splines)

**Author:** Brandon J. Woodard\
**Institution:** Brown University

This script fits modeled laser return waveforms to NASA GEDI
full-waveform LiDAR data using **cubic B-splines** and nonlinear least
squares.

------------------------------------------------------------------------

## What is GEDI?

GEDI (Global Ecosystem Dynamics Investigation) is a NASA full-waveform
LiDAR instrument aboard the International Space Station.\
It measures forest vertical structure by recording transmitted and
returned laser pulse energy over nanoseconds.

![GEDI on
ISS](https://www.eoportal.org/api/cms/documents/163813/6584772/GEDI.jpeg/b48570e3-15f7-e4d9-1e6d-09cbb91496fb?t=1670129680165)
![Example GEDI
Waveform](https://gedi.umd.edu/wp-content/uploads/2020/01/New_GEDI_data-259x258.png)

------------------------------------------------------------------------

## What This Script Does

-   Loads GEDI waveform CSV data\
-   Builds **cubic B-spline models** of signals
    (`scipy.interpolate.splrep`, k=3)\
-   Convolves transmit pulse with a Gaussian\
-   Fits modeled signal to GEDI return waveform\
-   Estimates:
    -   Vertical & horizontal shifts
    -   Scaling parameters
    -   Peak locations
    -   Goodness-of-fit (Chi-square)

------------------------------------------------------------------------

## Requirements

Install dependencies:

``` bash
pip install numpy scipy matplotlib pandas scikit-learn numba
```

------------------------------------------------------------------------

## How to Run

1.  Place your GEDI waveform CSV in the same directory.
2.  Set filename in the script:

``` python
filename = "your_file.csv"
```

3.  Run:

``` bash
python script_name.py
```

The script will:

-   Plot the GEDI return waveform
-   Plot the fitted model
-   Print fitted parameters and chi-square error

------------------------------------------------------------------------

## Core Modeling Idea

The modeled return signal:

R(t) = vs + vsc · Spline(hsc · t + hs)

Where parameters control vertical shift, horizontal shift, and scaling.

The transmit pulse is convolved with a Gaussian to approximate system
broadening.

------------------------------------------------------------------------

## Notes

-   Uses cubic B-splines (not raw waveforms).
-   Designed for GEDI-like CSV waveform exports.
-   Intended for research / signal analysis experimentation.

------------------------------------------------------------------------
