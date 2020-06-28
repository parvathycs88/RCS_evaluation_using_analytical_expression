# Evaluation of RCS using analytical expression

RCS needs to be calculated for an array of 10 by 10. The elements are 2 v-shaped elements with phase difference 180 degree.

## Data Pre-processing

Input dataset - Nearly 1200000 data instanpresent in 2 excel files each with 600,000 rows. Dataframe helps to merge the two excel files and work with for further processing.

Columns - Opening angle, Length, Frequency and Reflection phase.

Phase warping results in phase discontinuity. This is removed with help of numpy unwrap() function. It will work only for angles in radians. Hence angles converted from degrees to radians.

Dictionary helps to apply unwrapping to every combination individually.

### Prerequisites

* Python 

* Input dataset
