# TODO
- [] If wanted create an env variable for output paths, models name and params.
compute_midprice_direction_matrix  in metrics.py expects a price series, but it's currently being passed a list of tick changes. Additionally, it should use the conditioning state quantities ($Q_t$) rather than the next state quantities ($Q_{t+1}$) to correlate imbalance with the subsequent move.

# COMPLETED TASKS TO REMEMBER 
- To remove jumps between days, offset the data to the last datapoint and make it "start" from there
- The txt in the data folder has now only the last stock, and it's already transposed. If interested in reconstructing from the beginning, please download the full FI-2010 dataset. The csv instead can be re-created by chaning the number of ticks for jumps to be considered.