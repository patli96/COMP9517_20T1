# The Ground Truth

These files are grabbed directly from https://motchallenge.net/data/2D_MOT_2015/#download

## Structure of `det.txt`

For each line, it looks like:
```csv
748,-1,153,193,40.258,91.355,44.977,-10.9629,-2.60237,0
748,-1,613,212,47.48,107.74,34.517,-8.95412,-12.2105,0
748,-1,528,177,30.979,70.299,27.298,-5.0318,-8.51785,0
```

It is a comma separated array, where each value represents:
```csv
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

## Structure of `gt.txt`

For each line, it looks like:
```csv
780,8,244,170,27.63,71.997,1,-7.2877,-2.0376,0
781,1,308,199,26.227,75.578,1,-8.9816,-5.1148,0
781,2,611,204,29.915,94.195,1,-7.841,-11.566,0
```

It is a comma separated array, where each value represents:

```csv
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <is_considered>, <x>, <y>, <z>
```
