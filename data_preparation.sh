#!/bin/bash

# if want to fuse the multi csv
task=tile
documents_root=...
slide_max_tiles=...

python utils/data_preparation.py --task $task --documents_root $documents_root --slide_max_tiles $slide_max_tiles

#

