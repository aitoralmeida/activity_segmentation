#!/bin/bash

python3 activity_change_recognition_eval_summary.py --folder $1 --offset 1 --train_or_test $2 --threshold_num $3
python3 activity_change_recognition_eval_summary.py --folder $1 --offset 5 --train_or_test $2 --threshold_num $3
python3 activity_change_recognition_eval_summary.py --folder $1 --offset 10 --train_or_test $2 --threshold_num $3