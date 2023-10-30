DATASET_LIST="
ETTh1
"


GPU=0

SEED=42
MAX_TRAIN_LENGTH=200
MAX_SHIFT_SIZE=3
TEMPORAL_MASK="binomial"
FEATURE_MASK="all_true"
RUN_NAME="ETT"
ITERS_FACTOR=2

i=1
STARTDATE=`date +"%Y-%m-%d %T"`
for DATASET in $DATASET_LIST ; do
    echo "dataset" $i ":" $DATASET
    i=`expr $i + 1`
    RESULT_NAME="ETT_${DATASET}_seed_${SEED}"
    echo ">>> RUN NAME " $RESULT_NAME
    python -u train.py --dataset $DATASET --run_name $RUN_NAME --loader forecast_csv --batch_size 8\
                      --repr_dims 320 --max_threads 8 --seed $SEED --eval --gpu $GPU\
                      --result_name $RESULT_NAME \
                      --iters_factor $ITERS_FACTOR\
                      --max_shift_size $MAX_SHIFT_SIZE\
                      --temporal_mask_mode $TEMPORAL_MASK\
                      --feature_mask_mode $FEATURE_MASK\
                      --max_train_length $MAX_TRAIN_LENGTH
done
echo START Date and Time is: $STARTDATE
echo FINISH Date and Time is: `date +"%Y-%m-%d %T"`

