# Komplet pc path: C:\Users\admin\Documents\Alberto_transformers\OneDrive - NTNU\PhD\Research cases\Transformers and time-series\Code\LTSF-Linear-main
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Train each model 10 times with the 10 random seeds in paralel

# Random seeds
# seeds=($RANDOM $RANDOM $RANDOM)
seeds=(12456 45678 97467)

# model_name=DLinear
model_names="Linear DLinear NLinear"
start=`date +%s`
for model_name in $model_names
do
   ###### Dataset: ETTh2
   ###### Optimizing...
   for pred_len in 96 192 336 720
   do
      seq_len=336
      lr=0.05
      bs=32
      if [[ $model_name -eq "DLinear" ]]
      then
         bs=16
         if [[ $pred_len -ge 336 ]]
         then
            lr=0.005
            bs=32
         fi
      fi
      for seed in ${seeds[@]} 
      do
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTh2.csv \
            --model_id ETTh2_$seq_len'_'$pred_len \
            --model $model_name \
            --data ETTh2 \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 7 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size $bs --learning_rate $lr --individual >logs/LongForecasting/$model_name'_I_'ETTh2_$seq_len'_'$pred_len'_seed_'$seed.log &

         echo "Training $model_name on dataset ETTh2 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
      done
   done
   wait
done
wait
echo "Done."
end=`date +%s`
echo "Execution time was `expr $end - $start` seconds."