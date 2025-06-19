if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/Linear" ]; then
    mkdir ./logs/LongForecasting/Linear
fi

seeds=(15349)


model_names="NLinear"

for model_name in $model_names
do
   ###### Dataset: ETTh2 
   ###### Optimized (underperforming with respect to results from orignal work)
   for pred_len in 192
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
         echo "Training $model_name on dataset ETTh2 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTh2.csv \
            --model_id ETTh2_$seq_len'_'$pred_len \
            --model $model_name \
            --data ETTh2 \
            --save_pred_values True \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 7 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size $bs --learning_rate $lr --individual >logs/LongForecasting/Linear/$model_name'_I_'ETTh2_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      echo "Done."
   done
done