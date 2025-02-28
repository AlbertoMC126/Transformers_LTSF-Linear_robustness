if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Train each model 10 times with the 10 random seeds in paralel

# Random seeds
seeds=($RANDOM)

# model_name=DLinear
model_names="Linear DLinear NLinear"
start=`date +%s`
for model_name in $model_names
do

   ###### Dataset: exchange_rate 
   ###### Optimizing...
   for pred_len in 96 192 336 720
   do
      seq_len=336
      lr=0.0005
      for seed in ${seeds[@]} 
      do
         bs=8
         if [[ $pred_len -gt 300 ]]
         then
            bs=32
         fi

         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path exchange_rate.csv \
            --model_id Exchange_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 8 \
            --train_epochs 10 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size $bs --learning_rate $lr --individual >logs/LongForecasting/$model_name'_I_'exchange_$seq_len'_'$pred_len'_seed_'$seed.log

         echo "$model_name trained on dataset exchange_rate with seq_len $seq_len, pred_len $pred_len and seed $seed..."
      done
   done
done
wait
echo "Done."
end=`date +%s`
echo "Execution time was `expr $end - $start` seconds."