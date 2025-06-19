if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/Transformers" ]; then
    mkdir ./logs/LongForecasting/Transformers
fi

# Train each model 10 times with the 10 random seeds in paralel

# Random seeds (10)
# seeds=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
seeds=(10458 15227 3293 12890 15726 28649 25565 3144 32598 15349)
# seeds=(10458)

# Same seeds as Linear models

model_names="Transformer"

for model_name in $model_names
do
   pred_lengths=(96)
   ###### Dataset: exchange_rate 
   ###### Optimized
   for pred_len in ${pred_lengths[@]}
   do
      seq_len=25
      label_len=25
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset exchange rate with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path exchange_rate.csv \
            --model_id exchange_$seq_len_$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 8 \
            --dec_in 8 \
            --c_out 8 \
            --des 'Exp' \
            --itr 1 \
            --train_epochs 10 >logs/LongForecasting/Transformers/$model_name'_exchange_'$seq_len'_'$pred_len'_seed_'$seed.log
            # --train_epochs 1 >logs/LongForecasting/Transformers/$model_name'_exchange_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
      echo "Done."
   done


   ###### Dataset: ETTh1 
   ###### Optimized
   for pred_len in ${pred_lengths[@]}
   do
      seq_len=6
      label_len=6
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset ETTh1 with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path ETTh1.csv \
            --model_id ETTh1_$seq_len_$pred_len \
            --model $model_name \
            --data ETTh1 \
            --features M \
            --seq_len $seq_len \
            --label_len $label_len \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des 'Exp' \
            --itr 1  >logs/LongForecasting/Transformers/$model_name'_ETTh1_'$seq_len'_'$pred_len'_seed_'$seed.log
      done
      echo "Done."
   done

done