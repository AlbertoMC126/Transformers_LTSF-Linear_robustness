if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/Linear" ]; then
    mkdir ./logs/LongForecasting/Linear
fi

# Train each model 10 times with the 10 random seeds in paralel

# Random seeds (10)
# seeds=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
seeds=(10458 15227 3293 12890 15726 28649 25565 3144 32598 15349)

# For large datasets
# seeds_1=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
# seeds_1=(10458 15227 3293 12890 15726)
# seeds_2=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
# seeds_2=(28649 25565 3144 32598 15349)

model_names="Linear DLinear NLinear"

for model_name in $model_names
do
   ###### Dataset: electricity
   ###### Optimized
   for pred_len in 96 192 336 720
   # for pred_len in 96 720
   do
      seq_len=336
      for seed in ${seeds[@]} 
      do
         echo "Training $model_name on dataset electricity with seq_len $seq_len, pred_len $pred_len and seed $seed..."
         python -u run_longExp.py \
            --is_training 1 \
            --seed $seed \
            --num_workers 0 \
            --root_path ./dataset/ \
            --data_path electricity.csv \
            --model_id Electricity_$seq_len'_'$pred_len \
            --model $model_name \
            --data custom \
            --features M \
            --seq_len $seq_len \
            --pred_len $pred_len \
            --enc_in 321 \
            --train_epochs 20 --patience 5 \
            --des 'Exp' \
            --itr 1 --batch_size 16  --learning_rate 0.005 --individual >logs/LongForecasting/Linear/$model_name'_I_'electricity_$seq_len'_'$pred_len'_seed_'$seed.log
      done
      # wait
      # echo "Done."
   done
done
echo "Training of $model_name models on dataset electricity finished."
echo "Experiments finished"