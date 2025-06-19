# Desktop pc path: C:\Users\admin\Documents\Alberto_transformers\OneDrive - NTNU\PhD\Research cases\Transformers and time-series\Code\LTSF-Linear-main
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/Transformers" ]; then
    mkdir ./logs/LongForecasting/Transformers
fi

# Random seeds
# seeds=($RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM $RANDOM)
seeds=($RANDOM $RANDOM $RANDOM)
echo ${seeds[*]}

model_names="Autoformer Informer Transformer"
start0=`date +%s`

for model_name in $model_names
do
   start1=`date +%s`
   seq_len=96
   ###### Dataset: traffic 
   ###### Optimizing...
   for pred_len in 96
   do
      for seed in ${seeds[@]} 
      do
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
            --label_len 48 \
            --pred_len $pred_len \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --enc_in 8 \
            --dec_in 8 \
            --c_out 8 \
            --des 'Exp' \
            --itr 1 \
            --train_epochs 1 >logs/LongForecasting/Transformers/$model_name'_exchange_'$seq_len'_'$pred_len'_seed_'$seed.log &

         echo "Training $model_name on dataset exchange rate with seq_len $seq_len, pred_len $pred_len and seed $seed..."
      done
      # Finished seed loop
      # wait
   done
   # Finished pred_len loop
   wait
   end1=`date +%s`
   echo "Execution time for all $model_name was `expr $end1 - $start1` seconds."
done
wait
echo "Done."
end0=`date +%s`
echo "Execution time was `expr $end0 - $start0` seconds."