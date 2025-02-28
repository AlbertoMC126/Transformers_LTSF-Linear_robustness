cd FEDformer

if [ ! -d "../logs" ]; then
    mkdir ../logs
fi

if [ ! -d "../logs/LongForecasting" ]; then
    mkdir ../logs/LongForecasting
fi

if [ ! -d "../logs/LongForecasting/Transformers" ]; then
    mkdir ../logs/LongForecasting/Transformers
fi

seeds=(10458 15227 3293 12890 15726 28649 25565 3144 32598 15349)
# seeds=(10458)

for seed in ${seeds[@]} 
do

  for pred_len in 96 192 336 720
  do
  # ETTm1
  echo "Training FEDformer on dataset ETTm1 with seq_len 96, pred_len $pred_len and seed $seed..."
  python -u run.py \
    --is_training 1 \
    --seed $seed \
    --num_workers 0 \
    --data_path ETTm1.csv \
    --task_id ETTm1 \
    --model FEDformer \
    --data ETTm1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 512 \
    --itr 1 >../logs/LongForecasting/Transformers/FEDformer_ETTm1_'96'_$pred_len'_seed_'$seed.log

  # ETTh1
  echo "Training FEDformer on dataset ETTh1 with seq_len 96, pred_len $pred_len and seed $seed..."
  python -u run.py \
    --is_training 1 \
    --seed $seed \
    --num_workers 0 \
    --data_path ETTh1.csv \
    --task_id ETTh1 \
    --model FEDformer \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 512 \
    --itr 1 >../logs/LongForecasting/Transformers/FEDformer_ETTh1_'96'_$pred_len'_seed_'$seed.log

  # ETTm2
  echo "Training FEDformer on dataset ETTm2 with seq_len 96, pred_len $pred_len and seed $seed..."
  python -u run.py \
    --is_training 1 \
    --seed $seed \
    --num_workers 0 \
    --data_path ETTm2.csv \
    --task_id ETTm2 \
    --model FEDformer \
    --data ETTm2 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 512 \
    --itr 1 >../logs/LongForecasting/Transformers/FEDformer_ETTm2_'96'_$pred_len'_seed_'$seed.log

  # ETTh2
  echo "Training FEDformer on dataset ETTh2 with seq_len 96, pred_len $pred_len and seed $seed..."
  python -u run.py \
    --is_training 1 \
    --seed $seed \
    --num_workers 0 \
    --data_path ETTh2.csv \
    --task_id ETTh2 \
    --model FEDformer \
    --data ETTh2 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 512 \
    --save_pred_values false \
    --itr 1 >../logs/LongForecasting/Transformers/FEDformer_ETTh2_'96'_$pred_len'_seed_'$seed.log

  # electricity
  echo "Training FEDformer on dataset electricity with seq_len 96, pred_len $pred_len and seed $seed..."
  python -u run.py \
    --is_training 1 \
    --seed $seed \
    --num_workers 0 \
    --data_path electricity.csv \
    --task_id ECL \
    --model FEDformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --itr 1 >../logs/LongForecasting/Transformers/FEDformer_electricity_'96'_$pred_len'_seed_'$seed.log

  # exchange
  echo "Training FEDformer on dataset exchange rate with seq_len 96, pred_len $pred_len and seed $seed..."
  python -u run.py \
    --is_training 1 \
    --seed $seed \
    --num_workers 0 \
    --data_path exchange_rate.csv \
    --task_id Exchange \
    --model FEDformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --itr 1 >../logs/LongForecasting/Transformers/FEDformer_exchange_'96'_$pred_len'_seed_'$seed.log

  # traffic
  echo "Training FEDformer on dataset traffic with seq_len 96, pred_len $pred_len and seed $seed..."
  python -u run.py \
    --is_training 1 \
    --seed $seed \
    --num_workers 0 \
    --data_path traffic.csv \
    --task_id traffic \
    --model FEDformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --des 'Exp' \
    --itr 1 \
    --itr 1 >../logs/LongForecasting/Transformers/FEDformer_traffic_'96'_$pred_len'_seed_'$seed.log

  # weather
  echo "Training FEDformer on dataset weather with seq_len 96, pred_len $pred_len and seed $seed..."
  python -u run.py \
    --is_training 1 \
    --seed $seed \
    --num_workers 0 \
    --data_path weather.csv \
    --task_id weather \
    --model FEDformer \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 >../logs/LongForecasting/Transformers/FEDformer_weather_'96'_$pred_len'_seed_'$seed.log
  done


  for pred_len in 24 36 48 60
  do
  # illness
  echo "Training FEDformer on dataset national illness (ILI) with seq_len 96, pred_len $pred_len and seed $seed..."
  python -u run.py \
    --is_training 1 \
    --seed $seed \
    --num_workers 0 \
    --data_path national_illness.csv \
    --task_id ili \
    --model FEDformer \
    --data custom \
    --features M \
    --seq_len 36 \
    --label_len 18 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 >../logs/LongForecasting/Transformers/FEDformer_ILI_'36'_$pred_len'_seed_'$seed.log
  done

done