model="mlp_mixer"
ckpt_name="hidden_500"

python3.8 evaluation.py --model $model --ckpt_name $ckpt_name

./bin/evaluate_T ./23M/Y_train.txt ./result/$model/$ckpt_name.txt ./23M ./23M/trainParam.txt
