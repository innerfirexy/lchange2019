#!/usr/bin/env bash
input=$1
output_prefix=$2
output_postfix=$3
vocab=$4
model=$5
cbow=$6
wll=$7
mc=$8
iter=$9
# processes=${10}
# l2=${11}
lang=${10}
gpu=${11}

# For printing a new line
printf_new() {
    str=$1
    num=$2
    v=$(printf "%-${num}s" "$str")
    echo "${v// /*}"
}

if [[ -f "$vocab" ]]; then
    vocab_opt="--vocab_data"
else
    vocab_opt="--save_vocab"
fi

if [ ${cbow} = 1 ]; then
    cbow_str="cbow"
elif [ ${cbow} = 0 ]; then
    cbow_str="sg"
fi

pretrain_output="${output_prefix}_pretrain_${cbow_str}_wll${wll}_mc${mc}_iter${iter}.txt"
pretrain_emb0=$output_prefix"_pretrain_${cbow_str}_wll"$wll"_mc"$mc"_iter"$iter"_emb0.txt"
pretrain_emb1=$output_prefix"_pretrain_${cbow_str}_wll"$wll"_mc"$mc"_iter"$iter"_emb1.txt"
pretrain_emb0char=$output_prefix"_pretrain_${cbow_str}_wll"$wll"_mc"$mc"_iter"$iter"_emb0char.txt"

if [[ "$model" = "CWE" ]]; then
    final_output=$output_prefix"_CWE_${cbow_str}_wll"$wll"_mc"$mc"_iter"$iter"_"$output_postfix".txt"
elif [[ "$model" = "DSE" ]]; then
    final_output=$output_prefix"_DSE_${cbow_str}_wll"$wll"_mc"$mc"_iter"$iter"_"$output_postfix".txt"
fi

echo "input="$input
echo "output_prefix="$output_prefix
echo "output_postfix="$output_postfix
echo "vocab="$vocab
echo "model=${model}, cbow=${cbow}"
echo "wll="$wll", mc="$mc", iter="$iter", processes=4"
echo "Using pretrain_emb0: ${pretrain_emb0}"
echo "Using pretrain_emb1: ${pretrain_emb1}"
echo "Using pretrain_emb0char: ${pretrain_emb0char}"
echo "Final output file: "$final_output
printf_new "*" 64

# pretrain_word
echo "Start stage 1: pretrain word"
if [ -f "$pretrain_emb0" ] && [ -f "$pretrain_emb1" ]; then
    echo "Pretrained emb0 and emb1 already exist"
else
    python3 main.py --train $input $vocab_opt $vocab \
    --output $pretrain_output --output_emb1 1 --output_emb0_char 0 \
    --model CWE --cbow $cbow --pretrain_word \
    --wordlen_lim $wll --min_count $mc --iter $iter \
    --lang $lang --cuda --gpu_id $gpu
fi
echo "Done stage 1"
printf_new "*" 64

# pretrain_char
echo "Start stage 2: pretrain char"
if [ -f "$pretrain_emb0char" ] && [ -f "$pretrain_emb0char" ]; then
    echo "Pretrained emb0char already exists"
else
    python3 main.py --train $input --vocab_data $vocab \
    --output $pretrain_output --output_emb0_char 1 --output_emb0 0 \
    --pt_emb1 $pretrain_emb1 --fix_emb1 \
    --model CWE --cbow $cbow --pretrain_char \
    --wordlen_lim $wll --min_count $mc --iter $iter \
     --lang $lang --cuda --gpu_id $gpu
fi
echo "Done stage 2"
printf_new "*" 64

# train
echo "Start stage 3: training model"
python3 main.py --train $input --vocab_data $vocab \
--output $final_output \
--pt_emb0 $pretrain_emb0 \
--pt_emb1 $pretrain_emb1 \
--pt_emb0_char $pretrain_emb0char \
--model $model --cbow $cbow \
--wordlen_lim $wll --min_count $mc --iter $iter \
 --lang $lang --cuda --gpu_id $gpu
# --output_per_step
echo "Done stage 3"
echo "Final output file: "$final_output
