set -euf

DATA="SyntheticData3"
rm -rf $DATA
mkdir -p $DATA
#TrainDATA="data/multi_view_training_dslr_undistorted"
TrainDATA="data"
#TestDATA="data/multi_view_test_dslr_undistorted"
trainset=("courtyard" "electro" "kicker" "meadow" "office" "pipes" "playground" "relief" "relief_2" "terrace" "terrains" "botanical_garden" "boulders" "bridge" "door" "exhibition_hall" "lecture_room" "living_room" "lounge" "observatory" "old_computer" "statue" "terrace_2")
testset=("delivery_area" "facade")
anchorset=("office" "terrains")

# 生成train_data
#python3 sample_data.py $TrainDATA/courtyard/dslr_calibration_undistorted  $DATA/train_data_courtyard.txt 4000
#python3 sample_data.py $TrainDATA/playground/dslr_calibration_undistorted  $DATA/train_data_playground.txt 1000

for train in ${trainset[@]};do
    python3 sample_data.py $TrainDATA/$train/dslr_calibration_undistorted $DATA/train_data_$train.txt 1000000
done

# 生成test_data
#python3 sample_data.py $TrainDATA/relief/dslr_calibration_undistorted $DATA/test_data.txt 10000

for test in ${testset[@]};do
    python3 sample_data.py $TrainDATA/$test/dslr_calibration_undistorted $DATA/test_data_$test.txt 30000
done

# 生成anchor_data
#python3 sample_data.py $TrainDATA/courtyard/dslr_calibration_undistorted $DATA/anchor_data.txt 10000

#for anchor in ${anchorset[@]};do
#    python3 sample_data.py $TrainDATA/$anchor/dslr_calibration_undistorted $DATA/anchor_data_$anchor.txt 20000
#done

# 合并anchor_data
#echo "" > $DATA/anchor_data.txt
#total_lines=0
#for anchor in ${anchorset[@]};do
#    first_line=$(head -n 1 $DATA/anchor_data_$anchor.txt)
#    num_lines=$(echo $first_line | awk '{print $1}')
#    total_lines=$((total_lines + num_lines))
#    awk "NR>1" $DATA/anchor_data_$anchor.txt >> $DATA/anchor_data.txt
#done
#sed -i 1s/^/$total_lines/ $DATA/anchor_data.txt

#如果同伦延续能够从一个问题跟踪到另一个问题，则连接两个anchor的边
#./bin/connectivity ./$DATA/anchor_data.txt ./MODEL ./MODEL/trainParam.txt

#输入是上一步生成的连接图和anchor_data，输出为最终的anchor
./bin/anchors ./$DATA/anchor_data.txt ./MODEL ./MODEL/trainParam.txt

# label 是生成数据集
# X_train.txt 包含训练数据的 14 维向量
# Y_train.txt 为对应的anchor的id
# 生成训练集
#./bin/labels ./$DATA/train_data_courtyard.txt ./MODEL ./$DATA train_courtyard ./MODEL/trainParam.txt
#./bin/labels ./$DATA/train_data_playground.txt ./MODEL ./$DATA train_playground ./MODEL/trainParam.txt
for train in ${trainset[@]};do
    ./bin/labels ./$DATA/train_data_$train.txt ./MODEL ./$DATA train_$train ./MODEL/trainParam.txt
done

# 将来自不同数据集的训练集合并
#cat ./$DATA/X_train_courtyard.txt ./$DATA/X_train_playground.txt > ./MODEL/X_train.txt
#cat ./$DATA/Y_train_courtyard.txt ./$DATA/Y_train_playground.txt > ./MODEL/Y_train.txt
echo -e "\c" > ./MODEL/X_train.txt
echo -e "\c" > ./MODEL/Y_train.txt
for train in ${trainset[@]};do
    cat ./$DATA/X_train_$train.txt >> ./MODEL/X_train.txt
    cat ./$DATA/Y_train_$train.txt >> ./MODEL/Y_train.txt
done

# 生成测试集
#./bin/labels ./$DATA/test_data.txt ./MODEL ./MODEL val ./MODEL/trainParam.txt
for test in ${testset[@]};do
    ./bin/labels ./$DATA/test_data_$test.txt ./MODEL ./$DATA test_$test ./MODEL/trainParam.txt
done
echo -e "\c" > ./MODEL/X_val.txt
echo -e "\c" > ./MODEL/Y_val.txt
for test in ${testset[@]};do
    cat ./$DATA/X_test_$test.txt >> ./MODEL/X_val.txt
    cat ./$DATA/Y_test_$test.txt >> ./MODEL/Y_val.txt
done

# 训练
#python3 train_nn.py ./MODEL ./MODEL/trainParam.txt