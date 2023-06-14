set -euf

DATA="SyntheticData"
rm -rf $DATA
mkdir -p $DATA
#TrainDATA="data/multi_view_training_dslr_undistorted"
TrainDATA="data"
#TestDATA="data/multi_view_test_dslr_undistorted"

# 生成train_data
python3 sample_data.py $TrainDATA/courtyard/dslr_calibration_undistorted  $DATA/train_data_courtyard.txt 4000
python3 sample_data.py $TrainDATA/playground/dslr_calibration_undistorted  $DATA/train_data_playground.txt 1000

# 生成test_data
python3 sample_data.py $TrainDATA/relief/dslr_calibration_undistorted $DATA/test_data.txt 10000

# 生成anchor_data
python3 sample_data.py $TrainDATA/courtyard/dslr_calibration_undistorted $DATA/anchor_data.txt 10000

#如果同伦延续能够从一个问题跟踪到另一个问题，则连接两个anchor的边
./BIN/connectivity ./$DATA/anchor_data.txt ./MODEL ./MODEL/trainParam.txt

#输入是上一步生成的连接图和anchor_data，输出为最终的anchor
./BIN/anchors ./$DATA/anchor_data.txt ./MODEL ./MODEL/trainParam.txt

# label 是生成数据集
# X_train.txt 包含训练数据的 14 维向量
# Y_train.txt 为对应的anchor的id
# 生成训练集
./BIN/labels ./$DATA/train_data_courtyard.txt ./MODEL ./$DATA train_courtyard ./MODEL/trainParam.txt
./BIN/labels ./$DATA/train_data_playground.txt ./MODEL ./$DATA train_playground ./MODEL/trainParam.txt

# 将来自不同数据集的训练集合并
cat ./$DATA/X_train_courtyard.txt ./$DATA/X_train_playground.txt > ./model/X_train.txt
cat ./$DATA/Y_train_courtyard.txt ./$DATA/Y_train_playground.txt > ./model/Y_train.txt

# 生成测试集
./BIN/labels ./$DATA/test_data.txt ./MODEL ./MODEL val ./MODEL/trainParam.txt

# 训练
python3 train_nn.py ./MODEL ./MODEL/trainParam.txt

# 验证
./BIN/evaluate ./$DATA/test_data.txt ./MODEL/ ./MODEL/trainParam.txt