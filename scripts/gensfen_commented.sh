#!/bin/bash

# Using this commit:
# https://github.com/Sopel97/Stockfish.git
# commit d7d4ec211f7ef35ff39fe8aea54623a468b36c7d

DEPTH=5
GAMES=128000000
SEED=$RANDOM
 
options="
uci  # uci模式
setoption name PruneAtShallowDepth value false     # 关闭浅层剪枝, 让引擎在低深度时保留更多搜索分支, 利于多样化局面
setoption name Use NNUE value pure                 # 强制使用 NNUE(神经网络评估)模式, 而非传统评估函数
setoption name Threads value 250                   # 使用 250 个线程(显然是为高并发环境或集群机器设计)
setoption name Hash value 10240                    # 分配 10GB 哈希表内存(10240 MB)
setoption name SyzygyPath value /dev/shm/vjoost/3-4-5-6/WDL/:/dev/shm/vjoost/3-4-5-6/DTZ/  # 设置残局表库路径, 用于精确残局评估
isready  # 等待初始化完成

# gensfen 是 Stockfish 的一个扩展命令(不属于标准 UCI), 
# 用于批量生成随机或自对弈的 SFEN(Shogi FEN, 或类似格式) 局面。
# 在强化学习中, 这些局面可作为训练样本。
gensfen
    set_recommended_uci_options  # 使用推荐的 UCI 参数
    ensure_quiet  # 确保输出时不包含调试或噪声信息
    random_multi_pv 4  # 随机生成 4 个不同的主变(PV)线路
    random_multi_pv_diff 50  # 允许多PV之间评估差异最大为50分(增加多样性)
    random_move_count 8  # 每局最多随机8步
    random_move_maxply 20  # 随机探索最多20个半回合(ply)
    write_minply 5  # 只写出至少经过5个半回合的局面
    eval_limit 1000  # 限制评估节点上限, 控制计算开销
    seed $SEED  # 随机种子
    depth $DEPTH  # 搜索深度
    loop $GAMES   # 迭代次数(生成局面数量)
    output_file_name d${DEPTH}_${GAMES}_${SEED}"  # 输出文件名, 例如：d5_128000000_12345

printf "$options" | ./stockfish