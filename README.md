# AI4S CNS挑战赛 - 靶向分子研发与合成规划智能体

## 项目概述
参加第四届世界科学智能大赛AI4S智能体CNS挑战赛，任务2：靶向分子研发与合成规划智能体。

## 核心功能
1. **分子生成**: 基于靶点结构生成候选药物分子
2. **虚拟筛选**: 评估分子与靶点的结合能
3. **性质筛选**: 过滤不合理的分子结构
4. **逆合成分析**: 规划分子的合成路线
5. **结果输出**: 生成符合要求的CSV和LOG文件

## 技术架构
- **分子生成**: REINVENT / MolGPT / 基于规则的生成
- **分子对接**: AutoDock Vina / DiffDock
- **分子处理**: RDKit
- **逆合成分析**: Retro* / ASKCOS / LLM辅助
- **LLM API**: 支持多种大语言模型

## 项目结构
```
ai4s-drug-agent/
├── main.py              # 主入口
├── core/                # 核心模块
│   ├── agent.py         # 智能体主逻辑
│   ├── molecule_generator.py  # 分子生成
│   ├── docking.py       # 分子对接
│   ├── retrosynthesis.py # 逆合成分析
│   ├── evaluator.py     # 分子评估
│   └── utils.py         # 工具函数
├── data/                # 数据目录
│   └── target.pdb       # 靶点PDB文件
├── output/              # 输出目录
├── logs/                # 日志目录
└── models/              # 预训练模型
```

## 使用方法
```bash
# 安装依赖
pip install -r requirements.txt

# 运行智能体
python main.py --target data/target.pdb --output output/result.csv
```

## 评分维度
1. **药物小分子**
   - 与靶点的结合能 (越低越好)
   - 分子结构合理性
   - 可合成性

2. **合成路线**
   - 起始原料的可及性
   - 合成路线的经济性
