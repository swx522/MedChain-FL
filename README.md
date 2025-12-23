BC-FL-Med: Blockchain-Logged Federated LLM for Trustworthy Medical AI
课程项目 · 区块链导论（2025）
基于去中心化联邦学习 + 轻量级区块链日志的可信医疗大语言模型微调系统

📌 项目简介
本项目实现了一个去中心化的联邦学习系统，用于在不共享原始医疗数据的前提下，协作微调 BioBERT 大语言模型，完成医疗文本分类任务（40 类专科分类）。
我们聚焦真实挑战：
Non-IID 数据：模拟各医院数据分布不均（如某院只看“儿科”）
异常客户端：存在低质量或恶意节点
通信效率：大模型参数传输开销大
为此，我们集成PageRank 异常检测机制，自动识别并剔除不可靠客户端，并通过轻量级区块链日志记录关键事件（如模型提交、节点剔除），实现可审计、抗抵赖、去信任化的联邦学习过程。
✅ 本项目基于论文 Building Communication Efficient Asynchronous Peer-to-Peer Federated LLMs with Blockchain (BC-FL) 改进，非简单复现，而是聚焦“可信日志”与“鲁棒聚合”两大核心。

🚀 核心特性
功能说明
✅ 联邦学习框架
使用 Flower + Hugging Face，支持 BioBERT 模型微调
✅ Non-IID 数据划分
医疗数据按客户端切分，贴近真实场景
✅ PageRank 异常检测
基于客户端本地准确率构建图，自动剔除低分节点
✅ 区块链日志（关键！）
所有关键事件（提交、剔除、路径更新）记录为不可篡改的 JSON 日志，体现区块链在联邦学习中的审计价值
✅ 轻量高效
无需部署真实区块链（如 Hyperledger），用模拟日志满足教学目标

📂 项目结构
BC-FL-Med/
├── src/
│   └── Serverlesscase/
│       ├── Serverless_NonIID_Medical_transcriptions.py  <-- 主程序
│       └── ...                                           # 其他实验脚本
├── data/
│   └── mtsamples/                                        # 医疗文本数据集说明
├── results/
│   ├── blockchain_log.json                              <-- 区块链日志（关键产出！）
│   └── global_accuracies.txt                            # 全局准确率曲线
├── demo_video.mp4                                       # 演示视频
└── README.md

▶️ 快速运行
环境依赖
pip install -r requirements.txt
# 主要依赖：torch, transformers, datasets, flwr, evaluate, networkx
运行主实验
cd src/Serverlesscase
python Serverless_NonIID_Medical_transcriptions.py
输出说明
控制台将打印每轮 全局准确率（如 Global Model Accuracy: 74.20%）
最终生成 blockchain_log.json，包含：
[
  {"event": "pagerank_scores", "details": {"scores": [0.92, 0.21, 0.88, ...]}},
  {"event": "clients_removed", "details": {"client_ids": [1]}}
]

📝 课程交付物
可运行原型：Serverless_NonIID_Medical_transcriptions.py
区块链日志：blockchain_log.json（体现区块链作用）
技术文档：3–5 页说明（系统设计、PageRank 改进、日志机制）
演示视频：demo_video.mp4（含代码运行 + 日志展示）

🙏 致谢
感谢 Flower 和 Hugging Face 提供的联邦学习与 LLM 工具链
感谢论文作者开源 BC-FL 的工作
本项目为 《区块链导论》课程小组作业，仅供教学演示
