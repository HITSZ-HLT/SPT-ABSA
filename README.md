[**中文说明**](https://github.com/HITSZ-HLT/SPT-ABSA/) | [**English**](https://github.com/HITSZ-HLT/SPT-ABSA/blob/master/README_EN.md)

# SPT-ABSA

本仓库开源了以下论文的代码：

- 标题：An Empirical Study of Sentiment-Enhanced Pre-Training for Aspect-Based Sentiment Analysis
- 作者：Yice Zhang, Yifan Yang, Bin Liang, Shiwei Chen, Bing Qin, and Ruifeng Xu
- 会议：ACL-2023 Finding (Long)

模型权重可以在 [https://huggingface.co/zhang-yice/spt-absa-bert-400k](https://huggingface.co/zhang-yice/spt-absa-bert-400k) 获取。

### 简介


方面级情感分析（Aspect-Based Sentiment Analysis，ABSA）是情感分析中的一个重要问题。
其目标是从用户生成的内容中识别对特定方面的观点和情感。
许多研究工作利用预训练技术学习情感感知表示，并在多个ABSA任务上取得显著的性能提升。
我们对SPT-ABSA进行了实验性的分析，系统地探究和分析现有方法的有效性。

我们主要关注以下问题：
- (a) 不同类型的情感知识对下游的ABSA任务有何影响？
- (b) 哪种知识融合方法最有效？
- (c) 在预训练中注入非情感特定的语言知识（例如词性标记和句法关系）是否有积极影响？

基于对这些问题的探究，我们最终获得了一个强大的情感增强预训练模型。
这个强大的情感增强预训练模型有两个版本，分别是[zhang-yice/spt-absa-bert-400k](https://huggingface.co/zhang-yice/spt-absa-bert-400k) 和 [zhang-yice/spt-absa-bert-10k](https://huggingface.co/zhang-yice/spt-absa-bert-10k)，它集成了三种类型的知识：
- 方面词：掩码方面词的上下文并对其进行预测。
- 评论的评分：评分预测。
- 句法知识：
    - 词性，
    - 依存关系的方向，
    - 依存距离。
 
### 实验结果

<img width="75%" alt="image" src="https://github.com/HITSZ-HLT/SPT-ABSA/assets/9134454/38fc2db0-6ccf-47a7-a93c-cf54667e1a23">

<img width="75%" alt="image" src="https://github.com/HITSZ-HLT/SPT-ABSA/assets/9134454/20c5a976-014e-433f-a2ec-4bb259e5a382">

### 引用我们

@inproceedings{zhang-etal-2023-spt-absa,  
&emsp;&emsp;&emsp;&emsp;title = "An Empirical Study of Sentiment-Enhanced Pre-Training for Aspect-Based Sentiment Analysis",  
&emsp;&emsp;&emsp;&emsp;author = "Zhang, Yice  and  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Yang, yifan  and  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Liang, Bin  and  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Chen, Siwei  and  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Qin, Bing",  
&emsp;&emsp;&emsp;&emsp;booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",  
&emsp;&emsp;&emsp;&emsp;month = july,  
&emsp;&emsp;&emsp;&emsp;year = "2023",  
&emsp;&emsp;&emsp;&emsp;address = "Toronto, Canada",  
&emsp;&emsp;&emsp;&emsp;publisher = "Association for Computational Linguistics",  
}  
